import librosa
import librosa.filters
import math
import numpy as np
import tensorflow as tf
import soundfile as sf
import scipy
from hparams import hparams
from scipy.signal import resample

def load_wav(path):
  return librosa.core.load(path, sr=hparams.sample_rate)[0]


def save_wav_resmaple(wav, path):
    # Resample to 44 kHz
    target_sample_rate = 44000
    if hparams.sample_rate != target_sample_rate:
        num_samples = int(len(wav) * float(target_sample_rate) / hparams.sample_rate)
        wav_resampled = resample(wav, num_samples)
    else:
        wav_resampled = wav
    playback_speed = 1.15
    new_length = int(len(wav_resampled) * playback_speed)
    wav_playback_speed_changed = resample(wav_resampled, new_length)
    # Normalize the audio
    wav_resampled  *= 32767 / max(0.01, np.max(np.abs(wav_resampled)))

    # Save the WAV file
    sf.write(file=path, data=wav_resampled.astype(np.int16), samplerate=target_sample_rate, format='WAV')

def save_wav(wav, path):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  sf.write(file=path, data=wav.astype(np.int16), samplerate=hparams.sample_rate, format='WAV')


def preemphasis(x):
  return scipy.signal.lfilter([1, -hparams.preemphasis], [1], x)


def inv_preemphasis(x):
  return scipy.signal.lfilter([1], [1, -hparams.preemphasis], x)


def spectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
  return _normalize(S)


def inv_spectrogram(spectrogram):
  '''Converts spectrogram to waveform using librosa'''
  S = _db_to_amp(_denormalize(spectrogram) + hparams.ref_level_db)  # Convert back to linear
  return inv_preemphasis(_griffin_lim(S ** hparams.power))          # Reconstruct phase


def inv_spectrogram_tensorflow(spectrogram):
  '''Builds computational graph to convert spectrogram to waveform using TensorFlow.

  Unlike inv_spectrogram, this does NOT invert the preemphasis. The caller should call
  inv_preemphasis on the output after running the graph.
  '''
  S = _db_to_amp_tensorflow(_denormalize_tensorflow(spectrogram) + hparams.ref_level_db)
  return _griffin_lim_tensorflow(tf.pow(S, hparams.power))


def melspectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
  return _normalize(S)


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(hparams.sample_rate * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = _db_to_amp(threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
      return x + hop_length
  return len(wav)


def _griffin_lim(S):
  '''librosa implementation of Griffin-Lim
  Based on https://github.com/librosa/librosa/issues/434
  '''
  angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
  S_complex = np.abs(S).astype(np.complex)
  y = _istft(S_complex * angles)
  for i in range(hparams.griffin_lim_iters):
    angles = np.exp(1j * np.angle(_stft(y)))
    y = _istft(S_complex * angles)
  return y


def _griffin_lim_tensorflow(S):
  '''TensorFlow implementation of Griffin-Lim
  Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
  '''
  with tf.variable_scope('griffinlim'):
    # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
    S = tf.expand_dims(S, 0)
    S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
    y = _istft_tensorflow(S_complex)
    for i in range(hparams.griffin_lim_iters):
      est = _stft_tensorflow(y)
      angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
      y = _istft_tensorflow(S_complex * angles)
    return tf.squeeze(y, 0)


def _stft(y):
  n_fft, hop_length, win_length = _stft_parameters()
  return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
  _, hop_length, win_length = _stft_parameters()
  return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_tensorflow(signals):
  n_fft, hop_length, win_length = _stft_parameters()
  return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)


def _istft_tensorflow(stfts):
  n_fft, hop_length, win_length = _stft_parameters()
  return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)


def _stft_parameters():
  n_fft = (hparams.num_freq - 1) * 2
  hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
  win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
  return n_fft, hop_length, win_length


# Conversions:

_mel_basis = None

def _linear_to_mel(spectrogram):
  global _mel_basis
  if _mel_basis is None:
    _mel_basis = _build_mel_basis()
  return np.dot(_mel_basis, spectrogram)

def _build_mel_basis():
    n_fft = (hparams.num_freq - 1) * 2
    return librosa.filters.mel(hparams.sample_rate, n_fft, hparams.num_mels)

def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
  return np.power(10.0, x * 0.05)

def _db_to_amp_tensorflow(x):
  return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

def _normalize(S):
  return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)

def _denormalize(S):
  return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db

def _denormalize_tensorflow(S):
  return (tf.clip_by_value(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db


def process_audio(audio_data, sample_rate=hparams.sample_rate):
    """
    Processes an audio NumPy array using various techniques.

    Args:
        audio_data (np.ndarray): The audio data as a NumPy array.
        sample_rate (int): The sampling rate of the audio data.

    Returns:
        np.ndarray: The processed audio data as a NumPy array.
    """

    # Pre-emphasis (optional)
    audio_data, _ = librosa.effects.preemphasis(audio_data)

    # Choose and apply denoising techniques
    # Example: Spectral subtraction
    noise_estimate = librosa.estimator.compute_noise_params(y=audio_data, sr=sample_rate)
    cleaned_data = librosa.effects.subtractive_noise(audio_data, noise_estimate=noise_estimate)

    # Choose and apply amplification techniques
    # Example: Simple gain adjustment
    cleaned_data *= 1.5  # Adjust gain according to your needs

    # Choose and apply smoothing techniques
    # Example: Spectral envelope smoothing with window size 20
    smooth_data = librosa.effects.sox_suppress_clicks(cleaned_data, sr=sample_rate, window_size=20)

    # Choose and apply low-amplitude speech enhancement techniques
    # Example: Spectral tilt compensation
    tilt_factor = librosa.filters.compute_tilt(cleaned_data, sr=sample_rate)
    comp_data = librosa.effects.tilt_correction(smooth_data, tilt_factor, sr=sample_rate)

    # Post-deemphasis (optional)
    comp_data, _ = librosa.effects.preemphasis(comp_data, sr=sample_rate, invert=True)

    return comp_data
