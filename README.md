# Tacotron

An implementation of Tigrinya Tacotron speech synthesis in TensorFlow.


### Audio Samples

  * **[Audio Samples](https://keithito.github.io/audio-samples/)** from model trained using this research.
    * The first set was trained for 600K steps on the Audio book dataset prepared.
      * Speech started to become intelligible around 30K steps.

## Quick Start

### Installing dependencies

1. Install Python 3.

2. Install the latest version of [TensorFlow](https://www.tensorflow.org/install/) for your platform. For better
   performance, install with GPU support if it's available. This code works with TensorFlow 1.3 and later.

3. Install requirements:
   ```
   pip install -r requirements.txt
   ```


### Using a pre-trained model

1. **Run the demo server**:
   ```
   python3 gradio_server.py
   ```

3. **Point your browser at http://127.0.0.1:7860**
   * Type what you want to synthesize




