import gradio as gr
import argparse
from hparams import hparams, hparams_debug_string
import os
from synthesizer import Synthesizer
import librosa
# Load the synthesizer model



# Define the Gradio interface
synthesizer = Synthesizer()

def synthesis_for_gradio(text):
  return synthesizer.gradio_synthesize_with_resample(text)


iface = gr.Interface(
    fn=synthesis_for_gradio,  # Function to call for synthesis
    inputs=[gr.Text( label="Text to synthesize")],  # Input text field
    outputs=gr.components.Audio(label='Generated audio'),
    title="ትግርኛ ምሉእ ሓሳባት ናብ ድምጺ እትቅይር ሰብ-ሰርሖ ብልሒ ፥ Tigrinya Speech Synthesizer AI (/ ተኽሊት)",
    css='div {margin-left: auto; margin-right: auto; width: 100%;\
            background-image: url("https://drive.google.com/file/d/1DSdjzmHrOsBr3cg9X9ecadY6TUTPM05L/view?usp=sharing"); repeat 0 0;}',
    examples=["ፕረዚደንት ሃገረ ኤርትራ ናብ ጥልያን በጺሑ መጽዩ ፡ ኣዝዩ ትስፉው ዝኾነ ምብጻሕ እዩ ኔሩ","መንእሰያት ኤርትራ ኣይትትሓለሉ ጌና ብሩህ መጻኢ ይጽበየኩም ኣሎ",
              "እታ ዘንበባ ጽሕፍቲ ብዙሕ ሓበሬታ ኣይነበራን።","ናተይ ንዕቀት ድዩ ወይስ ናይ ጸሓይ ድምቀት ፡", " ኣጆኽን ኣንስቲ ጻዕዳ ጣፍ ለወስቲ ፡"],
    description = 'እዚኣ ኣብ ሰብ-ሰርሖ ብልሒ ተሞርኲሳ እትሰርሕ፥ ምሉእ ሓሳባት ትግርኛ ናብ ድምጺ እትቅይር ሶፍትዌር ኮይና፣ ጌና ኣብ ምምዕባል ትርከብ። Today we present to you Tigrinya Speech synthesizer AI. This a beta version of the ongoing work.',
    thumbnail = 'https://drive.google.com/file/d/1puFt2RezT9ZvsWegjnXn8brCofAfLzFw/view?usp=sharing',
    allow_flagging="manual"  
    )

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', default=".\\train_output\\model.ckpt-598000", help='Full path to model checkpoint')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  synthesizer.load(args.checkpoint)
  iface.launch(share=True)


# Launch the Gradio interface
