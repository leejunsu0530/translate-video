from pywhispercpp.model import Model  # type: ignore
from pywhispercpp.utils import output_srt  # type: ignore

SAMPLE_PATH = r"C:\Users\user\Desktop\whisper.cpp\samples\jfk.wav"
VAD_PATH = r'C:\Users\user\Desktop\whisper.cpp\models\ggml-silero-v5.1.2.bin'

model = Model('base.en', n_threads=4)
segments = model.transcribe(
    SAMPLE_PATH, language='en', new_segment_callback=print)
