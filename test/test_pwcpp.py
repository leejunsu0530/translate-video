from pywhispercpp.model import Model  # type: ignore
from pywhispercpp.utils import output_srt  # type: ignore

# from openvino.runtime import Core
# print(Core().available_devices)

SAMPLE_PATH = r"C:\Users\user\Desktop\whisper.cpp\samples\jfk.wav"
# VAD_PATH = r'C:\Users\user\Desktop\whisper.cpp\models\ggml-silero-v5.1.2.bin'

model = Model('base.en', print_realtime=True,
              print_progress=True)
model.system_info()
segments = model.transcribe(SAMPLE_PATH)

# model = Model('base.en', print_realtime=False, print_progress=False)
# segments = model.transcribe(SAMPLE_PATH, new_segment_callback=print)
