from pywhispercpp.model import Model  # type: ignore
from pywhispercpp.utils import output_srt  # type: ignore

from openvino.runtime import Core
print(Core().available_devices)

SAMPLE_PATH = r"C:\Users\user\Desktop\whisper.cpp\samples\jfk.wav"
# VAD_PATH = r'C:\Users\user\Desktop\whisper.cpp\models\ggml-silero-v5.1.2.bin'

model = Model('base.en', n_threads=4)
model.system_info()
