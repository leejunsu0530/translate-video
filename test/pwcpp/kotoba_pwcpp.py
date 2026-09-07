"""
https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-ggml
1. 자동 다운로드 안됨
2. kotoba는 리포에 .bin 있음. 아마 whisper.cpp에 그거 변환 코드가 있으니까 그걸로 일단 테스트"""
from pywhispercpp.model import Model

# 경로 지정하면 그거 불러옴
model = Model(  # 여기 params는 PARAMS_SCHEMA임. 아래 transcribe도 그렇다고 함
    'base',  # 'large-v3-turbo-q8_0',
    # r"C:\Users\leeju\Projects\videosubX\test\pwcpp\ggml-kotoba-whisper-v2.0-q5_0.bin",
    # use_openvino=True,
    # openvino_device='CPU',
    print_realtime=True,
    print_progress=True,
    print_timestamps=True,
    # vad=True,
    # redirect_whispercpp_logs_to=r"C:\Users\leeju\Projects\videosubX\test\pwcpp\log.txt",
    redirect_whispercpp_logs_to=False,
    n_threads=4
)

segments = model.transcribe(
    r'C:\Users\leeju\Projects\videosubX\test\sample1.mp4',
    n_processors=4,
    new_segment_callback=print
)

for segment in segments:
    print(segment.text)
