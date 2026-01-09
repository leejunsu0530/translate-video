from translatevideo import WhisperXTranscriber
from pprint import pprint
from pathlib import Path

# transcriber = WhisperXTranscriber(
# "tiny", device="cpu", num_workers=4,  compute_type="auto", language_code="en")
#
# results = transcriber.auto_transcribe(
# r"C:\\Users\\leeju\\Desktop\\test-files\\sample01.mp4")
# (Path.cwd()/"test"/"test_output.json").write_text(str(results), encoding="utf-8")


class MyClass:
    def __init__(self):
        # 클래스명.함수명 출력
        print(f"{self.__class__.__name__}.{self.__init__.__name__} called")


m = MyClass()
