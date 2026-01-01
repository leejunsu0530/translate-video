from translatevideo import WhisperXTranscriber
from pprint import pprint

transcriber = WhisperXTranscriber(
    "tiny", device="cpu", num_workers=4,  compute_type="auto", language_code="ja")
results = transcriber.auto_transcribe(
    r"C:\\Users\\leeju\\Desktop\\test-files\\sample01.mp4")
pprint(results)
