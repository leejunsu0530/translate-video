from translatevideo import WhisperXTranscriber
from pprint import pprint

transcriber = WhisperXTranscriber("tiny", "cpu", 4, 4, "int8", "ja")
results = transcriber.auto_transcribe(
    r"C:\\Users\\leeju\\Desktop\\test-files\\sample01.mp4")
pprint(results)
