from typing import Any, Generator, Iterable, Literal

from faster_whisper import WhisperModel
from time import time
from random import randint

from faster_whisper.transcribe import Segment, TranscriptionInfo

from rich.pretty import pprint
# model_size = "large-v3"

# Run on GPU with FP16
model1 = WhisperModel("lorneluo/whisper-small-ct2-int8",
                      device="cpu", compute_type="int8")

model2 = WhisperModel("small",
                      device="cpu", compute_type="int8")

file = r"C:\Users\leeju\Projects\videosubX\test\sample1.mp3"


def run(model: WhisperModel):
    srt = time()
    segments, info = model.transcribe(
        file, beam_size=5, vad_filter=True
    )
    return segments, info, time()-srt


def run_gen(times) -> Generator[tuple[
    int, Literal['pre-converted',
                 'not-converted'], Iterable[Segment], TranscriptionInfo, float
]]:

    for i in range(times):
        if randint(0, 1): # 먼저하는 거에 따라 바뀌나?
            yield (i, "not-converted") + run(model2)
            yield (i, "pre-converted") + run(model1)
        else:
            yield (i, "pre-converted") + run(model1)
            yield (i, "not-converted") + run(model2)


lst = [
    {
        "case": case,
        "model": model_name,
        "segments": segments,
        "info": info,
        "time": time_
    } for case, model_name, segments, info, time_ in run_gen(10)
]

# pprint(lst)
lst_pre = [f"{i["time"]:.2f}" for i in lst if i["model"] == "pre-converted"]
lst_not = [f"{i["time"]:.2f}" for i in lst if i["model"] == "not-converted"]

print(f"pre: {', '.join(lst_pre)}")
print(sum(map(float,lst_pre))/len(lst_pre))

print(f"not: {', '.join(lst_not)}")
print(sum(map(float,lst_not))/len(lst_not))
# print(f"Detected language '{info.language}' with probability {info.language_probability}")

# for segment in segments:
# print(f"[{segment.start} -> {segment.end}] {segment.text}")
