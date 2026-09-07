
import whisperx
from whisperx.asr import WhisperModel
import gc
# from whisperx.diarize import DiarizationPipeline
import torch
from shutil import which
import os
from pathlib import Path

ffmpeg_path = which("ffmpeg")
if ffmpeg_path is None:
    raise RuntimeError("ffmpeg is not installed or not found in PATH.")
os.add_dll_directory(str(Path(ffmpeg_path).parent))

MODEL_PATH = r"C:\Users\leeju\Projects\.ct2-workspace\outputs\kotoba-whisper-v2.2-ct2-int8"
MODEL_ID = "kotoba-tech/kotoba-whisper-v2.0-faster"


device = "cuda" if torch.cuda.is_available() else "cpu"
audio_file = r"C:\Users\leeju\Desktop\test_15min.mp4"
batch_size = 1  # 16 # reduce if low on GPU mem
# change to "int8" if low on GPU mem (may reduce accuracy)
compute_type = "int8"  # "float16"

print("모델 로딩")
# 1. Transcribe with original whisper (batched)
model = whisperx.load_model(
    # "large-v2",
    # MODEL_PATH,
    MODEL_ID,
    device, compute_type=compute_type,
    vad_method='silero',  # 기본은 pyannote
    language='ja',

)  # wx는 성능 좋지만, 자세한 인자 설정은 fw가 앞서는 듯(예시로, 여긴 multilingual 설정이 있음). 내가 만들어야 하나?

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
print("오디오 로딩 완료")
result = model.transcribe(audio, batch_size=batch_size,
                          print_progress=True,
                          combined_progress=True,
                          verbose=True)  # combined는 progress 켜져야 사용됨. verbose면 내용 등 자세히 출력됨
print(result["segments"])  # before alignment

# delete model if low on GPU resources
gc.collect()
torch.cuda.empty_cache()
del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a,
                        metadata, audio, device, return_char_alignments=False)

print(result["segments"])  # after alignment

# delete model if low on GPU resources
gc.collect()
torch.cuda.empty_cache()
del model_a

# 3. Assign speaker labels
# diarize_model = DiarizationPipeline(token=YOUR_HF_TOKEN, device=device)

# add min/max number of speakers if known
# diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

# result = whisperx.assign_word_speakers(diarize_segments, result)
# print(diarize_segments)
# print(result["segments"])  # segments are now assigned speaker IDs
