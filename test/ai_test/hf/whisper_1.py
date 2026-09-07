"""
https://huggingface.co/openai/whisper-large-v3
https://huggingface.co/docs/optimum-intel/openvino/inference
https://huggingface.co/docs/optimum-intel/openvino/optimization
dll 못찾는 문제는 토치나 그런거 문제가 아님. init에 박아놓은 코드 사용
"""

from transformers import AutoConfig, AutoProcessor, pipeline
from optimum.intel import OVModelForSpeechSeq2Seq
from datasets import load_dataset
from shutil import which
import os
from pathlib import Path
from openvino import Core


def check_available_devices_ov() -> list[str]:
    core = Core()
    devices = core.available_devices
    return devices


def freeze_batch_dim_to_one(ov_model) -> None:
    """
    NPU compiler rejects dynamic upper-bounds on batch dim for this Whisper graph.
    Force input dim0 to static 1 where it is dynamic.
    """
    shapes = {}
    for inp in ov_model.inputs:
        pshape = inp.get_partial_shape()
        if pshape.rank.is_static and len(pshape) > 0 and pshape[0].is_dynamic:
            pshape[0] = 1
            shapes[inp] = pshape
    if shapes:
        ov_model.reshape(shapes)

ffmpeg_path = which("ffmpeg")
if ffmpeg_path is None:
    raise RuntimeError("ffmpeg is not installed or not found in PATH.")
os.add_dll_directory(str(Path(ffmpeg_path).parent))

# ----------------------------------- 이 아래가 huggingface에서 가져온 코드

# Hub model id example: "openai/whisper-tiny" / "openai/whisper-large-v3-turbo"
# Local path also works: r"C:\Users\leeju\Projects\.HF_repo_example\whisper-base"
model_id = "openai/whisper-tiny"

cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
mel_bins = getattr(cfg, "num_mel_bins", None)
max_src = getattr(cfg, "max_source_positions", None)
print(f"[info] model config: num_mel_bins={mel_bins}, max_source_positions={max_src}")

preferred_ov_device = "NPU"
available_devices = check_available_devices_ov()
ov_device = preferred_ov_device if preferred_ov_device in available_devices else "CPU"
if ov_device != preferred_ov_device:
    print(f"[warn] NPU not found in OpenVINO devices {available_devices}. Falling back to CPU.")
else:
    print(f"[info] Using OpenVINO device: {ov_device}")

model = OVModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    export=True,  # HF -> OpenVINO IR export (first run only)
    compile=False,
    dynamic_shapes=False,  # Whisper input channel(80 mel bins) must stay static for NPU compiler
    batch_size=1,  # NPU compiler dislikes dynamic batch upper-bound for Whisper
    use_cache=False,  # avoid decoder-with-past dynamic path
    stateful=False,  # export non-stateful graph to reduce dynamic shape requirements
    ov_config={"PERFORMANCE_HINT": "LATENCY"},
)
# Force static batch across encoder/decoder OpenVINO graphs for NPU compatibility.
freeze_batch_dim_to_one(model.encoder.model)
freeze_batch_dim_to_one(model.decoder.model)
if model.decoder_with_past is not None:
    freeze_batch_dim_to_one(model.decoder_with_past.model)

try:
    model.to(ov_device)
    model.compile()
except RuntimeError as e:
    # Full Whisper graph can still contain dynamic parts that standalone NPU rejects.
    # HETERO keeps NPU execution where possible and falls back unsupported parts to CPU.
    if ov_device == "NPU":
        fallback_device = "CPU"
        print(f"[warn] NPU compile failed, retrying with {fallback_device}")
        print(f"[warn] original error: {e}")
        model.to(fallback_device)
        model.compile()
    else:
        raise

# Explicitly set to avoid future default-change warning from transformers.
processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    # `device` is a Torch device string; OVModel already uses OpenVINO device via model.to(...)
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]
# sample = r'C:\Users\leeju\Projects\videosubX\test\sample1.mp3',
# r"C:\Users\leeju\Desktop\test_15min.mp4",
# print(sample)

generate_kwargs = {
    "max_new_tokens": 448,
    "num_beams": 1,
    "condition_on_prev_tokens": False,
    # zlib compression ratio threshold (in token space)
    "compression_ratio_threshold": 1.35,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "return_timestamps": True,
}

result = pipe(
    sample,
    generate_kwargs=generate_kwargs,
    return_timestamps=True,
)

print(result["text"])
