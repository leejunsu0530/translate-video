import whisperx  # type: ignore
import gc
import torch
from whisperx.diarize import DiarizationPipeline  # type: ignore
from whisperx.schema import AlignedTranscriptionResult  # type: ignore
from typing import Literal, Callable, Optional
from pathlib import Path

AVAILABLE_WHISPER_MODELS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "distil-small.en",
    "medium",
    "medium.en",
    "distil-medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "distil-large-v2",
    "distil-large-v3",
    "large-v3-turbo",
    "turbo"
]


def delete_model(model) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    del model


class WhisperXTranscriber:
    def __init__(self,
                 whisper_model_name: Literal["tiny", "tiny.en", "base", "base.en", "small",
                                             "small.en", "distil-small.en", "medium", "medium.en",
                                             "distil-medium.en", "large-v1", "large-v2", "large-v3",
                                             "large", "distil-large-v2", "distil-large-v3", "large-v3-turbo",
                                             "turbo"] = "large-v2",
                 device: Literal["cpu", "cuda", "auto"] = "auto",
                 batch_size: int = 4,
                 compute_type: Literal['default', 'auto', 'int8', 'int8_float32', 'int8_float16',
                                       'int8_bfloat16', 'int16', 'float16', 'float32', 'bfloat16'] = "float16",
                 print_results: Optional[Callable[[str], None]] = None
                 ) -> None:
        """
        Some part of this code is adapted from github of whisperx

        Args:
            whisper_model_name: Size of the model to use (tiny, tiny.en, base, base.en, small, small.en, distil-small.en, medium, medium.en, distil-medium.en, large-v1,large-v2, large-v3, large, distil-large-v2, distil-large-v3, large-v3-turbo, or turbo)
            batch_size: reduce if low on GPU mem
            compute_type:
                change to "int8" if low on GPU mem (may reduce accuracy)
                - default: keep the same quantization that was used during model conversion
                - auto: use the fastest computation type that is supported on this system and device
                for cpu, default, auto, float32, int8, int8_float32 would be appropriate
            print_results: Optional function to print results (for logging).
            """
        self.device = device
        self.model = whisperx.load_model(
            whisper_model_name, device, compute_type=compute_type)
        self.batch_size = batch_size
        self.print_results = print_results

        self.aligned_segments: Optional[AlignedTranscriptionResult] = None

    def transcribe(self, audio_file: str | Path, delete_used_models: bool = True) -> whisperx.TranscriptionResult:
        """
        Because whisperx itself preprocesses audio file, any type of audio file can be given.
        """
        audio = whisperx.load_audio(str(audio_file))
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        if self.print_results is not None:
            self.print_results(result["segments"])  # before alignment

        # delete model if low on GPU resources
        if delete_used_models:
            delete_model(self.model)

        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=self.device)
        # language code가 필요해서 이건 init에서 미리 로드하지 않음. 직접 전달하는 경우에 한해 미리 로드하는 것도 고려는 해봄.
        aligned_result = whisperx.align(
            result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

        if self.print_results is not None:
            self.print_results(aligned_result["segments"])  # after alignment

        # delete model if low on GPU resources
        delete_model(model_a)

        self.aligned_segments = aligned_result

        # 3. Assign speaker labels
        # diarize_model = DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)

        # add min/max number of speakers if known
        # diarize_segments = diarize_model(audio)
        # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

        # result = whisperx.assign_word_speakers(diarize_segments, result)
        # print(diarize_segments)
        # print(result["segments"]) # segments are now assigned speaker IDs

        return aligned_result

     def make_srt(self, output_path: str | Path=Path().cwd()) -> Path:
        """
        Save the aligned segments as an SRT file.
        """
        if self.aligned_segments is None:
            raise ValueError("No aligned segments available. Please run transcribe() first.")

        pass # whisperx 자체도 서브타이틀 제작기능이 있는데 이거 쓸까?