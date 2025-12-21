import whisperx  # type: ignore
import gc
import torch
from whisperx.diarize import DiarizationPipeline  # type: ignore
from typing import Literal, Callable, Optional

from whisperx.asr import FasterWhisperPipeline  # type: ignore
from whisperx.schema import AlignedTranscriptionResult, TranscriptionResult

from pathlib import Path
import numpy as np

AVAILABLE_MODELS = [
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
                 compute_type: Literal["float16",
                                       "float32", "int8"] = "float16",
                 print_results: Optional[Callable[[str], None]] = None
                 ) -> None:
        """
        Some part of this code is adapted from github of whisperx

        Args:
            - whisper_model_name: Size of the model to use (**tiny, tiny.en, base, base.en,
            small, small.en, distil-small.en, medium, medium.en, distil-medium.en, large-v1,
            large-v2, large-v3, large, distil-large-v2, distil-large-v3, large-v3-turbo,** or **turbo**)
            - batch_size: reduce if low on GPU mem
            - compute_type: change to "int8" if low on GPU mem (may reduce accuracy)
            - print_results: Optional function to print results (for logging). 
            """
        self.device = device
        self.model: FasterWhisperPipeline = whisperx.load_model(
            whisper_model_name, device, compute_type=compute_type)
        self.batch_size = batch_size
        self.print_results = print_results

    def transcribe(self, audio_file: str | Path):
        """
        Because whisperx itself preprocesses audio file, we just need to give any type of audio file.
        """
        audio: np.ndarray = whisperx.load_audio(str(audio_file))
        result: TranscriptionResult = self.model.transcribe(
            audio, batch_size=self.batch_size)
        if self.print_results is not None:
            self.print_results(result["segments"])  # before alignment

        # delete model if low on GPU resources
        delete_model(self.model)

        # 2. Align whisper output

        """여기 아래부터 타입힌트 필요"""

        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=self.device)
        aligned_result: AlignedTranscriptionResult = whisperx.align(
            result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

        if self.print_results is not None:
            self.print_results(aligned_result["segments"])  # after alignment

        # delete model if low on GPU resources
        delete_model(model_a)

        return result

        # 3. Assign speaker labels
        # diarize_model = DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)

        # add min/max number of speakers if known
        # diarize_segments = diarize_model(audio)
        # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

        # result = whisperx.assign_word_speakers(diarize_segments, result)
        # print(diarize_segments)
        # print(result["segments"]) # segments are now assigned speaker IDs
