# pylint: disable=W0613
from typing import TypedDict, Literal, Dict, Optional, Union, Iterable
import torch
from whisperx.schema import AlignedTranscriptionResult, SingleSegment
from whisperx.asr import FasterWhisperPipeline
import numpy as np


class AlignMetadata(TypedDict):
    language: str
    dictionary: Dict[str, int]
    type: Literal["torchaudio", "huggingface"]


def load_align_model(language_code: str,
                     device: str,
                     model_name: Optional[str] = ...,
                     model_dir: Optional[str] = ...) -> tuple[torch.nn.Module, AlignMetadata]: ...


def align(transcript: Iterable[SingleSegment],
          model: torch.nn.Module,
          align_model_metadata: AlignMetadata,
          audio: Union[str, np.ndarray, torch.Tensor],
          device: str,
          interpolate_method: str = ...,
          return_char_alignments: bool = ...,
          print_progress: bool = ...,
          combined_progress: bool = ...
          ) -> AlignedTranscriptionResult: ...


def load_model(whisper_arch: str,
               device: str,
               device_index: int = ...,
               compute_type: str = ...,
               asr_options: dict[str, object] | None = ...,
               language: str | None = ...,
               vad_model: Vad | None = ...,
               vad_method: Literal["pyannote", "silero"] | None = ...,
               vad_options: dict[str, object] | None = ...,
               model: WhisperModel | None = ...,
               task: Literal["transcribe", "translate"] = ...,
               download_root: str | None = ...,
               local_files_only: bool = ...,
               threads: int = ...,
               ) -> FasterWhisperPipeline: ...


def load_audio(): ...


def assign_word_speakers(): ...
