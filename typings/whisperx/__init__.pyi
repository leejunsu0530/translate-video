# pylint: disable=W0613
from typing import TypedDict, Literal, Dict, Optional, Union, Iterable
import torch
from whisperx.schema import AlignedTranscriptionResult, TranscriptionResult, SingleSegment  # type: ignore
from whisperx.asr import FasterWhisperPipeline, WhisperModel  # type: ignore
from whisperx.vads.vad import Vad  # type: ignore
import numpy as np
import pandas as pd


class AlignMetadata(TypedDict):
    language: str
    dictionary: Dict[str, int]
    type: Literal["torchaudio", "huggingface"]


def load_align_model(language_code: str,
                     device: str,
                     model_name: Optional[str] = ...,
                     model_dir: Optional[str] = ...
                     ) -> tuple[torch.nn.Module, AlignMetadata]: ...


def align(transcript: Iterable[SingleSegment],
          model: torch.nn.Module,
          align_model_metadata: AlignMetadata,
          audio: Union[str, np.ndarray, torch.Tensor],
          device: str,
          interpolate_method: str = ...,
          return_char_alignments: bool = ...,
          print_progress: bool = ...,
          combined_progress: bool = ...
          ) -> AlignedTranscriptionResult:
    """
     Align phoneme recognition predictions to known transcription.
     """
    ...


def load_model(whisper_arch: str,
               device: str,
               device_index: int = ...,
               compute_type: str = ...,
               asr_options: Optional[dict[str, object]] = ...,
               language: Optional[str] = ...,
               vad_model: Optional[Vad] = ...,
               vad_method: Optional[str] = ...,
               vad_options: Optional[dict[str, object]] = ...,
               model: Optional[WhisperModel] = ...,
               task: str = ...,
               download_root: Optional[str] = ...,
               local_files_only: bool = ...,
               threads: int = ...,
               ) -> FasterWhisperPipeline:
    """
    Load a Whisper model for inference.

    Args:
        whisper_arch: The name of the Whisper model to load.
        device: The device to load the model on.
        compute_type: The compute type to use for the model.
        vad_model: The vad model to manually assign.
        vad_method: The vad method to use. vad_model has a higher priority if it is not None.
        options: A dictionary of options to use for the model.
        language: The language of the model. (use English for now)
        model: The WhisperModel instance to use.
        download_root: The root directory to download the model to.
        local_files_only: If `True`, avoid downloading the file and return the path to the local cached file if it exists.
        threads: The number of cpu threads to use per worker, e.g. will be multiplied by num workers.
    
    Returns:
        A Whisper pipeline.
    """
    ...


def load_audio(file: str,
               sr: int = ...
               ) -> np.ndarray:
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    ...


def assign_word_speakers(diarize_df: pd.DataFrame,
                         transcript_result: Union[AlignedTranscriptionResult, TranscriptionResult],
                         speaker_embeddings: Optional[dict[str,
                                                           list[float]]] = ...,
                         fill_nearest: bool = ...
                         ) -> Union[AlignedTranscriptionResult, TranscriptionResult]:
    """
    Assign speakers to words and segments in the transcript.

    Args:
        diarize_df: Diarization dataframe from DiarizationPipeline
        transcript_result: Transcription result to augment with speaker labels
        speaker_embeddings: Optional dictionary mapping speaker IDs to embedding vectors
        fill_nearest: If True, assign speakers even when there's no direct time overlap

    Returns:
        Updated transcript_result with speaker assignments and optionally embeddings
    """
    ...
