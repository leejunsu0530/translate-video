"""
part of the code is adapted from example code of whisperx
TODO:
- 오디오가 길 경우 메모리 관리를 위해 쪼개서 처리
- 불러오기 함수에 모델 이름 지정 기능 등 있는데 그거 활용할 수 있게 하기
- 나중에 다중상속 고려한 설계
"""
import whisperx  # type: ignore
import gc
import torch
import platform
from whisperx.vads import Vad   # type: ignore
from whisperx.diarize import DiarizationPipeline  # type: ignore
from whisperx.schema import AlignedTranscriptionResult, TranscriptionResult  # type: ignore
from whisperx.utils import LANGUAGES  # type: ignore
from typing import Literal, Optional, Any
from pathlib import Path
from numpy import ndarray
from rich import print

from translatevideo.utils.type_hints import LanguageNames
from translatevideo.utils.type_hints import LanguageCodes
from translatevideo.utils.type_hints import WhisperModels


class WhisperXTranscriber:
    def __init__(self,
                 whisper_model_name: WhisperModels = "large-v2",
                 vad_model: Optional[Vad] = None,
                 vad_method: Literal["pyannote", "silero"] = "silero",
                 device: Literal["cpu", "cuda", "xpu"] = "cpu",
                 num_workers: int = 0,
                 batch_size: int = 4,
                 compute_type: Literal['default', 'auto', 'int8', 'int8_float32', 'int8_float16',
                                       'int8_bfloat16', 'int16', 'float16', 'float32', 'bfloat16'] = "auto",
                 language_code: LanguageCodes | None = None,
                 print_progress: bool = True,
                 combined_progress: bool = False,
                 hf_token: Optional[str] = None,
                 min_speakers: Optional[int] = None,
                 max_speakers: Optional[int] = None,
                 delete_used_models: bool = True
                 ) -> None:
        """
        Some part of this code is adapted from github of whisperx

        Args:
            whisper_model_name: Size of the model to use (tiny, tiny.en, base, base.en, small, small.en, distil-small.en, medium, medium.en, distil-medium.en, large-v1,large-v2, large-v3, large, distil-large-v2, distil-large-v3, large-v3-turbo, or turbo)
            vad_model: The vad model to manually assign.
            vad_method: The vad method to use. vad_model has a higher priority if it is not None. **currently, torch higher than 2.6 causes error with pyannote vad, so please use silero vad instead**
            device: device to run the model on (cpu, cuda, xpu). "auto" is not supported here. "xpu" is not tested yet.
            num_workers: number of workers for **transcript** method. **Can't be used at windows and it will automatically be 0.**
            batch_size: number of batches for **transcript** method. reduce if low on GPU mem
            compute_type:
                change to "int8" if low on GPU mem (may reduce accuracy)
                - default: keep the same quantization that was used during model conversion
                - auto: use the fastest computation type that is supported on this system and device
                for **cpu**, default, auto, float32, int8, int8_float32 would be appropriate
            language_code: language code for **transcribe** and **align** method. If None, language will be detected automatically.
            print_progress: Whether to print progress through whisperx at **transcribe** and **align** method.
            combined_progress: Whether to use combined progress.
            hf_token: HuggingFace authentication token for **diarization** model download.
            min_speakers: Minimum number of speakers for **diarize** method. Add it if known.
            max_speakers: Maximum number of speakers for **diarize** method. Add it if known.
            delete_used_models: Whether to delete models after use to free up memory.
            """
        self.device = device
        if platform.system() == "Windows" and num_workers != 0:
            print(
                f"[yellow][Warning][/] {self.__class__.__name__}.{self.__class__.__init__.__name__}: num_workers can't be used at Windows OS. Setting num_workers to 0.")
            self.num_workers = 0
        else:
            self.num_workers = num_workers
        self.batch_size = batch_size
        self.language_code = language_code
        self.print_progress = print_progress
        self.combined_progress = combined_progress
        self.hf_token = hf_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.delete_used_models = delete_used_models
        # with torch.serialization.safe_globals([ListConfig]): # num workers 0이면 상관 x
        self.model = whisperx.load_model(whisper_model_name,
                                         device,
                                         compute_type=compute_type,
                                         language=language_code,
                                         vad_model=vad_model,
                                         vad_method=vad_method)

    def delete_model(self, model: Any) -> None:
        if self.delete_used_models:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    def auto_transcribe(self, audio_file: str | Path, use_diarization: bool = True) -> tuple[TranscriptionResult, LanguageNames]:
        """
        Because whisperx itself preprocesses audio file, any type of audio file can be given.
        """
        audio = whisperx.load_audio(str(audio_file))
        # 1. Transcribe with whisper
        print("[green][Info][/] Starting transcription...")
        result = self.transcribe(audio)
        language_name = self.load_language_name(result)

        # 2. Align whisper output
        print("[green][Info][/] Starting alignment...")
        result = self.align(result, audio)

        # 3. Assign speaker labels
        if use_diarization:
            print("[green][Info][/] Starting diarization...")
            result = self.diarize(audio, result)

        return result, language_name

    def load_language_name(self, transciption_result: TranscriptionResult) -> LanguageNames:
        """
        load language name from language code
        """
        language_code = transciption_result["language"] or self.language_code
        return LANGUAGES[language_code]

    def load_audio(self, audio_file: str | Path | ndarray) -> ndarray:
        """
        load audio file into ndarray
        """
        if isinstance(audio_file, (str, Path)):
            audio = whisperx.load_audio(str(audio_file))
        else:
            audio = audio_file
        return audio

    def transcribe(self, audio_file: str | Path | ndarray, additional_args: Optional[dict] = None) -> TranscriptionResult:
        """
        you can also use this function by itself if you don't need alignment and diarization
        """
        audio = self.load_audio(audio_file)
        if additional_args is None:
            additional_args = {}

        result = self.model.transcribe(
            audio,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # language=self.language_code,
            print_progress=self.print_progress,
            combined_progress=self.combined_progress,
            **additional_args
        )

        self.delete_model(self.model)
        return result

    def align(self,
              transcription_result: TranscriptionResult,
              audio: str | Path | ndarray,
              additional_args: Optional[dict] = None,
              ) -> AlignedTranscriptionResult:
        """
        you can also use this function by itself if you have transcription result and don't need diarization.
        """
        audio = self.load_audio(audio)
        if additional_args is None:
            additional_args = {}

        language_code = transcription_result["language"] or self.language_code

        model_a, metadata = whisperx.load_align_model(
            language_code=language_code, device=self.device)
        aligned_result = whisperx.align(
            transcription_result["segments"], model_a, metadata,
            audio, self.device,
            return_char_alignments=False,
            print_progress=self.print_progress,
            combined_progress=self.combined_progress,
            **additional_args
        )

        self.delete_model(model_a)
        return aligned_result

    def diarize(self,
                audio: str | Path | ndarray,
                transcription_result: TranscriptionResult | AlignedTranscriptionResult,
                additional_args: Optional[dict] = None
                ) -> AlignedTranscriptionResult | TranscriptionResult:
        """
        you can also use this function by itself if you have transcription result.
        **if hf_token is not provided, diarization will be skipped.**
        """
        if self.hf_token is None:
            print(
                f"[yellow][Warning][/] {self.__class__.__name__}.{self.diarize.__name__}: HuggingFace token must be provided for diarization model download. Skipping diarization.")
            return transcription_result

        audio = self.load_audio(audio)
        if additional_args is None:
            additional_args = {}

        diarize_model = DiarizationPipeline(
            use_auth_token=self.hf_token, device=self.device)

        diarize_segments = diarize_model(audio)
        diarize_model(audio,
                      min_speakers=self.min_speakers,
                      max_speakers=self.max_speakers
                      )

        diarized_result = whisperx.assign_word_speakers(
            diarize_segments,
            transcription_result,
            **additional_args
        )

        self.delete_model(diarize_model)
        return diarized_result


"""설치가 어렵기도 하고, 현재는 메리트가 없으므로 제거"""
# class PwcppTranscriber(WhisperXTranscriber):
# def __init__(self,
#  whisper_model_name="large-v2",
#  vad_model=None,
#  vad_method="silero",
#  device="auto",
#  num_workers=0,
#  batch_size=4,
#  compute_type="auto",
#  language_code=None,
#  print_progress=True,
#  combined_progress=False,
#  hf_token=None,
#  min_speakers=None,
#  max_speakers=None,
#  delete_used_models=True
#  ) -> None:
# pass
# vad 호출(위에서 정한대로)
# pwcpp 호출. ov 버전도 이제 통합됨
# 추후 pwcpp 관련 초기화 코드 추가 가능
