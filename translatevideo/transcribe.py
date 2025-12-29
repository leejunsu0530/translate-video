"""
part of the code is adapted from example code of whisperx
TODO:
- ~init에서 model_a, metadata 튜플로 묶어서 처리, 언어코드 없을 경우만 객체 다시 생성~
    - 이 부분은 불러오기 함수가 그다지 무거워보이지 않아서 상관x. 언어 코드 강제 지정만 추가?
- SRT 저장 기능 구현
- 불러오기 함수에 모델 이름 지정 기능 등 있는데 그거 활용할 수 있게 하기
- 나중에 다중상속 고려한 설계
"""
import whisperx  # type: ignore
import gc
import torch
from whisperx.vads import Vad   # type: ignore
from whisperx.diarize import DiarizationPipeline  # type: ignore
from whisperx.schema import AlignedTranscriptionResult, TranscriptionResult  # type: ignore
from whisperx.utils import LANGUAGES
from typing import Literal, Optional, Any
from pathlib import Path
from numpy import ndarray

from translatevideo.utils.type_hints import LanguageNames
from translatevideo.utils.type_hints import LanguageCodes
from translatevideo.utils.type_hints import WhisperModels


class WhisperXTranscriber:
    def __init__(self,
                 whisper_model_name: WhisperModels = "large-v2",
                 vad_model: Optional[Vad] = None,
                 vad_method: Literal["pwcpp", "silero"] = "silero",
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
            num_workers: number of workers for **transcript** method
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
        self.model = whisperx.load_model(
            whisper_model_name, device, compute_type=compute_type, vad_model=vad_model, vad_method=vad_method)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.language_code = language_code
        self.print_progress = print_progress
        self.combined_progress = combined_progress
        self.hf_token = hf_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.delete_used_models = delete_used_models

    def delete_model(self, model: Any) -> None:
        if self.delete_used_models:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    def auto_transcribe(self, audio_file: str | Path) -> tuple[TranscriptionResult, LanguageNames]:
        """
        Because whisperx itself preprocesses audio file, any type of audio file can be given.
        """
        audio = whisperx.load_audio(str(audio_file))
        # 1. Transcribe with whisper
        result = self.transcribe(audio)
        language_name = self.load_language_name(result)

        # 2. Align whisper output
        result = self.align(result, audio)

        # 3. Assign speaker labels
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

    def transcribe(self, audio_file: str | Path | ndarray) -> TranscriptionResult:
        """
        you can also use this function by itself if you don't need alignment and diarization
        """
        audio = self.load_audio(audio_file)

        result = self.model.transcribe(
            audio,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            language=self.language_code,
            print_progress=self.print_progress,
            combined_progress=self.combined_progress
        )

        self.delete_model(self.model)
        return result

    def align(self,
              transcription_result: TranscriptionResult,
              audio: str | Path | ndarray
              ) -> AlignedTranscriptionResult:
        """
        you can also use this function by itself if you have transcription result and don't need diarization.
        """
        audio = self.load_audio(audio)

        language_code = transcription_result["language"] or self.language_code

        model_a, metadata = whisperx.load_align_model(
            language_code=language_code, device=self.device)
        aligned_result = whisperx.align(
            transcription_result["segments"], model_a, metadata,
            audio, self.device,
            return_char_alignments=False,
            print_progress=self.print_progress,
            combined_progress=self.combined_progress
        )

        self.delete_model(model_a)
        return aligned_result

    def diarize(self,
                audio: str | Path | ndarray,
                transcription_result: TranscriptionResult | AlignedTranscriptionResult
                ) -> AlignedTranscriptionResult | TranscriptionResult:
        """
        you can also use this function by itself if you have transcription result.
        **if hf_token is not provided, diarization will be skipped.**
        """
        if self.hf_token is None:
            print(
                "[Warning] HuggingFace token must be provided for diarization model download. Skipping diarization.")
            return transcription_result

        if isinstance(audio, (str, Path)):
            audio = whisperx.load_audio(str(audio))

        diarize_model = DiarizationPipeline(
            use_auth_token=self.hf_token, device=self.device)

        diarize_segments = diarize_model(audio)
        diarize_model(audio,
                      min_speakers=self.min_speakers,
                      max_speakers=self.max_speakers
                      )

        diarized_result = whisperx.assign_word_speakers(
            diarize_segments,
            transcription_result
        )

        self.delete_model(diarize_model)
        return diarized_result


class PwcppTranscriber(WhisperXTranscriber):
    def __init__(self,
                 whisper_model_name: WhisperModels = "large-v2",
                 vad_model: Optional[Vad] = None,
                 vad_method: Literal["pwcpp", "silero"] = "silero",
                 device: Literal["cpu", "cuda", "auto"] = "auto",
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
        super().__init__(whisper_model_name, vad_model, vad_method, device, num_workers, batch_size, compute_type,
                         language_code, print_progress, combined_progress, hf_token,
                         min_speakers, max_speakers, delete_used_models)
        # 추후 pwcpp 관련 초기화 코드 추가 가능

    def transcribe(self, audio_file: str | Path | ndarray) -> TranscriptionResult:
        """
        overrides faster_whisper transcribe method to use pwcpp and vad
        """
