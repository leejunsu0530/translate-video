"""
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
from whisperx.diarize import DiarizationPipeline  # type: ignore
from whisperx.schema import AlignedTranscriptionResult, TranscriptionResult  # type: ignore
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
    del model
    gc.collect()
    torch.cuda.empty_cache()
    


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
                 print_results: Optional[Callable[[str], None]] = None,
                 use_diarization: bool = False,
                 hf_token: Optional[str] = None,
                 min_speakers: Optional[int] = None,
                 max_speakers: Optional[int] = None
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
            use_diarization: Whether to use speaker diarization.    
            hf_token: HuggingFace authentication token for diarization model download.
            min_speakers: Minimum number of speakers for diarization. Add it if known.
            max_speakers: Maximum number of speakers for diarization. Add it if known.
            """
        self.device = device
        self.model = whisperx.load_model(
            whisper_model_name, device, compute_type=compute_type)
        self.batch_size = batch_size
        self.print_results = print_results
        self.use_diarization = use_diarization
        self.hf_token = hf_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def transcribe(self, audio_file: str | Path, delete_used_models: bool = True) -> TranscriptionResult:
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
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

        if self.print_results is not None:
            self.print_results(result["segments"])  # after alignment

        # delete model if low on GPU resources
        delete_model(model_a)

        # 3. Assign speaker labels
        if self.use_diarization:
            if self.hf_token is None:
                raise ValueError("HuggingFace token must be provided for diarization model download.")
                
            diarize_model = DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)

            diarize_segments = diarize_model(audio)
            diarize_model(audio, min_speakers=self.min_speakers, max_speakers=self.max_speakers)

            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            if self.print_results is not None: 
                self.print_results(diarize_segments)
                self.print_results(result["segments"]) # segments are now assigned speaker IDs

            delete_model(diarize_model)

        return result
