"""
part of the code is adapted from of whisperx
TODO:
- 인자들을 묶어서 받는 방식 사용, asr이랑 transcribe 보고 수정. 지금 인자들 말고 추가로 원래 지원하는 인자들도 여기로 지원 
- 모델 사용 후 삭제 코드에서, 삭제를 어느 시점에 해야 하는지 gpt 물어보기
- 임시 gui 추가, 로깅 모듈로 출력 변경 및 연동
- ytdlp 기반 영상 불러오기 클래스
- 오디오가 길 경우 메모리 관리를 위해 쪼개서 처리 - 테스트 필요
- 여러 오디오 파일을 지원 < 이건 x
- 전사 결과를 파일로 기록하고 중단시 불러옴. 전부 완성되면 다음 처리 단계로 - 나중에
- 자막 제작(init, main, SubtitleProcessor, utils, transcribe 파일 참조) + subtitlesprocessor 사용 고려
- 불러오기 함수에 모델 이름 지정 기능 등 있는데 그거 활용할 수 있게 하기
- 나중에 다중상속 고려한 설계
"""

import whisperx  # type: ignore
import gc
import torch
import platform
import subprocess
from whisperx.vads import Vad   # type: ignore
from whisperx.schema import AlignedTranscriptionResult, TranscriptionResult  # type: ignore
from whisperx.utils import LANGUAGES, optional_int, str2bool  # type: ignore
from typing import Literal, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from rich import print

# from whisperx.diarize import DiarizationPipeline  # type: ignore


from translatevideo.utils.type_hints import LanguageNames
from translatevideo.utils.type_hints import LanguageCodes
from translatevideo.utils.type_hints import WhisperModels


@dataclass
class AsrOptions:
    """
    parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=1.0, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")
    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")

    여기 인자에서, 이걸 전해줄 때 asr에 직접 주는게 아니라 함수 내 인자로 전해줌. 사용가능한 인자의 종류가 어떻게 되는지 확인 필요
    여기서 init으로 값을 받는 것뿐만 아니라 연산도 행할 수 있는지, 전해줄 때는 어떻게 값을 주고 받아야 하는지
    parser에 넣은 help값 등은 건들 수 있는지?

    여기 인자는 main에서 호출한 transcribe.py에서 인자로 준걸 해석해서 아래의 asr options로 
    전해주고 여기서 load model 함수의 asr options 인자에 넣음. 
    그러면 load model에서 default asr options를 업데이트함.
    이론상으로 나는 아래의, cli에서 전해주는 인자뿐만 아니라 asr.py의 모든 인자를 전부 여기 정의해놓아도 됨. 
    내가 help를 적을 수 있는 건 cli의 인자만 되지만

    asr_options = {
        "beam_size": args.pop("beam_size"),
        "patience": args.pop("patience"),
        "length_penalty": args.pop("length_penalty"),
        "temperatures": temperature,
        "compression_ratio_threshold": args.pop("compression_ratio_threshold"),
        "log_prob_threshold": args.pop("logprob_threshold"),
        "no_speech_threshold": args.pop("no_speech_threshold"),
        "condition_on_previous_text": False,
        "initial_prompt": args.pop("initial_prompt"),
        "hotwords": args.pop("hotwords"),
        "suppress_tokens": [int(x) for x in args.pop("suppress_tokens").split(",")],
        "suppress_numerals": args.pop("suppress_numerals"),
    }
    """
    beam_size: optional_int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    temperatures: list[float] = 0
    compression_ratio_threshold
    log_prob_threshold
    no_speech_threshold
    condition_on_previous_text
    initial_prompt
    hotwords
    suppress_tokens
    suppress_numerals


@dataclass
class VadArgs:
    vad_model: Optional[Vad] = None
    vad_method: Literal["pyannote", "silero"] = "silero"


@dataclass
class DiarizeArgs:
    hf_token: Optional[str] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    num_speakers: Optional[int] = None


class WhisperXTranscriber:
    def __init__(self,
                 whisper_model_name: WhisperModels = "medium",
                 chunk_audio_minutes: Optional[float] = None,
                 language_code: LanguageCodes | None = None,
                 compute_type: Literal['default', 'auto', 'int8', 'int8_float32', 'int8_float16',
                                       'int8_bfloat16', 'int16', 'float16', 'float32', 'bfloat16'] = "auto",
                 device: Literal["cpu", "cuda", "xpu"] = "cpu",
                 batch_size: int = 4,
                 num_workers: int = 0,
                 vad_model: Optional[Vad] = None,
                 vad_method: Literal["pyannote", "silero"] = "silero",
                 print_progress: bool = True,
                 combined_progress: bool = True,
                 hf_token: Optional[str] = None,
                 min_speakers: Optional[int] = None,
                 max_speakers: Optional[int] = None,
                 num_speakers: Optional[int] = None,
                 ) -> None:
        """
        Some part of this code is adapted from github of whisperx

        Args:
            whisper_model_name: Size of the model to use (tiny, tiny.en, base, base.en, small, small.en, distil-small.en, medium, medium.en, distil-medium.en, large-v1,large-v2, large-v3, large, distil-large-v2, distil-large-v3, large-v3-turbo, or turbo)
            chunk_audio_minutes: If provided, audio will be chunked into segments of the given length (in minutes) for transcription to reduce memory usage. If None, the entire audio will be processed at once.
            language_code: language code for **transcribe** and **align** method. If None, language will be detected automatically.
            compute_type: change to "int8" if low on GPU mem (may reduce accuracy). When using cpu, default, auto, float32, int8, int8_float32 would be appropriate
            device: device to run the model on (cpu, cuda, xpu). "auto" is not supported here. "xpu" is not tested yet.
            batch_size: number of batches for **transcript** method. reduce if low on GPU mem
            num_workers: number of workers for **transcript** method. **Can't be used at windows and it will automatically be 0.**
            vad_model: The vad model to manually assign.
            vad_method: The vad method to use. vad_model has a higher priority if it is not None. **currently, torch higher than 2.6 causes error with pyannote vad, so please use silero vad instead**
            print_progress: Whether to print progress through whisperx at **transcribe** and **align** method.
            combined_progress: Whether to use combined progress.
            hf_token: HuggingFace authentication token for **diarization** model download.
            min_speakers: Minimum number of speakers for **diarize** method. Add it if known.
            max_speakers: Maximum number of speakers for **diarize** method. Add it if known.
            num_speakers: Number of speakers for **diarize** method. Add it if known.

            """
        # with torch.serialization.safe_globals([ListConfig]): # num workers 0이면 상관 x
        self._asr_model = None
        self._align_model_and_metadeta = None
        self._diarize_model = None

        self.whisper_model_name = whisper_model_name
        self.device = device
        self.compute_type = compute_type
        self.vad_model = vad_model
        self.vad_method = vad_method
        self.language_code = language_code
        if isinstance(chunk_audio_minutes, float) and chunk_audio_minutes <= 0:
            print(
                f"[Warning] {self.__class__.__name__}.chunk_audio_minutes must be positive value or None. It will be set to None.")
            self.chunk_audio_minutes = None
        else:
            self.chunk_audio_minutes = chunk_audio_minutes
        self.batch_size = batch_size
        if platform.system() == "Windows" and num_workers != 0:
            print(
                f"[yellow][Warning][/] {self.__class__.__name__}.{self.__class__.__init__.__name__}: num_workers can't be used at Windows OS. Setting num_workers to 0.")
            self.num_workers = 0
        else:
            self.num_workers = num_workers
        self.print_progress = print_progress
        self.combined_progress = combined_progress
        self.hf_token = hf_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.num_speakers = num_speakers
        # self.delete_used_models = delete_used_models

    @property
    def asr_model(self):
        """lazy import of asr model."""
        if self._asr_model is None:
            self._asr_model = whisperx.load_model(
                self.whisper_model_name,
                self.device,
                compute_type=self.compute_type,
                language=self.language_code,
                vad_model=self.vad_model,
                vad_method=self.vad_method
            )
        return self._asr_model

    @property
    def align_model_tuple(self):
        """
        lazy import of align model and metadata of it.
        language_code가 없다는 문제는 아래 align을 실행할 때 위 줄에서 지정하기 때문에 상관 없음
        """
        if self.language_code is None:
            raise ValueError(
                f"{self.__class__.__name__}.language_code must be setted before calling align model.")
        if self._align_model_and_metadeta is None:
            self._align_model_and_metadeta = whisperx.load_align_model(
                self.language_code,
                self.device,
            )
        return self._align_model_and_metadeta

    @property
    def diarize_model(self):
        """lazy import of diarize model."""
        if self._diarize_model is None:
            from whisperx.diarize import DiarizationPipeline  # type: ignore
            self._diarize_model = DiarizationPipeline(
                use_auth_token=self.hf_token, device=self.device
            )
        return self._diarize_model

    @staticmethod
    def _delete_object(*objects: Any) -> None:
        for obj in objects:
            del obj
        gc.collect()
        torch.cuda.empty_cache()

    def delete_model(self, model: Literal["asr_model", "align_model", "diarize_model"]) -> None:
        if model == "asr_model":
            self._delete_object(self._asr_model)
            self._asr_model = None
        elif model == "align_model":
            self._delete_object(self._align_model_and_metadeta)
            self._align_model_and_metadeta = None
        elif model == "diarize_model":
            self._delete_object(self._diarize_model)
            self._diarize_model = None

    def auto_transcribe(self, audio_file: str | Path, use_diarization: bool = True) -> tuple[TranscriptionResult, LanguageNames]:
        """
        Automatically chunks audio, transcribes, aligns, and diarizes (if specified) the given audio file.
        Because whisperx itself preprocesses audio file, any type of audio file can be given.
        """
        if self.chunk_audio_minutes is None:
            audio = self.load_audio(audio_file)
            print("[green][Info][/] Starting transcription without chunking...")
            result = self.transcribe(audio)
            language_name = self.return_language_name(result)
            self.delete_model("asr_model")
            print("[green][Info][/] Starting alignment...")
            result = self.align(result, audio)
            self.delete_model("align_model")
            if use_diarization:
                print("[green][Info][/] Starting diarization...")
                result = self.diarize(audio, result)
                self.delete_model("diarize_model")
            return result, language_name
        # ---------------------------
    # 스트리밍 청크 전사
    # ---------------------------
        all_segments = []
        detected_language = None

        for seg, lang in self.chunk_transcribe_generator(audio_file):
            all_segments.append(seg)
            detected_language = lang
        self.delete_model("asr_model")

        merged_result = {
            "segments": all_segments,
            "language": detected_language,
        }

        language_name = LANGUAGES[detected_language]

        # ---------------------------
        # 전체 정렬
        # ---------------------------
        print("[green][Info][/] Running global alignment...")
        full_audio = self.load_audio(audio_file)
        aligned = self.align(merged_result, full_audio)

        # ---------------------------
        # 전체 화자 분리
        # ---------------------------
        if use_diarization:
            print("[green][Info][/] Running global diarization...")
            aligned = self.diarize(full_audio, aligned)

        return aligned, language_name

    def load_audio(self, audio_file: str | Path | np.ndarray,
                   start: Optional[float] = None,
                   duration: Optional[float] = None,
                   sr: int = 16000) -> np.ndarray:
        """
        part of the code is adapted from of whisperx
        """
        if isinstance(audio_file, np.ndarray):
            return audio_file

        try:
            # Launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI to be installed.
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-threads",
                "0"]
            if start is not None and duration is not None:
                cmd += ["-ss", str(start), "-t", str(duration)]
            cmd += ["-i",
                    str(audio_file),
                    "-f",
                    "s16le",
                    "-ac",
                    "1",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    str(sr),
                    "-",
                    ]

            out = subprocess.run(cmd, capture_output=True, check=True).stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to load audio: {e.stderr.decode()}") from e

        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    def get_audio_duration(self, audio_file: str | Path) -> float:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_file)
        ]
        try:
            out = subprocess.check_output(
                cmd, stderr=subprocess.STDOUT).decode().strip()
            if not out:
                raise RuntimeError("ffprobe returned empty output.")
            duration = float(out)
            if duration <= 0:
                raise RuntimeError(f"Invalid duration value: {duration}.")
            return duration
        except FileNotFoundError as e:
            raise RuntimeError(
                "ffprobe not found. Please install FFmpeg and ensure ffprobe is in PATH."
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ffprobe failed for file '{audio_file}'.\n"
                f"ffprobe output:\n{e.output.decode(errors='ignore')}"
            ) from e
        except ValueError as e:
            raise RuntimeError(
                f"ffprobe returned a non-numeric duration: '{out}'"
            ) from e

    def transcribe(self, audio_file: str | Path | np.ndarray,
                   #    additional_args: Optional[dict] = None
                   _print_progress: bool = True,
                   _combined_progress: bool = True
                   ) -> TranscriptionResult:
        """
        you can also use this function by itself if you don't need alignment and diarization.
        To lower memory use, please use 'delete_model' method after using this method.

        _print_progress and _combined_progress are for internal use when called from chunk_transcribe_generator.
        """
        audio = self.load_audio(audio_file)
        # if additional_args is None:
        # additional_args = {}

        result = self.asr_model.transcribe(
            audio,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # language=self.language_code,
            print_progress=self.print_progress and _print_progress,
            combined_progress=self.combined_progress and _combined_progress,
            # **additional_args
        )
        return result

    def chunk_transcribe_generator(self, audio_file: str | Path):
        """
        Generator that:
        - loads small audio windows
        - runs WhisperX
        - shifts timestamps
        - yields segments in absolute time
        - deletes segment after each chunk

        *Note*: it doesn't delete ASR model. To lower memory use, please use 'delete_model' method after processing all chunks.
        """
        if self.chunk_audio_minutes is None:
            raise ValueError(
                "if chunk_audio_minutes is None, please use 'transcribe' method.")

        chunk_sec = self.chunk_audio_minutes * 60
        total_duration = self.get_audio_duration(audio_file)
        starts = np.arange(0, total_duration, chunk_sec)

        detected_language = None

        for idx, start in enumerate(starts):
            end = min(start + chunk_sec, total_duration)
            print(
                f"[cyan]Transcribing chunk {idx}/{len(starts)} ({start:.1f}s → {end:.1f}s)[/]")

            audio_chunk = self.load_audio(
                audio_file, start=start, duration=chunk_sec)
            chunk_result = self.transcribe(audio_chunk,
                                           # 나중에 여기에 여러 청크를 처리할 때의 출력을 따로 지정 가능
                                           )

            if detected_language is None:
                detected_language = chunk_result["language"]

            for seg in chunk_result["segments"]:
                seg["start"] += start
                seg["end"] += start
                yield seg, detected_language

            # hard memory cleanup
            self._delete_object(audio_chunk, chunk_result)

    def align(self,
              transcription_result: TranscriptionResult,
              audio: str | Path | np.ndarray,
              #   additional_args: Optional[dict] = None,
              ) -> AlignedTranscriptionResult:
        """
        you can also use this function by itself if you have transcription result and don't need diarization.
        To lower memory use, please use 'delete_model' method after using this method.
        """
        audio = self.load_audio(audio)
        # if additional_args is None:
        # additional_args = {}

        # language_code = transcription_result["language"] or self.language_code
        if self.language_code is None:
            print(
                "[green][Info][/] No default language code was set. Using detected language from transcription.")
            self.language_code = transcription_result["language"]

        aligned_result = whisperx.align(
            transcription_result["segments"],
            *self.align_model_tuple,
            audio, self.device,
            return_char_alignments=False,
            print_progress=self.print_progress,
            combined_progress=self.combined_progress,
            # **additional_args
        )

        # self.delete_object(model_a)
        return aligned_result

    def diarize(self,
                audio: str | Path | np.ndarray,
                transcription_result: TranscriptionResult | AlignedTranscriptionResult,
                # additional_args: Optional[dict] = None
                ) -> AlignedTranscriptionResult | TranscriptionResult:
        """
        you can also use this function by itself if you have transcription result.
        **if hf_token is not provided, diarization will be skipped.**
        To lower memory use, please use 'delete_model' method after using this method.
        """
        if self.hf_token is None:
            print(
                f"[yellow][Warning][/] {self.__class__.__name__}.{self.diarize.__name__}: HuggingFace token must be provided for diarization model download. Skipping diarization.")
            return transcription_result

        audio = self.load_audio(audio)
        # if additional_args is None:
        # additional_args = {}

        diarize_model = self.diarize_model

        diarize_segments = diarize_model(
            audio,
            num_speakers=self.num_speakers,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers
        )

        diarized_result = whisperx.assign_word_speakers(
            diarize_segments,
            transcription_result,
            # **additional_args
        )

        # self.delete_object(diarize_model)
        return diarized_result

    def return_language_name(self, transciption_result: TranscriptionResult) -> LanguageNames:
        """
        load language name from language code
        """
        language_code = transciption_result["language"] or self.language_code
        return LANGUAGES[language_code]


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
