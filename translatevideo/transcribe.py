"""
part of the code is adapted from of whisperx
TODO:
- 오디오가 길 경우 메모리 관리를 위해 쪼개서 처리
- 전사 결과를 파일로 기록하고 중단시 불러옴. 전부 완성되면 다음 처리 단계로
- 자막 제작(init, main, SubtitleProcessor, utils, transcribe 파일 참조) 
- 나중에 logging 모듈로 출력 변경
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
from whisperx.utils import LANGUAGES  # type: ignore
from typing import Literal, Optional, Any
from pathlib import Path
import numpy as np
from rich import print

# from whisperx.diarize import DiarizationPipeline  # type: ignore


from translatevideo.utils.type_hints import LanguageNames
from translatevideo.utils.type_hints import LanguageCodes
from translatevideo.utils.type_hints import WhisperModels


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
                 #  delete_used_models: bool = True
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
        if chunk_audio_minutes is not None and chunk_audio_minutes <= 0:
            raise ValueError(
                f"{self.__class__.__name__}.chunk_audio_minutes must be positive value or None.")
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

    def _delete_object(self, object_: Any) -> None:
        # if self.delete_used_models:
        if object_:
            del object_
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
        # # 오디오 청킹 및 제너레이터 순회를 여기서 담당. 아래 함수들은 그대로 유지
        # # 아래 전체에 for 문 씌움(오디오를 그냥 다 청킹해서 들고오면 메모리상으로 다를바가 없을거임 아마)
        # if self.chunk_audio_minutes is None:
        #     audio = self.load_audio(audio_file)
        #     print("[green][Info][/] Starting transcription without chunking...")
        #     result = self.transcribe(audio)
        #     language_name = self.return_language_name(result)
        #     print("[green][Info][/] Starting alignment...")
        #     result = self.align(result, audio)
        #     if use_diarization:
        #         print("[green][Info][/] Starting diarization...")
        #         result = self.diarize(audio, result)
        #     return result, language_name
        # ---------------------------
    # 스트리밍 청크 전사
    # ---------------------------
        all_segments = []
        detected_language = None

        for seg, lang in self._chunk_transcription_generator(audio_file):
            all_segments.append(seg)
            detected_language = lang

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

        # self.delete_object(self.asr_model)
        return result

    def chunk_transcribe_generator(self, audio_file: str | Path):
        """
        Generator that:
        - loads small audio windows
        - runs WhisperX
        - shifts timestamps
        - yields segments in absolute time
        - frees memory after each chunk
        """
        if self.chunk_audio_minutes is None:
            raise ValueError("if chunk_audio_minutes is None, please use 'transcribe' method.")
        
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
            del audio_chunk, chunk_result
            gc.collect()
            torch.cuda.empty_cache()

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
