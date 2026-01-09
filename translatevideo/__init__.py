# pylint: disable=wrong-import-position
from importlib.metadata import version, PackageNotFoundError
import warnings  # surpress torchaudio deprecation warning
print("[Note] torchaudio._backend.list_audio_backends deprecation warning is suppressed.")
warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*torchaudio._backend.list_audio_backends has been deprecated.*")
from .transcribe import WhisperXTranscriber  # noqa: E402

# versioning
try:
    __version__ = version("translate-video")  # pyproject.toml의 name과 일치시킬 것
except PackageNotFoundError:  # 빌드 실패시
    __version__ = "0.0.0.dev0"


__all__ = ["WhisperXTranscriber"]


def main():
    pass
