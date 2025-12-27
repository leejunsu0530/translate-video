from importlib.metadata import version, PackageNotFoundError
from .transcribe import WhisperXTranscriber

# versioning
try:
    __version__ = version("translate-video") # pyproject.toml의 name과 일치시킬 것
except PackageNotFoundError:  # 빌드 실패시
    __version__ = "0.0.0.dev0"


__all__ = ["WhisperXTranscriber"]
