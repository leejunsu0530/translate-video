import subprocess
from pathlib import Path
import sys
import os
from typing import Callable


def pwcpp_openvino_update(binary_path: str | Path, print_: Callable = print) -> None:
    if not binary_path:
        raise ValueError("openvino binary_path must be provided")
    setupvars_path = Path(binary_path) / "setupvars.bat"
    if not setupvars_path.exists():
        raise FileNotFoundError(f"setupvars.bat not found in {binary_path}")

    env = os.environ.copy()
    env["WHISPER_OPENVINO"] = "1"
    cmd = f"echo $env:WHISPER_OPENVINO && call {setupvars_path} && {sys.executable} -m pip install git+https://github.com/absadiki/pywhispercpp --no-cache --force-reinstall"
    print_(f"Running command: {cmd}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,  # 문자열로 디코딩
        bufsize=1,  # 줄 단위로 출력
        shell=True,
        env=env
    )

    if proc.stdout:
        for line in proc.stdout:
            print_(line, end='')  # 실시간 출력

        proc.wait()


if __name__ == "__main__":
    pwcpp_openvino_update(
        r"C:\Users\user\w_openvino_toolkit_windows_2024.6.0.17404.4c0f47d2335_x86_64")
