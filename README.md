# translate-video

오디오를 전사하고, 이미지와 오디오의 텍스트를 번역하여 ass 자막파일로 만들거나 영상에 삽입하는 파이썬 라이브러리

## features

1. [WhisperX](https://github.com/m-bain/whisperX) 기반 오디오 전사
2. PaddleOCR을 통한 비디오 텍스트 검출 및 인식
3. HuggingFace Hub를 통한 다국어 텍스트 번역
4. 생성, 번역 결과를 원본 비디오의 글자를 impaint하고 삽입하거나, ass 형식의 자막 파일로 생성

## Installation
> **Note**: 현재 개발은 **Windows**와 **Intel GPU** 사용 환경을 중점적으로 진행하고 있으며, 추후 cuda도 지원할 예정임.

1. Outer Dependencies

   yt-dlp를 위해서 FFmpeg, FFprove, Deno를 설치해야 한다.
   
   <details open>
      <summary><b>FFmpeg & FFprove 설치</b></summary>
      FFprove는 FFmpeg를 설치하면 같이 설치된다.
      Windows에서는 아래의 명령어로 설치할 수 있다.

      ```powershell
      # on Windows using Chocolatey (https://chocolatey.org/)
      choco install ffmpeg

      # on Windows using Scoop (https://scoop.sh/)
      scoop install ffmpeg
      ```
      위의 명령어에 실패하면 수동 설치를 할 수도 있다.
      [Windows builds from gyan.dev](https://www.gyan.dev/ffmpeg/builds/) 또는 [Windows builds by BtbN](https://github.com/BtbN/FFmpeg-Builds/releases)에 들어가서 압축 파일을 다운받아 원하는 폴더에 압축 해제하고 해당 폴더 내 bin 폴더를 시스템 환경변수의 Path에 설정하면 된다.

      yt-dlp는 FFmpeg의 일부 버그를 수정한 [빌드](https://github.com/yt-dlp/FFmpeg-Builds#ffmpeg-static-auto-builds)를 제공한다. 해당 링크에서 다운받는 것이 더 원할한 동작을 지원할 수 있다.
   </details>

   <details open>
      <summary><b>Deno 설치</b></summary>
      Deno는 yt-dlp의 JS 관련 의존성인 [yt-dlp-ejs](https://github.com/yt-dlp/ejs)의 의존성으로, 링크에서 알 수 있듯이 Node, Bun, QuickJS로 대체 가능하지만 Deno가 추천된다.

      https://docs.deno.com/runtime/getting_started/installation/ 에서 설치 방법을 알 수 있다. Windows의 경우에는 아래 명령어로 설치한다.

      ```powershell
      # Using PowerShell (Windows):
      irm https://deno.land/install.ps1 | iex

      # Using Scoop:
      scoop install deno

      # Using Chocolatey:
      choco install deno
      ```

      추가 정보는 yt-dlp의 [Wiki](https://github.com/yt-dlp/yt-dlp/wiki/EJS)를 참조하라.
   </details>

2. Installing Dependency Library

   다른 종속성 라이브러리는 이 라이브러리에서 자동으로 설치하지만 paddlepaddle과 torch는 최상의 결과를 위해서 수동 설치가 필요하다.

   <details open>
      <summary><b>paddlepaddle 설치</b></summary>
      
      [공식 사이트 설치 가이드](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/windows-pip_en.html)
   에서 자신의 환경에 맞게 설정하면 설치 명령어가 아래와 같이 나온다.

   ```powershell
      python -m pip install paddlepaddle==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
   ```    
   </details>

   <details open>
      <summary><b>torch 설치</b></summary>
      자신에게 맞는 index url을 지정해서 설치하면 된다. 예를 들어, Intel GPU를 사용할 경우에는 아래 명령어를 사용하면 된다.
      
      ```powershell
      python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
      ```
   </details>
   

3. Installing translate-video
   
   추후 pypi에서도 지원할 예정
   ```powershell
   python -m pip install git+https://github.com/leejunsu0530/translate-video
   ```

## Usage
### python
### GUI(Gradio)

## TODO

## License
이 프로젝트는 MIT 라이선스 하에 보호받고 있습니다. 자세한 정보는 LICENSE 파일을 참조하세요.

This project is licensed under the MIT License.
See the LICENSE file for details.

