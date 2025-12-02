# translate-video

오디오를 전사하고, 이미지와 오디오의 텍스트를 번역하여 ass 자막파일로 만들거나 영상에 삽입하는 파이썬 라이브러리

## features

1. whisper.cpp와 whisperX 기반 오디오 전사
2. PaddleOCR을 통한 비디오 텍스트 검출 및 인식
3. HuggingFace Hub를 통한 다국어 텍스트 번역
4. 생성, 번역 결과를 원본 비디오의 글자를 impaint하고 삽입하거나, ass 형식의 자막 파일로 생성

## Installation

    - **Note**: 현재 개발은 **Windows**와 **Intel GPU** 사용 환경을 중점적으로 진행하고 있으며, 추후 cuda도 지원할 예정임.

1. Dependency library

   다른 종속성은 라이브러리에서 자동으로 설치하지만 paddlepaddle, torch, pywhispercpp는 최상의 결과를 위해서는 수동 설치가 필요하다.

   - paddlepaddle 설치: [공식 사이트 설치 가이드](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/windows-pip_en.html)에서 자신의 환경에 맞게 설정하면 설치 명령어가 아래와 같이 나온다.
