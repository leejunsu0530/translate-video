# TOC
- [기능](#기능)
- [TODO](#TODO)
- [전사 파이프라인](#전사-파이프라인)
    - [오디오 로딩](#오디오-로딩)
    - [VAD](#vad)
    - [배치 전사](#배치-전사)
    - [강제 정렬](#강제-정렬)
    - [화자 분리](#화자-분리)
- [whisper 모델 성능 개선](whisper-모델-성능-개선)
    - [가속 방법](#가속-방법)
    - [파인튜닝](#파인튜닝)
        - [어댑터(lora)](#어댑터lora)
        - [kotoba-whisper](#kotoba-whisper)
        - [distil-whisper](#distil-whisper)
    - [양자화](#양자화)
- [번역](#번역)
- [기타](#기타)

---
# 기능
- 유튜브나 영상의 창을 띄우고, 투입할 오디오 부분을 선택하거나 특정 시점을 지정할 용도의, 1점, 2점 포인트.
- 오디오 분리 등의 ffmpeg 기능
- 다운로드 등의 yt-dlp 기능
    - 더 효율적인 다운로드 과정 등
    - 챕터 제작, 챕터별 썸네일 등 추가 PP
- 전처리(배경음악 제거, VAD 등)
- 파인튜닝 프레임워크
    - 오디오-원문-번역문의 자동 생성 및 수정 기능
- 전사
- 후처리(화자분리, 강제 정렬)
    - 자막을 문장 단위로 구분.
    - ai 기반 기능(문장 부호를 넣어주는 등의 후처리)

## 구현
1. 원래 계획: 오디오를 넣으면 wx가 기본 처리-VAD,전사-를 해서 편집 프로그램같은 창을 띄움. 여기서 사용자는 잘못된 VAD 구간을 수정하거나 잘못 전사된 부분을 바로잡음. 
2. ~~문제: vad 부분이 wx의 전사에 통합되어 있어서 이걸 어떻게 처리할지 모르겠다.~~ 
    - 그냥 전사 깔끔하게 한번 처리하고, 그 뒤론 아예 수동으로. 사용자가 수정한 걸 다시 전사하거나 할 필요 없다. 
    - 이 경우엔, VAD가 되고 바로 내부로 데이터가 들어가서 VAD 처리가 잘못 되었을 때를 전사가 끝나기 전까지 수정할 수 없다는 문제가 발생
    - 해결책: wx에 vad를 끄고, 배경음 제거와 vad를 거친 데이터를 넣는다. 가능할진 모름.
3. 별로 중요한 건 아니긴 한데, wx를 쓰지 않고 손으로 슬라이더를 움직이고 타이핑을 하고 싶으면, 그냥 할수 있도록 한다.
4. 추가로, 배경음 제거는 없어서 내가 하는걸로.

# TODO
- 다른 기초 작업 하기
- 파이프라인: 특정 인물 특화 어댑터 적용 kotoba whisper > wx로 전사(옵션 프리셋 추가) > [llama나 qwen으로 수정](https://gemini.google.com/share/e9fa3524b71d)(사용자 정의 사전 등)


- [로깅 만들기](https://gemini.google.com/share/bcf7563b2530)
- 유튜브 부분 제작
- colab 부분 제작
- 통합 파이프라인 패이지 외에 각 기능.별도 페이지 제작. 유튜브 다운로드, 이미 있는 문장으로 강제정렬 등
- 플리 정보는 올바르게 가져오면서 영상 정보는 안 가져오거나 하는게 있나? extract-flat이면 상한에 잘 안 걸릴 수는 있는데 정보가 부족해짐. 플리로써가 아니라 개별 영상으로써 다운받는 거면 상관없지만.
    - 현재 목표했던, 자세한 다운로드 방법: **flat없이, 내가 기록하는 게 아니라 write-info-json 사용해서 개별 영상 기록.** 혹시 시점을 뒤로 당기면(이게 되는지부터 의문이지만) 내가 기록할 때도 플리 데이터도 들어가나?(물론 이 경우에도 flat 없는 기준)
    - ~~[flat없이, 자동 기록으로 플리&영상 데이터 적힌 경우](./test/output/auto_write_no_flat/문과&이과%20[PLDUgRz_joPOUCB0kBGAM9nVA8LEISGiDi].info.json)에서, 플리만 자세하게 저장하고 영상은 안 불러온 채 끝낼 수 있으면?~~ 이건 그냥 flat한 플리데이터임. 내부 구조 분석해서 수동과 무슨 차이인건지 찾기. 그리고 플리 있도록 하는 경우에 추가되는 데이터는 아마 이게 전부인 것 같은데, 플리 업로더는 그냥 플리에서의 업로더(채널이 아니라)로 해서, 인덱스(스레딩하는 경우에 애매함)만 빼고 나머지는 확실하게 내가 수동으로 넣을 수 있음.
        ```json
            "playlist_count": 7,
            "playlist": "문과&이과",
            "playlist_id": "PLDUgRz_joPOUCB0kBGAM9nVA8LEISGiDi",
            "playlist_title": "문과&이과",
            "playlist_uploader": "문과vs이과",
            "playlist_uploader_id": "@vs-bi8bi",
            "playlist_channel": "문과vs이과",
            "playlist_channel_id": "UCa4Dk0DO3-n7rpeeGYNHnSg",
            "playlist_webpage_url": "https://www.youtube.com/playlist?list=PLDUgRz_joPOUCB0kBGAM9nVA8LEISGiDi",
            "n_entries": 7,
            "playlist_index": 1,
            "display_id": "5LaheFrkDjA",

        ```
    - 테스트할 것:
        - 수동 기록의 경우에 옵션을 넣어도 왜 개별 영상을 작성 안하는지(왜 달라지는지)
        - 개별 영상에 url 등의 방법으로 플리 데이터를 추가할 수 있는지
        - 가속 되는지
        - codex 사용해서 구조 분석
        - extract_info와 download에서의 기능 차이가 생기는지
        - extract_info의 download 옵션을 끄면 생기는 일(skip-download 있을 경우)
        - 자동기록일때 스레딩 같은거 될지
- [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz)사용한 yt-dlp 명령어 검색 및 자동완성 기능
    - 동적 gradio gui도 만들기
    - 카테고리별 검색기능
    - 설명 등으로도 검색
    - 옵션 추가인자 띄우기, 타입도 지원
    - alias, 프리셋 지원
    - 
- 정보 파일 다운로드는 저정 안하고 flat을 가져온 후, 폴더 내 파일 갯수로 진도 파악하는 방법도 있음
- 이미지 잔뜩 동적으로 gradio에서 띄워도 되나 -> O
- yt-dlp 업뎃마다 옵션 파싱해서 json 등으로 저장(런타임마다 하긴 무거울거니까).
- cli2api는 몽키패치 무시하려면 subprocess로?

- fw에서 배치와 배치 없이 두 경우에서 여러 모델 성능 테스트
- 프롬프트 기능 테스트
- fw 배치 파이프라인, wx 파이프라인 분석 후 배치별 문맥 고려형 범용 파이프라인 제작
- 코토바, wx, faster, pwcpp 등 긴 오디오에 밴치마크. 특히 코토바는 빠르긴 한데 성능도 나은지 확인
- wx 파이프라인 구조 분석. temp fallback나 문맥 유지가 끊겨있는 듯
- accelerate와 플래시 어텡션
---
# 전사 파이프라인
- 전사 파이프라인 만들기: 메모리 고려한 로딩, VAD, 배치 전사, 강제 정렬, 화자 분리

## 오디오 로딩
- 메모리를 고려하는 로딩 필요. 너무 크면 터지니까.
- vad를 거친 후에야 문장이 잘리지 않게 할 수 있음.

## VAD

## 배치 전사
- 사용 고려 중인 엔진: hf, pwcpp, wx, faster, ~~insanely~~(이건 분석은 해보겠지만 그냥 hf 최적화인듯)
- 가벼워야 배치 많이 만들어서 빨라진다.

## 강제 정렬
- 여기나 화자분리에서도 optimum, ct2 등의 최적화를 내가 적용할 수 있는지 >> 되면 아예 새 파이프라인으로 만들기

## 화자 분리

---

# whisper 모델 성능 개선
- 일단 hf는 복잡해서 보류. 특히 optimum intel은. 

## 가속 방법
일단 NPU는 사용 실패.
- ctranslate2
- optimum(ct2와 양립 불가) -> intel or cuda 둘다 가능.
    - intel cpu 말고 xpu 사용 
- onnx
- ggml > pwcpp
- 양자화
- torch의 intel 또는 cuda
- ~~파인튜닝(kotoba를 사용해서? https://ysg2997.tistory.com/53 참고)~~ 사실 이건 가속은 아니지. 중첩되니까.
- accelerate (?)
- flash_attention_2 (?)

참고:

```python
# insanely-faster-whisper에 있는 허깅페이스의 코드.
    model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
```

## 파인튜닝
1. kotoba를 어떻게든 가볍게 만듦
2. 어댑터를 적용함
3. 데이터셋을 사용하는 양자화로 각 체크포인트를 int8 양자화함
4. ct2 변환
**주의: 양자화는 lora 효과를 감소시킴. 특히 여러 겹일때는.**
작게 모델을 만들면 int8 필요 없을지도?

### 어댑터(LoRA)
일본어 모델에 구어체 어댑터, 특정 인물 어댑터 차례대로 적용

### kotoba-whisper
> 참고: https://github.com/kotoba-tech/kotoba-whisper

1. kotoba 더 최적화(더 잘라내기?)
2. 작은 모델(base, small)로 다시 학습
참고로 ct2 변환에 특정 파일들 만드는 명령어 없으면 터진다.

### distil-whisper
- distil-whisper ct2나 ggml 있는지 확인
영어만 가능함. [hf hub](https://huggingface.co/distil-whisper) 참고.

## 양자화
ct2는 자체 양자화 된다.

---

# 번역
- mbart, marian, nllb, JapaneseBart 등으로 품질 테스트, 일본어 구어체 튜닝 + 특정 인물/도메인 튜닝. 
- marianMT는 새 언어쌍 만들려면 몇백~몇천만 문장 데이터셋이 필요하다 함.
- nllb 또는 mbart + lora fine-tuning + int8로 사용
- 구글, deepl, gpt 지원

---

# 기타

## 이미지 처리 관련
- [ ] paddleocr 기반 이미지/영상 자막 생성 지원

## yt-dlp 관련
- [x] 플레이리스트를 개별 영상 파일들로 쪼개기
- [x] pp에 인자 전달 방법 및 기본 시점 정의하는 방법 찾기
- [ ] cli to api 몽키패치 해결    
- [ ] optparse로 ytdlp의 인자 설명 추가
- [ ] 커스텀 pp, 필터 추가
    - [ ] 챕터별 썸네일 pp
    - [ ] 자동자막 생성 pp
    - [ ] 챕터 자동 생성 pp 및 사용자 지정 챕터 삽입 pp
- [ ] 더 빠른 info 다운로드(멀티스레딩)
- [ ] 다운로드 중 유튜브 차단이 걸리면 해당 프로세스를 종료하고 다른 프로세스에서 이어감
- [ ] exe 파일로 사용할 경우에 라이브러리 업데이트가 안되니까, 외부 ytdlp 경로와 연결 가능하게 하기
- [ ] 버전 확인 및 업데이트 스크립트
- [ ] 외부 경로 쪽 관리 스크립트


