from huggingface_hub import list_models
# 나중에 한번에 처리할 토큰수도 설정
# 자동 언어 인식은 marian에서는 애매함


class Translater:
    def __init__(self) -> None:
        pass

    def translate(self, text: str) -> str:
        return ""


class HFTranslaterVino(Translater):
    def __init__(self):
        pass


class MarianMT():
    def __init__(self, src: str, tgt: str, device: str = "CPU") -> None:
        self.model_name = self.select_model(src, tgt)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.ov_model: Optional[OVModelForSeq2SeqLM] = None

    def check_models(self, src: str = "", tgt: str = "") -> list[str]:
        model_list = list_models()

        org = "Helsinki-NLP"
        model_ids: list[str] = [
            x.id for x in model_list if x.id.startswith(org)]

        model_ids = [
            mid for mid in model_ids
            if (not src or mid.rsplit("-", 2)[1] == src)
            and (not tgt or mid.rsplit("-", 2)[2] == tgt)
        ]

        # suffix = [x.split("/")[1] for x in model_ids]
        # old_style_multi_models = [
        # f"{org}/{s}" for s in suffix if s != s.lower()]
        return model_ids

    def select_model(self, src: str, tgt: str) -> str:
        name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        if not name in self.check_models():
            raise ValueError(f"Model {name} not found.")
        return name

    def translate(self, text):
        pass


from optimum.intel.openvino import OVModelForSeq2SeqLM
from transformers import (
    BertJapaneseTokenizerFast,
    PreTrainedTokenizerFast,
)
import torch

# ✅ 모델과 토크나이저 이름
encoder_model_name = "cl-tohoku/bert-base-japanese-v2"
decoder_model_name = "skt/kogpt2-base-v2"
model_id = "sappho192/aihub-ja-ko-translator"

# ✅ 인코더용 Fast 토크나이저 사용 (Rust 기반)
src_tokenizer = BertJapaneseTokenizerFast.from_pretrained(encoder_model_name)
# ✅ 디코더용 Fast 토크나이저 (GPT2 계열은 이미 빠름)
trg_tokenizer = PreTrainedTokenizerFast.from_pretrained(decoder_model_name)

# ✅ OpenVINO 모델 로드 (FP16 반정밀도 사용)
model = OVModelForSeq2SeqLM.from_pretrained(
    model_id,
    export=True,          # 아직 IR 파일이 없으면 자동 변환
    ov_config={"INFERENCE_PRECISION_HINT": "f16"},  # FP16 모드
)

# ✅ 디바이스 선택 (CPU, GPU, AUTO)
model.to("AUTO")

# ✅ 번역 함수
def translate_batch(texts):
    # 효율적인 토큰화
    inputs = src_tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=256,
        num_threads=4,
    )

    # 추론 시 불필요한 그래디언트 계산 비활성화
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
        )

    # 결과 디코딩 (배치로 한 번에)
    decoded = trg_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    return decoded

# ✅ 사용 예시
texts = [
    "初めまして。よろしくお願いします。",
    "これはテストです。",
    "天気がいいですね。"
]

results = translate_batch(texts)
for ja, ko in zip(texts, results):
    print(f"{ja} → {ko}")
