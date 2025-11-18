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