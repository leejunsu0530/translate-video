from transformers import (
    EncoderDecoderModel,
    BertJapaneseTokenizer,
    GPT2TokenizerFast
)
from .translate import Tr
import torch


class AIHubTranslator:
    def __init__(self, max_length: int = 500) -> None:
        encoder_model_name = "cl-tohoku/bert-base-japanese-v2"
        decoder_model_name = "skt/kogpt2-base-v2"
        model_name = "sappho192/aihub-ja-ko-translator"

        self.max_length = max_length

        self.src_tokenizer = BertJapaneseTokenizer.from_pretrained(
            encoder_model_name)  # 이거 타입힌트 필요
        self.trg_tokenizer = GPT2TokenizerFast.from_pretrained(
            decoder_model_name)
        self.model = EncoderDecoderModel.from_pretrained(model_name)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # 추론 모드로 변경(훈련용 동작 끔)

    def translate(self, texts: list[str]) -> list[str]:
        embeddings = self.src_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        embeddings = {k: v.to(self.device) for k, v in embeddings.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **embeddings,
                max_length=self.max_length,
                num_beams=1,
                do_sample=False  # 이 옵션들이 뭔지 모르겠음. 설명 필요
            )
            # text_trg = trg_tokenizer.decode(output.cpu()) # 여긴 output을 cpu로 올렸는데 이거 안함?
        result_texts = self.trg_tokenizer.batch_decode
        # 여기는 첫번째 질문까지임. 추가적인 작업 및 테스트 필요
        return texts
