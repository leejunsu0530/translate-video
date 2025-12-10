print("실행1")

from transformers import(
    EncoderDecoderModel,
    PreTrainedTokenizerFast,
    BertJapaneseTokenizer
)

# import torch
print("아니왜노트북에선메모리가터져")


encoder_model_name = "cl-tohoku/bert-base-japanese-v2"
decoder_model_name = "skt/kogpt2-base-v2"

src_tokenizer = BertJapaneseTokenizer.from_pretrained(encoder_model_name)
trg_tokenizer = PreTrainedTokenizerFast.from_pretrained(decoder_model_name)

model = EncoderDecoderModel.from_pretrained("sappho192/aihub-ja-ko-translator")

text = "初めまして。よろしくお願いします。"

def translate(text_src):
    embeddings = src_tokenizer(text_src, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
    embeddings = {k: v for k, v in embeddings.items()}
    output = model.generate(**embeddings, max_length=500)[0, 1:-1]
    text_trg = trg_tokenizer.decode(output.cpu())
    return text_trg

print(translate(text))
