# from huggingface_hub import list_models
# 나중에 한번에 처리할 토큰수도 설정
# 자동 언어 인식은 marian에서는 애매함


class Translator:
    def __init__(self) -> None:
        pass

    def translate(self, texts: list[str]) -> str:
        raise NotImplementedError("Translator.translate() method must be overrided.")

# 
# class HFTranslaterVino(Translater):
    # def __init__(self):
        # pass
# 

# class MarianMT():
    # def __init__(self, src: str, tgt: str, device: str = "CPU") -> None:
        # self.model_name = self.select_model(src, tgt)
        # self.device = device
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.ov_model: Optional[OVModelForSeq2SeqLM] = None
# 
    # def check_models(self, src: str = "", tgt: str = "") -> list[str]:
        # model_list = list_models()
# 
        # org = "Helsinki-NLP"
        # model_ids: list[str] = [
            # x.id for x in model_list if x.id.startswith(org)]
# 
        # model_ids = [
            # mid for mid in model_ids
            # if (not src or mid.rsplit("-", 2)[1] == src)
            # and (not tgt or mid.rsplit("-", 2)[2] == tgt)
        # ]
# 
        # suffix = [x.split("/")[1] for x in model_ids]
        # old_style_multi_models = [
        # f"{org}/{s}" for s in suffix if s != s.lower()]
        # return model_ids
# 
    # def select_model(self, src: str, tgt: str) -> str:
        # name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        # if not name in self.check_models():
            # raise ValueError(f"Model {name} not found.")
        # return name
# 
    # def translate(self, text):
        # pass
# 
