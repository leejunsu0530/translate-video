from deep_translator import GoogleTranslator
from .base import TranslatorBase
from ..utils.type_hints import LanguageNames

"""언어는 언어 이름(english)이나 약어(en)로 전달 가능"""

class GoogleTR(TranslatorBase):
    def __init__(self,
                 src_lang="auto",
                 tgt_lang="en",
                 batch_size=4
                 ):
        self.translator = GoogleTranslator() # convert 함수에서 사용하기 위해 미리 생성
        super().__init__(src_lang, tgt_lang, batch_size)
        
        
    def _convert_lang_code(self, language_name) -> str:
        supported:dict[LanguageNames,str] = self.translator.get_supported_languages(as_dict=True)
        language_code = supported.get(language_name)
        if language_code is None:
            raise ValueError(f"Google Translator does not support language: {language_name}")

        return language_code

    def translate(self, texts):
        return super().translate(texts)

    def batch_translate(self, texts):
        return super().batch_translate(texts)
