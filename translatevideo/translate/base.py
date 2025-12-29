from pathlib import Path
from ..utils.type_hints import LanguageNames


class TranslatorBase:
    def __init__(self,
                 src_lang: LanguageNames,
                 tgt_lang: LanguageNames,
                 batch_size: int = 4
                 ) -> None:
        """set source and target language codes. automatically converts language codes if necessary 
        """
        self.src_lang = self._convert_lang_code(src_lang)
        self.tgt_lang = self._convert_lang_code(tgt_lang)
        self.batch_size = batch_size

    def _convert_lang_code(self, language_name: LanguageNames) -> str:
        """번역기 별로 언어코드 변환. 필요없는 경우에는 사용하지 않음.
        만일 자식의 init에서 super 호출 이후에 여기서 사용할 변수를 지정하면 불러오지 못하므로 주의"""
        # 클래스 메서드로 딕셔너리 구현해두고 여기서 구체적으로 확인 후 전달/오류 띄우기까지 구현할까?
        return language_name

    def translate(self, texts: str) -> str:
        raise NotImplementedError(
            "Translator.translate method must be overrided.")

    def batch_translate(self, texts: list[str]) -> list[str]:
        """self.batch_size 단위로 texts를 나누어 번역 수행"""
        raise NotImplementedError(
            "Translator.batch_translate method must be overrided.")

    def translate_srt(self, srt_path: Path | str, output_path: Path | str) -> Path:
        """
        translate .srt subtitle file and return translated srt file path
        """
        # srt를 읽고 배치번역으로 번역함
        srt_path = Path(srt_path)
        with srt_path.open('r', encoding='utf-8') as f:
            # self.batch_translate()
            pass  # 읽고 형태 변환해서 처리
        with output_path.open('w', encoding='utf-8') as f:
            pass  # 번역된 내용 기록

        return Path(output_path)  # 번역된 srt 파일 경로 반환


class HFTranslatorBase(TranslatorBase):
    def __init__(self,
                 src_lang,
                 tgt_lang,
                 model_name: str,
                 device: str = 'cpu',
                 batch_size: int = 4
                 ) -> None:
        super().__init__(src_lang, tgt_lang, batch_size)
        self.model_name = model_name
        # 모델 로드 등 초기화 작업
