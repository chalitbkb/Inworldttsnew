"""Input text normalization."""

import abc
import logging as base_logging

import lingua
import unidecode
from nemo_text_processing.text_normalization import normalize

base_logging.getLogger("NeMo-text-processing").setLevel(base_logging.CRITICAL)

# Language codes for text normalization.
# Language codes for text normalization.
_ENGLISH = "en"
_JAPANESE = "ja"
_CHINESE = "zh"
_SPANISH = "es"
_FRENCH = "fr"
_GERMAN = "de"
_THAI = "th"  # Adding Thai language identifier

# Try to import pythainlp for Thai normalization
try:
    from pythainlp.util import normalize as thai_normalize
    _THAI_SUPPORTED = True
except ImportError:
    base_logging.warning(
        "pythainlp is not installed. "
        "Thai text normalization will act as a fallback No-Op wrapper."
    )


class TextNormalizer(metaclass=abc.ABCMeta):
    """Text normalization class for normalizers to implement."""

    @abc.abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize"""
        raise NotImplementedError("|normalize| is not implemented.")

    @abc.abstractmethod
    def get_supported_languages(self) -> list[str]:
        """Get supported languages."""
        raise NotImplementedError("|get_supported_languages| is not implemented.")

    @abc.abstractmethod
    def normalize_with_language(self, text: str, language: str) -> str:
        """Normalize text with a specific language."""
        raise NotImplementedError("|normalize_with_language| is not implemented.")


class NoOpTextNormalizer(TextNormalizer):
    """No-op text normalizer."""

    def normalize(self, text: str) -> str:
        return text

    def get_supported_languages(self) -> list[str]:
        return []

    def normalize_with_language(self, text: str, language: str) -> str:
        return text


class ThaiTextNormalizer(TextNormalizer):
    """Text normalizer for Thai language using PyThaiNLP."""

    def __init__(self):
        super().__init__()
        self._supported_languages = [_THAI]

    def get_supported_languages(self) -> list[str]:
        return self._supported_languages

    def normalize(self, text: str) -> str:
        if _THAI_SUPPORTED:
            # use pythainlp to clean up duplicate vowels/tones
            return thai_normalize(text)
        return text

    def normalize_with_language(self, text: str, language: str) -> str:
        if language == _THAI:
            return self.normalize(text)
        return text


class MultiLingualTextNormalizer(TextNormalizer):
    """Router text normalizer for dispatching to specific normalizers."""

    def __init__(self):
        super().__init__()
        self.nemo_normalizer = NemoTextNormalizer()
        self.thai_normalizer = ThaiTextNormalizer()

        self._supported_languages = (
            self.nemo_normalizer.get_supported_languages()
            + self.thai_normalizer.get_supported_languages()
        )

    def get_supported_languages(self) -> list[str]:
        return self._supported_languages

    def normalize(self, text: str) -> str:
        # Check through lang detector used by nemo
        try:
            if self.nemo_normalizer.lang_detector is None:
                self.nemo_normalizer.init_lang_detector()
            self.nemo_normalizer.lang_detector.detect_language_of(text)

            # Note: lingua currently does not support Thai language natively,
            # so dynamically detecting Thai text using lingua will often fall to
            # English or unclassified. If multi-lingual inference is needed,
            # a better language detector might be necessary.

            return self.nemo_normalizer.normalize(text)
        except Exception:
            return text

    def normalize_with_language(self, text: str, language: str) -> str:
        if language == _THAI:
            return self.thai_normalizer.normalize_with_language(text, language)
        return self.nemo_normalizer.normalize_with_language(text, language)


class NemoTextNormalizer(TextNormalizer):
    """Text normalizer for different languages using Nvidias NeMo text normalization
    library."""

    def __init__(self):
        super().__init__()
        self._supported_languages = [
            _ENGLISH,
            _JAPANESE,
            _CHINESE,
            _SPANISH,
            _FRENCH,
            _GERMAN,
        ]
        self._normalize_text = {
            lang: normalize.Normalizer(input_case="cased", lang=lang)
            for lang in self._supported_languages
        }
        self.lang_detector = None

    def init_lang_detector(self):
        self.lang_detector = lingua.LanguageDetectorBuilder.from_languages(
            lingua.Language.KOREAN,
            lingua.Language.JAPANESE,
            lingua.Language.CHINESE,
            lingua.Language.ENGLISH,
            lingua.Language.SPANISH,
            lingua.Language.FRENCH,
            lingua.Language.GERMAN,
        ).build()

    def convert_to_ascii(self, text: str) -> str:
        return unidecode.unidecode(text)

    def get_supported_languages(self) -> list[str]:
        return self._supported_languages

    def normalize(self, text: str) -> str:
        # detect language and normalize text
        try:
            # Only initialize the language detector if it's not already initialized
            # (dynamic language detection).
            if self.lang_detector is None:
                self.init_lang_detector()
            language = self.lang_detector.detect_language_of(text)
            if language == lingua.Language.ENGLISH:
                return self.normalize_with_language(text, _ENGLISH)
            elif language == lingua.Language.JAPANESE:
                return self.normalize_with_language(text, _JAPANESE)
            elif language == lingua.Language.CHINESE:
                return self.normalize_with_language(text, _CHINESE)
            elif language == lingua.Language.SPANISH:
                return self.normalize_with_language(text, _SPANISH)
            elif language == lingua.Language.FRENCH:
                return self.normalize_with_language(text, _FRENCH)
            elif language == lingua.Language.GERMAN:
                return self.normalize_with_language(text, _GERMAN)
            else:
                return text
        except Exception:
            return text

    def normalize_with_language(self, text: str, language: str) -> str:
        if language not in self._supported_languages:
            return text

        if language == _ENGLISH:
            text = self.convert_to_ascii(text)

        try:
            text = self._normalize_text[language].normalize(text)
        except Exception:
            # return the unnormalized text if error
            return text
        return text


def create_text_normalizer(enable_text_normalization: bool) -> TextNormalizer:
    """Create text normalizer for all languages."""
    if enable_text_normalization:
        return MultiLingualTextNormalizer()
    else:
        return NoOpTextNormalizer()
