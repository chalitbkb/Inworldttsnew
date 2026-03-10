"""Input text normalization."""

import abc
import logging as base_logging

import lingua
import unidecode
from nemo_text_processing.text_normalization import normalize

base_logging.getLogger("NeMo-text-processing").setLevel(base_logging.CRITICAL)

try:
    from pythainlp.util import normalize as thai_normalize
    _THAI_SUPPORTED = True
except ImportError:
    base_logging.warning(
        "pythainlp is not installed. "
        "Thai text normalization will act as a fallback No-Op wrapper."
    )
    _THAI_SUPPORTED = False

# Mapping from ISO 639-1 language codes to lingua Language enum.
# Add new languages here to automatically support them across the pipeline.
_LINGUA_LANG_MAP: dict[str, lingua.Language] = {
    "en": lingua.Language.ENGLISH,
    "ja": lingua.Language.JAPANESE,
    "zh": lingua.Language.CHINESE,
    "es": lingua.Language.SPANISH,
    "fr": lingua.Language.FRENCH,
    "de": lingua.Language.GERMAN,
    "ko": lingua.Language.KOREAN,
    "th": lingua.Language.THAI,
    "pt": lingua.Language.PORTUGUESE,
    "ru": lingua.Language.RUSSIAN,
    "it": lingua.Language.ITALIAN,
    "nl": lingua.Language.DUTCH,
    "pl": lingua.Language.POLISH,
    "ar": lingua.Language.ARABIC,
    "hi": lingua.Language.HINDI,
    "vi": lingua.Language.VIETNAMESE,
    "id": lingua.Language.INDONESIAN,
    "tr": lingua.Language.TURKISH,
    "sv": lingua.Language.SWEDISH,
    "lo": lingua.Language.LAO,
    "my": lingua.Language.BURMESE,
    "km": lingua.Language.KHMER,
}

# Reverse mapping: lingua Language -> language code.
_LINGUA_LANG_REVERSE_MAP: dict[lingua.Language, str] = {
    v: k for k, v in _LINGUA_LANG_MAP.items()
}

# Languages that NeMo text normalizer supports natively.
_NEMO_SUPPORTED_LANGS = {"en", "ja", "zh", "es", "fr", "de"}

# Languages where ASCII conversion should be applied before normalization.
_ASCII_CONVERT_LANGS = {"en"}


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


class MultiLingualTextNormalizer(TextNormalizer):
    """Router text normalizer for dispatching to specific normalizers."""

    def __init__(self):
        super().__init__()
        # All languages registered in _LINGUA_LANG_MAP are considered "supported".
        self._supported_languages = list(_LINGUA_LANG_MAP.keys())

        # Only create NeMo normalizers for languages that NeMo actually supports.
        self._nemo_normalizers = {
            lang: normalize.Normalizer(input_case="cased", lang=lang)
            for lang in _NEMO_SUPPORTED_LANGS
        }
        self.lang_detector = None

    def init_lang_detector(self):
        self.lang_detector = lingua.LanguageDetectorBuilder.from_languages(
            *_LINGUA_LANG_MAP.values()
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
            detected_lang = self.lang_detector.detect_language_of(text)

            # Look up the ISO language code from the detected lingua Language
            lang_code = _LINGUA_LANG_REVERSE_MAP.get(detected_lang)
            if lang_code:
                return self.normalize_with_language(text, lang_code)
            else:
                return text
        except Exception:
            return text

    def normalize_with_language(self, text: str, language: str) -> str:
        # If the language isn't in our supported set, pass through.
        if language not in _LINGUA_LANG_MAP:
            return text

        # Apply ASCII conversion for languages that need it.
        if language in _ASCII_CONVERT_LANGS:
            text = self.convert_to_ascii(text)

        # Use NeMo normalizer if available for this language
        if language in self._nemo_normalizers:
            try:
                text = self._nemo_normalizers[language].normalize(text)
            except Exception:
                return text
        # Special case wrapper for Thai via PyThaiNLP
        elif language == "th" and _THAI_SUPPORTED:
            return thai_normalize(text)

        return text


def create_text_normalizer(enable_text_normalization: bool) -> TextNormalizer:
    """Create text normalizer."""
    if enable_text_normalization:
        return MultiLingualTextNormalizer()
    else:
        return NoOpTextNormalizer()
