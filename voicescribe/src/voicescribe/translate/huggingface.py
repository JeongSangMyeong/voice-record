"""Hugging Face transformers 기반 번역기.

기본 모델은 ``facebook/m2m100_418M`` — MIT 라이선스(상업적 이용 가능), 100개 언어를
모델 하나로 처리한다. 필요하면 ``--translation-model`` 로 다른 모델을 지정할 수 있다.

주의: NLLB-200 계열은 성능이 좋지만 **CC-BY-NC(비상업)** 라이선스이므로 기본값으로
쓰지 않는다. 사용자가 명시적으로 지정할 때만 쓴다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import Translator, TranslatorNotAvailableError

#: 기본 모델(상업적 이용 가능).
DEFAULT_MODEL = "facebook/m2m100_418M"

#: 참고용 모델 목록과 라이선스.
MODEL_NOTES: dict[str, str] = {
    "facebook/m2m100_418M": "MIT · 100개 언어 · 약 1.9GB · 상업적 이용 가능",
    "facebook/m2m100_1.2B": "MIT · 100개 언어 · 약 5GB · 더 정확하지만 느림",
    "facebook/nllb-200-distilled-600M": "CC-BY-NC(비상업) · 200개 언어 · 약 2.5GB",
}

#: M2M100 은 대부분 ISO 639-1 코드를 그대로 쓰지만 일부만 다르다.
_M2M100_OVERRIDES = {"jw": "jv", "yue": "zh"}

#: NLLB 는 독자적인 코드 체계(FLORES-200)를 쓴다. 자주 쓰는 것만 매핑한다.
_NLLB_CODES = {
    "ko": "kor_Hang", "en": "eng_Latn", "ja": "jpn_Jpan", "zh": "zho_Hans",
    "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn", "ru": "rus_Cyrl",
    "pt": "por_Latn", "it": "ita_Latn", "vi": "vie_Latn", "th": "tha_Thai",
    "id": "ind_Latn", "ar": "arb_Arab", "hi": "hin_Deva", "tr": "tur_Latn",
    "pl": "pol_Latn", "nl": "nld_Latn", "uk": "ukr_Cyrl",
}


class HuggingFaceTranslator(Translator):
    """transformers 다국어 번역 모델 래퍼."""

    name = "hf"
    description = "무료·로컬 신경망 번역(M2M100 기본, 100개 언어). PyTorch 필요, 모델 약 1.9GB."
    needs_download = True

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded_name: str | None = None

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def install_hint(self) -> str:
        return (
            "설치 방법:\n"
            '  pip install "voicescribe[translate-hf]"\n'
            "  (또는 직접: pip install transformers torch sentencepiece)\n"
            "※ PyTorch 를 함께 내려받으므로 용량이 큽니다."
        )

    # ------------------------------------------------------------------ #

    @property
    def _is_nllb(self) -> bool:
        return "nllb" in self.model_name.lower()

    def _lang_code(self, code: str) -> str:
        """모델이 요구하는 언어 코드로 바꾼다."""
        if self._is_nllb:
            mapped = _NLLB_CODES.get(code)
            if mapped is None:
                raise TranslatorNotAvailableError(
                    f"NLLB 모델에서 '{code}' 언어 코드를 아직 매핑하지 않았습니다. "
                    "M2M100(기본값)을 사용하거나 코드를 직접 지정하세요."
                )
            return mapped
        return _M2M100_OVERRIDES.get(code, code)

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._loaded_name == self.model_name:
            return self._tokenizer, self._model

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        except Exception as exc:
            raise TranslatorNotAvailableError(
                f"번역 모델 '{self.model_name}' 을(를) 불러오지 못했습니다: {exc}\n"
                "인터넷 연결과 디스크 여유 공간을 확인하세요."
            ) from exc

        model.eval()
        self._tokenizer, self._model, self._loaded_name = tokenizer, model, self.model_name
        return tokenizer, model

    def _forced_bos_id(self, tokenizer: Any, target_code: str) -> int | None:
        """생성 시 강제할 목표 언어 토큰 id 를 구한다(모델 계열마다 방식이 다르다)."""
        getter = getattr(tokenizer, "get_lang_id", None)
        if callable(getter):  # M2M100
            return int(getter(target_code))
        lang_map = getattr(tokenizer, "lang_code_to_id", None)
        if isinstance(lang_map, dict) and target_code in lang_map:  # 구버전 NLLB
            return int(lang_map[target_code])
        token_id = tokenizer.convert_tokens_to_ids(target_code)  # 최신 NLLB
        unk = getattr(tokenizer, "unk_token_id", None)
        if token_id is not None and token_id != unk:
            return int(token_id)
        return None

    # ------------------------------------------------------------------ #

    def translate_batch(self, texts: Sequence[str], source: str, target: str) -> list[str]:
        self.ensure_available()
        if source == target:
            return list(texts)

        import torch

        tokenizer, model = self._load()
        src_code = self._lang_code(source)
        tgt_code = self._lang_code(target)
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = src_code

        forced_bos = self._forced_bos_id(tokenizer, tgt_code)
        results: list[str] = []
        batch_size = 8  # CPU 메모리를 고려한 보수적인 크기

        for start in range(0, len(texts), batch_size):
            chunk = [t.strip() for t in texts[start : start + batch_size]]
            non_empty = [(i, t) for i, t in enumerate(chunk) if t]
            translated = [""] * len(chunk)
            if non_empty:
                encoded = tokenizer(
                    [t for _, t in non_empty], return_tensors="pt", padding=True, truncation=True, max_length=512
                )
                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        forced_bos_token_id=forced_bos,
                        max_new_tokens=512,
                        num_beams=2,
                    )
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                for (idx, _), text in zip(non_empty, decoded, strict=False):
                    translated[idx] = text.strip()
            results.extend(translated)

        return results
