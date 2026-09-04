"""언어 코드 정규화 테스트."""

from __future__ import annotations

import pytest

from voicescribe.languages import (
    LANGUAGES,
    UnknownLanguageError,
    language_name,
    normalize_language,
    supported_languages,
)


class TestNormalize:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("ko", "ko"), ("KO", "ko"), ("Korean", "ko"), ("한국어", "ko"),
            ("한국", "ko"), ("kor", "ko"), ("kr", "ko"),
            ("ja", "ja"), ("일본어", "ja"), ("jp", "ja"),
            ("en-US", "en"), ("en_GB", "en"), ("english", "en"),
            ("zh-CN", "zh"), ("중국어", "zh"), ("mandarin", "zh"),
            ("yue", "yue"), ("cantonese", "yue"),
            ("nb", "no"), ("iw", "he"), ("jv", "jw"),
        ],
    )
    def test_aliases(self, given, expected):
        assert normalize_language(given) == expected

    @pytest.mark.parametrize("given", [None, "", "  ", "auto", "AUTO", "자동", "자동감지"])
    def test_auto_detect_returns_none(self, given):
        assert normalize_language(given) is None

    def test_real_code_wins_over_alias(self):
        """'uk' 는 우크라이나어의 실제 코드다. 영국 영어로 오해하면 안 된다."""
        assert normalize_language("uk") == "uk"
        assert language_name("uk") == "우크라이나어"

    def test_unknown_raises_with_helpful_message(self):
        with pytest.raises(UnknownLanguageError) as exc:
            normalize_language("클링온어")
        assert "voicescribe langs" in str(exc.value)


class TestCatalog:
    def test_covers_whisper_language_set(self):
        assert len(LANGUAGES) >= 99

    def test_every_entry_has_both_names(self):
        for code, (english, korean) in LANGUAGES.items():
            assert code and english and korean, code
            assert english.islower() or " " in english, code

    def test_supported_languages_is_sorted(self):
        codes = [code for code, _, _ in supported_languages()]
        assert codes == sorted(codes)

    def test_language_name_falls_back_to_code(self):
        assert language_name("xx") == "xx"
        assert language_name("unknown") == "알 수 없음"
        assert language_name(None) == "자동 감지"
        assert language_name(None, korean=False) == "auto-detect"
