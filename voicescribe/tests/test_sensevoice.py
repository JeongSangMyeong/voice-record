"""SenseVoice(sherpa-onnx) 엔진 테스트.

모델을 내려받지 않고도 확인할 수 있는 것만 검사한다.
실제 인식 정확도는 모델이 있을 때만 검사한다(``VOICESCRIBE_SENSEVOICE_DIR`` 환경변수).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from voicescribe.engines import TranscribeOptions, get_engine
from voicescribe.engines.base import EngineNotAvailableError
from voicescribe.engines.sensevoice_engine import (
    MODEL_DIR_NAME,
    MODEL_URL,
    SUPPORTED_LANGUAGES,
    VAD_URL,
)


class TestModelIdentity:
    def test_uses_the_correct_2024_build(self):
        """2025-09-09 빌드는 광둥어 전용 파인튜닝이라 한국어가 깨진다. 절대 바꾸면 안 된다."""
        assert MODEL_DIR_NAME == "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
        assert "2025-09-09" not in MODEL_URL
        assert MODEL_URL.endswith(f"{MODEL_DIR_NAME}.tar.bz2")

    def test_model_urls_point_to_github_releases(self):
        """Hugging Face 는 사내망·프록시에서 막히는 경우가 많아 GitHub 릴리스를 쓴다."""
        for url in (MODEL_URL, VAD_URL):
            assert url.startswith("https://github.com/k2-fsa/sherpa-onnx/releases/")

    def test_supported_language_set(self):
        assert set(SUPPORTED_LANGUAGES) == {"ko", "ja", "zh", "en", "yue"}


class TestGuards:
    @pytest.fixture
    def engine(self):
        engine = get_engine("sensevoice")
        if not engine.is_available():
            pytest.skip("sherpa-onnx 가 설치되지 않았습니다")
        return engine

    def test_rejects_unsupported_language_before_downloading(self, engine, two_speaker_wav):
        """지원하지 않는 언어면 1GB 를 내려받기 전에 막아야 한다."""
        from voicescribe.audio import load_audio

        audio = load_audio(two_speaker_wav)
        with pytest.raises(EngineNotAvailableError, match="faster-whisper"):
            engine.transcribe(audio, TranscribeOptions(language="fr", model="sensevoice"))

    def test_rejects_whisper_translate_task(self, engine, two_speaker_wav):
        from voicescribe.audio import load_audio

        audio = load_audio(two_speaker_wav)
        with pytest.raises(EngineNotAvailableError, match="translate"):
            engine.transcribe(
                audio, TranscribeOptions(task="translate", language="ko", model="sensevoice")
            )

    def test_registered_with_aliases(self):
        assert get_engine("sensevoice").name == "sensevoice"
        assert get_engine("fast").name == "sensevoice"
        assert get_engine("sherpa").name == "sensevoice"

    def test_install_hint_mentions_no_torch(self):
        assert "PyTorch" in get_engine("sensevoice").install_hint()


_MODEL_DIR = os.environ.get("VOICESCRIBE_SENSEVOICE_DIR", "")


@pytest.mark.skipif(not _MODEL_DIR, reason="VOICESCRIBE_SENSEVOICE_DIR 이 설정되지 않았습니다")
class TestRealRecognition:
    """실제 모델이 있을 때만 도는 인식 정확도 테스트."""

    @pytest.mark.parametrize(
        ("wav", "expected_language", "must_contain"),
        [
            ("ko.wav", "ko", "생각"),
            ("en.wav", "en", "chieftain"),
            ("zh.wav", "zh", "开饭"),
        ],
    )
    def test_recognizes_real_speech(self, wav, expected_language, must_contain):
        from voicescribe.transcriber import transcribe_file

        path = Path(_MODEL_DIR) / "test_wavs" / wav
        if not path.exists():
            pytest.skip(f"{path} 가 없습니다")
        result = transcribe_file(path, engine="sensevoice", model="sensevoice")
        assert result.language == expected_language
        assert must_contain in result.text


class TestWhisperModelCatalog:
    """Hugging Face Hub 에서 확인한 저장소 정보와 코드가 일치하는지 검사한다."""

    def test_every_model_has_a_repo(self):
        from voicescribe.engines.faster_whisper_engine import MODEL_CATALOG, MODEL_REPOS

        assert set(MODEL_CATALOG) == set(MODEL_REPOS)

    def test_repo_ids_look_valid(self):
        from voicescribe.engines.faster_whisper_engine import MODEL_REPOS

        for name, repo in MODEL_REPOS.items():
            owner, _, model = repo.partition("/")
            assert owner and model, f"{name} 의 저장소 형식이 잘못됨: {repo}"

    def test_turbo_points_at_the_ct2_conversion(self):
        """`large-v3-turbo` 는 CTranslate2 로 변환된 저장소여야 한다(원본 openai/ 저장소가 아님)."""
        from voicescribe.engines.faster_whisper_engine import MODEL_REPOS

        assert MODEL_REPOS["large-v3-turbo"] == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
        assert not MODEL_REPOS["large-v3-turbo"].startswith("openai/")
