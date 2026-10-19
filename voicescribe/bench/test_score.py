"""채점기 검증. 실행: .venv/bin/python -m pytest bench/test_score.py -q"""

from score import normalize_words, score_pair


def test_normalize_drops_punctuation_and_fillers() -> None:
    assert normalize_words("어/ 음, 그래서 회의는... 3시에!") == ["그래서", "회의는", "3시에"]


def test_normalize_drops_kspon_noise_tags() -> None:
    assert normalize_words("b/ 오늘 n/ 밥 먹고 o/ 가자") == ["오늘", "밥", "먹고", "가자"]


# 빠뜨림 1 + 덧붙임 1 은 틀림 2 와 비용이 같아 정렬이 갈린다. 그래서 종류별로 따로 검사한다.
def test_normalize_keeps_spelling_side_of_kspon_dual_transcription() -> None:
    assert normalize_words("(30만 원)/(삼십만 원) 줬어 (그러니까)/(근까) 그+ 그거") == ["30만", "원", "줬어", "그러니까", "그거"]


def test_dual_transcription_accepts_either_spelling_or_pronunciation() -> None:
    # '40번' 으로 적든 '사십 번' 으로 적든 맞힌 것으로 본다.
    assert score_pair("(40)/(사십) 번까지 나왔네", "사십 번까지 나왔네").char_errors == 0
    assert score_pair("(40)/(사십) 번까지 나왔네", "40 번까지 나왔네").char_errors == 0


def test_counts_deletion() -> None:
    result = score_pair("내일 회의 시간 정해", "내일 시간 정해")
    assert (result.ref_words, result.deletions, result.substitutions, result.insertions) == (4, 1, 0, 0)


def test_counts_substitution() -> None:
    result = score_pair("내일 회의 시간 정해", "내린 회의 시간 정해")
    assert (result.deletions, result.substitutions, result.insertions) == (0, 1, 0)


def test_counts_insertion() -> None:
    result = score_pair("내일 회의 시간 정해", "내일 회의 시간 정말 정해")
    assert (result.deletions, result.substitutions, result.insertions) == (0, 0, 1)
    assert result.wer == 0.25


def test_cer_ignores_spacing() -> None:
    result = score_pair("조금만 생각을 하면", "조 금만 생각을하면")
    assert result.char_errors == 0
    assert result.substitutions + result.deletions + result.insertions > 0  # 어절로는 틀림


def test_empty_hypothesis_is_all_deletions() -> None:
    result = score_pair("가 나 다", "")
    assert result.deletions == 3 and result.char_errors == 3
