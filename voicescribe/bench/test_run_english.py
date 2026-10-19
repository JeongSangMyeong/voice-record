"""영어 채점 맞추기 검사: 정답(AMI)과 결과(Whisper·Qwen)가 같은 말이면 같은 낱말 목록이 돼야 한다."""

from run_english import normalize_en


def test_numbers_fillers_and_case_are_aligned():
    ref = "MM-HMM YEAH I THINK TWENTY FIVE PERCENT OF IT UM IS FINE"
    hyp = "Mm-hmm. Yeah, I think 25% of it is fine."
    assert normalize_en(ref) == normalize_en(hyp) == "yeah i think twenty five percent of it is fine"


def test_contractions_and_thousands():
    assert normalize_en("DON'T SPEND ONE THOUSAND TWO HUNDRED") == normalize_en("Don't spend 1,200")


def test_letters_spelled_with_underscores_match_plain_letters():
    assert normalize_en("THE T_V_ REMOTE") == normalize_en("the TV remote")
