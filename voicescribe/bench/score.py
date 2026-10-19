"""받아쓰기 결과 채점.

두 가지를 함께 잰다(CLAUDE.md '인식 품질을 판단하는 방법').

- 글자 오류율(CER): 띄어쓰기를 무시하고 글자만 비교한다. 띄어쓰기 차이에 흔들리지 않는다.
- 어절 단위 빠뜨림·틀림·덧붙임: 읽기 쉬운지를 본다. 단어를 잘게 쪼개는 모델은 여기서 드러난다.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

#: 채점에서 빼는 군말. 모델마다 받아 적는지가 달라 비교를 흐린다.
FILLERS = frozenset({"어", "음", "아", "응", "으", "에", "엄", "흠", "음음", "어어", "아아", "으음"})
#: KsponSpeech 잡음 표시(b/ 숨소리, n/ 잡음, o/ 겹침, l/ 웃음, u/ 알아들을 수 없음).
_KSPON_TAG = re.compile(r"(?<!\S)[bnolu]/(?!\S)")
#: KsponSpeech 이중 전사 '(철자)/(발음)'. 후보 모델이 모두 표준 표기로 적으므로 철자 쪽을 쓴다.
_KSPON_DUAL = re.compile(r"\(([^)]*)\)/\([^)]*\)")
_KSPON_DUAL_BOTH = re.compile(r"\(([^)]*)\)/\(([^)]*)\)")
#: MagicHub 표시([LAUGHTER], [*] 등).
_BRACKET_TAG = re.compile(r"\[[^\]]*\]")
_NON_WORD = re.compile(r"[^\w\s]|_")


def normalize_words(text: str) -> list[str]:
    """문장부호·잡음 표시·군말을 지우고 어절 목록으로 바꾼다."""
    text = unicodedata.normalize("NFC", text).lower()
    text = _KSPON_DUAL.sub(r"\1", text)
    text = _KSPON_TAG.sub(" ", text)
    text = _BRACKET_TAG.sub(" ", text)
    words = []
    for raw in text.split():
        # KsponSpeech 는 군말을 '어/', 더듬은 조각을 '그+' 처럼 표시한다.
        marked = raw.endswith(("/", "+"))
        word = _NON_WORD.sub("", raw)
        if not word or marked or word in FILLERS:
            continue
        words.append(word)
    return words


@dataclass(frozen=True)
class Score:
    ref_words: int
    substitutions: int
    deletions: int
    insertions: int
    ref_chars: int
    char_errors: int

    def __add__(self, other: Score) -> Score:
        return Score(*(a + b for a, b in zip(self._values(), other._values(), strict=True)))

    def _values(self) -> tuple[int, ...]:
        return (
            self.ref_words,
            self.substitutions,
            self.deletions,
            self.insertions,
            self.ref_chars,
            self.char_errors,
        )

    @property
    def wer(self) -> float:
        return (self.substitutions + self.deletions + self.insertions) / max(1, self.ref_words)

    @property
    def cer(self) -> float:
        return self.char_errors / max(1, self.ref_chars)


EMPTY = Score(0, 0, 0, 0, 0, 0)


def score_pair(reference: str, hypothesis: str) -> Score:
    """정답과 인식 결과 한 쌍을 채점한다.

    KsponSpeech 이중 전사 '(40)/(사십)' 는 철자·발음 어느 쪽으로 적어도 맞힌 것으로 본다.
    문장 안의 이중 전사를 모두 철자로 읽은 것과 모두 발음으로 읽은 것 중 덜 틀린 쪽을 쓴다.
    """
    if not _KSPON_DUAL.search(reference):
        return _score_words(normalize_words(reference), normalize_words(hypothesis))
    hyp = normalize_words(hypothesis)
    spelling = _KSPON_DUAL.sub(r"\1", reference)
    pronunciation = _KSPON_DUAL_BOTH.sub(r"\2", reference)
    candidates = [_score_words(normalize_words(ref), hyp) for ref in (spelling, pronunciation)]
    return min(candidates, key=lambda s: (s.char_errors, s.substitutions + s.deletions + s.insertions))


def _score_words(ref: list[str], hyp: list[str]) -> Score:
    ops = Levenshtein.editops(ref, hyp)
    counts = {"replace": 0, "delete": 0, "insert": 0}
    for op in ops:
        counts[op.tag] += 1
    ref_chars, hyp_chars = "".join(ref), "".join(hyp)
    return Score(
        ref_words=len(ref),
        substitutions=counts["replace"],
        deletions=counts["delete"],
        insertions=counts["insert"],
        ref_chars=len(ref_chars),
        char_errors=Levenshtein.distance(ref_chars, hyp_chars),
    )
