"""Whisper 계열 모델이 지원하는 언어 목록과 언어 코드 정규화.

의존성이 전혀 없어 모델 미설치 환경에서도 항상 import 된다.
"""

from __future__ import annotations

#: ISO 639-1(일부 639-3) 코드 -> (영어 이름, 한국어 이름)
LANGUAGES: dict[str, tuple[str, str]] = {
    "af": ("afrikaans", "아프리칸스어"),
    "am": ("amharic", "암하라어"),
    "ar": ("arabic", "아랍어"),
    "as": ("assamese", "아삼어"),
    "az": ("azerbaijani", "아제르바이잔어"),
    "ba": ("bashkir", "바시키르어"),
    "be": ("belarusian", "벨라루스어"),
    "bg": ("bulgarian", "불가리아어"),
    "bn": ("bengali", "벵골어"),
    "bo": ("tibetan", "티베트어"),
    "br": ("breton", "브르타뉴어"),
    "bs": ("bosnian", "보스니아어"),
    "ca": ("catalan", "카탈루냐어"),
    "cs": ("czech", "체코어"),
    "cy": ("welsh", "웨일스어"),
    "da": ("danish", "덴마크어"),
    "de": ("german", "독일어"),
    "el": ("greek", "그리스어"),
    "en": ("english", "영어"),
    "es": ("spanish", "스페인어"),
    "et": ("estonian", "에스토니아어"),
    "eu": ("basque", "바스크어"),
    "fa": ("persian", "페르시아어"),
    "fi": ("finnish", "핀란드어"),
    "fo": ("faroese", "페로어"),
    "fr": ("french", "프랑스어"),
    "gl": ("galician", "갈리시아어"),
    "gu": ("gujarati", "구자라트어"),
    "ha": ("hausa", "하우사어"),
    "haw": ("hawaiian", "하와이어"),
    "he": ("hebrew", "히브리어"),
    "hi": ("hindi", "힌디어"),
    "hr": ("croatian", "크로아티아어"),
    "ht": ("haitian creole", "아이티 크리올어"),
    "hu": ("hungarian", "헝가리어"),
    "hy": ("armenian", "아르메니아어"),
    "id": ("indonesian", "인도네시아어"),
    "is": ("icelandic", "아이슬란드어"),
    "it": ("italian", "이탈리아어"),
    "ja": ("japanese", "일본어"),
    "jw": ("javanese", "자바어"),
    "ka": ("georgian", "조지아어"),
    "kk": ("kazakh", "카자흐어"),
    "km": ("khmer", "크메르어"),
    "kn": ("kannada", "칸나다어"),
    "ko": ("korean", "한국어"),
    "la": ("latin", "라틴어"),
    "lb": ("luxembourgish", "룩셈부르크어"),
    "ln": ("lingala", "링갈라어"),
    "lo": ("lao", "라오어"),
    "lt": ("lithuanian", "리투아니아어"),
    "lv": ("latvian", "라트비아어"),
    "mg": ("malagasy", "말라가시어"),
    "mi": ("maori", "마오리어"),
    "mk": ("macedonian", "마케도니아어"),
    "ml": ("malayalam", "말라얄람어"),
    "mn": ("mongolian", "몽골어"),
    "mr": ("marathi", "마라티어"),
    "ms": ("malay", "말레이어"),
    "mt": ("maltese", "몰타어"),
    "my": ("myanmar", "미얀마어"),
    "ne": ("nepali", "네팔어"),
    "nl": ("dutch", "네덜란드어"),
    "nn": ("nynorsk", "노르웨이어(뉘노르스크)"),
    "no": ("norwegian", "노르웨이어"),
    "oc": ("occitan", "오크어"),
    "pa": ("punjabi", "펀자브어"),
    "pl": ("polish", "폴란드어"),
    "ps": ("pashto", "파슈토어"),
    "pt": ("portuguese", "포르투갈어"),
    "ro": ("romanian", "루마니아어"),
    "ru": ("russian", "러시아어"),
    "sa": ("sanskrit", "산스크리트어"),
    "sd": ("sindhi", "신드어"),
    "si": ("sinhala", "싱할라어"),
    "sk": ("slovak", "슬로바키아어"),
    "sl": ("slovenian", "슬로베니아어"),
    "sn": ("shona", "쇼나어"),
    "so": ("somali", "소말리어"),
    "sq": ("albanian", "알바니아어"),
    "sr": ("serbian", "세르비아어"),
    "su": ("sundanese", "순다어"),
    "sv": ("swedish", "스웨덴어"),
    "sw": ("swahili", "스와힐리어"),
    "ta": ("tamil", "타밀어"),
    "te": ("telugu", "텔루구어"),
    "tg": ("tajik", "타지크어"),
    "th": ("thai", "태국어"),
    "tk": ("turkmen", "투르크멘어"),
    "tl": ("tagalog", "타갈로그어"),
    "tr": ("turkish", "튀르키예어"),
    "tt": ("tatar", "타타르어"),
    "uk": ("ukrainian", "우크라이나어"),
    "ur": ("urdu", "우르두어"),
    "uz": ("uzbek", "우즈베크어"),
    "vi": ("vietnamese", "베트남어"),
    "yi": ("yiddish", "이디시어"),
    "yo": ("yoruba", "요루바어"),
    "yue": ("cantonese", "광둥어"),
    "zh": ("chinese", "중국어"),
}

#: 자주 쓰는 별칭 -> 표준 코드. 사용자가 무엇을 입력하든 최대한 받아준다.
#: 주의: 실제 언어 코드(``uk`` = 우크라이나어)가 우선이라 별칭에 넣지 않는다.
_ALIASES: dict[str, str] = {
    "kor": "ko", "kr": "ko", "korea": "ko", "hangul": "ko",
    "한국": "ko", "한국말": "ko", "국어": "ko",
    "eng": "en", "us": "en", "gb": "en", "en-gb": "en", "영문": "en",
    "jpn": "ja", "jp": "ja", "일어": "ja", "일본": "ja",
    "chi": "zh", "cn": "zh", "zho": "zh", "mandarin": "zh",
    "zh-cn": "zh", "zh-tw": "zh", "zh-hans": "zh", "zh-hant": "zh", "중국": "zh", "중문": "zh",
    "yue-hant": "yue", "cantonese": "yue", "광둥": "yue",
    "spa": "es", "esp": "es", "fra": "fr", "fre": "fr", "deu": "de", "ger": "de",
    "rus": "ru", "por": "pt", "pt-br": "pt", "ita": "it", "nld": "nl", "vie": "vi",
    "tha": "th", "ind": "id", "ara": "ar", "hin": "hi", "tur": "tr", "pol": "pl",
    "nb": "no", "nob": "no", "iw": "he", "in": "id", "ji": "yi", "jv": "jw",
    "auto": "auto", "자동": "auto", "자동감지": "auto", "detect": "auto",
}

# 영어 이름과 한국어 이름도 그대로 코드로 변환할 수 있게 역인덱스를 만든다.
_NAME_TO_CODE: dict[str, str] = {}
for _code, (_en, _ko) in LANGUAGES.items():
    _NAME_TO_CODE.setdefault(_en, _code)
    _NAME_TO_CODE.setdefault(_ko, _code)
    if _ko.endswith("어") and len(_ko) > 2:
        _NAME_TO_CODE.setdefault(_ko[:-1], _code)  # "한국어" -> "한국"


class UnknownLanguageError(ValueError):
    """알 수 없는 언어 코드/이름."""


def normalize_language(value: str | None) -> str | None:
    """사용자 입력을 Whisper 언어 코드로 정규화한다.

    ``"ko"``, ``"Korean"``, ``"한국어"``, ``"kor"`` 모두 ``"ko"`` 가 된다.
    ``None`` / ``"auto"`` / 빈 문자열은 자동 감지를 뜻하는 ``None`` 을 반환한다.

    Raises:
        UnknownLanguageError: 어떤 규칙으로도 해석할 수 없는 값일 때.
    """
    if value is None:
        return None
    key = str(value).strip().lower().replace("_", "-")
    if not key or key == "auto":
        return None
    if key in LANGUAGES:
        return key
    alias = _ALIASES.get(key)
    if alias is not None:
        return None if alias == "auto" else alias
    if key in _NAME_TO_CODE:
        return _NAME_TO_CODE[key]
    base = key.split("-", 1)[0]  # "en-US" -> "en"
    if base in LANGUAGES:
        return base
    if base in _ALIASES:
        alias = _ALIASES[base]
        return None if alias == "auto" else alias
    raise UnknownLanguageError(
        f"'{value}' 는 지원하지 않는 언어입니다. "
        "`voicescribe langs` 명령으로 지원 목록을 확인하세요."
    )


def language_name(code: str | None, *, korean: bool = True) -> str:
    """언어 코드를 사람이 읽는 이름으로 바꾼다."""
    if not code:
        return "자동 감지" if korean else "auto-detect"
    if str(code).lower() == "unknown":
        return "알 수 없음" if korean else "unknown"
    entry = LANGUAGES.get(str(code).lower())
    if entry is None:
        return str(code)
    return entry[1] if korean else entry[0]


def supported_languages() -> list[tuple[str, str, str]]:
    """(코드, 영어 이름, 한국어 이름) 목록을 코드순으로 반환한다."""
    return [(code, en, ko) for code, (en, ko) in sorted(LANGUAGES.items())]
