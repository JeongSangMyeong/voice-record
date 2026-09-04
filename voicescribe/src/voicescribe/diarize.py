"""화자 분리(누가 언제 말했는지).

세 가지 방법을 지원하며, ``method="auto"`` 면 좋은 것부터 자동으로 시도한다.

1. **sherpa-onnx (권장)** — 무료, 토큰 불필요, PyTorch 불필요(약 45MB).
   모델은 GitHub 릴리스에서 받는다. 1시간 오디오를 4코어 CPU 로 약 10분에 처리한다.
2. **pyannote.audio** — 가장 정확하지만 Hugging Face 무료 토큰이 필요하고,
   모델 페이지에서 약관에 동의해야 하며 PyTorch(약 2.5GB)를 끌고 온다.
3. **간이 방식** — 추가 설치·다운로드가 전혀 없다. numpy 로 MFCC 를 뽑아
   목소리가 비슷한 구간끼리 묶는다. "2~3명이 번갈아 말하는 회의록" 정도는 잘 나눈다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from .audio import AudioBuffer
    from .types import TranscriptionResult

#: 화자 수를 자동으로 정할 때 시도할 최대 인원.
_DEFAULT_MAX_SPEAKERS = 6
#: 특징 추출 설정.
_FRAME_LENGTH = 0.025  # 25ms
_FRAME_HOP = 0.010  # 10ms
_N_MELS = 26
_N_MFCC = 13


class DiarizationError(RuntimeError):
    """화자 분리에 실패했을 때."""


#: sherpa-onnx 화자 분리에 필요한 모델(모두 무료, 토큰 불필요).
_SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/"
    "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
#: 주의: 아래 URL 의 'recongition' 오타는 업스트림 릴리스 태그 그대로다. 고치면 404 가 난다.
_EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
)
#: 군집 임계값. 기본값 0.5 는 같은 사람을 여러 명으로 쪼개는 경향이 강하다
#: (4명짜리 파일에서 10명이 나옴). 0.8 이 훨씬 안정적이다.
_CLUSTER_THRESHOLD = 0.8


# --------------------------------------------------------------------------- #
# 특징 추출 (numpy 만 사용)
# --------------------------------------------------------------------------- #


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    import numpy as np

    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    import numpy as np

    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_filterbank(n_filters: int, n_fft: int, sample_rate: int) -> np.ndarray:
    """삼각형 멜 필터뱅크를 만든다."""
    import numpy as np

    low_mel = _hz_to_mel(0.0)
    high_mel = _hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fbank = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_filters + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center == left:
            center = min(left + 1, n_fft // 2)
        if right == center:
            right = min(center + 1, n_fft // 2)
        for k in range(left, center):
            fbank[i - 1, k] = (k - left) / max(1, center - left)
        for k in range(center, right):
            fbank[i - 1, k] = (right - k) / max(1, right - center)
    return fbank


def _mfcc(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """MFCC 특징을 (프레임 수, _N_MFCC) 모양으로 계산한다."""
    import numpy as np

    frame_len = max(1, int(sample_rate * _FRAME_LENGTH))
    hop = max(1, int(sample_rate * _FRAME_HOP))
    if samples.size < frame_len:
        samples = np.pad(samples, (0, frame_len - samples.size))

    n_frames = 1 + (samples.size - frame_len) // hop
    if n_frames <= 0:
        return np.zeros((1, _N_MFCC), dtype=np.float32)

    indices = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = samples[indices] * np.hamming(frame_len).astype(np.float32)

    n_fft = 1
    while n_fft < frame_len:
        n_fft *= 2
    spectrum = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2 / n_fft

    fbank = _mel_filterbank(_N_MELS, n_fft, sample_rate)
    mel_energy = np.maximum(spectrum @ fbank.T, 1e-10)
    log_mel = np.log(mel_energy)

    # DCT-II 로 MFCC 를 만든다(scipy 없이 직접 계산).
    n = np.arange(_N_MELS)
    k = np.arange(_N_MFCC)[:, None]
    dct_matrix = np.cos(np.pi * k * (2 * n + 1) / (2 * _N_MELS)).astype(np.float32)
    return (log_mel @ dct_matrix.T).astype(np.float32)


def _segment_embedding(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """한 발화 구간을 고정 길이 벡터로 요약한다(평균 + 표준편차)."""
    import numpy as np

    mfcc = _mfcc(samples, sample_rate)
    embedding = np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0)])
    norm = float(np.linalg.norm(embedding))
    return embedding / norm if norm > 0 else embedding


# --------------------------------------------------------------------------- #
# 군집화
# --------------------------------------------------------------------------- #


def _cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    import numpy as np

    similarity = embeddings @ embeddings.T
    return np.clip(1.0 - similarity, 0.0, 2.0)


def _agglomerative(distances: np.ndarray, n_clusters: int) -> list[int]:
    """평균 연결(average linkage) 병합 군집화. 라벨 리스트를 돌려준다."""
    import numpy as np

    n = distances.shape[0]
    clusters: list[list[int]] = [[i] for i in range(n)]
    while len(clusters) > max(1, n_clusters):
        best = (float("inf"), 0, 1)
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                block = distances[np.ix_(clusters[a], clusters[b])]
                dist = float(block.mean())
                if dist < best[0]:
                    best = (dist, a, b)
        _, a, b = best
        clusters[a] = clusters[a] + clusters[b]
        clusters.pop(b)

    labels = [0] * n
    for label, members in enumerate(clusters):
        for idx in members:
            labels[idx] = label
    return labels


def _silhouette(distances: np.ndarray, labels: list[int]) -> float:
    """군집 품질 점수(-1~1, 클수록 좋음). 화자 수를 자동으로 고를 때 쓴다."""
    import numpy as np

    unique = sorted(set(labels))
    if len(unique) < 2 or len(labels) <= len(unique):
        return -1.0

    label_array = np.array(labels)
    scores: list[float] = []
    for i in range(len(labels)):
        same = (label_array == label_array[i]) & (np.arange(len(labels)) != i)
        if not same.any():
            continue
        a = float(distances[i, same].mean())
        b = min(
            float(distances[i, label_array == other].mean())
            for other in unique
            if other != label_array[i]
        )
        denominator = max(a, b)
        if denominator > 0:
            scores.append((b - a) / denominator)
    return float(np.mean(scores)) if scores else -1.0


# --------------------------------------------------------------------------- #
# 공개 API
# --------------------------------------------------------------------------- #


def diarize_simple(
    audio: AudioBuffer,
    result: TranscriptionResult,
    *,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[str]:
    """추가 설치 없이 동작하는 간이 화자 분리. 구간별 화자 라벨을 돌려준다."""
    import numpy as np

    segments = result.segments
    if len(segments) < 2:
        return ["화자1"] * len(segments)

    rate = audio.sample_rate
    embeddings = []
    for seg in segments:
        start = max(0, int(seg.start * rate))
        end = min(len(audio.samples), int(seg.end * rate))
        chunk = audio.samples[start:end]
        if chunk.size < int(0.1 * rate):  # 0.1초 미만은 특징이 불안정하다.
            chunk = np.pad(chunk, (0, max(0, int(0.1 * rate) - chunk.size)))
        embeddings.append(_segment_embedding(chunk, rate))

    matrix = np.vstack(embeddings)
    distances = _cosine_distance_matrix(matrix)

    lower = max(1, min_speakers or 1)
    upper = min(max_speakers or _DEFAULT_MAX_SPEAKERS, len(segments))
    if min_speakers and max_speakers and min_speakers == max_speakers:
        best_labels = _agglomerative(distances, min_speakers)
    else:
        # k 후보별 점수를 모두 구한 뒤, 점수가 비슷하면 '사람이 더 적은 쪽'을 고른다.
        # (실루엣 점수만 보면 같은 사람을 둘로 쪼개는 경향이 있다.)
        scored: list[tuple[int, float, list[int]]] = []
        for k in range(max(2, lower), max(2, upper) + 1):
            labels = _agglomerative(distances, k)
            scored.append((k, _silhouette(distances, labels), labels))

        if not scored:
            best_labels = [0] * len(segments)
        else:
            best_score = max(score for _, score, _ in scored)
            # 최고점의 92% 이상(또는 0.03 이내)이면 동급으로 보고 가장 적은 화자 수를 택한다.
            threshold = min(best_score * 0.92, best_score - 0.03) if best_score > 0 else best_score
            eligible = [item for item in scored if item[1] >= threshold]
            best_labels = min(eligible, key=lambda item: item[0])[2]
            # 군집이 뚜렷하지 않으면 전부 한 사람으로 본다.
            if best_score < 0.05 and not min_speakers:
                best_labels = [0] * len(segments)

    # 처음 말한 사람이 화자1 이 되도록 번호를 다시 매긴다.
    order: dict[int, int] = {}
    for label in best_labels:
        if label not in order:
            order[label] = len(order) + 1
    return [f"화자{order[label]}" for label in best_labels]


def _model_cache_dir() -> Path:
    """모델을 저장할 폴더."""
    env = os.environ.get("VOICESCRIBE_MODEL_DIR")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "voicescribe"


def _ensure_sherpa_models() -> tuple[str, str]:
    """sherpa-onnx 화자 분리 모델을 확보한다(없으면 내려받는다)."""
    import tarfile
    import urllib.request

    root = _model_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    segmentation = root / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
    embedding = root / "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"

    if not segmentation.exists():
        archive = root / "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
        try:
            urllib.request.urlretrieve(_SEGMENTATION_URL, archive)  # noqa: S310
            with tarfile.open(archive, "r:bz2") as tar:
                for member in tar.getmembers():  # 경로 탈출 방지
                    if (root / member.name).resolve().is_relative_to(root.resolve()) is False:
                        raise DiarizationError(f"안전하지 않은 경로: {member.name}")
                tar.extractall(root)  # noqa: S202
            archive.unlink(missing_ok=True)
        except DiarizationError:
            raise
        except Exception as exc:
            raise DiarizationError(
                f"화자 분리 모델을 내려받지 못했습니다: {exc}\n직접 받으려면: {_SEGMENTATION_URL}"
            ) from exc

    if not embedding.exists():
        try:
            urllib.request.urlretrieve(_EMBEDDING_URL, embedding)  # noqa: S310
        except Exception as exc:
            raise DiarizationError(
                f"화자 임베딩 모델을 내려받지 못했습니다: {exc}\n직접 받으려면: {_EMBEDDING_URL}"
            ) from exc

    return str(segmentation), str(embedding)


def diarize_sherpa(
    audio: AudioBuffer,
    result: TranscriptionResult,
    *,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[str]:
    """sherpa-onnx 로 화자를 나눈다(토큰·PyTorch 불필요)."""
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise DiarizationError(
            "sherpa-onnx 가 설치되지 않았습니다.\n"
            '설치: pip install "voicescribe[fast]"  (또는 pip install sherpa-onnx)'
        ) from exc

    import numpy as np

    segmentation, embedding = _ensure_sherpa_models()
    known_speakers = min_speakers if min_speakers and min_speakers == max_speakers else None

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=segmentation, window_shift_ratio=0.1
            )
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=embedding, num_threads=min(8, os.cpu_count() or 4)
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=known_speakers or -1, threshold=_CLUSTER_THRESHOLD
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise DiarizationError("sherpa-onnx 화자 분리 설정이 올바르지 않습니다.")

    engine = sherpa_onnx.OfflineSpeakerDiarization(config)
    if audio.sample_rate != engine.sample_rate:
        raise DiarizationError(
            f"이 모델은 {engine.sample_rate}Hz 오디오가 필요합니다(현재 {audio.sample_rate}Hz)."
        )

    turns = [
        (float(r.start), float(r.end), f"SPK{int(r.speaker):02d}")
        for r in engine.process(np.asarray(audio.samples, dtype=np.float32)).sort_by_start_time()
    ]
    return _labels_from_turns(turns, result)


def _labels_from_turns(
    turns: list[tuple[float, float, str]], result: TranscriptionResult
) -> list[str]:
    """화자 구간 목록을 받아쓰기 구간에 매핑한다(가장 많이 겹치는 화자를 고른다)."""
    mapping: dict[str, str] = {}
    labels: list[str] = []
    for seg in result.segments:
        overlaps: dict[str, float] = {}
        for start, end, speaker in turns:
            overlap = min(seg.end, end) - max(seg.start, start)
            if overlap > 0:
                overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap
        raw = max(overlaps, key=lambda k: overlaps[k]) if overlaps else "SPK00"
        if raw not in mapping:
            mapping[raw] = f"화자{len(mapping) + 1}"
        labels.append(mapping[raw])
    return labels


def diarize_pyannote(
    audio: AudioBuffer,
    result: TranscriptionResult,
    *,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    token: str | None = None,
) -> list[str]:
    """pyannote.audio 로 정확한 화자 분리를 수행한다.

    Hugging Face 무료 토큰(``HF_TOKEN`` 환경변수)과 모델 약관 동의가 필요하다.
    """
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizationError(
            "pyannote.audio 가 설치되지 않았습니다.\n"
            '설치: pip install "voicescribe[diarize]"\n'
            "그리고 https://huggingface.co/pyannote/speaker-diarization-3.1 에서 약관에 동의한 뒤\n"
            "HF_TOKEN 환경변수에 무료 토큰을 넣어 주세요."
        ) from exc

    hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise DiarizationError(
            "HF_TOKEN 환경변수가 없습니다. huggingface.co 에서 무료 토큰을 만들어 설정하세요."
        )

    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    except Exception as exc:
        raise DiarizationError(f"pyannote 모델을 불러오지 못했습니다: {exc}") from exc

    waveform = torch.from_numpy(audio.samples).unsqueeze(0)
    kwargs: dict[str, int] = {}
    if min_speakers:
        kwargs["min_speakers"] = min_speakers
    if max_speakers:
        kwargs["max_speakers"] = max_speakers
    annotation = pipeline({"waveform": waveform, "sample_rate": audio.sample_rate}, **kwargs)

    turns = [
        (float(turn.start), float(turn.end), str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    return _labels_from_turns(turns, result)


def apply_diarization(
    audio: AudioBuffer,
    result: TranscriptionResult,
    *,
    method: str = "auto",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> TranscriptionResult:
    """화자 라벨을 결과에 채워 넣는다(제자리 수정).

    Args:
        method: ``"auto"``(좋은 것부터 자동 선택), ``"sherpa"``, ``"pyannote"``, ``"simple"``.
    """
    if not result.segments:
        return result

    kwargs = {"min_speakers": min_speakers, "max_speakers": max_speakers}
    labels: list[str] | None = None

    if method == "auto":
        # 정확도·설치 편의를 함께 고려한 순서로 시도한다.
        for backend in (diarize_pyannote, diarize_sherpa):
            try:
                labels = backend(audio, result, **kwargs)  # type: ignore[operator]
                break
            except DiarizationError:
                continue
        if labels is None:
            labels = diarize_simple(audio, result, **kwargs)  # type: ignore[arg-type]
    elif method == "sherpa":
        labels = diarize_sherpa(audio, result, **kwargs)  # type: ignore[arg-type]
    elif method == "pyannote":
        labels = diarize_pyannote(audio, result, **kwargs)  # type: ignore[arg-type]
    elif method == "simple":
        labels = diarize_simple(audio, result, **kwargs)  # type: ignore[arg-type]
    else:
        raise DiarizationError(
            f"'{method}' 는 알 수 없는 화자 분리 방식입니다. "
            "auto / sherpa / pyannote / simple 중에서 고르세요."
        )

    for seg, label in zip(result.segments, labels, strict=False):
        seg.speaker = label
    result.speakers = sorted(set(labels), key=lambda s: (len(s), s))
    return result
