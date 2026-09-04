"""브라우저에서 쓰는 로컬 웹 UI.

``voicescribe web`` 으로 실행하면 http://127.0.0.1:7860 이 열린다.
기본값은 내 컴퓨터에서만 접속 가능하며, 음성 파일이 외부로 나가지 않는다.

긴 오디오도 다룰 수 있도록 작업을 백그라운드 스레드에서 돌리고
진행률은 SSE(Server-Sent Events)로 실시간 전달한다.
"""

# 주의: 이 모듈에서는 ``from __future__ import annotations`` 를 쓰지 않는다.
# FastAPI 는 함수 정의 시점에 타입 어노테이션을 실제로 평가하는데, 어노테이션이
# 문자열로 바뀌면 create_app() 안에서 지연 import 한 UploadFile 을 찾지 못한다.

import contextlib
import json
import queue
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: 동시에 보관할 작업 수(오래된 것부터 정리한다).
_MAX_JOBS = 40
#: 업로드 허용 최대 크기(바이트). 기본 1GB.
_MAX_UPLOAD_BYTES = 1024 * 1024 * 1024

_STATIC_DIR = Path(__file__).parent


class WebDependencyError(RuntimeError):
    """웹 UI 에 필요한 패키지가 없을 때."""


@dataclass
class Job:
    """받아쓰기 작업 하나."""

    id: str
    filename: str
    status: str = "대기 중"
    fraction: float = 0.0
    message: str = "대기 중"
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    result: Any = None
    outputs: dict[str, str] = field(default_factory=dict)
    listeners: list["queue.Queue[str]"] = field(default_factory=list)
    temp_dir: str | None = None

    def emit(self) -> None:
        """현재 상태를 모든 구독자에게 보낸다."""
        payload = json.dumps(self.snapshot(), ensure_ascii=False)
        for listener in list(self.listeners):
            # 구독자가 이미 끊겼거나 큐가 가득 찼어도 나머지에게는 계속 보낸다.
            with contextlib.suppress(Exception):
                listener.put_nowait(payload)

    def snapshot(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "fraction": round(self.fraction, 4),
            "message": self.message,
            "error": self.error,
            "formats": sorted(self.outputs),
        }
        if self.result is not None:
            data["result"] = self.result.to_dict()
        return data

    def cleanup(self) -> None:
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None


_JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


def _register(job: Job) -> None:
    with _JOBS_LOCK:
        _JOBS[job.id] = job
        while len(_JOBS) > _MAX_JOBS:
            oldest = min(_JOBS.values(), key=lambda j: j.created_at)
            _JOBS.pop(oldest.id, None)
            oldest.cleanup()


def _run_job(job: Job, audio_path: Path, request: Any, formats: list[str], render_options: dict[str, Any]) -> None:
    """백그라운드 스레드에서 실제 받아쓰기를 수행한다."""
    from ..audio import load_audio
    from ..output import render
    from ..transcriber import transcribe_buffer

    def progress(fraction: float, message: str) -> None:
        job.fraction = fraction
        job.message = message
        job.status = "처리 중"
        job.emit()

    try:
        progress(0.01, "오디오 읽는 중")
        audio = load_audio(audio_path)
        result = transcribe_buffer(audio, request, progress)
        job.result = result
        for fmt in formats:
            job.outputs[fmt] = render(result, fmt, **render_options)
        job.status = "완료"
        job.fraction = 1.0
        job.message = "완료"
    except Exception as exc:
        job.status = "실패"
        job.error = f"{type(exc).__name__}: {exc}"
        job.message = "실패"
    finally:
        job.emit()
        job.cleanup()


def create_app() -> Any:
    """FastAPI 앱을 만든다."""
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
    except ImportError as exc:
        raise WebDependencyError(
            "웹 UI 에 필요한 패키지가 없습니다.\n"
            '설치: pip install "voicescribe[web]"'
        ) from exc

    from .. import __version__

    app = FastAPI(title="VoiceScribe", version=__version__, docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> Any:
        page = _STATIC_DIR / "index.html"
        if not page.exists():
            raise HTTPException(status_code=500, detail="index.html 을 찾을 수 없습니다.")
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/api/config")
    def config() -> Any:
        from ..engines import list_engines
        from ..engines.faster_whisper_engine import MODEL_CATALOG as WHISPER_MODELS
        from ..engines.sensevoice_engine import MODEL_CATALOG as SENSEVOICE_MODELS
        from ..languages import supported_languages
        from ..output import FORMAT_DESCRIPTIONS
        from ..translate import list_translators

        # 설치된 엔진의 모델만 보여 준다(없으면 기본으로 Whisper 목록).
        installed = {e.name for e in list_engines() if e.is_available()}
        models: dict[str, str] = {}
        if "faster-whisper" in installed:
            models.update(WHISPER_MODELS)
        if "sensevoice" in installed:
            models.update(
                {k: f"{v} · 한·일·중·영·광둥어 전용" for k, v in SENSEVOICE_MODELS.items()}
            )
        if not models:
            models = dict(WHISPER_MODELS)

        return JSONResponse(
            {
                "version": __version__,
                "languages": [
                    {"code": code, "en": en, "ko": ko} for code, en, ko in supported_languages()
                ],
                "formats": FORMAT_DESCRIPTIONS,
                "models": models,
                "engines": [
                    {"name": e.name, "available": e.is_available(), "description": e.description}
                    for e in list_engines()
                ],
                "translators": [
                    {"name": t.name, "available": t.is_available(), "description": t.description}
                    for t in list_translators()
                ],
            }
        )

    @app.post("/api/jobs")
    async def create_job(
        file: UploadFile = File(...),
        language: str = Form("auto"),
        model: str = Form("base"),
        engine: str = Form("auto"),
        formats: str = Form("txt"),
        translate_to: str = Form(""),
        translator: str = Form(""),
        diarize: str = Form("false"),
        timestamps: str = Form("false"),
        bilingual: str = Form("false"),
        prompt: str = Form(""),
        beam_size: int = Form(5),
    ) -> Any:
        from ..output import FORMATTERS
        from ..transcriber import TranscribeRequest

        temp_dir = tempfile.mkdtemp(prefix="voicescribe-")
        safe_name = Path(file.filename or "audio").name or "audio"
        audio_path = Path(temp_dir) / safe_name

        written = 0
        with audio_path.open("wb") as sink:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > _MAX_UPLOAD_BYTES:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    raise HTTPException(status_code=413, detail="파일이 너무 큽니다(최대 1GB).")
                sink.write(chunk)

        if written == 0:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="빈 파일입니다.")

        wanted = [f for f in formats.split(",") if f.strip() in FORMATTERS] or ["txt"]
        job = Job(id=uuid.uuid4().hex[:12], filename=safe_name, temp_dir=temp_dir)
        _register(job)

        request = TranscribeRequest(
            path=audio_path,
            language=None if language.lower() in ("", "auto") else language,
            engine=None if engine.lower() in ("", "auto") else engine,
            model=model,
            beam_size=max(1, int(beam_size)),
            initial_prompt=prompt or None,
            translate_to=translate_to or None,
            translator=translator or None,
            diarize=_as_bool(diarize),
        )
        render_options = {
            "timestamps": _as_bool(timestamps),
            "speakers": True,
            "bilingual": _as_bool(bilingual),
            "translated_only": False,
        }

        thread = threading.Thread(
            target=_run_job, args=(job, audio_path, request, wanted, render_options), daemon=True
        )
        thread.start()
        return JSONResponse({"id": job.id})

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> Any:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
        return JSONResponse(job.snapshot())

    @app.get("/api/jobs/{job_id}/events")
    def job_events(job_id: str) -> Any:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")

        listener: queue.Queue[str] = queue.Queue(maxsize=200)
        job.listeners.append(listener)

        def stream():  # noqa: ANN202
            # 종료 판정은 반드시 '방금 내보낸 이벤트'의 status 로 한다.
            # 살아 있는 job.status 로 판정하면, 큐에 마지막 완료 이벤트가 남아 있는데도
            # 스트림이 먼저 닫혀 클라이언트가 결과를 받지 못한다.
            def is_final(payload_json: str) -> bool:
                try:
                    return json.loads(payload_json).get("status") in ("완료", "실패")
                except (ValueError, AttributeError):
                    return False

            try:
                first = json.dumps(job.snapshot(), ensure_ascii=False)
                yield f"data: {first}\n\n"
                if is_final(first):  # 접속 전에 이미 끝난 작업
                    return
                while True:
                    try:
                        payload = listener.get(timeout=15)
                    except queue.Empty:
                        yield ": keep-alive\n\n"  # 프록시가 연결을 끊지 않도록
                        if job.status in ("완료", "실패"):
                            # 상태는 끝났는데 큐가 비었다면 최종 스냅샷을 한 번 더 보낸다.
                            yield f"data: {json.dumps(job.snapshot(), ensure_ascii=False)}\n\n"
                            break
                        continue
                    yield f"data: {payload}\n\n"
                    if is_final(payload):
                        break
            finally:
                if listener in job.listeners:
                    job.listeners.remove(listener)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str, format: str = "txt") -> Any:
        job = _JOBS.get(job_id)
        if job is None or format not in job.outputs:
            raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다.")
        stem = Path(job.filename).stem or "transcript"
        temp = Path(tempfile.mkdtemp(prefix="voicescribe-dl-")) / f"{stem}.{format}"
        temp.write_text(job.outputs[format], encoding="utf-8-sig" if format == "csv" else "utf-8")
        return FileResponse(temp, filename=temp.name, media_type="application/octet-stream")

    return app


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on", "y")


def serve(
    host: str = "127.0.0.1",
    port: int = 7860,
    open_browser: bool = True,
    use_https: bool = False,
) -> int:
    """웹 서버를 실행한다.

    Args:
        host: ``0.0.0.0`` 이면 같은 와이파이의 휴대폰에서도 접속할 수 있다.
        use_https: 자체 서명 인증서로 https 를 켠다. 휴대폰에서 마이크 녹음을
            하려면 필요하다(브라우저가 보안 연결에서만 마이크를 허용한다).
    """
    try:
        import uvicorn
    except ImportError:
        print('웹 UI 에 필요한 패키지가 없습니다.\n설치: pip install "voicescribe[web]"')
        return 1

    try:
        app = create_app()
    except WebDependencyError as exc:
        print(str(exc))
        return 1

    from .lan import access_notice, detect_lan_ip, ensure_self_signed_cert

    ssl_options: dict[str, str] = {}
    if use_https:
        names = ["localhost", "127.0.0.1"]
        lan_ip = detect_lan_ip()
        if lan_ip:
            names.append(lan_ip)
        try:
            cert_file, key_file = ensure_self_signed_cert(names)
        except RuntimeError as exc:
            print(str(exc))
            return 1
        ssl_options = {"ssl_certfile": str(cert_file), "ssl_keyfile": str(key_file)}

    scheme = "https" if use_https else "http"
    local_url = f"{scheme}://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"

    print("\n  VoiceScribe 웹 UI 실행 중\n")
    print(access_notice(host, port, use_https))
    if use_https:
        print("\n  ※ 브라우저가 '안전하지 않음' 이라고 경고합니다.")
        print("     내 컴퓨터가 직접 만든 인증서라 그렇습니다.")
        print("     '고급' → '계속 진행' 을 누르면 됩니다.")
    print("\n  종료하려면 Ctrl+C 를 누르세요.\n")

    if open_browser:
        threading.Timer(1.0, lambda: _open_browser(local_url)).start()

    uvicorn.run(app, host=host, port=port, log_level="warning", **ssl_options)
    return 0


def _open_browser(url: str) -> None:
    try:
        import webbrowser

        webbrowser.open(url)
    except Exception:
        pass
