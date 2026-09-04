"""배포용 파일 검사.

받는 사람이 더블클릭해서 쓰는 파일들이라, 깨지면 바로 사용자 문제로 이어진다.
문법·줄바꿈·필수 안내 문구가 유지되는지 확인한다.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PC_DIR = PROJECT_ROOT / "deploy" / "pc"
HF_DIR = PROJECT_ROOT / "deploy" / "huggingface"


class TestPcLaunchers:
    @pytest.mark.parametrize("name", ["시작-리눅스.sh", "시작-맥.command", "압축만들기.sh"])
    def test_shell_scripts_are_valid(self, name):
        path = PC_DIR / name
        assert path.exists(), f"{name} 이 없습니다"
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash 가 없습니다")
        result = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, f"{name} 문법 오류: {result.stderr}"

    @pytest.mark.parametrize("name", ["시작-리눅스.sh", "시작-맥.command"])
    def test_shell_scripts_are_executable(self, name):
        assert (PC_DIR / name).stat().st_mode & 0o111, f"{name} 에 실행 권한이 없습니다"

    @pytest.mark.parametrize("name", ["시작-윈도우.bat", "압축만들기.bat"])
    def test_batch_files_use_crlf(self, name):
        """윈도우 배치 파일이 LF 로 저장되면 goto/label 이 깨질 수 있다."""
        data = (PC_DIR / name).read_bytes()
        assert b"\r\n" in data, f"{name} 이 CRLF 가 아닙니다"
        lone_lf = data.replace(b"\r\n", b"").count(b"\n")
        assert lone_lf == 0, f"{name} 에 CRLF 가 아닌 줄이 {lone_lf}개 있습니다"

    @pytest.mark.parametrize(
        "name", ["시작-윈도우.bat", "시작-리눅스.sh", "시작-맥.command"]
    )
    def test_launcher_checks_for_project_files(self, name):
        """폴더를 잘못 잡았을 때 '인터넷 문제'로 오해하게 두면 안 된다."""
        text = (PC_DIR / name).read_text(encoding="utf-8")
        assert "pyproject.toml" in text, f"{name} 에 폴더 확인 단계가 없습니다"
        assert "압축" in text, f"{name} 에 압축 관련 안내가 없습니다"

    def test_launchers_install_the_fast_engine(self):
        """받는 사람이 첫 실행에서 바로 받아쓰기가 되어야 한다."""
        for name in ("시작-윈도우.bat", "시작-리눅스.sh", "시작-맥.command"):
            text = (PC_DIR / name).read_text(encoding="utf-8")
            assert "fast" in text and "web" in text, f"{name} 의 설치 대상 확인 필요"

    def test_handover_guide_exists(self):
        guide = PC_DIR / "넘겨주는방법.md"
        assert guide.exists()
        text = guide.read_text(encoding="utf-8")
        assert "시작-윈도우.bat" in text
        assert "인터넷으로 나가지 않습니다" in text  # 개인정보 안내가 빠지면 안 된다


class TestHuggingFaceSpace:
    def test_required_files_exist(self):
        for name in ("app.py", "requirements.txt", "README.md"):
            assert (HF_DIR / name).exists(), f"{name} 이 없습니다"

    def test_readme_has_space_metadata(self):
        text = (HF_DIR / "README.md").read_text(encoding="utf-8")
        assert text.startswith("---"), "Space 메타데이터 블록이 없습니다"
        for key in ("sdk: gradio", "app_file: app.py"):
            assert key in text, f"{key} 가 없습니다"

    def test_app_compiles(self):
        result = subprocess.run(
            ["python3", "-m", "py_compile", str(HF_DIR / "app.py")],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr

    def test_app_pins_the_correct_sensevoice_build(self):
        """2025-09-09 빌드는 광둥어 전용이라 한국어가 깨진다.

        주석에는 경고 목적으로 그 이름이 나올 수 있으니, 실제로 쓰이는
        모델 이름과 다운로드 주소만 검사한다.
        """
        text = (HF_DIR / "app.py").read_text(encoding="utf-8")
        code_lines = [
            line for line in text.splitlines() if not line.lstrip().startswith("#")
        ]
        code = "\n".join(code_lines)
        assert "2024-07-17" in code
        assert "2025-09-09" not in code

    def test_requirements_match_the_app_imports(self):
        requirements = (HF_DIR / "requirements.txt").read_text(encoding="utf-8")
        for package in ("gradio", "sherpa-onnx", "faster-whisper", "av", "numpy"):
            assert package in requirements, f"{package} 가 requirements.txt 에 없습니다"


class TestPhoneLaunchers:
    """휴대폰 접속용 실행 파일 검사."""

    @pytest.mark.parametrize(
        "name",
        ["시작-휴대폰도쓰기.bat", "시작-휴대폰도쓰기-맥.command", "시작-휴대폰도쓰기-리눅스.sh"],
    )
    def test_exists_and_enables_lan_and_https(self, name):
        text = (PC_DIR / name).read_text(encoding="utf-8")
        assert "--lan" in text, f"{name} 에 --lan 이 없습니다"
        assert "--https" in text, f"{name} 에 --https 가 없습니다(휴대폰 마이크에 필요)"
        assert "lan]" in text, f"{name} 이 [lan] 옵션 패키지를 설치하지 않습니다"

    @pytest.mark.parametrize(
        "name", ["시작-휴대폰도쓰기-맥.command", "시작-휴대폰도쓰기-리눅스.sh"]
    )
    def test_shell_syntax(self, name):
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash 가 없습니다")
        result = subprocess.run(
            [bash, "-n", str(PC_DIR / name)], capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stderr

    def test_batch_uses_crlf(self):
        data = (PC_DIR / "시작-휴대폰도쓰기.bat").read_bytes()
        assert b"\r\n" in data
        assert data.replace(b"\r\n", b"").count(b"\n") == 0


WEB_DIR = PROJECT_ROOT / "deploy" / "web"


class TestBrowserOnlyWebApp:
    """기기 안에서만 도는 정적 웹앱 검사.

    이 방식의 핵심은 '녹음이 서버로 안 간다' 는 것이다.
    업로드 코드가 실수로 들어오면 그 약속이 깨지므로 검사한다.
    """

    def test_splits_audio_itself_to_show_real_progress(self):
        """라이브러리에 통째로 맡기면 내부에서 30초씩 자르는데 진행을 알려 주지 않는다.

        5분 넘게 진행률이 55% 에 멈춰 있어 멈춘 것처럼 보였다.
        직접 잘라서 돌리면 구간별 진행과 남은 시간을 보여 줄 수 있고,
        조용한 구간을 건너뛰어 실제로 빨라진다.
        """
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "splitIntoWindows" in engine
        assert 'type: "progress"' in engine
        assert "남음" in html          # 남은 시간 표시

    def test_never_exceeds_the_thirty_second_limit(self):
        """Whisper 는 한 번에 30초까지만 본다. 넘기면 뒷부분을 조용히 버린다.

        쉬지 않고 말하는 녹음에서 '말하는 구간' 을 하나도 못 찾아
        전체를 한 창으로 반환하던 경로가 있었다(3분 녹음에서 2분 30초 유실).
        어떤 경로로 가든 toWindows 를 거치도록 고쳤다.
        """
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        assert "function toWindows" in engine
        # 제한을 우회하는 조기 반환이 없어야 한다
        body = engine[engine.index("export function splitIntoWindows") : engine.index("function toWindows")]
        for line in body.splitlines():
            if line.strip().startswith("return "):
                assert "toWindows" in line, f"toWindows 를 거치지 않는 반환: {line.strip()}"

    def test_every_requested_model_file_actually_exists(self):
        """모델·기기별로 실제 존재하는 파일만 요청하는지, 용량 안내가 맞는지 본다.

        q4f16 은 large-v3-turbo 에만 있는데 전부에 적용해서
        "Could not locate file" 로 실패한 적이 있다.
        또 화면에 안내하는 용량이 실제와 다르면 사용자가 데이터를 예상보다
        많이 쓰게 되므로, 표에 적은 sizeMB 도 실제 파일 크기와 대조한다.
        """
        import json

        actual = json.loads(
            (Path(__file__).parent / "whisper_onnx_files.json").read_text(encoding="utf-8")
        )
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        block = re.search(r"export const MODEL_PROFILES = \{(.*?)\n\};", engine, re.S)
        assert block, "MODEL_PROFILES 를 찾지 못했습니다"

        # 라이브러리 소스에서 확인한 이름 -> 파일 접미사 대응
        suffix = {
            "fp32": "", "fp16": "_fp16", "int8": "_int8", "uint8": "_uint8",
            "q8": "_quantized", "q4": "_q4", "q4f16": "_q4f16", "bnb4": "_bnb4",
        }

        entries = re.findall(
            r'(?:"([^"]+)":\s*\{)|'
            r'(\w+):\s*\{\s*dtype:\s*\{\s*encoder_model:\s*"(\w+)",\s*'
            r'decoder_model_merged:\s*"(\w+)"\s*\},\s*sizeMB:\s*(\d+)',
            block.group(1),
        )
        model = None
        checked = 0
        for model_name, profile, enc, dec, size_mb in entries:
            if model_name:
                model = model_name
                assert model in actual, f"파일 목록에 없는 모델: {model}"
                continue
            assert model, "모델 이름보다 먼저 나온 항목이 있습니다"
            assert {enc, dec} <= set(suffix), f"모르는 정밀도 이름: {enc}, {dec}"
            files = actual[model]
            total = 0
            for part, name in (("encoder_model", enc), ("decoder_model_merged", dec)):
                filename = f"{part}{suffix[name]}.onnx"
                assert filename in files, (
                    f"{model} 에 없는 파일을 요청합니다: {filename} ({profile} 칸)"
                )
                total += files[filename]
            expected = round(total / 1048576)
            assert int(size_mb) == expected, (
                f"{model} {profile}: 안내 용량 {size_mb}MB, 실제 {expected}MB"
            )
            checked += 1
        assert checked == 12, f"검사한 조합이 {checked}개뿐입니다(모델 4 x 칸 3 이어야 함)"

    def test_gpu_never_uses_int8_decoder(self):
        """그래픽 가속에서 q8 디코더를 쓰면 GPU 커널이 없어 CPU 로 떨어진다.

        디코더는 글자마다 도는 가장 무거운 부분이라 여기서 CPU 로 내려가면
        빠른 모드를 켜 놓고도 느리다. 실제로 이것 때문에 느렸다.
        """
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        block = re.search(r"export const MODEL_PROFILES = \{(.*?)\n\};", engine, re.S)
        assert block
        for profile, dec in re.findall(
            r'(\w+):\s*\{\s*dtype:\s*\{[^}]*decoder_model_merged:\s*"(\w+)"',
            block.group(1),
        ):
            if profile.startswith("webgpu"):
                assert dec != "q8", f"{profile} 칸이 GPU 에서 못 도는 q8 디코더를 씁니다"

    def test_threads_are_enabled_when_isolated(self):
        """헤더가 붙었는데도 CPU 를 1개만 쓰면 몇 배 느려진다."""
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        assert "crossOriginIsolated" in engine, "스레드 조건을 확인하지 않습니다"
        assert "numThreads" in engine, "스레드 수를 지정하지 않습니다"

        worker_headers = (WEB_DIR / "coi-serviceworker.js").read_text(encoding="utf-8")
        assert "Cross-Origin-Embedder-Policy" in worker_headers
        assert "Cross-Origin-Opener-Policy" in worker_headers
        # 외부 요청까지 가로채면 모델 내려받기를 방해할 수 있다
        assert "self.location.origin" in worker_headers, "같은 출처만 처리해야 합니다"

        page = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "coi-serviceworker.js" in page, "서비스 워커를 등록하지 않습니다"
    def test_retries_with_a_safe_precision_when_a_file_is_missing(self):
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        assert "FALLBACK_DTYPE" in engine
        assert "could not locate" in engine.lower()

    def test_checks_fp16_support_not_just_webgpu(self):
        """WebGPU 가 있어도 fp16(shader-f16) 을 못 쓰는 기기가 있다."""
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        assert "shader-f16" in engine

    def test_user_can_stop_a_running_job(self):
        """멈출 방법이 없으면 휴대폰이 뜨거워져도 탭을 닫는 수밖에 없다."""
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert 'id="stop"' in html
        assert "cancelled" in html

    def test_ignores_results_that_arrive_after_stopping(self):
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "if (cancelled) return;" in html

    def test_warns_before_a_long_recording_on_the_big_model(self):
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "뜨거워지고" in html or "뜨거워질" in html
        assert "confirm(" in html

    def test_diarization_uses_a_fast_transform(self):
        """정의대로 계산하면(DFT) 휴대폰이 눈에 띄게 버벅인다. FFT 여야 한다."""
        diarize = (WEB_DIR / "diarize.js").read_text(encoding="utf-8")
        assert "fftInPlace" in diarize

    def test_diarization_is_on_by_default(self):
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert 'id="diarize" checked' in html

    def test_progress_message_matches_the_real_phase(self):
        """예전에는 이미 다 받은 뒤에도 '모델을 내려받는 중' 이라고 표시했다."""
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        assert "currentPhase" in html
        assert 'type: "phase"' in engine
        # 단계와 무관하게 다운로드 중이라고 단정하는 문구가 없어야 한다
        assert "아직 모델을 내려받는 중입니다" not in html

    def test_warns_when_the_device_cannot_run_the_big_model_well(self):
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "requestAdapter" in html
        assert "뜨거워질 수 있습니다" in html

    def test_keeps_the_screen_awake_while_working(self):
        """휴대폰은 화면이 꺼지면 다운로드를 멈춘다. 560MB 를 받는 중이면 치명적이다."""
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "wakeLock" in html
        assert "visibilitychange" in html

    def test_translates_machine_errors_into_plain_korean(self):
        """'network error' 같은 메시지는 받는 사람이 무엇을 해야 할지 알 수 없다."""
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "friendlyError" in html
        assert "다운로드가 중간에 끊겼습니다" in html
        assert "메모리가 부족합니다" in html

    def test_warns_before_a_large_download(self):
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "다른 앱으로 나가면 중단됩니다" in html

    def test_offers_an_accurate_model_for_korean(self):
        """base 만으로는 한국어 정확도가 부족하다. 더 정확한 선택지가 있어야 한다."""
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "whisper-large-v3-turbo" in html

    def test_model_sizes_are_not_hardcoded_in_the_page(self):
        """내려받는 용량은 기기(그래픽 가속 여부)에 따라 다르다.

        화면에 숫자를 박아 두면 어느 한쪽에서는 반드시 틀린 안내가 된다.
        실제 값을 engine.js 의 표에서 읽어 오는지 확인한다.
        """
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "MODEL_PROFILES" in html, "표에서 용량을 읽어 오지 않습니다"
        assert "pickProfileKey" in html, "기기에 맞는 칸을 고르지 않습니다"
        hardcoded = re.findall(r"(?:약 )?(\d{2,4})\s?MB", html)
        assert not hardcoded, f"화면에 박아 둔 용량이 남아 있습니다: {hardcoded}"

    def test_picks_efficient_precision_per_device(self):
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        assert "q4f16" in engine        # WebGPU 에서 가장 작고 빠르다
        assert "webgpu" in engine

    def test_supports_speaker_separation(self):
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        assert (WEB_DIR / "diarize.js").exists()
        assert 'id="diarize"' in html
        assert "diarize.js" in engine

    def test_diarization_failure_does_not_lose_the_transcript(self):
        """화자 구분은 부가 기능이다. 실패해도 받아쓴 내용은 나와야 한다."""
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        index = engine.index("assignSpeakers")
        assert "catch" in engine[index : index + 600]

    def test_required_files_exist(self):
        for name in ("index.html", "worker.js", "engine.js", "diarize.js", "올리는방법.md"):
            assert (WEB_DIR / name).exists(), f"{name} 이 없습니다"

    def test_falls_back_to_the_main_thread(self):
        """작업자가 어떤 이유로든 뜨지 않아도 동작해야 한다.

        실제 안드로이드 기기에서 작업자가 원인 메시지도 없이 죽는 일이 있었다.
        원인을 따지기 전에 우선 동작하도록 우회 경로를 둔다.
        """
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "runOnMainThread" in html
        assert "engine.js" in html

    def test_busts_the_browser_cache_on_update(self):
        """파일을 고쳐도 브라우저가 예전 것을 쓰면 고친 의미가 없다."""
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "APP_VERSION" in html
        assert "worker.js?v=" in html

    def test_engine_is_shared_by_both_paths(self):
        """작업자와 화면 쪽이 같은 코드를 쓰도록 해 로직이 갈라지지 않게 한다."""
        worker = (WEB_DIR / "worker.js").read_text(encoding="utf-8")
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        assert "./engine.js" in worker
        assert "export async function runTranscription" in engine

    def test_never_uploads_audio(self):
        """서버로 오디오를 보내는 코드가 없어야 한다."""
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        worker = (WEB_DIR / "worker.js").read_text(encoding="utf-8")
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        for name, text in (("index.html", html), ("worker.js", worker), ("engine.js", engine)):
            assert "FormData" not in text, f"{name} 에 업로드 코드가 있습니다"
            assert "XMLHttpRequest" not in text, f"{name} 에 업로드 코드가 있습니다"
            # fetch 는 모델을 받을 때만 쓰이므로, 직접 호출이 없어야 한다.
            assert "fetch(" not in text, f"{name} 에 직접 fetch 호출이 있습니다"

    def test_worker_pins_an_exact_library_version(self):
        """CDN 버전을 고정하지 않으면 어느 날 갑자기 깨질 수 있다."""
        worker = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        assert "@huggingface/transformers@" in worker
        assert "@latest" not in worker

    def test_worker_uses_the_browser_bundle_not_the_bundler_build(self):
        """dist/transformers.web.js 는 번들러용이라 브라우저에서 즉시 죽는다.

        그 파일은 최상위에 `import "onnxruntime-web/webgpu"` 같은 이름 참조를
        남겨 두는데 브라우저가 이를 풀지 못해, 원인 메시지도 없이
        "작업자 오류: undefined" 만 뜬다. 실제로 겪었던 문제다.
        의존성이 모두 합쳐진 dist/transformers.min.js 를 써야 한다.
        """
        worker = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        code = "\n".join(
            line for line in worker.splitlines() if not line.lstrip().startswith("//")
        )
        assert "transformers.min.js" in code
        assert "transformers.web.js" not in code

    def test_worker_has_a_fallback_cdn(self):
        worker = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        assert "jsdelivr" in worker and "unpkg" in worker

    def test_page_discards_a_dead_worker(self):
        """죽은 워커를 재사용하면 두 번째 시도가 '모델 준비 중' 에서 멈춘다."""
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "killWorker" in html
        assert "terminate()" in html

    def test_handles_audio_longer_than_thirty_seconds(self):
        """Whisper 는 한 번에 30초만 본다. 잘라서 처리하도록 설정해야 한다."""
        worker = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        assert "chunk_length_s" in worker

    def test_falls_back_when_webgpu_is_missing(self):
        """아이폰·구형 기기는 WebGPU 가 없을 수 있다."""
        worker = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        assert "wasm" in worker and "webgpu" in worker

    def test_supports_iphone_recording_format(self):
        """아이폰 사파리는 audio/mp4 로 녹음한다."""
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "audio/mp4" in html

    def test_warns_when_not_served_over_https(self):
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "isSecureContext" in html

    def test_korean_is_the_default_language(self):
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert 'value="ko" selected' in html

    def test_mobile_viewport_is_declared(self):
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert 'name="viewport"' in html
        assert "width=device-width" in html

    def test_guide_covers_both_hosting_options(self):
        guide = (WEB_DIR / "올리는방법.md").read_text(encoding="utf-8")
        assert "GitHub Pages" in guide
        assert "Cloudflare" in guide
        assert "file://" in guide  # 파일 직접 열기가 안 된다는 안내
