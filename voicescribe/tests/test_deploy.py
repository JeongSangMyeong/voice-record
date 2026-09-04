"""배포용 파일 검사.

받는 사람이 더블클릭해서 쓰는 파일들이라, 깨지면 바로 사용자 문제로 이어진다.
문법·줄바꿈·필수 안내 문구가 유지되는지 확인한다.
"""

from __future__ import annotations

import json
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

    def test_diarization_uses_a_real_speaker_model(self):
        """직접 만든 음색 특징으로는 화자를 못 가른다(실측).

        같은 사람끼리 유사도 0.99, 다른 사람끼리 0.98 이라 사실상 구별이
        되지 않았고, 한 사람을 다섯 명으로 쪼개는 일이 잦았다.
        목소리를 전문으로 배운 모델을 써야 한다.
        """
        diarize = (WEB_DIR / "diarize.js").read_text(encoding="utf-8")
        assert "wespeaker" in diarize, "화자 인식 모델을 쓰지 않습니다"
        assert "AutoProcessor" in diarize and "AutoModel" in diarize
        # 되살아나면 안 되는 옛 방식
        assert "fftInPlace" not in diarize, "직접 만든 MFCC 방식이 되살아났습니다"
        assert "silhouette" not in diarize, "화자 수를 실루엣 점수로 고르면 안 됩니다"

    def test_diarization_can_answer_one_speaker(self):
        """옛 코드는 화자 수 후보를 2명부터 셌다.

        그래서 한 사람만 말한 녹음도 반드시 둘 이상으로 쪼갰다.
        임계값으로 멈추는 방식이라야 '한 명' 이라는 답이 나온다.
        """
        diarize = (WEB_DIR / "diarize.js").read_text(encoding="utf-8")
        assert "clusterByAffinity" in diarize
        assert "for (let k = 2;" not in diarize, "아직도 2명부터 세고 있습니다"

    def test_diarization_does_not_guess_when_the_model_is_missing(self):
        """모델을 못 받으면 지어내지 말고 화자 구분만 빼야 한다.

        옛 방식은 실측상 '무조건 한 명' 이라고 답하는 것보다도 점수가 낮았다.
        틀린 화자 표시는 사용자를 오히려 헷갈리게 한다.
        """
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "diarize-skipped" in engine
        assert "화자 구분을 하지 못했습니다" in html, "실패를 사용자에게 알리지 않습니다"

    def test_segment_times_survive_to_diarization(self):
        """구간 시각을 두 번 푸는 코드가 있으면 화자 구분이 통째로 죽는다.

        창을 직접 자르도록 바꾸면서 collected 가 이미 {start, end, text} 가 되었는데,
        뒤쪽에 남아 있던 c.timestamp 를 다시 읽는 코드가 모든 시각을 0 으로 만들었다.
        그러면 모든 구간의 길이가 0 이라 화자 구분이 전부 건너뛰어지고
        타임스탬프도 전부 00:00 으로 나온다.
        """
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        after = engine[engine.index("async function runWhisper") :]
        # 왜 그랬는지 적어 둔 주석은 검사에서 뺀다
        after = "\n".join(
            line for line in after.splitlines() if not line.lstrip().startswith("//")
        )
        # 창을 자를 때 한 번 푸는 것은 정상이다. 그 뒤에 또 풀면 시각이 0 이 된다.
        tail = after[after.index("return {") :]
        assert "c.timestamp" not in tail, (
            "구간을 만든 뒤에 c.timestamp 를 또 읽고 있습니다. 시각이 0 이 됩니다."
        )

    def test_clustering_matches_a_plain_implementation(self):
        """빠르게 고친 병합이 정의대로 계산한 결과와 같아야 한다."""
        node = shutil.which("node")
        if not node:
            pytest.skip("node 가 없어 건너뜁니다")
        script = f"""
        const {{ clusterByAffinity }} = await import("{(WEB_DIR / 'diarize.js').as_posix()}");
        function naive(V, th = 0.35, maxS = 8) {{
          const S = V.map((a) => V.map((b) => {{
            let d = 0; for (let i = 0; i < a.length; i++) d += a[i] * b[i]; return d; }}));
          let g = V.map((_, i) => [i]);
          while (g.length > 1) {{
            let best = {{ v: -Infinity, a: -1, b: -1 }};
            for (let a = 0; a < g.length; a++) for (let b = a + 1; b < g.length; b++) {{
              let s = 0; for (const i of g[a]) for (const j of g[b]) s += S[i][j];
              s /= g[a].length * g[b].length;
              if (s > best.v) best = {{ v: s, a, b }};
            }}
            if (best.v < th && g.length <= maxS) break;
            g[best.a] = g[best.a].concat(g[best.b]); g.splice(best.b, 1);
          }}
          const l = new Array(V.length).fill(0);
          g.forEach((m, i) => m.forEach((j) => (l[j] = i)));
          return l;
        }}
        const key = (l) => {{ const m = new Map();
          return l.map((x) => {{ if (!m.has(x)) m.set(x, m.size); return m.get(x); }}).join(","); }};
        let seed = 7;
        const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648 - 0.5);
        function make(n, k, noise) {{
          const base = [...Array(k)].map(() => {{
            const v = Float32Array.from({{ length: 48 }}, rand);
            let s = 0; v.forEach((x) => (s += x * x)); s = Math.sqrt(s);
            v.forEach((_, i) => (v[i] /= s)); return v; }});
          return [...Array(n)].map((_, i) => {{
            const c = base[i % k], v = new Float32Array(48); let s = 0;
            for (let j = 0; j < 48; j++) {{ v[j] = c[j] + rand() * noise; s += v[j] * v[j]; }}
            s = Math.sqrt(s); for (let j = 0; j < 48; j++) v[j] /= s; return v; }});
        }}
        let bad = 0, total = 0;
        for (const n of [2, 3, 5, 8, 15, 30, 60])
          for (const k of [1, 2, 3, 5])
            for (const noise of [0.1, 0.5, 1.5, 3.0]) {{
              if (k > n) continue;
              const V = make(n, k, noise); total++;
              if (key(clusterByAffinity(V)) !== key(naive(V))) bad++;
            }}
        // 극단 입력에서도 길이와 상한을 지켜야 한다
        const many = [...Array(20)].map((_, i) => {{ const v = new Float32Array(20); v[i] = 1; return v; }});
        const l = clusterByAffinity(many);
        console.log(JSON.stringify({{ total, bad, empty: clusterByAffinity([]).length,
          one: clusterByAffinity([Float32Array.from([1, 0])]).length,
          distinct: new Set(l).size, length: l.length }}));
        """
        result = subprocess.run(
            [node, "--input-type=module", "-e", script],
            capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 0, result.stderr
        r = json.loads(result.stdout.strip().splitlines()[-1])
        assert r["bad"] == 0, f"{r['total']}개 중 {r['bad']}개가 정의대로 계산한 결과와 다릅니다"
        assert r["empty"] == 0 and r["one"] == 1
        assert r["distinct"] <= 8, "화자 수 상한을 넘었습니다"
        assert r["length"] == 20, "결과 길이가 입력과 다릅니다"

    def test_short_segments_go_to_the_closest_speaker(self):
        """짧은 구간을 앞 구간 화자로 미루면 화자가 바뀐 직후가 전부 틀린다.

        실측(구간 3개 중 1개를 짧다고 가정): 앞 구간 따라가기 0.814 → 가까운 화자 1.000.
        """
        diarize = (WEB_DIR / "diarize.js").read_text(encoding="utf-8")
        assert "nearestCentroid" in diarize and "centroidsOf" in diarize

    def test_saved_file_opens_without_broken_korean(self):
        """UTF-8 파일에 BOM 이 없으면 윈도우 메모장·엑셀이 cp949 로 읽어 한글이 깨진다.

        옛 메모장은 LF 만 있으면 전부 한 줄로 붙여 버리므로 줄바꿈도 CRLF 로 맞춘다.
        """
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        save = html[html.index('$("save").addEventListener') :]
        save = save[: save.index("});") + 3]
        assert "\\uFEFF" in save, "저장 파일에 BOM 을 붙이지 않습니다 (한글이 깨집니다)"
        assert "\\r\\n" in save, "줄바꿈을 CRLF 로 바꾸지 않습니다"

    def test_saved_bytes_really_start_with_a_bom(self):
        """문자열 검사만으로는 부족하다. 실제로 나오는 바이트를 확인한다."""
        node = shutil.which("node")
        if not node:
            pytest.skip("node 가 없어 건너뜁니다")
        script = r"""
        const text = "\uFEFF" + "\uc548\ub155\ud558\uc138\uc694\n\ub458\uc9f8 \uc904".replace(/\r?\n/g, "\r\n");
        const bytes = new TextEncoder().encode(text);
        console.log(JSON.stringify({
          head: [...bytes.slice(0, 3)],
          crlf: [...bytes].some((b, i) => b === 13 && bytes[i + 1] === 10),
          // TextDecoder 는 BOM 을 알아서 떼어내므로 첫 글자부터 본문이다
          roundtrip: new TextDecoder("utf-8").decode(bytes).slice(0, 5),
        }));
        """
        result = subprocess.run(
            [node, "--input-type=module", "-e", script],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        r = json.loads(result.stdout.strip().splitlines()[-1])
        assert r["head"] == [239, 187, 191], f"BOM 이 아닙니다: {r['head']}"
        assert r["crlf"], "CRLF 줄바꿈이 없습니다"
        assert r["roundtrip"] == "안녕하세요", f"한글이 깨졌습니다: {r['roundtrip']}"

    def test_notification_goes_through_the_service_worker(self):
        """안드로이드 크롬은 new Notification() 을 막는다.

        서비스 워커의 showNotification 을 써야 휴대폰에 알림이 뜬다.
        """
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "registration.showNotification" in html, "서비스 워커로 알림을 띄우지 않습니다"
        assert "requestPermission" in html
        # 권한 요청은 사용자가 직접 누른 순간에만 가능하다
        assert 'addEventListener("change"' in html
        worker = (WEB_DIR / "coi-serviceworker.js").read_text(encoding="utf-8")
        assert "notificationclick" in worker, "알림을 눌러도 앱으로 돌아오지 않습니다"

    def test_notification_never_breaks_unsupported_browsers(self):
        """알림은 덤이다. 지원하지 않는 브라우저에서 오류가 나면 안 된다."""
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert 'typeof Notification !== "undefined"' in html
        assert "isSecureContext" in html
        # 서비스 워커가 늦거나 없을 때 영영 기다리면 안 된다
        assert "Promise.race" in html, "navigator.serviceWorker.ready 에 시간 제한이 없습니다"

    def test_installable_so_iphone_can_get_notifications(self):
        """아이폰은 홈 화면에 추가해야만 알림을 받을 수 있다(iOS 16.4 이상)."""
        import json as _json

        manifest_path = WEB_DIR / "manifest.json"
        assert manifest_path.exists(), "manifest.json 이 없어 홈 화면 앱이 되지 않습니다"
        manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["display"] == "standalone"
        for icon in manifest["icons"]:
            assert (WEB_DIR / icon["src"]).exists(), f"아이콘 파일이 없습니다: {icon['src']}"
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert 'rel="manifest"' in html
        assert "apple-touch-icon" in html

    def test_saving_offers_share_on_phones(self):
        """홈 화면에 추가한 아이폰 앱에서는 그냥 내려받기가 막히는 경우가 있다.

        공유가 되면 공유를 먼저 쓰고, 안 되면 내려받기로 돌아가야 한다.
        """
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "navigator.canShare" in html and "navigator.share" in html
        assert "AbortError" in html, "사용자가 공유를 취소한 경우를 오류로 처리하면 안 됩니다"
        # 공유가 안 되는 기기를 위해 내려받기 경로는 남아 있어야 한다
        assert "a.download = name" in html

    def test_screen_stays_awake_through_the_whole_job(self):
        """받아쓰기가 제일 오래 걸리는데, 그 단계에서 화면 꺼짐 방지를 풀고 있었다.

        그러면 휴대폰이 자동으로 화면을 끄고 작업이 멈춘다(아이폰은 특히 빠르다).
        화면 꺼짐 방지는 시작부터 끝(완료·실패·중지)까지 유지해야 한다.
        """
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        start = html.index('d.phase === "transcribing"')
        branch = html[start : start + 500]
        assert "releaseAwake()" not in branch, (
            "받아쓰기 단계에서 화면 꺼짐 방지를 풀고 있습니다. 작업이 중간에 멈춥니다."
        )
        assert "if (wakeLock && !wakeLock.released) return;" in html, (
            "화면 꺼짐 방지가 겹쳐 쌓입니다"
        )

    def test_background_keepalive_uses_audible_gain(self):
        """가려진 탭을 멈추지 않게 하려면 '소리가 나는' 상태여야 한다.

        크기가 정확히 0 이면 브라우저가 소리 없음으로 보고 그대로 멈춘다.
        들리지는 않지만 0 은 아닌 크기를 써야 한다.
        """
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "startKeepAlive" in html and "stopKeepAlive" in html
        match = re.search(r"gain\.gain\.value\s*=\s*([0-9.]+)", html)
        assert match, "배경 유지용 소리 크기를 찾지 못했습니다"
        value = float(match.group(1))
        assert 0 < value <= 0.01, f"크기가 {value} 입니다. 0 이면 소용없고 너무 크면 들립니다"
        # 작업이 끝나면 반드시 꺼야 한다
        for marker in ("function showResult", "function showError", '$("stop").addEventListener'):
            spot = html.index(marker)
            assert "stopKeepAlive()" in html[spot : spot + 400], f"{marker} 에서 소리를 끄지 않습니다"

    def test_keepalive_is_opt_in_and_disclosed(self):
        """소리를 몰래 재생하면 안 된다. 사용자가 켤 때만, 사실을 알리고 켠다."""
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert 'if ($("notify")?.checked) startKeepAlive();' in html, "동의 없이 켜집니다"
        assert "들리지 않는 소리" in html, "소리를 재생한다는 사실을 알리지 않습니다"
        assert "아이폰" in html, "아이폰에서는 안 된다는 안내가 없습니다"

    def test_sensevoice_features_match_the_reference(self):
        """소리에서 특징을 뽑는 계산이 조금만 틀어져도 인식이 통째로 망가진다.

        파이썬 기준 구현으로 미리 계산해 둔 값과 자바스크립트 결과를 대조한다.
        (기준 구현 자체는 sherpa-onnx 의 출력과 4개 언어에서 일치하는 것을 확인했다)
        """
        node = shutil.which("node")
        if not node:
            pytest.skip("node 가 없어 건너뜁니다")

        reference = Path(__file__).parent / "sensevoice_reference.json"
        script = f"""
        import fs from "node:fs";
        const m = await import("{(WEB_DIR / 'sensevoice.js').as_posix()}");
        const gold = JSON.parse(fs.readFileSync("{reference.as_posix()}", "utf8"));
        const meta = m.parseMeta(JSON.parse(
          fs.readFileSync("{(WEB_DIR / 'sensevoice-meta.json').as_posix()}", "utf8")));
        const audio = Float32Array.from(gold["소리표본"]);
        const mel = m.computeFbank(audio, 16000);
        const lfr = m.applyLfr(mel, meta.lfrWindowSize, meta.lfrWindowShift);
        m.applyCmvn(lfr, meta.negMean, meta.invStddev);
        const diff = (a, b) => Math.max(...a.map((x, i) => Math.abs(x - b[i])));
        console.log(JSON.stringify({{
          melFrames: mel.length,
          melDim: mel[0]?.length,
          lfrFrames: lfr.length,
          lfrDim: lfr[0]?.length,
          firstMelDiff: diff([...mel[0]], gold["멜첫줄"]),
          normHeadDiff: diff([...lfr[0]].slice(0, 20), gold["정규화첫줄앞20"]),
        }}));
        """
        result = subprocess.run(
            [node, "--input-type=module", "-e", script],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        r = json.loads(result.stdout.strip().splitlines()[-1])
        gold = json.loads(reference.read_text(encoding="utf-8"))
        # 표본을 잘라 넣었으므로 프레임 수는 잘린 길이에 맞게 나온다
        assert r["melDim"] == 80 and r["lfrDim"] == 560
        assert r["melFrames"] > 0 and r["lfrFrames"] > 0
        assert r["firstMelDiff"] < 1e-3, f"멜 특징이 어긋납니다: {r['firstMelDiff']}"
        assert r["normHeadDiff"] < 1e-2, f"정규화 결과가 어긋납니다: {r['normHeadDiff']}"

    def test_sensevoice_preprocessing_rules_are_complete(self):
        """브라우저는 ONNX 안의 규칙을 못 읽어서 따로 빼 두었다. 빠지면 인식이 안 된다."""
        import json as _json

        meta = _json.loads((WEB_DIR / "sensevoice-meta.json").read_text(encoding="utf-8"))
        assert len(meta["neg_mean"]) == 560
        assert len(meta["inv_stddev"]) == 560
        assert meta["lfr_window_size"] == "7" and meta["lfr_window_shift"] == "6"
        for code in ("lang_auto", "lang_ko", "lang_en", "lang_ja", "lang_zh", "lang_yue"):
            assert code in meta, f"언어 코드가 빠졌습니다: {code}"
        # sherpa-onnx 2024-07-17 판. 신판(2025-09-09)은 한국어가 깨진다.
        assert meta["크기"] == 239233841, "확인한 것과 다른 모델을 가리키고 있습니다"

    def test_sensevoice_ctc_decoding(self):
        """같은 글자가 이어지면 하나로 합치고 빈칸은 버려야 한다."""
        node = shutil.which("node")
        if not node:
            pytest.skip("node 가 없어 건너뜁니다")
        script = f"""
        const m = await import("{(WEB_DIR / 'sensevoice.js').as_posix()}");
        const tokens = ["<blank>", "▁안녕", "하", "세", "요", "<|ko|>"];
        // 프레임마다 고를 번호: 빈칸(0)과 반복을 섞어 둔다
        const ids = [0, 5, 1, 1, 0, 2, 2, 2, 3, 0, 4, 4];
        const vocab = tokens.length;
        const logits = new Float32Array(ids.length * vocab);
        ids.forEach((id, t) => (logits[t * vocab + id] = 10));
        const pieces = m.ctcGreedy(logits, ids.length, vocab, tokens);
        console.log(JSON.stringify({{ pieces, text: m.detokenize(pieces),
          tokens: m.parseTokens("<blank> 0\\n▁안녕 1\\n하 2\\n") }}));
        """
        result = subprocess.run(
            [node, "--input-type=module", "-e", script],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        r = json.loads(result.stdout.strip().splitlines()[-1])
        assert r["text"] == "안녕하세요", f"해독 결과가 다릅니다: {r['text']!r}"
        assert r["tokens"] == ["<blank>", "▁안녕", "하"], r["tokens"]

    def test_sensevoice_windows_are_short_enough_for_speakers(self):
        """SenseVoice 는 글자별 시각을 주지 않는다.

        구간을 길게 잡으면 그 구간 전체가 한 화자로 뭉쳐 버린다.
        """
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        match = re.search(r"SENSEVOICE_WINDOW_SECONDS = (\d+)", engine)
        assert match, "SenseVoice 용 구간 길이를 찾지 못했습니다"
        assert int(match.group(1)) <= 15, "구간이 길어 화자 구분이 뭉개집니다"
        assert "tileAtPauses(audio, sampleRate, SENSEVOICE_WINDOW_SECONDS)" in engine

    def test_diarization_works_with_sensevoice_too(self):
        """SenseVoice 경로에서는 화자 구분용 라이브러리를 아직 안 불러온 상태다.

        그대로 두면 화자 구분이 항상 실패한다(실제로 그랬다).
        """
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        spot = engine.index('const { assignSpeakers } = await import("./diarize.js")')
        before = engine[max(0, spot - 400) : spot]
        assert "if (!libRef) await loadLibrary();" in before, (
            "SenseVoice 로 받아쓰면 화자 구분이 실패합니다"
        )

    def test_engine_choice_is_understandable(self):
        """어느 것이 무슨 엔진인지 화면만 보고 알 수 있어야 한다.

        예전에는 칸 이름이 '정확도' 였고 그 밑에 '한국어는 가장 정확을 권합니다'
        라는 옛 안내가 남아 있어서, 기본값과 정반대로 안내하고 있었다.
        사용자가 '뭐가 SenseVoice 냐' 고 물어본 이유다.
        """
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        select = html[html.index('<select id="model">') : html.index("</select>", html.index('<select id="model">'))]
        assert "SenseVoice" in select, "엔진 이름이 화면에 없습니다"
        assert "Whisper" in select, "엔진 이름이 화면에 없습니다"
        assert "optgroup" in select, "어느 것이 어느 언어용인지 묶여 있지 않습니다"
        assert 'value="sensevoice-small" selected' in select, "권장 엔진이 기본값이 아닙니다"
        # 기본값과 어긋나는 옛 안내가 남아 있으면 안 된다
        assert "한국어는 <b>가장 정확</b>을 권합니다" not in html, "옛 안내가 남아 있습니다"

    def test_onnx_engine_is_served_from_our_own_site(self):
        """CDN 주소로 불러오면 브라우저가 작업자를 못 만들어 통째로 죽는다.

        실제 오류: SecurityError: Failed to construct 'Worker': Script at
        'https://cdn.jsdelivr.net/.../ort.bundle.min.mjs' cannot be accessed
        from origin 'https://jeongsangmyeong.github.io'.
        CPU 를 여러 개 쓰려면 작업자가 필요하므로 같은 사이트에 두어야 한다.
        """
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        block = engine[engine.index("const ORT_DIR") : engine.index("let senseVoice")]
        assert "cdn.jsdelivr.net" not in block and "unpkg.com" not in block, (
            "onnxruntime 을 CDN 에서 불러오면 작업자를 만들지 못해 죽습니다"
        )
        assert 'const ORT_DIR = "./ort/"' in engine

        root = WEB_DIR.parent.parent.parent
        for name in ("ort.bundle.min.mjs", "ort-wasm-simd-threaded.mjs",
                     "ort-wasm-simd-threaded.wasm"):
            assert (root / "ort" / name).exists(), f"엔진 파일이 없습니다: ort/{name}"

    def test_sensevoice_windows_cover_everything(self):
        """조용한 부분을 버리고 자르면 말끝이 잘려 결과가 나빠진다.

        실측: 파일 통째로 넣으면 '조금만 생각을 하면서 살면 훨씬 편할 거야.' 인데
        무음을 버리고 자르면 '조 금만 생각 을 하 면서 살 면 훨씬 편할 거야.' 가 된다.
        SenseVoice 용 구간은 소리 전체를 빠짐없이 덮어야 한다.
        """
        node = shutil.which("node")
        if not node:
            pytest.skip("node 가 없어 건너뜁니다")
        script = f"""
        const {{ tileAtPauses }} = await import("{(WEB_DIR / 'engine.js').as_posix()}");
        const sr = 16000;
        const make = (seconds, speaking) => {{
          const a = new Float32Array(sr * seconds);
          for (let i = 0; i < a.length; i++) {{
            const t = i / sr;
            if (speaking(t)) a[i] = Math.sin(i * 0.05) * 0.3;
          }}
          return a;
        }};
        const cases = [
          ["짧음", make(5, () => true)],
          ["쉼표 있는 긴 소리", make(90, (t) => t % 7 < 5)],
          ["끊김 없는 긴 소리", make(90, () => true)],
          ["거의 무음", make(40, (t) => t > 39.5)],
        ];
        const out = [];
        for (const [name, audio] of cases) {{
          for (const max of [12, 28]) {{
            const w = tileAtPauses(audio, sr, max);
            const total = audio.length / sr;
            const covered = w.reduce((s, x) => s + (x.end - x.start), 0);
            const gap = w.slice(1).some((x, i) => Math.abs(x.start - w[i].end) > 1e-6);
            out.push({{ name, max, windows: w.length,
              coverGap: Math.abs(covered - total),
              gap, longest: Math.max(...w.map((x) => x.end - x.start)) }});
          }}
        }}
        console.log(JSON.stringify(out));
        """
        result = subprocess.run(
            [node, "--input-type=module", "-e", script],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        for row in json.loads(result.stdout.strip().splitlines()[-1]):
            label = f"{row['name']} 상한 {row['max']}초"
            assert row["coverGap"] < 0.05, f"{label}: 소리 일부가 빠졌습니다"
            assert not row["gap"], f"{label}: 구간 사이에 틈이 있습니다"
            assert row["longest"] <= row["max"] + 0.05, f"{label}: 구간이 상한을 넘었습니다"

    def test_sensevoice_does_not_drop_silence(self):
        """무음을 버리는 방식(Whisper 용)을 SenseVoice 에 쓰면 안 된다."""
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        spot = engine.index("async function runSenseVoice")
        body = engine[spot : engine.index("async function runWhisper")]
        assert "tileAtPauses(" in body, "전체를 덮는 방식을 쓰지 않습니다"
        assert "splitIntoWindows(" not in body, "무음을 버리는 방식을 쓰고 있습니다"

    def test_web_files_at_the_root_match_the_deploy_copy(self):
        """뿌리의 파일이 실제로 배포되는 파일이다. 테스트는 배포본만 본다.

        둘이 어긋나면 검사를 통과했는데도 사용자에게는 옛 파일이 간다.
        """
        root = WEB_DIR.parent.parent.parent
        for name in ("index.html", "engine.js", "diarize.js", "worker.js",
                     "coi-serviceworker.js", "manifest.json", "sensevoice.js",
                     "sensevoice-meta.json",
                     "icon-192.png", "icon-512.png", "apple-touch-icon.png"):
            here, there = root / name, WEB_DIR / name
            if not here.exists():
                continue
            assert here.read_bytes() == there.read_bytes(), f"{name} 이 배포본과 다릅니다"

    def test_speaker_clustering_scores_perfectly_on_real_voices(self):
        """실제 사람 목소리로 만든 평가셋에서 성능이 떨어지면 잡아낸다.

        임베딩은 브라우저가 쓰는 모델과 같은 가중치·같은 전처리로 미리 뽑아 두었다.
        모델을 내려받지 않고도 묶는 논리를 그대로 검증할 수 있다.
        옛 구현 점수: 쌍 F1 0.663, 화자 수 정확 4/11.
        """
        import json
        import shutil
        import subprocess

        node = shutil.which("node")
        if not node:
            pytest.skip("node 가 없어 건너뜁니다")

        fixture = Path(__file__).parent / "speaker_embeddings.json"
        script = f"""
        import fs from "node:fs";
        const {{ clusterByAffinity, toSpeakerNames }} =
          await import("{(WEB_DIR / 'diarize.js').as_posix()}");
        const d = JSON.parse(fs.readFileSync("{fixture.as_posix()}", "utf8"));
        const out = [];
        for (const [name, info] of Object.entries(d["시나리오"])) {{
          const V = d["임베딩"][name].map((v) => Float32Array.from(v));
          const labels = toSpeakerNames(V.length, V.map((_, i) => i), clusterByAffinity(V));
          let tp = 0, fp = 0, fn = 0;
          const t = info["정답"];
          for (let i = 0; i < t.length; i++) for (let j = i + 1; j < t.length; j++) {{
            const same = t[i] === t[j], psame = labels[i] === labels[j];
            if (same && psame) tp++; else if (psame) fp++; else if (same) fn++;
          }}
          const pr = tp + fp ? tp / (tp + fp) : 1, rc = tp + fn ? tp / (tp + fn) : 1;
          out.push({{ name, f1: pr + rc ? (2 * pr * rc) / (pr + rc) : 0,
                     k: new Set(labels).size, want: info["화자수"] }});
        }}
        console.log(JSON.stringify(out));
        """
        result = subprocess.run(
            [node, "--input-type=module", "-e", script],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        rows = json.loads(result.stdout.strip().splitlines()[-1])
        assert len(rows) == 11

        wrong = [r for r in rows if r["k"] != r["want"]]
        assert not wrong, "화자 수를 틀린 경우: " + ", ".join(
            f"{r['name']} {r['k']}명(정답 {r['want']}명)" for r in wrong
        )
        average = sum(r["f1"] for r in rows) / len(rows)
        assert average > 0.95, f"쌍 F1 평균이 {average:.3f} 로 떨어졌습니다"

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
        # 받아쓰기 모델만 기기에 따라 용량이 달라진다.
        # 화자 구분 모델은 어떤 기기에서도 같은 파일(26MB)이라 적어 두어도 된다.
        without_speaker_note = re.sub(r"처음 한 번 \d+MB 더 받음", "", html)
        hardcoded = re.findall(r"(?:약 )?(\d{2,4})\s?MB", without_speaker_note)
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
        assert "catch" in engine[index : index + 1500]

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
        """녹음이 기기 밖으로 나가지 않아야 한다. 이 약속이 이 도구의 존재 이유다.

        모델을 내려받으려면 fetch 가 필요하므로 무조건 금지할 수는 없다.
        대신 모든 요청이 (1) 알려진 모델 주소이고 (2) 보내는 내용이 없는지 본다.
        """
        files = ["index.html", "worker.js", "engine.js", "sensevoice.js", "diarize.js"]
        allowed = ("huggingface.co", "cdn.jsdelivr.net", "unpkg.com", "./", "`${")
        for name in files:
            path = WEB_DIR / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            for banned in ("FormData", "XMLHttpRequest", "sendBeacon", "WebSocket", "RTCPeer"):
                assert banned not in text, f"{name} 에 밖으로 보내는 코드가 있습니다: {banned}"

            for match in re.finditer(r"fetch\(", text):
                call = text[match.end() : match.end() + 220]
                assert not re.search(r"\bbody\s*:", call), (
                    f"{name} 의 fetch 가 무언가를 보내고 있습니다: {call[:90]}"
                )
                assert not re.search(r"method\s*:\s*[\"']()(?!GET)", call), (
                    f"{name} 의 fetch 가 GET 이 아닙니다: {call[:90]}"
                )
                assert any(token in call for token in allowed), (
                    f"{name} 에 알 수 없는 곳으로 가는 fetch 가 있습니다: {call[:90]}"
                )

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
