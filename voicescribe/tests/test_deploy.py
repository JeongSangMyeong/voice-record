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

    def test_default_model_is_the_most_accurate(self):
        """기본값은 가장 정확한 것이어야 하고, 안내가 기본값과 어긋나면 안 된다.

        예전에 '한국어는 가장 정확을 권합니다' 라는 옛 안내가 기본값과 정반대로
        남아 있어서 사용자가 어느 것을 골라야 하는지 물어본 적이 있다.
        """
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        select = html[html.index('<select id="model">') : html.index("</select>", html.index('<select id="model">'))]
        # 실제 회의 녹음으로 재 보니 Whisper 가 확실히 정확했다. 그쪽이 기본값이어야 한다.
        assert 'value="onnx-community/whisper-large-v3-turbo" selected' in select, (
            "가장 정확한 엔진이 기본값이 아닙니다"
        )
        assert "한국어는 <b>가장 정확</b>을 권합니다" not in html, "옛 안내가 남아 있습니다"

    def test_windows_cover_the_whole_recording(self):
        """조용한 부분을 버리고 자르면 경계에서 말이 통째로 사라진다.

        실제 회의 녹음(2분)으로 잰 결과다. 단어 오류율 / 빠뜨린 단어 수:
          무음 버림 28초   40.9%  32개
          무음 버림 20초   51.1%  61개   ← 구간 길이만 바꿔도 무너진다
          전체 덮기 28초   39.2%  30개
          전체 덮기 15초   39.2%  30개   ← 어느 길이든 일정하다
        전체를 덮되 경계는 쉬는 자리로 잡아야 한다.
        """
        node = shutil.which("node")
        if not node:
            pytest.skip("node 가 없어 건너뜁니다")

        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        assert "tileAtPauses(audio, sampleRate)" in engine, "전체를 덮는 방식을 쓰지 않습니다"

        script = f"""
        const {{ tileAtPauses }} = await import("{(WEB_DIR / 'engine.js').as_posix()}");
        const sr = 16000;
        const make = (seconds, speaking) => {{
          const a = new Float32Array(sr * seconds);
          for (let i = 0; i < a.length; i++) if (speaking(i / sr)) a[i] = Math.sin(i * 0.05) * 0.3;
          return a;
        }};
        const cases = [
          ["쉼 있는 긴 소리", make(120, (t) => t % 7 < 5), 120],
          ["끊김 없는 긴 소리", make(120, () => true), 120],
          ["짧은 소리", make(5, () => true), 5],
          ["말이 가운데만", make(40, (t) => t > 10 && t < 25), 40],
        ];
        const out = [];
        for (const [name, audio, total] of cases) {{
          const w = tileAtPauses(audio, sr);
          // 말이 있는 부분이 모두 어느 구간엔가 들어 있는지 확인한다
          let uncovered = 0;
          for (let t = 0; t < total; t += 0.1) {{
            const i = Math.floor(t * sr);
            if (Math.abs(audio[i]) < 0.01) continue;              // 조용한 지점은 넘어간다
            if (!w.some((x) => t >= x.start - 0.05 && t <= x.end + 0.05)) uncovered++;
          }}
          out.push({{ name, windows: w.length, uncovered,
            longest: w.length ? Math.max(...w.map((x) => x.end - x.start)) : 0 }});
        }}
        console.log(JSON.stringify(out));
        """
        result = subprocess.run(
            [node, "--input-type=module", "-e", script],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        for row in json.loads(result.stdout.strip().splitlines()[-1]):
            assert row["uncovered"] == 0, f"{row['name']}: 말이 있는 부분을 {row['uncovered']}군데 빠뜨렸습니다"
            assert row["longest"] <= 28.05, f"{row['name']}: 구간이 30초 제한을 넘습니다"
            assert row["windows"] > 0, f"{row['name']}: 구간이 하나도 없습니다"

    def test_web_files_at_the_root_match_the_deploy_copy(self):
        """뿌리의 파일이 실제로 배포되는 파일이다. 테스트는 배포본만 본다.

        둘이 어긋나면 검사를 통과했는데도 사용자에게는 옛 파일이 간다.
        """
        root = WEB_DIR.parent.parent.parent
        for name in ("index.html", "engine.js", "diarize.js", "worker.js",
                     "coi-serviceworker.js", "manifest.json",
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
        files = ["index.html", "worker.js", "engine.js", "diarize.js"]
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


def _run_node(script: str) -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("node 가 없어 건너뜁니다")
    result = subprocess.run(
        [node, "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


class TestIphoneDoesNotReload:
    """아이폰에서 파일을 올리면 '로딩하다가 새로고침' 되던 문제(2026-10-07 제보).

    사파리는 메모리를 너무 쓰는 탭을 강제로 끄고 다시 연다. 원인이 둘 겹쳐 있었다.
      1) 작업자 안에서는 navigator.vendor 가 없어(웹킷 NavigatorID.idl 의 [Exposed=Window])
         라이브러리가 사파리를 못 알아보고 asyncify 판을 고른다. 이 판은 iOS 26.2 이후
         웹킷 JIT 메모리 폭주 버그를 일으킨다(onnxruntime#26827, WebKit 304810).
      2) '큼' 모델은 실측 메모리 +4GB. 아이폰 탭 한도는 약 1.5GB 다.
    """

    def test_webkit_is_recognised_inside_a_worker(self):
        """아이폰의 크롬·파이어폭스도 속은 웹킷이라 같은 대우를 받아야 한다."""
        engine = (WEB_DIR / "engine.js").as_posix()
        out = _run_node(f"""
        const {{ isWebKit, isAppleMobile }} = await import("{engine}");
        const ua = {{
          iphoneSafari: "Mozilla/5.0 (iPhone; CPU iPhone OS 26_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.1 Mobile/15E148 Safari/604.1",
          iphoneChrome: "Mozilla/5.0 (iPhone; CPU iPhone OS 26_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/140.0.0.0 Mobile/15E148 Safari/604.1",
          macSafari: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.1 Safari/605.1.15",
          desktopChrome: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
          androidChrome: "Mozilla/5.0 (Linux; Android 15; SM-S928N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Mobile Safari/537.36",
          firefox: "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:143.0) Gecko/20100101 Firefox/143.0",
        }};
        console.log(JSON.stringify({{
          webkit: Object.fromEntries(Object.entries(ua).map(([k, v]) => [k, isWebKit(v)])),
          mobile: {{
            iphone: isAppleMobile(ua.iphoneSafari, "iPhone", 5),
            ipadDesktopMode: isAppleMobile(ua.macSafari, "MacIntel", 5),
            mac: isAppleMobile(ua.macSafari, "MacIntel", 0),
            android: isAppleMobile(ua.androidChrome, "Linux armv8l", 5),
          }},
        }}));
        """)
        got = json.loads(out)
        assert got["webkit"] == {
            "iphoneSafari": True, "iphoneChrome": True, "macSafari": True,
            "desktopChrome": False, "androidChrome": False, "firefox": False,
        }
        assert got["mobile"] == {"iphone": True, "ipadDesktopMode": True, "mac": False, "android": False}

    def test_webkit_gets_the_build_without_the_memory_bug(self):
        """웹킷에서는 asyncify 판 대신 일반 판을 쓴다. 일반 판은 그래픽 가속이 없으므로 CPU 로 돈다."""
        engine_path = WEB_DIR / "engine.js"
        out = _run_node(f"""
        const {{ useWebKitSafeBuild }} = await import("{engine_path.as_posix()}");
        const base = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/";
        const make = () => ({{ backends: {{ onnx: {{ wasm: {{ wasmPaths: {{
          mjs: base + "ort-wasm-simd-threaded.asyncify.mjs",
          wasm: base + "ort-wasm-simd-threaded.asyncify.wasm",
        }} }} }} }} }});
        const iphone = make(), chrome = make();
        useWebKitSafeBuild(iphone, "Mozilla/5.0 (iPhone; CPU iPhone OS 26_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.1 Mobile/15E148 Safari/604.1");
        useWebKitSafeBuild(chrome, "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36");
        console.log(JSON.stringify({{ iphone: iphone.backends.onnx.wasm.wasmPaths, chrome: chrome.backends.onnx.wasm.wasmPaths }}));
        """)
        got = json.loads(out)
        assert got["iphone"]["wasm"].endswith("/ort-wasm-simd-threaded.wasm")
        assert got["iphone"]["mjs"].endswith("/ort-wasm-simd-threaded.mjs")
        assert "asyncify" in got["chrome"]["wasm"], "웹킷이 아닌 곳은 건드리지 않아야 합니다"

        engine = engine_path.read_text(encoding="utf-8")
        load = engine[engine.index("async function loadLibrary") : engine.index("function tuneThreads")]
        assert "useWebKitSafeBuild(" in load, "라이브러리를 불러온 직후에 판을 바꾸지 않습니다"
        pick = engine[engine.index("async function pickDevice") : engine.index("async function loadLibrary")]
        assert "isWebKit()" in pick, "웹킷에서 그래픽 가속을 고르면 일반 판에는 없어서 실패합니다"

    def test_light_models_can_start_without_graphics_acceleration(self):
        """q8 디코더는 onnxruntime-web 1.26 의 기본 최적화 단계에서 세션을 못 만든다.

        실제 오류: "TransposeDQWeightsForMatMulNBits Missing required scale".
        그래서 그래픽 가속이 없는 기기에서는 '보통·작음·아주 작음' 이 아예 시작하지 못했다.
        최적화 단계를 basic 으로 낮추면 된다(실측: 보통 +1.19GB, 28초 소리를 23초에 처리).
        q4 로 바꾸는 방법도 되지만 CPU 에서는 세 배 느리고 메모리도 더 쓴다(+2.0GB, 49초).
        """
        engine_path = WEB_DIR / "engine.js"
        out = _run_node(f"""
        const {{ MODEL_PROFILES, sessionOptionsFor }} = await import("{engine_path.as_posix()}");
        const rows = [];
        for (const [model, profiles] of Object.entries(MODEL_PROFILES)) {{
          for (const [key, p] of Object.entries(profiles)) {{
            rows.push({{ model, key, dtype: p.dtype, options: sessionOptionsFor(p.dtype) ?? null }});
          }}
        }}
        rows.push({{ model: "fallback", key: "any", dtype: {{ encoder_model: "q8", decoder_model_merged: "q8" }},
                     options: sessionOptionsFor({{ encoder_model: "q8", decoder_model_merged: "q8" }}) ?? null }});
        console.log(JSON.stringify(rows));
        """)
        for row in json.loads(out):
            uses_q8 = "q8" in row["dtype"].values()
            if uses_q8:
                assert row["options"] == {"graphOptimizationLevel": "basic"}, row
            else:
                assert row["options"] is None, f"q8 이 아닌데 최적화를 낮춥니다(느려짐): {row}"

        engine = engine_path.read_text(encoding="utf-8")
        assert "session_options" in engine, "세션 옵션을 넘기지 않습니다"
        # 아이폰이 실제로 쓰는 '보통' 은 디코더가 CPU 에서 빠르고 가벼운 q8 이어야 한다.
        # 인코더는 q4 여야 한다. q8 인코더는 '아주 작음' 에서 "베베베…" 만 낸다(실측).
        for name in ("whisper-small", "whisper-base", "whisper-tiny"):
            block = re.search(rf'"onnx-community/{name}":\s*\{{(.*?)\n  \}}', engine, re.S).group(1)
            assert re.search(r'wasm:\s*\{\s*dtype:\s*\{\s*encoder_model:\s*"q4",\s*decoder_model_merged:\s*"q8"', block), name

    def test_iphone_never_offers_the_big_model(self):
        """'큼' 은 아이폰 메모리에 들어가지 않는다. 고르면 반드시 새로고침된다."""
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "isAppleMobile(" in html
        block = html[html.index("function adaptModelsToPhone") :]
        block = block[: block.index("\n}\n")]
        assert "whisper-large-v3-turbo" in block and ".remove()" in block, "아이폰에서 '큼' 을 빼지 않습니다"
        assert "whisper-small" in block, "아이폰 기본값을 '보통' 으로 두지 않습니다"

    def test_audio_is_decoded_straight_to_16khz(self):
        """48kHz 로 한 번 펼쳤다가 줄이면 1시간 스테레오가 1.3GB 다. 바로 16kHz 로 읽는다."""
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        decode = html[html.index("async function toMono16k") : html.index("/* ---------- 실행 ---------- */")]
        assert "(1, 1, SAMPLE_RATE)" in decode, "16kHz 컨텍스트로 바로 디코딩하지 않습니다"
        assert "numberOfChannels" in decode

    def test_audio_is_handed_to_the_worker_not_copied(self):
        """작업자에 넘길 때 복사하면 같은 소리가 메모리에 두 벌 남는다."""
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert re.search(r"worker\.postMessage\(request,\s*\[", html), "소리를 복사해서 넘깁니다"
        main = html[html.index("async function runOnMainThread") :]
        main = main[: main.index("\n}\n")]
        assert "toMono16k(" in main, "넘겨서 비어 버린 소리를 다시 읽지 않습니다(대비 경로가 빈 소리로 돈다)"

    def test_tells_the_user_when_the_phone_killed_the_page(self):
        """새로고침이 또 일어나도 조용히 처음 화면으로 돌아가지 않고, 이유와 대책을 알린다."""
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert html.count("WORK_MARK") >= 4, "작업 중 표시를 남기거나 지우지 않습니다"
        recover = html[html.index("function recoverFromKilledPage") :]
        recover = recover[: recover.index("\n}\n")]
        assert "메모리" in recover
        assert "SMALLER_MODEL" in recover, "다음에는 더 작은 모델을 고르지 않습니다"
        for fn in ("function showError", "function showResult"):
            body = html[html.index(fn) :]
            assert "clearWorkMark()" in body[: body.index("\n}\n")], f"{fn} 에서 표시를 지우지 않습니다"
        stop = html[html.index('$("stop").addEventListener') :]
        assert "clearWorkMark()" in stop[: stop.index("\n});\n")]

    def test_gpu_loss_continues_in_a_fresh_worker(self):
        """라이브러리는 실행을 한 줄로 묶어 두어, 한 번 실패하면 같은 작업자에서는 계속 실패한다.

        (transformers.js 의 webInferenceChain/webInitChain) 그래서 같은 작업자 안에서
        일반 모드로 갈아타는 예전 방식은 한 번도 동작할 수 없었다. 새 작업자에서 이어서 한다.
        """
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        worker = (WEB_DIR / "worker.js").read_text(encoding="utf-8")
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "resume" in engine and "forceDevice" in engine
        assert '"gpu-lost"' in worker and "audio.buffer" in worker, "소리를 돌려주지 않으면 이어서 할 수 없습니다"
        handler = html[html.index('d.type === "gpu-lost"') :][:900]
        assert "killWorker()" in handler and 'forceDevice: "wasm"' in handler

    def test_switching_models_releases_the_old_one(self):
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        body = engine[engine.index("async function getTranscriber") :]
        body = body[: body.index("\n}\n")]
        assert ".dispose()" in body

    def test_worker_error_reruns_the_current_request(self):
        """작업자를 재사용하므로, 오류 처리기가 처음 요청을 붙잡고 있으면 엉뚱한 파일을 다시 돈다."""
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        start = html[html.index("function startWithWorker") :]
        start = start[: start.index("\n}\n")]
        assert "runOnMainThread(activeRequest)" in start
        assert "runOnMainThread(request)" not in start

    def test_service_worker_does_not_reload_on_webkit(self):
        """사파리는 COEP credentialless 를 모른다. 새로고침해도 격리되지 않고, 파일을 읽는 중에 새로고침될 수 있다."""
        coi = (WEB_DIR / "coi-serviceworker.js").read_text(encoding="utf-8")
        page_part = coi[coi.index("// --- 페이지에서 불릴 때") :]
        assert "AppleWebKit" in page_part
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        go = html[html.index('$("go").addEventListener') :]
        assert go.index("window.__transcribing = true") < go.index("toMono16k("), (
            "소리를 읽는 동안에는 새로고침 금지 표시가 없습니다"
        )

    def test_picking_a_file_mid_job_does_not_start_a_second_job(self):
        """작업 중에 다른 파일을 고르면 시작 버튼이 다시 켜져 두 작업이 한 작업자에서 겹쳤다.

        또 대비 경로가 '지금 고른 파일' 을 다시 읽어, 앞 작업의 이어하기에 엉뚱한 파일이 붙을 수 있었다.
        """
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        set_file = html[html.index("function setFile") :]
        set_file = set_file[: set_file.index("\n}\n")]
        assert "__transcribing" in set_file, "작업 중에도 시작 버튼을 다시 켭니다"
        main = html[html.index("async function runOnMainThread") :]
        main = main[: main.index("\n}\n")]
        assert "toMono16k(activeFile)" in main, "작업을 시작한 파일이 아니라 지금 고른 파일을 읽습니다"

    def test_gpu_loss_on_the_main_thread_does_not_reuse_the_dead_model(self):
        """화면에서 직접 처리하다 그래픽 가속이 끊기면, 끊긴 모델을 붙잡고 있어 다시 해도 계속 실패했다."""
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        lost = engine[engine.index("if (device === \"webgpu\" && isDeviceLost(error))") :][:500]
        assert "transcriber = null" in lost and "loadedKey = null" in lost
        assert "새로" in lost, "화면 쪽에서는 이어서 할 수 없으니 새로고침을 안내해야 합니다"


class TestCallRecordingsInTheBrowser:
    """웹판도 '통화녹음_이름' 같은 파일은 두 사람으로 본다(최대 2명).

    웹의 묶기는 기준값으로 멈추는 방식이라, 상한을 2로 두면 한 사람을 여럿으로 쪼개는
    실수를 막으면서 혼잣말 파일은 그대로 한 명으로 남는다.
    """

    def test_file_name_hint(self):
        from .test_engines_and_pipeline import CALL_NAME_CASES

        names = json.dumps([name for name, _ in CALL_NAME_CASES], ensure_ascii=False)
        out = _run_node(f"""
        const {{ speakerLimitFromFileName }} = await import("{(WEB_DIR / 'diarize.js').as_posix()}");
        console.log(JSON.stringify({names}.map((n) => speakerLimitFromFileName(n) ?? null)));
        """)
        assert json.loads(out) == [expected for _, expected in CALL_NAME_CASES], "PC판과 판단이 다릅니다"

    def test_cap_stops_one_person_from_becoming_three(self):
        out = _run_node(f"""
        const {{ clusterByAffinity }} = await import("{(WEB_DIR / 'diarize.js').as_posix()}");
        const unit = (axis, wobble) => {{ const v = new Float32Array(8); v[axis] = 1; v[7] = wobble;
          const n = Math.hypot(...v); return v.map((x) => x / n); }};
        const three = [unit(0, 0), unit(0, 0.1), unit(1, 0), unit(1, 0.1), unit(2, 0), unit(2, 0.1)];
        const one = [unit(0, 0), unit(0, 0.05), unit(0, 0.1)];
        const count = (labels) => new Set(labels).size;
        console.log(JSON.stringify([count(clusterByAffinity(three)), count(clusterByAffinity(three, undefined, 2)),
                                    count(clusterByAffinity(one, undefined, 2))]));
        """)
        assert json.loads(out) == [3, 2, 1]

    def test_the_hint_reaches_the_clustering(self):
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        diarize = (WEB_DIR / "diarize.js").read_text(encoding="utf-8")
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "fileName: activeFile.name" in html
        assert "speakerLimitFromFileName(request.fileName)" in engine
        assert "options.maxSpeakers" in diarize[diarize.index("export async function assignSpeakers") :]

    def test_user_can_say_how_many_people_spoke(self):
        """누나와 둘이 한 통화가 화자 6명으로 나왔다(2026-10-07 제보). 인원을 고를 수 있어야 한다."""
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        select = html[html.index('<select id="speakers"') : html.index("</select>", html.index('<select id="speakers"'))]
        assert '<option value="" selected>자동</option>' in select
        for n in range(2, 7):
            assert f'value="{n}"' in select
        assert 'maxSpeakers: Number($("speakers").value) || null' in html
        set_file = html[html.index("function setFile") :]
        set_file = set_file[: set_file.index("\n}\n")]
        assert "speakerLimitFromFileName" in set_file, "통화 녹음 파일을 골라도 2명으로 맞춰 주지 않습니다"
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        assert "request.maxSpeakers || speakerLimitFromFileName(request.fileName)" in engine


class TestRunawayRepeatsInTheBrowser:
    def test_squash(self):
        out = _run_node(f"""
        const {{ squashRepeats }} = await import("{(WEB_DIR / 'engine.js').as_posix()}");
        console.log(JSON.stringify([squashRepeats("아 ".repeat(116) + "근데 그건"), squashRepeats("네 네 네 알겠습니다"), squashRepeats("")]));
        """)
        assert json.loads(out) == ["아 아 근데 그건", "네 네 네 알겠습니다", ""]
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        loop = engine[engine.index("for (let i = startIndex; i < windows.length; i++)") :][:2500]
        assert "squashRepeats(" in loop, "받아쓴 구간마다 정리하지 않습니다"

    def test_same_cases_as_the_pc_version(self):
        from .test_engines_and_pipeline import SQUASH_CASES

        texts = json.dumps([text for text, _ in SQUASH_CASES], ensure_ascii=False)
        out = _run_node(f"""
        const {{ squashRepeats }} = await import("{(WEB_DIR / 'engine.js').as_posix()}");
        console.log(JSON.stringify({texts}.map((t) => squashRepeats(t))));
        """)
        assert json.loads(out) == [expected for _, expected in SQUASH_CASES], "PC판과 정리 결과가 다릅니다"


class TestRepeatedSentencesAreSquashed:
    """문장(구간) 단위로 같은 말을 되풀이하는 것도 정리한다(2026-10-09, 실제 브라우저로 확인).

    알아듣기 어려운 영어 녹음에서 "No, what?" 가 구간 12개로, "Welcome to the start." / "Okay." 가 번갈아
    세 번 나왔다. 한 문장 안의 되풀이(squashRepeats)로는 잡히지 않는다.
    두 낱말 이상 문장은 세 번부터 한 번만, 한 낱말("네.")은 여섯 번부터 두 번만, 여러 문장 묶음은 세 번부터 한 번만.
    """

    def _squash(self, texts: list[str]) -> list[str]:
        chunks = json.dumps([{"start": i, "end": i + 1, "text": t} for i, t in enumerate(texts)], ensure_ascii=False)
        out = _run_node(f"""
        const {{ squashRepeatedChunks }} = await import("{(WEB_DIR / 'engine.js').as_posix()}");
        console.log(JSON.stringify(squashRepeatedChunks({chunks}).map((c) => c.text)));
        """)
        return json.loads(out)

    def test_same_sentence_many_times(self):
        assert self._squash(["Is there a guitar?"] + ["No, what?"] * 12 + ["Stop!"]) == [
            "Is there a guitar?", "No, what?", "Stop!",
        ]

    def test_alternating_pair(self):
        texts = ["Welcome to the start.", "Sorry.", "Okay.", "Welcome to the start.", "Okay.",
                 "Welcome to the start.", "Okay.", "Welcome to the start.", "Is there a guitar?"]
        assert self._squash(texts) == ["Welcome to the start.", "Sorry.", "Okay.", "Welcome to the start.",
                                       "Is there a guitar?"]

    def test_real_conversation_stays(self):
        for texts in (["네.", "네.", "네."], ["여보세요?", "여보세요?"],
                      ["알겠습니다 감사합니다.", "알겠습니다 감사합니다."], ["어.", "응.", "어.", "응."]):
            assert self._squash(texts) == texts

    def test_one_word_runaway(self):
        assert self._squash(["네."] * 7 + ["알겠습니다."]) == ["네.", "네.", "알겠습니다."]

    def test_applied_to_the_whole_result(self):
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        run = engine[engine.index("async function runWhisper") :]
        run = run[: run.index("\n}\n")]
        assert "squashRepeatedChunks(collected)" in run, "받아쓴 결과 전체에 적용하지 않습니다"


class TestUpdatesReachTheWorker:
    """작업자가 engine.js 를 버전 표시 없이 불러 와, 고친 뒤에도 브라우저가 옛 파일을 쓰고 있었다(2026-10-09).

    화면은 worker.js?v=버전 으로 띄우지만 그 안의 `import "./engine.js"` 에는 버전이 붙지 않는다.
    실제로 고친 코드를 브라우저로 돌렸는데 옛 결과가 나와서 알았다. GitHub Pages 는 10분까지 옛 파일을 준다.
    """

    def test_worker_loads_the_engine_with_the_version(self):
        worker = (WEB_DIR / "worker.js").read_text(encoding="utf-8")
        assert not re.search(r'^import .* from "\./engine\.js";', worker, re.M), "버전 없이 불러옵니다"
        assert "import.meta.url" in worker and "engine.js?v=" in worker

    def test_worker_still_falls_back_when_the_engine_fails_to_load(self):
        """불러오기에 실패하면 예전처럼 작업자 오류가 나야 화면 쪽이 직접 처리한다(index.html 의 error 처리)."""
        worker = (WEB_DIR / "worker.js").read_text(encoding="utf-8")
        assert "throw error" in worker[worker.index(".catch(") : worker.index(".catch(") + 200]

    def test_engine_loads_diarize_with_the_same_version(self):
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        assert 'await import("./diarize.js")' not in engine
        assert "./diarize.js${" in engine and "import.meta.url" in engine


class TestAutoLanguageReallyDetects:
    """'자동 감지' 가 실제로는 언어를 고르지 않고 영어로 받아쓰고 있었다(2026-10-09, 사용자 통화 녹음으로 확인).

    transformers.js 는 언어를 안 주면 경고만 남기고 영어로 정한다(modeling_whisper.js 'defaulting to English').
    그래서 한국어 통화가 영어 번역으로 나왔고, 같은 말을 되풀이했다('It's a stormy wind. It's a stormy wind.').
    Whisper 가 원래 하는 방식대로 첫 말소리를 듣고 언어 기호 중 가장 그럴듯한 것을 고른다.
    """

    def _detect(self, scores: dict[str, float], dtype: str = "float32") -> str:
        out = _run_node(f"""
        const {{ detectLanguage }} = await import("{(WEB_DIR / 'engine.js').as_posix()}");
        const ids = {{ "<|en|>": 1, "<|ko|>": 2, "<|ja|>": 3, "<|jw|>": 4, "<|zh|>": 5 }};
        const scores = {json.dumps(scores)};
        const logits = new Float32Array(8).fill(-100);
        for (const [code, value] of Object.entries(scores)) logits[ids[`<|${{code}}|>`]] = value;
        const seen = {{}};
        class Tensor {{ constructor(type, data, dims) {{ Object.assign(this, {{ type, data, dims }}); }} }}
        const asr = {{
          processor: async (audio) => {{ seen.audio = audio.length; return {{ input_features: "특징" }}; }},
          model: Object.assign(async (inputs) => {{
            seen.start = Number(inputs.decoder_input_ids.data[0]);
            const tensor = {{ type: "{dtype}", data: logits, to: () => ({{ type: "float32", data: logits }}) }};
            if ("{dtype}" !== "float32") tensor.data = null;
            return {{ logits: tensor }};
          }}, {{ generation_config: {{ decoder_start_token_id: 7, lang_to_id: ids }} }}),
        }};
        const language = await detectLanguage(asr, new Float32Array(16000), {{ Tensor }});
        console.log(JSON.stringify({{ language, seen }}));
        """)
        return json.loads(out)

    def test_picks_the_most_likely_language(self):
        result = self._detect({"en": 1.0, "ko": 3.0, "ja": 2.0})
        assert result["language"] == "ko"
        assert result["seen"]["start"] == 7, "시작 기호 다음에 올 말을 봐야 합니다"

    def test_only_languages_the_site_offers(self):
        """화면에 없는 언어(자바어 jw 등)로 잘못 고르면 엉뚱한 글이 나온다. 고를 수 있는 언어 중에서만 고른다."""
        assert self._detect({"jw": 9.0, "ko": 3.0, "en": 1.0})["language"] == "ko"

    def test_half_precision_scores(self):
        """그래픽 가속(q4f16)에서는 점수가 16비트로 나온다. 그대로 비교하면 틀린다."""
        assert self._detect({"en": 1.0, "ko": 3.0}, dtype="float16")["language"] == "ko"

    def test_offered_languages_match_the_screen(self):
        out = _run_node(f"""
        const {{ AUTO_LANGUAGES }} = await import("{(WEB_DIR / 'engine.js').as_posix()}");
        console.log(JSON.stringify(AUTO_LANGUAGES));
        """)
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        select = html[html.index('<select id="lang">') : html.index("</select>", html.index('<select id="lang">'))]
        offered = [v for v in re.findall(r'value="([a-z]+)"', select) if v != "auto"]
        assert sorted(json.loads(out)) == sorted(offered)

    def test_never_hands_null_language_to_the_library(self):
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        run = engine[engine.index("async function runWhisper") :]
        run = run[: run.index("\n}\n")]
        assert 'language === "auto" ? null' not in run, "자동이면 라이브러리가 영어로 정해 버립니다"
        assert "detectLanguage(" in run
        assert 'catch' in run[run.index("detectLanguage(") - 200 : run.index("detectLanguage(") + 400], (
            "언어를 못 알아내도 받아쓰기는 해야 합니다"
        )
        assert "language: chosen" in run[run.index("lost.resume") - 10 : run.index("lost.resume") + 120], (
            "그래픽 가속이 끊겨 이어서 할 때 언어를 다시 바꾸면 안 됩니다"
        )

    def test_shows_the_detected_language(self):
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        done = engine[engine.index('type: "done"') :][:400]
        assert "language" in done, "감지한 언어를 화면에 넘기지 않습니다"
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        show = html[html.index("function showResult") :][:2200]
        assert "d.language" in show, "감지한 언어를 보여 주지 않습니다"


#: 한국어 화자 구분 모델. 저장소(=웹사이트)에 같이 올려 같은 주소에서 받는다.
KOREAN_MODEL_DIR = WEB_DIR.parent.parent.parent / "speaker-model" / "eres2net-base"


def _fake_speaker_lib(fail_local: bool) -> str:
    """진짜 모델 대신 쓰는 가짜 라이브러리. 소리가 +면 A, -면 B 의 지문을 돌려준다."""
    return f"""
    const loaded = [];
    const env = {{ allowLocalModels: false }};
    const unit = (axis) => {{ const v = new Float32Array(4); v[axis] = 1; return v; }};
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const lib = {{
      env,
      // 진짜 라이브러리처럼 파일을 받는 데 시간이 걸린다. 그 사이의 설정 값을 적어 둔다.
      AutoProcessor: {{ from_pretrained: async (id) => {{ await sleep(0);
        loaded.push("processor " + id + " local=" + env.allowLocalModels);
        return async (audio) => ({{ input_features: audio }}); }} }},
      AutoModel: {{ from_pretrained: async (id) => {{
        loaded.push("model " + id + " local=" + env.allowLocalModels);
        await sleep(10);
        if ({str(fail_local).lower()} && !id.includes("wespeaker")) throw new Error("404");
        return async (inputs) => ({{ embeddings: {{ data: unit(inputs.input_features[0] > 0 ? 0 : 1) }} }}); }} }},
    }};
    const second = (sign) => new Float32Array(16000).fill(0.5 * sign);
    const audio = new Float32Array([...second(1), ...second(-1), ...second(1), ...second(-1)]);
    const segments = (text) => [0, 1, 2, 3].map((t) => ({{ start: t, end: t + 1, text }}));
    """


class TestKoreanSpeakerModel:
    """한국어 녹음은 ERes2Net, 그 밖은 지금 모델(WeSpeaker)로 화자를 가른다(2026-10-09 사용자 결정).

    웹과 같은 방식으로 재 보니 한국어 2인 대화(원음)는 ERes2Net 이 74% → 82% 로 나았지만
    영어 회의는 76% → 66~69% 로 나빴다. 한국어와 영어를 둘 다 많이 쓰므로 둘을 나눠 쓴다.
    """

    def test_picks_the_model_by_language(self):
        out = _run_node(f"""
        const {{ speakerModelFor }} = await import("{(WEB_DIR / 'diarize.js').as_posix()}");
        const seg = (text) => [{{ text }}];
        console.log(JSON.stringify([
          speakerModelFor("ko", seg("hello")),
          speakerModelFor("en", seg("안녕하세요")),
          speakerModelFor("ja", seg("こんにちは")),
          speakerModelFor("auto", seg("오늘 회의는 여기까지 하겠습니다")),
          speakerModelFor("auto", seg("Let's wrap up the meeting here, 감사")),
          speakerModelFor(null, seg("그래서 API 를 바꿨어요")),
          speakerModelFor("auto", seg("")),
        ]));
        """)
        assert json.loads(out) == ["korean", "default", "default", "korean", "default", "korean", "default"]

    def test_korean_recording_uses_eres2net_from_the_same_site(self):
        out = _run_node(_fake_speaker_lib(fail_local=False) + f"""
        const {{ assignSpeakers }} = await import("{(WEB_DIR / 'diarize.js').as_posix()}");
        const labels = await assignSpeakers(audio, segments("네 맞아요"), 16000, {{ transformers: lib, language: "ko" }});
        console.log(JSON.stringify({{ labels, loaded, localAfter: env.allowLocalModels }}));
        """)
        result = json.loads(out)
        assert result["labels"] == ["화자1", "화자2", "화자1", "화자2"]
        models = [x for x in result["loaded"] if x.startswith("model ")]
        assert len(models) == 1 and models[0].endswith("/speaker-model/eres2net-base local=true"), result["loaded"]
        assert result["localAfter"] is False, "다른 모델까지 이 사이트에서 찾게 됩니다. 설정을 되돌려야 합니다"
        processors = [x for x in result["loaded"] if x.startswith("processor ")]
        assert processors and all(x.endswith("local=false") for x in processors), (
            "전처리 설정을 받는 동안 '이 사이트에서 찾기' 가 켜져 있었습니다: " + str(processors)
        )

    def test_falls_back_to_wespeaker_when_the_korean_model_fails(self):
        out = _run_node(_fake_speaker_lib(fail_local=True) + f"""
        const {{ assignSpeakers }} = await import("{(WEB_DIR / 'diarize.js').as_posix()}");
        const labels = await assignSpeakers(audio, segments("네 맞아요"), 16000, {{ transformers: lib, language: "ko" }});
        console.log(JSON.stringify({{ labels, loaded, localAfter: env.allowLocalModels }}));
        """)
        result = json.loads(out)
        assert result["labels"] == ["화자1", "화자2", "화자1", "화자2"], "한국어 모델을 못 받으면 화자 구분이 통째로 빠집니다"
        assert any("wespeaker" in x for x in result["loaded"] if x.startswith("model "))
        assert result["localAfter"] is False

    def test_english_recording_keeps_wespeaker(self):
        out = _run_node(_fake_speaker_lib(fail_local=False) + f"""
        const {{ assignSpeakers }} = await import("{(WEB_DIR / 'diarize.js').as_posix()}");
        await assignSpeakers(audio, segments("sounds good"), 16000, {{ transformers: lib, language: "auto" }});
        console.log(JSON.stringify(loaded.filter((x) => x.startsWith("model "))));
        """)
        models = json.loads(out)
        assert len(models) == 1 and "wespeaker" in models[0], models

    def test_korean_thresholds_depend_on_whether_the_headcount_is_known(self):
        """ERes2Net 은 인원이 '자동' 일 때와 정해졌을 때 기준값이 다르다(2026-10-09 측정).

        한국어 평가셋 19개(실제 2인 대화 12·3~4인 회의 4·혼자 3, 웹과 같은 Whisper small 구간)에서
          자동     0.20~0.24 → 인원 정확 10~11/19, 0.26 부터 6/19 로 쪼개짐 → 가운데 0.22
          인원 지정 0.35      → 18/19, 2인 대화 12/12 (0.25 는 전화 음질 하나를 1명으로 합침)
        같은 자리에서 지금 웹(WeSpeaker 0.35) 자동은 3/19 였다 — 2인 통화를 4~8명으로 쪼갰다.
        """
        out = _run_node(f"""
        const {{ SPEAKER_MODELS }} = await import("{(WEB_DIR / 'diarize.js').as_posix()}");
        console.log(JSON.stringify(SPEAKER_MODELS));
        """)
        models = json.loads(out)
        assert models["korean"]["sameSpeaker"] == 0.22
        assert models["korean"]["sameSpeakerWithLimit"] == 0.35
        assert models["default"]["sameSpeaker"] == models["default"]["sameSpeakerWithLimit"] == 0.35, (
            "영어 등은 바꾸지 않기로 했습니다"
        )

    def test_known_headcount_uses_its_own_threshold(self):
        """두 사람 목소리 유사도가 0.3 이면: 자동(0.22)은 한 명으로 합치고, 2명으로 정하면(0.35) 둘로 남긴다."""
        lib = _fake_speaker_lib(fail_local=False).replace(
            "unit(inputs.input_features[0] > 0 ? 0 : 1)",
            "(inputs.input_features[0] > 0 ? unit(0) : Float32Array.from([0.3, Math.sqrt(1 - 0.09), 0, 0]))",
        )
        out = _run_node(lib + f"""
        const {{ assignSpeakers }} = await import("{(WEB_DIR / 'diarize.js').as_posix()}");
        const auto = await assignSpeakers(audio, segments("네 맞아요"), 16000, {{ transformers: lib, language: "ko" }});
        const two = await assignSpeakers(audio, segments("네 맞아요"), 16000, {{ transformers: lib, language: "ko", maxSpeakers: 2 }});
        console.log(JSON.stringify([new Set(auto).size, new Set(two).size]));
        """)
        assert json.loads(out) == [1, 2]

    def test_engine_passes_the_language(self):
        engine = (WEB_DIR / "engine.js").read_text(encoding="utf-8")
        call = engine[engine.index("await assignSpeakers(") :][:900]
        assert "language: result.language || request.language" in call, "'자동' 이면 알아낸 언어를 넘겨야 합니다"

    def test_model_files_are_in_the_repository(self):
        """GitHub 릴리스 파일은 브라우저가 직접 못 받는다(CORS). 그래서 사이트에 같이 올린다."""
        config = json.loads((KOREAN_MODEL_DIR / "config.json").read_text(encoding="utf-8"))
        assert config["model_type"] == "wespeaker-resnet", "transformers.js 가 읽는 형식이어야 합니다"
        onnx = KOREAN_MODEL_DIR / "onnx" / "model.onnx"
        head = onnx.read_bytes()[:200]
        assert not head.startswith(b"version https://git-lfs"), "GitHub Pages 는 LFS 파일을 내주지 않습니다"
        size = onnx.stat().st_size
        assert 30_000_000 < size < 95_000_000, f"{size} 바이트 — GitHub 는 100MB 넘는 파일을 거부합니다"
        assert b"input_features" in onnx.read_bytes()[-200_000:] or b"input_features" in onnx.read_bytes()[:200_000]
        notice = (KOREAN_MODEL_DIR / "README.md").read_text(encoding="utf-8")
        assert "Apache" in notice and "eres2net" in notice.lower(), "출처와 라이선스를 밝혀야 합니다"


class TestQuietSpeakersAreMerged:
    """인원 '자동' 에서 말을 아주 적게 한 무리는 가장 닮은 사람에게 합친다(2026-10-09).

    짧은 말 한두 마디가 따로 무리가 되어 2인 대화가 8명으로 나오곤 했다. 영어 회의 녹음(AMI) 24개에서
    WeSpeaker 0.35 그대로는 인원 정확 0/24(평균 4.8명 어긋남). 전체 말한 시간의 10% 미만 무리를 합치니
    9/24(0.8명), 한국어(ERes2Net) 19개는 10/19 → 12/19. 기준값을 바꾸는 것만으로는 영어가 나아지지 않았다
    (목소리 모델이 이 녹음을 잘 못 가르기 때문). 인원을 정했으면 그 사람 수를 믿고 합치지 않는다.
    """

    def test_merge_quiet_speakers(self):
        out = _run_node(f"""
        const {{ mergeQuietSpeakers }} = await import("{(WEB_DIR / 'diarize.js').as_posix()}");
        const unit = (...xs) => {{ const v = Float32Array.from(xs); const n = Math.hypot(...v); return v.map((x) => x / n); }};
        const a = unit(1, 0, 0), b = unit(0, 1, 0), nearB = unit(0.2, 1, 0.3);
        const vectors = [a, a, a, a, a, b, b, b, b, b, nearB];
        const labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2];
        const seconds = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 2];   // 세 번째 '사람' 은 2초(2%)뿐
        console.log(JSON.stringify([
          mergeQuietSpeakers(vectors, labels, seconds, 0.1),
          mergeQuietSpeakers(vectors, labels, seconds, 0),
          mergeQuietSpeakers([a, b], [0, 1], [9, 0.5], 0.1),
          mergeQuietSpeakers([a, a], [0, 0], [5, 5], 0.1),
        ]));
        """)
        merged, untouched, two_people, one_person = json.loads(out)
        assert merged == [0] * 5 + [1] * 6, "말을 적게 한 무리가 가장 닮은 사람에게 가야 합니다"
        assert untouched == [0] * 5 + [1] * 5 + [2]
        assert two_people == [0, 0], "10% 미만이면 두 명이라도 합칩니다"
        assert one_person == [0, 0]

    def test_only_when_the_headcount_is_automatic(self):
        lib = _fake_speaker_lib(fail_local=False).replace(
            "const audio = new Float32Array([...second(1), ...second(-1), ...second(1), ...second(-1)]);",
            "const audio = new Float32Array([...Array.from({ length: 19 }, () => [...second(1)]).flat(), ...second(-1)]);",
        ).replace(
            "const segments = (text) => [0, 1, 2, 3].map((t) => ({ start: t, end: t + 1, text }));",
            "const segments = (text) => [{ start: 0, end: 19, text }, { start: 19, end: 20, text }];",
        )
        assert "end: 19" in lib and "length: 19" in lib, "가짜 녹음을 바꾸지 못했습니다"
        out = _run_node(lib + f"""
        const {{ assignSpeakers }} = await import("{(WEB_DIR / 'diarize.js').as_posix()}");
        const auto = await assignSpeakers(audio, segments("hello"), 16000, {{ transformers: lib, language: "en" }});
        const two = await assignSpeakers(audio, segments("hello"), 16000, {{ transformers: lib, language: "en", maxSpeakers: 2 }});
        console.log(JSON.stringify([auto, two]));
        """)
        auto, two = json.loads(out)
        assert auto == ["화자1", "화자1"], "1초(5%)만 말한 무리가 따로 남았습니다"
        assert two == ["화자1", "화자2"], "2명으로 정했는데 합쳐 버렸습니다"
