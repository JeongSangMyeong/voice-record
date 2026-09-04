"""휴대폰에서 접속할 수 있게 도와주는 기능.

내 PC 를 서버로 쓰고 휴대폰이 같은 와이파이로 접속하는 방식이다.
녹음 파일은 내 PC 안에만 있고 외부 서비스로 나가지 않는다.

한 가지 주의할 점이 있다. 브라우저는 **보안 연결(https)에서만 마이크를 허용**한다.
따라서
  * 파일 올리기  → 그냥 http 로도 잘 된다
  * 마이크 녹음  → https 가 필요하다 (``--https`` 옵션으로 인증서를 만들어 준다)
"""

from __future__ import annotations

import contextlib
import ipaddress
import socket
from pathlib import Path

#: 자체 서명 인증서를 보관할 위치.
CERT_DIR = Path.home() / ".cache" / "voicescribe" / "certs"
CERT_FILE = CERT_DIR / "voicescribe-cert.pem"
KEY_FILE = CERT_DIR / "voicescribe-key.pem"


def detect_lan_ip() -> str | None:
    """이 컴퓨터가 같은 와이파이 안에서 갖는 주소를 찾는다.

    바깥으로 패킷을 보내지는 않는다. 어떤 랜카드를 쓰게 되는지만 물어본다.
    """
    for probe in ("8.8.8.8", "1.1.1.1"):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect((probe, 80))
            candidate = sock.getsockname()[0]
        except OSError:
            continue
        finally:
            sock.close()
        if _is_private(candidate):
            return candidate

    # 위 방법이 안 되면 호스트 이름으로 찾아본다.
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            candidate = info[4][0]
            if _is_private(candidate):
                return candidate
    except OSError:
        pass
    return None


def _is_private(address: str) -> bool:
    """집·회사 와이파이에서 쓰는 사설 주소인지 확인한다(127.x 는 제외)."""
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    return parsed.is_private and not parsed.is_loopback and not parsed.is_link_local


def ensure_self_signed_cert(host_names: list[str]) -> tuple[Path, Path]:
    """자체 서명 인증서를 만든다(이미 있으면 그대로 쓴다).

    휴대폰 브라우저에서 마이크를 쓰려면 https 가 필요한데, 집 안에서 쓰는
    서버라 정식 인증서를 받을 수 없다. 그래서 직접 만든 인증서를 쓴다.
    브라우저가 "안전하지 않음" 경고를 띄우지만, 내 PC 가 만든 것이므로
    한 번 통과시키면 된다.

    Raises:
        RuntimeError: cryptography 패키지가 없을 때(설치 방법을 담아 던진다).
    """
    if CERT_FILE.exists() and KEY_FILE.exists():
        return CERT_FILE, KEY_FILE

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError as exc:
        raise RuntimeError(
            "https 를 쓰려면 cryptography 패키지가 필요합니다.\n"
            '  설치: pip install "voicescribe[lan]"  (또는 pip install cryptography)\n'
            "  * 파일 올리기만 할 거라면 https 없이 그냥 쓰셔도 됩니다."
        ) from exc

    import datetime

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "VoiceScribe Local"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "VoiceScribe"),
    ])

    alt_names: list[x509.GeneralName] = []
    for name in dict.fromkeys(["localhost", *host_names]):  # 순서를 지키며 중복 제거
        try:
            alt_names.append(x509.IPAddress(ipaddress.ip_address(name)))
        except ValueError:
            alt_names.append(x509.DNSName(name))

    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=825))  # 브라우저가 받아주는 최대치
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    KEY_FILE.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    CERT_FILE.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    # 개인키는 나만 읽을 수 있게 한다(윈도우에서는 무시된다).
    with contextlib.suppress(OSError):
        KEY_FILE.chmod(0o600)
    return CERT_FILE, KEY_FILE


def render_qr(url: str) -> str | None:
    """터미널에 표시할 QR 코드를 만든다. 만들 수 없으면 None."""
    try:
        import segno
    except ImportError:
        return None
    import io

    buffer = io.StringIO()
    segno.make(url, error="m").terminal(out=buffer, border=1)
    return buffer.getvalue()


def access_notice(host: str, port: int, use_https: bool) -> str:
    """실행할 때 화면에 보여 줄 접속 안내문을 만든다."""
    scheme = "https" if use_https else "http"
    lines: list[str] = []

    if host in ("0.0.0.0", "::"):
        lan_ip = detect_lan_ip()
        lines.append(f"  이 컴퓨터에서 : {scheme}://127.0.0.1:{port}")
        if lan_ip:
            phone_url = f"{scheme}://{lan_ip}:{port}"
            lines.append(f"  휴대폰에서    : {phone_url}")
            lines.append("")
            lines.append("  휴대폰을 이 컴퓨터와 같은 와이파이에 연결한 뒤 위 주소를 여세요.")
            qr = render_qr(phone_url)
            if qr:
                lines.append("")
                lines.append("  휴대폰 카메라로 아래 QR 을 찍어도 됩니다:")
                lines.append("")
                lines.append(qr)
            else:
                lines.append('  (QR 로 열고 싶으면: pip install "voicescribe[lan]")')
        else:
            lines.append("  휴대폰에서    : 이 컴퓨터의 IP 주소를 찾지 못했습니다.")
            lines.append("                  와이파이에 연결되어 있는지 확인해 주세요.")
    else:
        lines.append(f"  접속 주소 : {scheme}://{host}:{port}")
        if host in ("127.0.0.1", "localhost"):
            lines.append("")
            lines.append("  휴대폰에서도 쓰려면 --lan 옵션을 붙여 실행하세요.")

    if not use_https and host in ("0.0.0.0", "::"):
        lines.append("")
        lines.append("  ※ 휴대폰에서 '마이크로 녹음' 을 쓰려면 --https 옵션이 필요합니다.")
        lines.append("     (브라우저가 보안 연결에서만 마이크를 허용합니다)")
        lines.append("     파일 올리기는 지금 이대로도 잘 됩니다.")

    return "\n".join(lines)
