"""휴대폰 접속 기능 테스트."""

from __future__ import annotations

import ipaddress

import pytest

from voicescribe.web.lan import _is_private, access_notice, detect_lan_ip, render_qr


class TestPrivateAddress:
    @pytest.mark.parametrize(
        "address", ["192.168.0.10", "10.0.0.5", "172.16.3.4", "192.0.2.2"]
    )
    def test_accepts_lan_addresses(self, address):
        assert _is_private(address)

    @pytest.mark.parametrize("address", ["127.0.0.1", "8.8.8.8", "169.254.1.1", "그냥문자열"])
    def test_rejects_others(self, address):
        assert not _is_private(address)


class TestDetectLanIp:
    def test_returns_none_or_a_private_address(self):
        found = detect_lan_ip()
        if found is not None:
            assert _is_private(found)
            ipaddress.ip_address(found)  # 형식이 올바른지


class TestAccessNotice:
    def test_localhost_mode_hints_at_lan_option(self):
        text = access_notice("127.0.0.1", 7860, False)
        assert "--lan" in text

    def test_lan_mode_shows_both_addresses(self):
        text = access_notice("0.0.0.0", 7860, False)
        assert "127.0.0.1:7860" in text
        assert "휴대폰" in text

    def test_http_lan_mode_warns_about_microphone(self):
        """http 로는 휴대폰 마이크를 못 쓴다는 걸 반드시 알려야 한다."""
        text = access_notice("0.0.0.0", 7860, False)
        assert "--https" in text
        assert "마이크" in text

    def test_https_mode_does_not_warn(self):
        text = access_notice("0.0.0.0", 7860, True)
        assert "https://" in text
        assert "--https 옵션이 필요합니다" not in text


class TestQr:
    def test_returns_text_or_none(self):
        result = render_qr("https://192.168.0.10:7860")
        if result is not None:
            assert len(result.splitlines()) > 10  # QR 답게 여러 줄이어야 한다


class TestCertificate:
    def test_includes_every_requested_host(self, tmp_path, monkeypatch):
        pytest.importorskip("cryptography", reason="[lan] 옵션 패키지입니다")
        from cryptography import x509

        from voicescribe.web import lan

        monkeypatch.setattr(lan, "CERT_DIR", tmp_path)
        monkeypatch.setattr(lan, "CERT_FILE", tmp_path / "c.pem")
        monkeypatch.setattr(lan, "KEY_FILE", tmp_path / "k.pem")

        cert_file, key_file = lan.ensure_self_signed_cert(["127.0.0.1", "192.168.0.10"])
        assert cert_file.exists() and key_file.exists()

        certificate = x509.load_pem_x509_certificate(cert_file.read_bytes())
        names = [
            str(entry.value)
            for entry in certificate.extensions.get_extension_for_class(
                x509.SubjectAlternativeName
            ).value
        ]
        assert names == ["localhost", "127.0.0.1", "192.168.0.10"]  # 중복 없이 순서대로

    def test_reuses_an_existing_certificate(self, tmp_path, monkeypatch):
        pytest.importorskip("cryptography", reason="[lan] 옵션 패키지입니다")
        from voicescribe.web import lan

        monkeypatch.setattr(lan, "CERT_DIR", tmp_path)
        monkeypatch.setattr(lan, "CERT_FILE", tmp_path / "c.pem")
        monkeypatch.setattr(lan, "KEY_FILE", tmp_path / "k.pem")

        first, _ = lan.ensure_self_signed_cert(["127.0.0.1"])
        content = first.read_bytes()
        second, _ = lan.ensure_self_signed_cert(["127.0.0.1"])
        assert second.read_bytes() == content  # 매번 새로 만들면 안 된다


class TestCliWiring:
    def test_lan_flag_binds_all_interfaces(self):
        from voicescribe.cli import build_parser

        args = build_parser().parse_args(["web", "--lan"])
        assert args.lan is True
        assert args.https is False

    def test_explicit_host_is_not_overridden(self, monkeypatch):
        from voicescribe import cli

        captured = {}

        def fake_serve(**kwargs):
            captured.update(kwargs)
            return 0

        monkeypatch.setattr("voicescribe.web.server.serve", fake_serve)
        args = cli.build_parser().parse_args(["web", "--lan", "--host", "10.1.2.3"])
        cli.cmd_web(args)
        assert captured["host"] == "10.1.2.3"  # 직접 지정한 host 가 우선이다

    def test_lan_flag_sets_all_interfaces(self, monkeypatch):
        from voicescribe import cli

        captured = {}
        monkeypatch.setattr(
            "voicescribe.web.server.serve", lambda **kw: captured.update(kw) or 0
        )
        cli.cmd_web(cli.build_parser().parse_args(["web", "--lan", "--https"]))
        assert captured["host"] == "0.0.0.0"
        assert captured["use_https"] is True
