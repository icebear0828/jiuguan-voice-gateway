"""
Unit tests for gateway server API (server.py)
Tests HTTP endpoints, CORS headers, error degradation, 503 upstream protection, and mock TTS generation.
"""

import json
import threading
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import pytest
from unittest.mock import patch, MagicMock

import server
from server import VoiceGatewayHandler


@pytest.fixture(scope="module")
def gateway_server():
    """Starts an in-process ThreadingHTTPServer on an ephemeral port for testing."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), VoiceGatewayHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    yield base_url
    httpd.shutdown()
    httpd.server_close()


def _request(url: str, method: str = "GET", data: dict = None, headers: dict = None):
    headers = headers or {}
    req_data = json.dumps(data).encode("utf-8") if data is not None else None
    if req_data:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


@pytest.mark.unit
class TestGatewayEndpoints:
    def test_health_check(self, gateway_server):
        status, body, headers = _request(f"{gateway_server}/health")
        assert status == 200
        data = json.loads(body.decode("utf-8"))
        assert data["status"] == "healthy"
        assert data["service"] == "jiuguan-voice-gateway"
        assert headers.get("Access-Control-Allow-Origin") == "*"

    def test_options_cors(self, gateway_server):
        status, body, headers = _request(f"{gateway_server}/api/audio/generate", method="OPTIONS")
        assert status == 204
        assert headers.get("Access-Control-Allow-Origin") == "*"
        assert "POST" in headers.get("Access-Control-Allow-Methods", "")

    def test_get_engines(self, gateway_server):
        status, body, _ = _request(f"{gateway_server}/api/audio/engines")
        assert status == 200
        engines = json.loads(body.decode("utf-8"))
        assert any(e["id"] == "edge-tts" for e in engines)
        assert any(e["id"] == "gpt-sovits" for e in engines)

    def test_get_voices(self, gateway_server):
        status, body, _ = _request(f"{gateway_server}/api/audio/voices")
        assert status == 200
        voices = json.loads(body.decode("utf-8"))
        assert isinstance(voices, list)

    def test_generate_missing_input_400(self, gateway_server):
        status, body, _ = _request(
            f"{gateway_server}/api/audio/generate",
            method="POST",
            data={"engine": "edge-tts"}  # missing "text" / "input"
        )
        assert status == 400
        resp = json.loads(body.decode("utf-8"))
        assert "error" in resp
        assert "Missing input text" in resp["error"]

    def test_generate_edge_tts_route(self, gateway_server):
        fake_audio = b"FAKE_MP3_STREAM_CONTENT"

        class DummyComm:
            def __init__(self, **kwargs):
                pass

            async def stream(self):
                yield {"type": "audio", "data": fake_audio}

        with patch("edge_tts.Communicate", side_effect=DummyComm):
            status, body, headers = _request(
                f"{gateway_server}/api/audio/generate",
                method="POST",
                data={
                    "text": "测试解说音色",
                    "engine": "edge-tts",
                    "voice": "zh-CN-YunxiNeural"
                }
            )
            assert status == 200
            assert body == fake_audio
            assert "audio/mpeg" in headers.get("Content-Type", "")

    @patch("core.voice_router.VoiceRouter.ensure_character_weights", return_value=True)
    def test_generate_gpt_sovits_upstream_503_protection(self, mock_weights, gateway_server):
        # Point GPT_SOVITS_URL to an unreachable port to test 503 timeout/protection
        with patch.object(server, "GPT_SOVITS_URL", "http://127.0.0.1:59999/tts"):
            status, body, _ = _request(
                f"{gateway_server}/api/audio/generate",
                method="POST",
                data={
                    "text": "测试上游离线",
                    "voice_mode": "rimuru"
                }
            )
            assert status == 503
            resp = json.loads(body.decode("utf-8"))
            assert "Cannot connect to GPT-SoVITS backend" in resp.get("error", "")

    @patch("core.voice_router.VoiceRouter.ensure_character_weights", return_value=True)
    def test_generate_gpt_sovits_success_forwarding(self, mock_weights, gateway_server):
        fake_wav = b"RIFF00WAVEfmt 00data00fake_pcm_content"

        class MockUpstreamHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(fake_wav)))
                self.end_headers()
                self.wfile.write(fake_wav)

            def log_message(self, format, *args):
                pass

        mock_upstream = ThreadingHTTPServer(("127.0.0.1", 0), MockUpstreamHandler)
        upstream_port = mock_upstream.server_address[1]
        t = threading.Thread(target=mock_upstream.serve_forever, daemon=True)
        t.start()

        try:
            with patch.object(server, "GPT_SOVITS_URL", f"http://127.0.0.1:{upstream_port}/tts"):
                status, body, headers = _request(
                    f"{gateway_server}/api/audio/generate",
                    method="POST",
                    data={
                        "text": "こんにちは！",
                        "voice_mode": "rimuru",
                        "language": "ja"
                    }
                )
                assert status == 200
                assert body == fake_wav
                assert "audio/wav" in headers.get("Content-Type", "")
        finally:
            mock_upstream.shutdown()
            mock_upstream.server_close()
