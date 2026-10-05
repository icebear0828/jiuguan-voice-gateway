#!/usr/bin/env python3
"""
jiuguan-voice-gateway: Decoupled Multi-Engine Voice Gateway Server
Supports Edge-TTS (Xiaoshuai/Yunxi, Xiaoxiao, etc.) & GPT-SoVITS (Mika, Kurisu, Rimuru, Utaha)
"""

import os
import sys
import json
import asyncio
import mimetypes
import http.client
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional

# Ensure core module can be imported
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SERVER_DIR)

from core.voice_router import VoiceRouter

PORT = int(os.environ.get("VOICE_GATEWAY_PORT", 8888))
HOST = os.environ.get("VOICE_GATEWAY_HOST", "0.0.0.0")
ASSETS_DIR = os.path.join(SERVER_DIR, "assets")
AUDIO_DIR = os.path.join(ASSETS_DIR, "audio")
MODELS_DIR = os.path.join(ASSETS_DIR, "models")
REGISTRY_FILE = os.path.join(MODELS_DIR, "registry.json")
STATIC_DIR = os.environ.get("STATIC_DIR", os.path.abspath(os.path.join(SERVER_DIR, "static")))
GPT_SOVITS_URL = os.environ.get("GPT_SOVITS_URL", "http://127.0.0.1:9880/tts")


class VoiceGatewayHandler(BaseHTTPRequestHandler):
    timeout = 60  # Socket timeout in seconds to prevent hung clients from blocking

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_HEAD(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path.startswith("/assets/audio/") or path.startswith("/api/audio/file/"):
            rel_path = path.replace("/assets/audio/", "").replace("/api/audio/file/", "").replace("training_dataset/", "")
            file_path = os.path.abspath(os.path.join(AUDIO_DIR, rel_path))
            if not os.path.isfile(file_path):
                file_path = os.path.abspath(os.path.join(AUDIO_DIR, "training_dataset", rel_path))
            if os.path.isfile(file_path):
                ctype, _ = mimetypes.guess_type(file_path)
                self.send_response(200)
                self.send_header("Content-Type", ctype or "application/octet-stream")
                self.send_header("Content-Length", str(os.path.getsize(file_path)))
                self.send_header("Accept-Ranges", "bytes")
                self._send_cors_headers()
                self.end_headers()
                return
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "healthy",
                "service": "jiuguan-voice-gateway",
                "version": "2.0.0",
                "docs": "/docs/UNIFIED_TTS_API.md"
            }, ensure_ascii=False).encode("utf-8"))

        elif path in ["/", "/index.html", "/studio", "/voices", "/audio"]:
            index_path = os.path.join(STATIC_DIR, "index.html")
            if os.path.isfile(index_path):
                with open(index_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "healthy",
                    "service": "jiuguan-voice-gateway",
                    "version": "2.0.0"
                }, ensure_ascii=False).encode("utf-8"))

        elif path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({
                "version": "2.0.0",
                "status": "IDLE",
                "cases": [],
                "progress": {"total": 0, "completed": 0, "passed": 0, "blocked": 0, "failed": 0}
            }, ensure_ascii=False).encode("utf-8"))

        elif path == "/api/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self._send_cors_headers()
            self.end_headers()

        elif path == "/api/audio/voices":
            voices = []
            try:
                if os.path.exists(REGISTRY_FILE):
                    with open(REGISTRY_FILE, "r", encoding="utf-8") as rf:
                        reg_data = json.load(rf)
                    for cid, cinfo in reg_data.get("characters", {}).items():
                        emotions_list = VoiceRouter.get_available_emotions(cid) if cid not in ["xiaoshuai", "anna"] else []
                        voices.append({
                            "id": cid,
                            "name": cinfo.get("character_name", cid),
                            "source": cinfo.get("character_source", ""),
                            "cv": cinfo.get("cv", ""),
                            "engine": cinfo.get("engine", "gpt-sovits"),
                            "version": cinfo.get("active_version", "v1.0"),
                            "architecture": cinfo.get("architecture", "GPT-SoVITS"),
                            "avatar_symbol": cinfo.get("avatar_symbol", "🎙️"),
                            "supported_languages": cinfo.get("supported_languages", ["ja"]),
                            "primary_language": cinfo.get("primary_language", "ja"),
                            "description": cinfo.get("description", ""),
                            "features": cinfo.get("features", ["live_tts", "preset_demos"]),
                            "emotions": emotions_list,
                            "demos": cinfo.get("demos", [])
                        })
            except Exception as e:
                print(f"Error reading voices: {e}")

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(voices, ensure_ascii=False).encode("utf-8"))

        elif path == "/api/audio/dataset":
            manifest_p = os.path.join(AUDIO_DIR, "training_dataset", "dataset_manifest.json")
            data = []
            if os.path.exists(manifest_p):
                with open(manifest_p, "r", encoding="utf-8") as f:
                    data = json.load(f)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

        elif path == "/api/audio/samples":
            sample_metadata = [
                {
                    "id": "ref02_ab_group",
                    "type": "AB_COMPARISON",
                    "badge": "AI 深度分离对照",
                    "title": "切片 01: 情绪爆发·狂热呼喊 (2.0s~7.0s 纯净原声)",
                    "character": "安娜·锦之宫 (CV: 松来未祐)",
                    "transcript": "愛の力ですわ！狸吉くんへの愛が溢れ出て止まらないのです！",
                    "original_file": "/assets/audio/ref02_original_with_bgm.wav",
                    "vocal_file": "/assets/audio/ref02_vocal_isolated.wav",
                    "bgm_file": "/assets/audio/ref02_bgm_stem.wav",
                    "duration": "5.00s",
                    "source": "第 04 话 18:27~18:32 (精确排除前后杂乱音，100% 松来未祐干声)",
                    "is_ab": True
                },
                {
                    "id": "ref01_ab_group",
                    "type": "AB_COMPARISON",
                    "badge": "AI 深度分离对照",
                    "title": "切片 02: 饼干制作·狂爱耳语 (4.2s~6.8s 纯净耳语)",
                    "character": "安娜·锦之宫 (CV: 松来未祐)",
                    "transcript": "どうにかなってしまいそうです…",
                    "original_file": "/assets/audio/ref01_original_with_bgm.wav",
                    "vocal_file": "/assets/audio/ref01_vocal_isolated.wav",
                    "bgm_file": "/assets/audio/ref01_bgm_stem.wav",
                    "duration": "2.60s",
                    "source": "第 04 话 18:16~18:19 (彻底剔除男主混音，100% 松来未祐独白)",
                    "is_ab": True
                }
            ]
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(sample_metadata, ensure_ascii=False).encode("utf-8"))

        elif path == "/api/audio/engines":
            engines_info = [
                {
                    "id": "edge-tts",
                    "name": "🎬 影视解说 · 小帅 (微软云希)",
                    "type": "commercial_narration",
                    "description": "全网经典影视解说、短视频速通、科普旁白标配音色，零延迟流式直出",
                    "voices": [
                        {"id": "zh-CN-YunxiNeural", "name": "小帅解说 · 影视速通 (云希)", "default_speed": 1.15, "default_pitch": "+0Hz"},
                        {"id": "zh-CN-XiaoxiaoNeural", "name": "温柔知性 · 有声书/客服 (晓晓)", "default_speed": 1.0, "default_pitch": "+0Hz"},
                        {"id": "zh-CN-YunjianNeural", "name": "沉稳厚重 · 纪录片/旁白 (云健)", "default_speed": 1.0, "default_pitch": "-2Hz"},
                        {"id": "zh-CN-XiaoyiNeural", "name": "活泼轻快 · 年轻日常 (晓伊)", "default_speed": 1.1, "default_pitch": "+0Hz"}
                    ]
                },
                {
                    "id": "gpt-sovits",
                    "name": "🎭 二次元角色克隆 · GPT-SoVITS",
                    "type": "anime_character_clone",
                    "description": "本地神经网络推理，1:1 复刻二次元声优独特声线与微表情",
                    "voices": []
                }
            ]

            try:
                if os.path.exists(REGISTRY_FILE):
                    with open(REGISTRY_FILE, "r", encoding="utf-8") as rf:
                        reg_data = json.load(rf)
                    sovits_voices = []
                    for cid, cinfo in reg_data.get("characters", {}).items():
                        emotions_list = VoiceRouter.get_available_emotions(cid)
                        sovits_voices.append({
                            "id": cid,
                            "name": f"{cinfo.get('character_name')} ({cinfo.get('active_version')})",
                            "lang": "ja",
                            "default_speed": 1.0,
                            "cv": cinfo.get("cv", ""),
                            "emotions": [
                                {
                                    "id": e["id"],
                                    "name": e["name"],
                                    "emotion": e["emotion"],
                                    "is_default": e["is_default"],
                                    "file_exists": e["file_exists"]
                                }
                                for e in emotions_list
                            ]
                        })
                    for eng in engines_info:
                        if eng["id"] == "gpt-sovits":
                            eng["voices"] = sovits_voices
            except Exception as e:
                print(f"Error loading models registry: {e}")

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(engines_info, ensure_ascii=False).encode("utf-8"))

        elif path == "/api/audio/characters":
            chars = []
            try:
                if os.path.exists(REGISTRY_FILE):
                    with open(REGISTRY_FILE, "r", encoding="utf-8") as rf:
                        reg_data = json.load(rf)
                    for cid, cinfo in reg_data.get("characters", {}).items():
                        chars.append({
                            "id": cid,
                            "name": cinfo.get("character_name"),
                            "cv": cinfo.get("cv", ""),
                            "supported_languages": cinfo.get("supported_languages", ["ja"]),
                            "primary_language": cinfo.get("primary_language", "ja"),
                            "emotions": VoiceRouter.get_available_emotions(cid)
                        })
            except Exception as e:
                print(f"Error reading characters: {e}")

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(chars, ensure_ascii=False).encode("utf-8"))

        elif path.startswith("/assets/audio/") or path.startswith("/api/audio/file/"):
            rel_path = path.replace("/assets/audio/", "").replace("/api/audio/file/", "").replace("training_dataset/", "")
            file_path = os.path.abspath(os.path.join(AUDIO_DIR, rel_path))
            if not os.path.isfile(file_path):
                file_path = os.path.abspath(os.path.join(AUDIO_DIR, "training_dataset", rel_path))
            if file_path.startswith(AUDIO_DIR) and os.path.isfile(file_path):
                ctype, _ = mimetypes.guess_type(file_path)
                self.send_response(200)
                self.send_header("Content-Type", ctype or "application/octet-stream")
                self.send_header("Content-Length", str(os.path.getsize(file_path)))
                self.send_header("Accept-Ranges", "bytes")
                self._send_cors_headers()
                self.end_headers()
                with open(file_path, "rb") as f:
                    while chunk := f.read(65536):
                        self.wfile.write(chunk)
        elif path.startswith("/assets/") or path in ["/favicon.svg", "/icons.svg"]:
            rel_path = path.lstrip("/")
            file_path = os.path.abspath(os.path.join(STATIC_DIR, rel_path))
            if file_path.startswith(STATIC_DIR) and os.path.isfile(file_path):
                ctype, _ = mimetypes.guess_type(file_path)
                self.send_response(200)
                self.send_header("Content-Type", ctype or "application/octet-stream")
                self.send_header("Content-Length", str(os.path.getsize(file_path)))
                self._send_cors_headers()
                self.end_headers()
                with open(file_path, "rb") as f:
                    while chunk := f.read(65536):
                        self.wfile.write(chunk)
            else:
                self.send_error(404, "Static asset not found")

        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/audio/generate":
            content_len = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                data = json.loads(post_data.decode("utf-8"))
            except Exception:
                data = {}

            input_text = data.get("input") or data.get("text", "")
            engine = data.get("engine", "").lower()
            voice = data.get("voice", "")
            voice_mode = data.get("voice_mode", "")
            language = data.get("language", "auto")
            speed = float(data.get("speed", data.get("speed_factor", 1.0)))
            pitch = data.get("pitch", "+0Hz")

            if not input_text:
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Missing input text"}).encode("utf-8"))
                return

            # Route Decision
            is_edge_tts = (
                engine in ["edge-tts", "edge", "yunxi", "xiaoshuai"] or
                voice.startswith("zh-CN") or
                voice_mode in ["xiaoshuai", "yunxi", "xiaoxiao", "yunjian", "xiaoyi"] or
                (not engine and not voice_mode.startswith("mika") and not voice.startswith("mika") and any(k in input_text for k in ["小帅", "注意看", "解说"]))
            )

            # Route 1: Edge-TTS Engine
            if is_edge_tts:
                edge_voice = voice if voice.startswith("zh-CN") else "zh-CN-YunxiNeural"
                if voice_mode == "xiaoxiao":
                    edge_voice = "zh-CN-XiaoxiaoNeural"
                elif voice_mode == "yunjian":
                    edge_voice = "zh-CN-YunjianNeural"
                elif voice_mode == "xiaoyi":
                    edge_voice = "zh-CN-XiaoyiNeural"

                speed_pct = int((speed - 1.0) * 100)
                rate_str = f"{'+' if speed_pct >= 0 else ''}{speed_pct}%"

                try:
                    import edge_tts

                    async def _run_edge_tts():
                        comm = edge_tts.Communicate(text=input_text, voice=edge_voice, rate=rate_str, pitch=pitch)
                        audio_chunks = []
                        async for chunk in comm.stream():
                            if chunk["type"] == "audio":
                                audio_chunks.append(chunk["data"])
                        return b"".join(audio_chunks)

                    audio_data = asyncio.run(_run_edge_tts())
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/mpeg")
                    self._send_cors_headers()
                    self.send_header("Content-Length", str(len(audio_data)))
                    self.end_headers()
                    self.wfile.write(audio_data)
                    return
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self._send_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": f"Edge-TTS synthesis error: {e}"}).encode("utf-8"))
                    return

            # Route 2: Native GPT-SoVITS Engine
            target_char = "kurisu"
            vm_lower = f"{(voice_mode or '')} {(voice or '')}".lower()
            if any(k in vm_lower for k in ["rimuru", "萌王", "利姆鲁"]):
                target_char = "rimuru"
            elif any(k in vm_lower for k in ["mika", "未花", "圣园"]):
                target_char = "mika"
            elif any(k in vm_lower for k in ["kurisu", "助手", "红莉栖", "christina", "assistant"]):
                target_char = "kurisu"
            elif any(k in vm_lower for k in ["utaha", "诗羽", "霞之丘"]):
                target_char = "utaha"
            elif any(k in vm_lower for k in ["chizuru", "千鹤", "水原", "mizuhara"]):
                target_char = "chizuru"

            req_emotion = data.get("emotion")
            plan = VoiceRouter.resolve_plan(
                character=target_char,
                text=input_text,
                explicit_emotion=req_emotion,
                user_params={
                    "temperature": data.get("temperature"),
                    "speed_factor": speed,
                    "top_k": data.get("top_k"),
                    "top_p": data.get("top_p")
                }
            )

            # Ensure model weights hot-loaded
            VoiceRouter.ensure_character_weights(target_char)

            target_lang = "ja" if language.lower() in ["ja", "japanese"] else "zh"

            gpt_sovits_payload = {
                "text": plan.clean_text,
                "text_lang": target_lang,
                "ref_audio_path": plan.ref_audio_path,
                "prompt_text": plan.prompt_text,
                "prompt_lang": plan.prompt_lang,
                "speed_factor": plan.speed_factor,
                "top_k": plan.top_k,
                "top_p": plan.top_p,
                "temperature": plan.temperature
            }

            try:
                # Parse host and port from GPT_SOVITS_URL
                parsed_sovits = urllib.parse.urlparse(GPT_SOVITS_URL)
                s_host = parsed_sovits.hostname or "127.0.0.1"
                s_port = parsed_sovits.port or 9880
                s_path = parsed_sovits.path or "/tts"

                conn = http.client.HTTPConnection(s_host, s_port, timeout=45)
                conn.request("POST", s_path, body=json.dumps(gpt_sovits_payload), headers={"Content-Type": "application/json"})
                resp = conn.getresponse()
                resp_data = resp.read()
                conn.close()

                if resp.status == 200:
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self._send_cors_headers()
                    self.send_header("Content-Length", str(len(resp_data)))
                    self.end_headers()
                    self.wfile.write(resp_data)
                else:
                    self.send_response(resp.status)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self._send_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "error": f"Upstream GPT-SoVITS responded with {resp.status}",
                        "upstream_details": resp_data.decode("utf-8", errors="ignore")
                    }).encode("utf-8"))
            except Exception as e:
                self.send_response(503)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": f"Cannot connect to GPT-SoVITS backend at {GPT_SOVITS_URL}: {e}"
                }).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint not found")


def main():
    server = ThreadingHTTPServer((HOST, PORT), VoiceGatewayHandler)
    print(f"🚀 [jiuguan-voice-gateway] Running on http://{HOST}:{PORT}")
    print(f"📚 Endpoints:")
    print(f"   GET  /health")
    print(f"   GET  /api/audio/engines")
    print(f"   GET  /api/audio/characters")
    print(f"   POST /api/audio/generate")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    main()
