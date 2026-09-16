# 🎙️ jiuguan-voice-gateway

> 酒馆解耦多引擎语音合成网关服务 (Multi-Engine Voice Gateway & Evaluation Suite)

专为 **SillyTavern（酒馆）** 及外部自动化脚本定制的多引擎语音合成中继网关与评测套件。

---

## 🌟 核心特性

1. **双引擎解耦调度**：
   - **商业级旁白引擎 (`edge-tts`)**：集成微软云希（`zh-CN-YunxiNeural` 影视解说小帅音色）、晓晓、云健、晓伊，毫秒级流式 MP3 输出，极度适合解说与剧情速通。
   - **二次元角色克隆引擎 (`gpt-sovits`)**：对接本地或远程 GPT-SoVITS 推理后端，支持二次元声优高保真 32kHz WAV 原生神经网络推理。
2. **多情绪智能路由 (`VoiceRouter`)**：
   - 支持自动情感启发式匹配（傲娇、科研、温柔、活泼、挑逗等）；
   - 自动选择最优参考音频切片并动态装配推理提示词；
   - 支持多角色声学模型权重热加载（Hot-swap）。
3. **内置多角色音色资产库 (`assets/`)**：
   - **利姆鲁（萌王）**：4 种情绪切片 + v2Pro 权重；
   - **牧濑红莉栖（助手）**：8 轨全场景情绪样本 + v2Pro 权重；
   - **圣园未花（Mika）**：6 轨专属音色样本 + 权重；
   - **霞之丘诗羽（学姐）**：精选情绪切片 + 权重；
   - **安娜·锦之宫**：4 话原声纯净干声切片与零样本克隆。

---

## 🚀 快速启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动网关服务 (默认端口 8888)

```bash
python3 server.py
```

指定端口或绑定地址：
```bash
VOICE_GATEWAY_PORT=8888 VOICE_GATEWAY_HOST=0.0.0.0 python3 server.py
```

---

## 🌐 核心 RESTful API 接口

| Method | Endpoint | 说明 |
| :--- | :--- | :--- |
| `GET` | `/health` | 服务健康检查 |
| `GET` | `/api/audio/engines` | 获取所有支持的引擎与音色清单 |
| `GET` | `/api/audio/characters` | 获取支持的角色与可用情绪列表 |
| `POST` | `/api/audio/generate` | 核心语音合成端点（支持 Edge-TTS 与 GPT-SoVITS） |
| `GET` | `/assets/audio/<path>` | 访问本地静态参考音频与评测样本 |

详细 API 字段规范与多语言调用示例请参阅：
- [UNIFIED_TTS_API.md](docs/UNIFIED_TTS_API.md)
- [YUNXI_NARRATION_API.md](docs/YUNXI_NARRATION_API.md)

---

## 🍷 SillyTavern（酒馆）插件配置

1. 打开酒馆 Extensions -> **TTS (Text-to-Speech)**；
2. 插件类型选择 **Custom (HTTP Endpoint)** 或 OpenAI Compatible；
3. 端点 URL 填入：`http://127.0.0.1:8888/api/audio/generate`；
4. 请求体模板（Body Payload）：
   ```json
   {
     "engine": "edge-tts",
     "voice": "zh-CN-YunxiNeural",
     "input": "{{text}}"
   }
   ```
