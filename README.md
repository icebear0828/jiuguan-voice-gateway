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

---

## 🛡️ 工业级 4-Stage 语音质检漏斗与自蒸馏飞轮

本项目内置面向二次元角色高保真语音合成的 **4-Stage (+ Stage 3.5) 质检漏斗流水线 (`QualityPipeline`)**，并结合 **Best-of-N 拒绝采样** 打造无损自蒸馏闭环：

```
待测合成语音 ──► [Stage 1: 声学物理与自愈] ──► [Stage 2: 语义与音素PER] ──► [Stage 3: 多情绪原型声纹] ──► [Stage 3.5: F0韵律与Siri腔拦截] ──► [Stage 4: UTMOS与破音跳变检测] ──► 极品合格语音 ──► 5:5 动态退火自蒸馏
```

- **Stage 1 (声学物理门禁)**：`pyloudnorm` Integrated LUFS（-18.0）+ True Peak 4x 过采样防爆音；`[whisper]` 弱音标签增益豁免；局部 VAD 句中卡壳检测（`max_internal_silence <= 600ms`）；20ms Cosine Fade 平滑自愈。
- **Stage 2 (语义对齐门禁)**：`cn2an` 前置双向口语正则化；`pypinyin` / `pyopenjtalk` 音素级 PER 序列校验（拦截多音字字对音错）；差集复读机与尾音死拖长拦截。
- **Stage 3 (音色一致性门禁)**：多情绪原型锚点池（Multi-Prototype Anchors）+ Max-Pooling；超短音频（$<1.5\text{s}$）自适应动态门限。
- **Stage 3.5 (韵律表现力门禁)**：基频起伏度（`Pitch Std >= 12Hz`）与半音范围，精准拦截平直冷漠的“Siri机器人棒读”；Prompt 情绪对齐防傲娇坍塌为 neutral。
- **Stage 4 (神经 MOS 门禁)**：生成式 TTS 专用评估器 `UTMOSVerifier`；高频金属电音与梳状滤波识别；RMVPE / F0 连续性瞬态破音跳变捕捉。
- **自蒸馏闭环飞轮**：5:5 动态退火配比（50% 原声 + 50% 极品合成，杜绝模型方差坍塌）+ 领域词特征加权 + **Best-of-N 拒绝采样（Rejection Sampling）** 择优录取。

📖 **深入了解算法原理与技术规范**：请参阅 [QUALITY_FUNNEL_AND_DISTILLATION.md](docs/QUALITY_FUNNEL_AND_DISTILLATION.md)。

---

## 🧪 自动化测试与工程验收

运行全套 113 项核心测试与系统级验收审计：

```bash
# 激活虚拟环境并安装开发依赖
pip install -r requirements.txt -r requirements-dev.txt

# 运行全量代码与质检门禁审计
make audit
```
