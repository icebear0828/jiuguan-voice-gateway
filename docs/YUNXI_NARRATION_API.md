# 🎬 经典影视解说（小帅 · 云希）语音合成服务 API 文档

本文档详细说明如何在本地或生产环境中调用、嵌入 **“小帅”（`zh-CN-YunxiNeural`）** 经典影视解说/中文科普语音合成引擎。

---

## 📌 一、核心特性与技术规格

* **音色标识**：`zh-CN-YunxiNeural`（云希 · 中文男声）
* **音色定位**：短视频解说、电影速通、科普知识讲解、有声书旁白、AI 助手
* **商业合规**：✅ **允许免费商用**（免版税、无肖像/声纹侵权风险）
* **硬件要求**：**零显卡/零显存依赖**（CPU / 内存几乎不占），支持秒级高清流式直出
* **音频规格**：24kHz / 48kHz 高采样率 MP3/WAV/AAC 编码

---

## 🛠️ 二、Python SDK 极简调用方式

项目虚拟环境中已预装核心引擎依赖，可通过 Python 直接调用：

```python
import asyncio
import edge_tts

async def generate_speech(
    text: str,
    output_path: str = "output.mp3",
    rate: str = "+10%",      # 语速调节：解说推荐 +10% ~ +20%，默认 +0%
    pitch: str = "+0Hz",     # 音调调节：默认 +0Hz
    volume: str = "+0%"      # 音量调节：默认 +0%
):
    """
    生成小帅（云希）解说语音
    """
    communicate = edge_tts.Communicate(
        text=text,
        voice="zh-CN-YunxiNeural",
        rate=rate,
        pitch=pitch,
        volume=volume
    )
    await communicate.save(output_path)
    print(f"✅ 音频生成成功: {output_path}")

# 执行合成
if __name__ == "__main__":
    text = "注意看，眼前这个男人叫小帅，他怎么也没想到，自己竟然穿越了！"
    asyncio.run(generate_speech(text, "xiaoshuai.mp3", rate="+15%"))
```

---

## 🌐 三、HTTP RESTful API 接口规范

支持标准 OpenAI / SillyTavern 兼容的 REST API 协议。

### 1. 语音合成接口

* **接口路径**：`POST /v1/audio/speech`
* **Content-Type**：`application/json`
* **返回类型**：`audio/mpeg`（二进制音频流）

#### 📥 请求参数（Request Body）

| 字段名 | 类型 | 必填 | 默认值 | 详细说明 |
| :--- | :--- | :---: | :--- | :--- |
| `input` / `text` | `string` | **是** | - | 待合成的中文文本内容（支持长文本） |
| `voice` | `string` | 否 | `zh-CN-YunxiNeural` | 音色标识，固定填 `zh-CN-YunxiNeural` |
| `rate` / `speed` | `string` / `float` | 否 | `+10%` / `1.1` | 语速：解说推荐 `+10%` 到 `+20%`（或数值 `1.1`~`1.2`） |
| `pitch` | `string` | 否 | `+0Hz` | 音调偏移，如 `+5Hz`、`-5Hz` |
| `volume` | `string` | 否 | `+0%` | 音量增益，如 `+10%` |
| `response_format` | `string` | 否 | `mp3` | 输出格式，可选 `mp3`, `wav`, `aac` |

#### 📤 响应格式（Response）

* **成功（HTTP 200）**：直接返回 `audio/mpeg` 二进制音频流数据；
* **失败（HTTP 400/500）**：返回 JSON 错误对象：
  ```json
  {
    "error": {
      "message": "Missing required field: input",
      "type": "invalid_request_error"
    }
  }
  ```

---

## 💻 四、多语言调用代码示例

### 1. cURL 命令行调用

```bash
curl -X POST http://127.0.0.1:8888/api/audio/generate \
  -H "Content-Type: application/json" \
  -d '{
    "input": "注意看，眼前这个男人叫小帅，他发现了事情的真相！",
    "voice": "zh-CN-YunxiNeural",
    "speed": 1.15
  }' \
  --output result.mp3
```

### 2. JavaScript / TypeScript (Fetch API / 前端)

```typescript
async function fetchXiaoshuaiVoice(text: string): Promise<Blob> {
  const response = await fetch("http://127.0.0.1:8888/api/audio/generate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      input: text,
      voice: "zh-CN-YunxiNeural",
      speed: 1.15,
    }),
  });

  if (!response.ok) {
    throw new Error(`TTS synthesis failed: ${response.statusText}`);
  }

  return await response.blob();
}
```

### 3. Python `requests` 调用

```python
import requests

url = "http://127.0.0.1:8888/api/audio/generate"
payload = {
    "input": "注意看，眼前这个男人叫小帅，整个宇宙的秘密即将揭晓！",
    "voice": "zh-CN-YunxiNeural",
    "speed": 1.15
}

response = requests.post(url, json=payload, timeout=30)
if response.status_code == 200:
    with open("xiaoshuai_story.mp3", "wb") as f:
        f.write(response.content)
    print("✅ 音频保存成功: xiaoshuai_story.mp3")
else:
    print("❌ 生成失败:", response.text)
```

---

## 🎚️ 五、不同解说场景参数调优指南（黄金预设）

| 解说场景 | 推荐语速 (`rate`) | 推荐音调 (`pitch`) | 听感与情绪特点 | 适用文案示例 |
| :--- | :---: | :---: | :--- | :--- |
| **经典短视频速通** | `+15%` ~ `+20%` | `+0Hz` | 节奏紧凑、抓耳、吸睛 | “注意看，眼前这个男人叫小帅…” |
| **悬疑反转高潮** | `+10%` | `+2Hz` | 紧张悬疑、略微拔高声调 | “谁知下一秒，门外突然传来了脚步声…” |
| **硬核科技/科普** | `+5%` ~ `+10%` | `+0Hz` | 清晰沉稳、逻辑严谨 | “其实量子力学并没有想象的那么玄乎…” |
| **历史/纪录片旁白** | `+0%` | `-2Hz` | 稳重大气、叙事感拉满 | “公元前221年，大秦帝国统一六国…” |

---

## 🍻 六、酒馆（SillyTavern）集成配置

若需要接入 SillyTavern 等 WebUI 客户端：
1. **TTS 插件类型**：选择 **OpenAI (Speech API)** 或 **Custom Edge-TTS**；
2. **API Endpoint**：配置为 `http://127.0.0.1:8888/v1/audio/speech`；
3. **Voice ID**：填写 `zh-CN-YunxiNeural`；
4. **Speed**：建议滑动到 `1.15x`。
