# 👑 利姆鲁 (Rimuru Tempest) 多情感黄金参考音频规范 (Multi-Emotion Prompt Library)

本目录为利姆鲁 GPT-SoVITS v2Pro 架构的多情感参考音频母带库。系统已部署智能情感路由引擎（`VoiceRouter`），支持根据**显式参数**、**文本内嵌标签**（如 `[playful]`、`[combat]`、`[gentle]`）或**自然语意启发式规则**自动调度对应的参考音频与推理超参。

---

## 🎭 5 维情感基准配置清单

| 情感分类 (ID) | 情绪标签 | 音频文件路径 | 配套台词文本 (.txt) | 采样温度 (Temp) | 语速 (Speed) | 状态 |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **`daily`** *(默认基准)* | 日常从容 / 盟主自介 | `01_Master_Daily.wav` | `自己紹介がまだだったな 俺はリムル＝テンペスト` | 0.68 | 1.00 | 🟢 已就绪 (SNR 21dB) |
| **`confident`** | 盟主威严 / 战斗霸气 | `02_Master_Confident.wav` | `その名にふさわしく、俺の役に立ってくれ！` | 0.70 | 1.02 | 🟢 已就绪 (SNR 21.7dB) |
| **`playful`** | 俏皮活力 / 萌王自夸 | `03_Master_Playful.wav` | `俺はスライムのリムル！悪いスライムじゃないよ！` | 0.74 | 1.05 | 🟡 预留槽位 (智能Fallback) |
| **`gentle`** | 温柔深情 / 羁绊承诺 | `04_Master_Gentle.wav` | `仲間たちと共に、誰もが笑って暮らせる国を必ず作ってみせるさ。` | 0.65 | 0.95 | 🟡 预留槽位 (智能Fallback) |
| **`slacking`** | 吐槽摸鱼 / 搞笑无奈 | `05_Master_Slacking.wav` | `ふぁ〜あ……たまには仕事をサボって温泉に行こう！` | 0.72 | 1.03 | 🟡 预留槽位 (智能Fallback) |

> 💡 **声学安全防护机制 (Acoustic Safety Fallback)**：
> 当预留槽位（如 `03_Master_Playful.wav`）尚未放入音频文件时，`VoiceRouter` 会自动安全回退至已就绪的母带音频，并同步将提示文本对齐至对应母带台词，彻底防止注意力对齐错乱（吞字/乱叫），同时提前生效专属的 Temperature 与 Speed 超参。

---

## 📥 如何导入新音频切片

当您提取或准备好新的音频切片时，只需按以下声学标准将文件放入本目录即可**即时热生效（无需重启服务）**：

### 1. 声学质检标准
* **格式**：16-bit Mono (单声道) PCM `.wav`
* **采样率**：`32000 Hz`（与 v2Pro 底模原生完全一致）或 `44100 / 48000 Hz`
* **长度**：严格控制在 **3.0 秒 ~ 6.0 秒**（过短特征不足，过长影响推理速度与自回归步长）
* **底噪**：纯净录音棚/原画母带干声（Dry Vocal），零背景音乐（No BGM）、零环境杂音
* **电平安全**：执行峰值正规化至 **`-1.5 dBFS`**（防止削顶爆音）

### 2. 文件命名与台词绑定
将音频及同名 `.txt` 文本存放在本目录下：
* 俏皮活力：`03_Master_Playful.wav` + `03_Master_Playful.txt`
* 温柔深情：`04_Master_Gentle.wav` + `04_Master_Gentle.txt`
* 吐槽摸鱼：`05_Master_Slacking.wav` + `05_Master_Slacking.txt`

若所剪辑音频的台词有所变动，同步修改配套 `.txt` 文本以及 `prompt_metadata.json` 中的 `prompt_text` 字段即可。

---

## 🚀 运行时调用与标签测试

在调用 `/api/audio/generate` 或对话服务时，支持以下调用方式：

1. **内嵌情感标签（自动提取情绪并剥离发音标签）**：
   ```json
   {
     "engine": "gpt-sovits",
     "voice_mode": "rimuru",
     "input": "[playful] やあ！スライムのリムルだよ！わはは！",
     "language": "ja"
   }
   ```
2. **显式参数传递**：
   ```json
   {
     "engine": "gpt-sovits",
     "voice_mode": "rimuru",
     "emotion": "confident",
     "input": "俺の仲間に手を出したこと、後悔させてやる！",
     "language": "ja"
   }
   ```
3. **自然语意启发式触发**：
   文本中含有 `わはは`、`スライム`、`後悔させてやる`、`約束`、`サボ`、`温泉` 等特征词，系统将自动分类并加载对应情感。
