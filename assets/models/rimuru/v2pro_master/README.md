---
language:
- ja
- zh
tags:
- text-to-speech
- tts
- gpt-sovits
- gpt-sovits-v2pro
- voice-cloning
- anime
- rimuru-tempest
license: apache-2.0
pipeline_tag: text-to-speech
---

# 👑 Rimuru Tempest (利姆鲁·特恩佩斯特) - GPT-SoVITS v2Pro V2.4 Master Release

<p align="center">
  <b>CV: 冈咲美保 (Miho Okasaki) | 动画《关于我转生变成史莱姆这档事》主人公</b>
</p>

---

## 📖 模型简介 (Model Overview)

本模型基于 **1,082 句无杂音录音棚官方母带干声（Native Studio Dry Vocals）**，在 RTX 5070 环境下严格执行 1.0x 原生分布微调训练完成：
* **GPT 自回归文本模型 (T2S)**：训练 12 轮（10 轮收敛导出，Top-3 Token Acc ~78%），彻底消除了旧版因过采样导致的死循环复读与吞字卡顿；
* **SoVITS 声学底模 (VITS)**：训练 25 轮（Mel Loss 11.64），底噪纯净，音质清脆透亮；
* **声学安全合规**：全流程采用 `-1.5 dBFS` 防削顶正规化，零爆音、零破音。

---

## 📁 仓库文件结构 (Repository Structure)

```plaintext
├── GPT_weights_v2Pro/
│   └── Rimuru_v2Pro_Native-e10.ckpt     # GPT 自回归语义模型 (155 MB)
├── SoVITS_weights_v2Pro/
│   └── Rimuru_v2Pro_e25_s6825.pth       # SoVITS V2Pro 声学底模 (135 MB)
├── reference_audio/                     # 严格声学质检的黄金母带参考音频
│   ├── 01_Master_Daily.wav              # 【全场景通用日常主基准】(SNR 21.0 dB, 3.82s)
│   ├── 01_Master_Daily.txt              # 台词: 自己紹介がまだだったな 俺はリムル＝テンペスト
│   ├── 02_Master_Confident.wav          # 【盟主坚定威严基准】(SNR 21.7 dB, 3.30s)
│   └── 02_Master_Confident.txt          # 台词: その名にふさわしく、俺の役に立ってくれ！
├── samples/                             # 实测生成的高保真试听样本
│   ├── 01_日常问候_日常从容自介.wav
│   ├── 02_自信得意_轻快傲娇自夸.wav
│   ├── 03_认真承诺_坚定盟主承诺.wav
│   └── 04_长篇剧情_联邦国演说_标准母本版.wav  # 26.5 秒长演说 (Whisper CER 6.1%)
├── infer_example.py                     # 开箱即用的极简 Python 推理代码
└── .gitattributes                       # Hugging Face Git LFS 配置
```

---

## 🎛️ 推荐推理参数 (Recommended Inference Settings)

| 参数项 (Parameter) | 推荐值 (Recommended) | 说明 (Notes) |
| :--- | :---: | :--- |
| **参考音频 (Reference Audio)** | `01_Master_Daily.wav` | 通用全场景首选，语调起伏自然 |
| **参考文本 (Prompt Text)** | `自己紹介がまだだったな 俺はリムル＝テンペスト` | 必须逐字严格匹配，防止交叉注意力混淆 |
| **采样温度 (Temperature)** | **`0.68 ~ 0.72`** | 最佳甜点区间（少年感灵动活泼，不平淡也不破音） |
| **Top-P** | `0.95` | 保持音素多样性 |
| **Top-K** | `5` | 抑制极端尾部异常采样 |
| **切分方式 (Text Split)** | 短句 `cut0` / 长文按意群 | 单句建议 15~25 字意群分句，句间添加 0.25s 换气 |
| **电平正规化** | `-1.5 dBFS` | 保持广播级安全动态余量 |

---

## 🚀 快速推理示例 (Quickstart Python Example)

```python
import soundfile as sf
from TTS_infer_pack.TTS import TTS, TTS_Config

# 1. 初始化配置
tts_config = TTS_Config()
tts_config.v2Pro = True
tts_config.t2s_weights_path = "GPT_weights_v2Pro/Rimuru_v2Pro_Native-e10.ckpt"
tts_config.vits_weights_path = "SoVITS_weights_v2Pro/Rimuru_v2Pro_e25_s6825.pth"
tts_config.device = "cuda:0"
tts_config.is_half = True

tts = TTS(tts_config)

# 2. 推理目标文本
req = {
    "text": "仲間たちと共に、誰もが笑って暮らせる国を必ず作ってみせるさ。",
    "text_lang": "ja",
    "ref_audio_path": "reference_audio/01_Master_Daily.wav",
    "prompt_text": "自己紹介がまだだったな 俺はリムル＝テンペスト",
    "prompt_lang": "ja",
    "top_k": 5,
    "top_p": 0.95,
    "temperature": 0.70,
    "text_split_method": "cut0",
    "sample_steps": 32,
    "speed_factor": 1.0
}

gen = tts.run(req)
sr, audio_data = next(gen)
sf.write("output_rimuru.wav", audio_data, sr)
print("✅ 合成成功！已保存至 output_rimuru.wav")
```

---

## ⚖️ 开源协议与免责声明 (License & Disclaimer)

* 本模型仅供 **个人学习、非商业学术研究与二创交流** 使用；
* 请勿将本模型用于任何违法违规、侵犯他人名誉权或商业诈骗等不当用途；
* 原始声音版权归属于原动画官方制作委员会及声优 **冈咲美保** 所有。
