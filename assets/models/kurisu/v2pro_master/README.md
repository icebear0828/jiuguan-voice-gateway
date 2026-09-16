---
language:
- ja
- zh
tags:
- gpt-sovits
- text-to-speech
- tts
- voice-clone
- steins-gate
- makise-kurisu
- anime
license: cc-by-nc-4.0
pipeline_tag: text-to-speech
inference: false
---

# 🧬 Makise Kurisu (牧瀬 紅莉栖) GPT-SoVITS v2Pro Voice Model

<div align="center">

![Steins Gate](https://img.shields.io/badge/Character-Makise%20Kurisu-critical?style=for-the-badge)
![Base Model](https://img.shields.io/badge/Architecture-GPT--SoVITS%20v2Pro-blue?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Corpus-2%2C268%20Clean%20Lines-success?style=for-the-badge)
![Sampling](https://img.shields.io/badge/Sample%20Rate-32%2C000Hz%20PCM-orange?style=for-the-badge)

</div>

High-fidelity Japanese & Multilingual voice model of **Makise Kurisu (牧瀬紅莉栖)** from *STEINS;GATE (命运石之门)*, CV: **Asami Imai (今井麻美)**.

Trained on **2,268 pure 32kHz studio voice lines** standardized with EBU R128 (`-16.0 LUFS`), True Peak (`-1.0 dBFS`), and Silero VAD silence trimming.

---

## 🌟 Key Features

- **1:1 Pure Phoneme Alignment**: Fully purged of artificial label pollution, preserving authentic prosodic shifts between intellectual reasoning and tsundere snark.
- **Dual-Stage Fine-Tuning**:
  - **SoVITS Acoustic Model**: 25 Epochs (9,475 steps) on `s2Gv2Pro.pth`
  - **GPT Prior Model**: 20 Epochs (7,420 steps, Top-3 Token Accuracy 86.3%) on `s1v3.ckpt`
- **Multi-Emotion Few-Shot Conditioning**:
  - 💥 **Tsundere / Snarky**: High-energy vocal dynamic with sharp inflections
  - 🔬 **Scientific / Analytical**: Calm, measured, articulate intellectual discourse
  - 🌸 **Gentle / Emotional**: Warm, resonant emotional depth
  - ☕ **Daily Lab Conversation**: Natural everyday banter with Okabe and Daru

---

## 📁 Repository Structure

```
.
├── README.md                      # Model Card & Documentation
├── .gitattributes                 # Git LFS tracking configuration
├── requirements.txt               # Runtime dependencies
├── inference_example.py           # Standalone Python inference script
├── GPT_weights/
│   └── Kurisu-e20.ckpt            # GPT Autoregressive Prior (148 MB)
├── SoVITS_weights/
│   └── Kurisu_e25_s9475.pth       # SoVITS Acoustic Generator (128 MB)
├── reference_audios/              # Golden Prompt Reference Audios
│   ├── tsundere_CRS_0030.wav      # "何が真実よ、この変態! 馬鹿なの!? 死ぬの!?" (4.10s)
│   ├── science_CRS_0015.wav       # "どうして?私、前にあなたと会ったことありますか?" (4.36s)
│   ├── gentle_CRS_0014.wav        # "まるで、今にも泣き出しそうで、それにすごく辛そうでした" (6.07s)
│   ├── daily_CRS_0038.wav         # "こういうのは初めてなので緊張してますが..." (6.80s)
│   └── prompt_metadata.json       # Metadata and prompt texts
└── demo_audios/                   # Sample audio previews
    ├── demo_01_tsundere_coffee.wav
    ├── demo_04_science_timemachine.wav
    ├── demo_07_gentle_worldline.wav
    ├── demo_10_daily_morning.wav
    └── demo_12_chinese_tsundere.wav
```

---

## 🚀 Quickstart & Usage

### 1. WebUI (GUI Mode)
1. Download `Kurisu-e20.ckpt` into your local `GPT_weights_v2Pro/` folder.
2. Download `Kurisu_e25_s9475.pth` into your local `SoVITS_weights_v2Pro/` folder.
3. In the 1B-Inference WebUI:
   - Select GPT model: `Kurisu-e20.ckpt`
   - Select SoVITS model: `Kurisu_e25_s9475.pth`
   - Set **Speed**: `1.0` (Do NOT modify)
   - Set **Sample Steps**: `32`

### 2. Python Script Mode
```python
python inference_example.py
```

---

## 🎭 Golden Reference Prompts

| Emotion Style | Reference Audio File | Prompt Transcript |
| :--- | :--- | :--- |
| **傲娇 / 吐槽 (Tsundere)** | `reference_audios/tsundere_CRS_0030.wav` | `何が真実よ、この変態! 馬鹿なの!? 死ぬの!?` |
| **知性 / 论证 (Science)** | `reference_audios/science_CRS_0015.wav` | `どうして?私、前にあなたと会ったことありますか?` |
| **温柔 / 深情 (Gentle)** | `reference_audios/gentle_CRS_0014.wav` | `まるで、今にも泣き出しそうで、それにすごく辛そうでした` |
| **Lab日常 (Daily)** | `reference_audios/daily_CRS_0038.wav` | `こういうのは初めてなので緊張してますが、ガチガチなのは多めに見てください。 どうぞよろしく。` |

---

## 📜 License & Citation

- **Original Character & Voice Copyright**: MAGES. / 5pb. / Nitroplus / Asami Imai.
- **Model Weights License**: [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (Non-commercial research & personal hobby use only).
