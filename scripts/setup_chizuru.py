#!/usr/bin/env python3
import os
import sys
import json
import shutil
import unicodedata

STAGE = "/Users/yutao/jiuguan-voice-gateway/assets/models/_staging_chizuru/GPT-SoVITS-1007-cu128/models/v4/租借女友-日语-水原千鹤"
MODELS_DIR = "/Users/yutao/jiuguan-voice-gateway/assets/models"
CHIZURU_DIR = os.path.join(MODELS_DIR, "chizuru")
V4_DIR = os.path.join(CHIZURU_DIR, "v4")
REF_DIR = os.path.join(V4_DIR, "reference_audios")
SAMPLES_DIR = "/Users/yutao/jiuguan-voice-gateway/assets/audio/chizuru_samples/v4"

os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# 1. Copy weights
ckpt_name = "水原千鹤-v4dpo-20260607-e10.ckpt"
pth_name = "水原千鹤-v4dpo-20260607_e16_s464_l32.pth"
shutil.copy2(os.path.join(STAGE, ckpt_name), os.path.join(V4_DIR, ckpt_name))
shutil.copy2(os.path.join(STAGE, pth_name), os.path.join(V4_DIR, pth_name))

# 2. Setup current symlink
current_link = os.path.join(CHIZURU_DIR, "current")
if os.path.islink(current_link) or os.path.exists(current_link):
    os.remove(current_link)
os.symlink("v4", current_link)

# 3. Process reference audios
src_emotions = os.path.join(STAGE, "reference_audios/日语/emotions")
files = os.listdir(src_emotions)

emotion_defs = [
    {
        "id": "daily",
        "name": "default_daily",
        "orig_file": "【默认】ちょっと考えてみて、よかったら教えて。.wav",
        "clean_file": "ref_default.wav",
        "emotion": "默认 / 日常从容",
        "text": "ちょっと考えてみて、よかったら教えて。",
        "lang": "ja",
        "is_default": True,
        "recommended_params": {"temperature": 0.8, "speed_factor": 1.0, "top_k": 5, "top_p": 0.95}
    },
    {
        "id": "gentle",
        "name": "gentle_confession",
        "orig_file": "【温柔-中音调】私、あなたの彼女…だろう？.wav",
        "clean_file": "ref_gentle.wav",
        "emotion": "温柔 / 恋人契约",
        "text": "私、あなたの彼女…だろう？",
        "lang": "ja",
        "is_default": False,
        "recommended_params": {"temperature": 0.75, "speed_factor": 0.95, "top_k": 5, "top_p": 0.95}
    },
    {
        "id": "intimate",
        "name": "intimate_date",
        "orig_file": "【亲密】じゃあ、いつもより伸び伸びしたデート、期待してるね。.wav",
        "clean_file": "ref_intimate.wav",
        "emotion": "亲密 / 约会期待",
        "text": "じゃあ、いつもより伸び伸びしたデート、期待してるね。",
        "lang": "ja",
        "is_default": False,
        "recommended_params": {"temperature": 0.8, "speed_factor": 1.0, "top_k": 5, "top_p": 0.95}
    },
    {
        "id": "tsundere",
        "name": "tsundere_denial",
        "orig_file": "【傲娇】別に、あなたと行くなんて言ってないけど。.wav",
        "clean_file": "ref_tsundere.wav",
        "emotion": "傲娇 / 口是心非",
        "text": "別に、あなたと行くなんて言ってないけど。",
        "lang": "ja",
        "is_default": False,
        "recommended_params": {"temperature": 0.85, "speed_factor": 1.05, "top_k": 5, "top_p": 0.95}
    },
    {
        "id": "shy",
        "name": "shy_heartbeat",
        "orig_file": "【害羞】私も久しぶりで、なんだかドキドキしてきちゃって.wav",
        "clean_file": "ref_shy.wav",
        "emotion": "害羞 / 心跳加速",
        "text": "私も久しぶりで、なんだかドキドキしてきちゃって",
        "lang": "ja",
        "is_default": False,
        "recommended_params": {"temperature": 0.8, "speed_factor": 0.98, "top_k": 5, "top_p": 0.95}
    },
    {
        "id": "annoyed",
        "name": "annoyed_hangup",
        "orig_file": "【不满】これで大した用じゃなかったらもう速攻で切るから。.wav",
        "clean_file": "ref_annoyed.wav",
        "emotion": "不满 / 嗔怒挂断",
        "text": "これで大した用じゃなかったらもう速攻で切るから。",
        "lang": "ja",
        "is_default": False,
        "recommended_params": {"temperature": 0.85, "speed_factor": 1.05, "top_k": 5, "top_p": 0.95}
    },
    {
        "id": "happy",
        "name": "happy_cheerful",
        "orig_file": "【开心-高音调】喜んでくれて、私も嬉しい！.wav",
        "clean_file": "ref_happy.wav",
        "emotion": "开心 / 阳光明媚",
        "text": "喜んでくれて、私も嬉しい！",
        "lang": "ja",
        "is_default": False,
        "recommended_params": {"temperature": 0.85, "speed_factor": 1.05, "top_k": 5, "top_p": 0.95}
    },
    {
        "id": "excited",
        "name": "excited_penguin",
        "orig_file": "【兴奋】わー見て見て、ペンギンいるよ！よちよち歩いてかわいい。.wav",
        "clean_file": "ref_excited.wav",
        "emotion": "兴奋 / 元气雀跃",
        "text": "わー見て見て、ペンギンいるよ！よちよち歩いてかわいい。",
        "lang": "ja",
        "is_default": False,
        "recommended_params": {"temperature": 0.9, "speed_factor": 1.1, "top_k": 5, "top_p": 0.95}
    },
    {
        "id": "sad",
        "name": "sad_sunset",
        "orig_file": "【消沉】楽しかった分、夕焼けを見てると、ちょっと寂しくなっちゃうね。.wav",
        "clean_file": "ref_sad.wav",
        "emotion": "消沉 / 黄昏落寞",
        "text": "楽しかった分、夕焼けを見てると、ちょっと寂しくなっちゃうね。",
        "lang": "ja",
        "is_default": False,
        "recommended_params": {"temperature": 0.75, "speed_factor": 0.92, "top_k": 5, "top_p": 0.95}
    },
    {
        "id": "angry",
        "name": "angry_flustered",
        "orig_file": "【生气】あやだ、本当だ！もう夜に干すんじゃなかった！.wav",
        "clean_file": "ref_angry.wav",
        "emotion": "生气 / 急躁抓狂",
        "text": "あやだ、本当だ！もう夜に干すんじゃなかった！",
        "lang": "ja",
        "is_default": False,
        "recommended_params": {"temperature": 0.9, "speed_factor": 1.08, "top_k": 5, "top_p": 0.95}
    }
]

ref_audios_json = []
for ed in emotion_defs:
    src_p = None
    for f in files:
        if unicodedata.normalize("NFC", f) == unicodedata.normalize("NFC", ed["orig_file"]):
            src_p = os.path.join(src_emotions, f)
            break
    if not src_p or not os.path.exists(src_p):
        print("Warning: not found", ed["orig_file"])
        continue
    dest_clean = os.path.join(REF_DIR, ed["clean_file"])
    shutil.copy2(src_p, dest_clean)
    shutil.copy2(src_p, os.path.join(REF_DIR, os.path.basename(src_p)))
    
    clean_name = ed["clean_file"]
    ref_audios_json.append({
        "id": ed["id"],
        "name": ed["name"],
        "audio_file": f"reference_audios/{clean_name}",
        "text": ed["text"],
        "lang": ed["lang"],
        "emotion": ed["emotion"],
        "is_default": ed["is_default"],
        "recommended_params": ed["recommended_params"]
    })

# Also copy 6 representative samples to chizuru_samples/v4 for official demos
demo_map = [
    ("01_gentle_girlfriend.wav", "ref_gentle.wav"),
    ("02_intimate_date.wav", "ref_intimate.wav"),
    ("03_tsundere_denial.wav", "ref_tsundere.wav"),
    ("04_shy_heartbeat.wav", "ref_shy.wav"),
    ("05_excited_penguin.wav", "ref_excited.wav"),
    ("06_sad_sunset.wav", "ref_sad.wav")
]

for demo_target, ref_src in demo_map:
    shutil.copy2(os.path.join(REF_DIR, ref_src), os.path.join(SAMPLES_DIR, demo_target))

# Write model_info.json
model_info = {
    "character_id": "chizuru",
    "character_name": "水原千鹤 (Mizuhara Chizuru / 一之濑千鹤)",
    "character_source": "租借女友 (彼女、お借りします / Rent-A-Girlfriend)",
    "cv": "雨宫天 (Sora Amamiya)",
    "version": "v4",
    "architecture": "GPT-SoVITS-v4",
    "status": "active",
    "created_at": "2026-06-07T12:00:00Z",
    "weights": {
        "gpt": ckpt_name,
        "sovits": pth_name
    },
    "training_stats": {
        "gpt_epochs": 10,
        "sovits_epochs": 16,
        "sovits_steps": 464,
        "if_dpo": True,
        "architecture_notes": "GPT-SoVITS v4 LoRA (layer 32) + DPO Alignment"
    },
    "recommended_params": {
        "temperature": 0.8,
        "top_k": 5,
        "top_p": 0.95,
        "speed_factor": 1.0
    },
    "reference_audios": ref_audios_json,
    "notes": "水原千鹤GPT-SoVITS v4 DPO对齐微调模型，覆盖默认、温柔、亲密、傲娇、害羞、不满、开心、兴奋、消沉、生气等多维度情感声线。"
}

with open(os.path.join(V4_DIR, "model_info.json"), "w", encoding="utf-8") as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)

with open(os.path.join(REF_DIR, "prompt_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(ref_audios_json, f, ensure_ascii=False, indent=2)

print(f"✅ Successfully set up Chizuru v4 model! Copied {len(ref_audios_json)} emotion references and {len(demo_map)} demos.")
