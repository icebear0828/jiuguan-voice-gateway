#!/usr/bin/env python3
"""
Mika Misono (聖園ミカ) Native GPT-SoVITS Japanese Voice Batch Generator
100% Local Inference using fine-tuned weights:
  - GPT: mika-e15.ckpt
  - SoVITS: mika_e8_s176.pth

Usage:
    python scripts/generate_mika_gpt_sovits.py
"""

import os
import sys
import json
import time
import soundfile as sf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", PROJECT_ROOT)
SOVITS_ROOT = os.environ.get("SOVITS_ROOT", os.path.abspath(os.path.join(PROJECT_ROOT, "..", "GPT-SoVITS")))
SOVITS_CORE = os.path.join(SOVITS_ROOT, "GPT_SoVITS")
sys.path.insert(0, SOVITS_ROOT)
sys.path.insert(0, SOVITS_CORE)
sys.path.insert(0, os.path.join(SOVITS_CORE, "eres2net"))

if os.path.exists(SOVITS_ROOT):
    os.chdir(SOVITS_ROOT)

from TTS_infer_pack.TTS import TTS, TTS_Config

MIKA_MODEL_DIR = os.path.join(WORKSPACE_DIR, "assets/models/mika")
REF_AUDIO_PATH = os.path.join(MIKA_MODEL_DIR, "reference_audio/2e7e7148e1.wav")
REF_TRANSCRIPT = "あけましておめでとう、先生！こんな私だけど、今年もよろしくね"

OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "assets/audio/mika_samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 6 Characteristic Japanese lines for Mika Misono (聖園ミカ)
MIKA_DIALOGUES = [
    {
        "id": "mika_01_greeting",
        "category": "日常问候",
        "text": "先生、おかえり！待ってたよ！ちゃーんといい子でお留守番してたんだから！",
        "desc": "元气迎接先生归来"
    },
    {
        "id": "mika_02_affection",
        "category": "撒娇互动",
        "text": "先生、私のこと、待っててくれた？わはは～☆ 今日もいっぱい甘やかしてね！",
        "desc": "未花标志性小恶魔撒娇"
    },
    {
        "id": "mika_03_date_invite",
        "category": "约会邀请",
        "text": "ん～！今日も良い天気だね！良かったら、仕事終わりにお散歩とかどうかな？☆",
        "desc": "邀请先生散步"
    },
    {
        "id": "mika_04_combat_serious",
        "category": "战斗严肃",
        "text": "ここは私が守るから！あなた達は、ここから先には通さないよ！",
        "desc": "战斗认真声线，保护先生"
    },
    {
        "id": "mika_05_entrance",
        "category": "登场台词",
        "text": "聖園ミカ、ついに登場～☆ って感じかな？これからもよろしくね、先生！",
        "desc": "招牌登场台词"
    },
    {
        "id": "mika_06_emotional",
        "category": "深情心声",
        "text": "ありがとね、先生。私、堂々と先生のお隣に立てるように頑張るね…！",
        "desc": "真挚深情轻语"
    }
]


def run_batch_generation():
    print("=" * 70)
    print("🌟 蔚蓝档案 · 聖園ミカ (Misono Mika) 原生 GPT-SoVITS 批量日文语音生成器")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"🎯 待生成台词数: {len(MIKA_DIALOGUES)} 条")
    print(f"🧠 GPT 权重: {os.path.join(MIKA_MODEL_DIR, 'mika-e15.ckpt')}")
    print(f"🎵 SoVITS 权重: {os.path.join(MIKA_MODEL_DIR, 'mika_e8_s176.pth')}")
    print("=" * 70)

    # 1. Initialize GPT-SoVITS Native Pipeline
    print("\n[1/3] ⚙️  正在加载本地 GPT-SoVITS 模型与特征提取器...")
    cfg = TTS_Config()
    cfg.bert_base_path = os.path.join(SOVITS_CORE, "pretrained_models/chinese-roberta-wwm-ext-large")
    cfg.cnhuhbert_base_path = os.path.join(SOVITS_CORE, "pretrained_models/chinese-hubert-base")
    cfg.t2s_weights_path = os.path.join(MIKA_MODEL_DIR, "mika-e15.ckpt")
    cfg.vits_weights_path = os.path.join(MIKA_MODEL_DIR, "mika_e8_s176.pth")
    cfg.device = "cpu"
    cfg.is_half = False

    tts = TTS(cfg)
    print("✅ GPT-SoVITS 原生模型加载成功!")

    # 2. Batch Inference
    print("\n[2/3] 🎙️  开始执行原生神经网络语音合成...")
    results = []

    for i, item in enumerate(MIKA_DIALOGUES, 1):
        clip_id = item["id"]
        text = item["text"]
        out_filename = f"{clip_id}.wav"
        out_path = os.path.join(OUTPUT_DIR, out_filename)

        print(f"\n[{i}/{len(MIKA_DIALOGUES)}] 🎙️  [{clip_id}] {item['category']}")
        print(f"   台词: 「{text}」")
        print(f"   说明: {item['desc']}")

        req = {
            "text": text,
            "text_lang": "ja",
            "ref_audio_path": REF_AUDIO_PATH,
            "prompt_text": REF_TRANSCRIPT,
            "prompt_lang": "ja",
            "top_k": 5,
            "top_p": 1.0,
            "temperature": 1.0,
            "speed_factor": 1.0,
        }

        start_t = time.time()
        audio_chunks = []
        sample_rate = 32000

        for sr, audio in tts.run(req):
            sample_rate = sr
            audio_chunks.append(audio)

        elapsed = time.time() - start_t

        if audio_chunks:
            import numpy as np
            final_audio = np.concatenate(audio_chunks)
            sf.write(out_path, final_audio, sample_rate)
            file_size_kb = os.path.getsize(out_path) / 1024
            duration_sec = len(final_audio) / sample_rate

            print(f"   ✅ 生成成功! -> {out_filename} ({file_size_kb:.1f} KB, 时长: {duration_sec:.2f}s, 耗时: {elapsed:.2f}s)")

            results.append({
                "id": clip_id,
                "category": item["category"],
                "description": item["desc"],
                "text": text,
                "filename": out_filename,
                "file_path": out_path,
                "duration_sec": round(duration_sec, 2),
                "size_kb": round(file_size_kb, 1),
                "model_type": "GPT-SoVITS (mika-e15.ckpt + mika_e8_s176.pth)",
                "status": "success"
            })
        else:
            print(f"   ❌ 生成失败: 没有输出音频")
            results.append({
                "id": clip_id,
                "text": text,
                "status": "failed"
            })

    # 3. Write Manifest
    manifest_path = os.path.join(OUTPUT_DIR, "mika_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(results, mf, ensure_ascii=False, indent=2)

    success_count = sum(1 for r in results if r.get("status") == "success")
    print("\n" + "=" * 70)
    print(f"🎉 批量生成全部完成! 成功: {success_count}/{len(MIKA_DIALOGUES)}")
    print(f"📋 清单索引已保存: {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_batch_generation()
