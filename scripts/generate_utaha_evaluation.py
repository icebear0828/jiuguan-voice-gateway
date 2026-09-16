#!/usr/bin/env python3
"""
Kasumigaoka Utaha (霞之丘诗羽 / 霞之丘詩羽) GPT-SoVITS v4 Batch Audio Generator & Evaluator
Generates high-fidelity Japanese voice evaluation samples across 5 emotions (neutral, intimate, shy, annoyed, sad).
Produces evaluation WAV samples and metadata report.
"""

import os
import sys
import time
import json
import numpy as np
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

MODEL_DIR = os.path.join(WORKSPACE_DIR, "assets/models/utaha/v4")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "assets/audio/utaha_samples/v4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REF_NEUTRAL = {
    "path": os.path.join(MODEL_DIR, "reference_audios/ref_neutral.wav"),
    "text": "最終決戦前、とうとう主人公への気持ちを我慢できなくなったヒロインは、彼を自分の部屋へと誘って、",
    "lang": "ja"
}

REF_INTIMATE = {
    "path": os.path.join(MODEL_DIR, "reference_audios/ref_intimate.wav"),
    "text": "あなたの重みを感じて、あなたの手のひらの温もりを感じて",
    "lang": "ja"
}

REF_SHY = {
    "path": os.path.join(MODEL_DIR, "reference_audios/ref_shy.wav"),
    "text": "別に激しくしてもいいの優しくされても",
    "lang": "ja"
}

REF_ANNOYED = {
    "path": os.path.join(MODEL_DIR, "reference_audios/ref_annoyed.wav"),
    "text": "どうして後片付けしようとしているのよええ確かにこのシーンは終わったわでもこのゲームには",
    "lang": "ja"
}

REF_SAD = {
    "path": os.path.join(MODEL_DIR, "reference_audios/ref_sad.wav"),
    "text": "ダメなの？こんなに近いのに、こんなに振り合ってるのに、こんなに強く…",
    "lang": "ja"
}

EVALUATION_DIALOGUES = [
    {
        "id": "01_neutral_creator",
        "title": "中立知性 · 创作者的觉悟",
        "category": "中立独白",
        "text": "物語の結末を決めるのは、作者の特権であり、義務よ。倫理君、覚悟はできている？",
        "translation": "决定故事结局是作者的特权，也是义务。伦理君，你做好心理准备了吗？",
        "ref": REF_NEUTRAL,
        "temperature": 0.8,
        "top_p": 0.95,
        "note": "诗羽学姐标志性知性作家声线，检验冷静叙述与语调沉稳感"
    },
    {
        "id": "02_intimate_whisper",
        "title": "亲密耳语 · 伦理君的心跳",
        "category": "亲密细腻",
        "text": "ふふっ、そんなに私の顔をじっと見つめて……何か言いたいことでもあるの、倫理君？",
        "translation": "呵呵，这么目不转睛地看着我的脸……是有什么想说的吗，伦理君？",
        "ref": REF_INTIMATE,
        "temperature": 0.8,
        "top_p": 0.95,
        "note": "近距离诱惑低语，检验极细腻气音与挑逗尾音"
    },
    {
        "id": "03_shy_tsundere",
        "title": "害羞傲娇 · 不是为你写的",
        "category": "害羞微嗔",
        "text": "べ、別にあなたに読んでもらいたくて書いたんじゃないわよ。勘違いしないでちょうだい。",
        "translation": "别、可不是为了想让你读才写的。可别自作多情了。",
        "ref": REF_SHY,
        "temperature": 0.85,
        "top_p": 0.95,
        "note": "轻度傲娇结巴与掩饰心绪，检验语气急促转换"
    },
    {
        "id": "04_annoyed_deadlock",
        "title": "不满嗔怒 · 截稿前的毒虫",
        "category": "不满斥责",
        "text": "締め切り直前になってプロットを全部白紙に戻すなんて……本気で言っているのかしら、この毒虫。",
        "translation": "临近截稿日居然把大纲全推翻重来……你这家伙是认真的吗，毒虫。",
        "ref": REF_ANNOYED,
        "temperature": 0.85,
        "top_p": 0.95,
        "note": "标志性毒舌昵称『毒虫』，检验冷酷压迫与嘲弄语调"
    },
    {
        "id": "05_sad_farewell",
        "title": "悲伤深情 · 再见了我的初恋",
        "category": "悲伤哀婉",
        "text": "さようなら、私の初恋……もう二度と、あの頃の二人には戻れないのね。",
        "translation": "再见了，我的初恋……我们已经再也回不到那个时候了吧。",
        "ref": REF_SAD,
        "temperature": 0.75,
        "top_p": 0.95,
        "note": "催泪经典独白，检验微颤泣声与哀婉拖音"
    }
]


def main():
    print("=" * 76)
    print("📚 霞之丘诗羽 (Kasumigaoka Utaha) GPT-SoVITS v4 评测语音生成器")
    print(f"📦 模型目录: {MODEL_DIR}")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    print(f"🎯 评测样条数: {len(EVALUATION_DIALOGUES)} 条")
    print("=" * 76)

    # 1. Init pipeline
    cfg = TTS_Config()
    cfg.bert_base_path = os.path.join(SOVITS_CORE, "pretrained_models/chinese-roberta-wwm-ext-large")
    cfg.cnhuhbert_base_path = os.path.join(SOVITS_CORE, "pretrained_models/chinese-hubert-base")
    cfg.t2s_weights_path = os.path.join(MODEL_DIR, "霞之丘诗羽_01-e10.ckpt")
    cfg.vits_weights_path = os.path.join(MODEL_DIR, "霞之丘诗羽_01_e16_s368_l32.pth")
    cfg.device = "cpu"
    cfg.is_half = False

    print("\n[1/3] ⚙️  加载霞之丘诗羽 GPT-SoVITS v4 模型...")
    t0_load = time.time()
    tts = TTS(cfg)
    print(f"✅ 模型加载完成! (耗时 {time.time() - t0_load:.2f}s)\n")

    # 2. Inference Loop
    print("[2/3] 🎙️  开始批量执行霞之丘诗羽高保真日文语音合成...")
    generated_records = []

    for idx, item in enumerate(EVALUATION_DIALOGUES, 1):
        dialogue_id = item["id"]
        out_filename = f"{dialogue_id}.wav"
        out_filepath = os.path.join(OUTPUT_DIR, out_filename)
        rel_filepath = f"assets/audio/utaha_samples/v4/{out_filename}"

        print(f"\n[{idx}/{len(EVALUATION_DIALOGUES)}] 🎵 生成样音: {item['title']} ({item['category']})")
        print(f"   日文台词: 「{item['text']}」")
        print(f"   中文释义: {item['translation']}")
        print(f"   母带基准: {os.path.basename(item['ref']['path'])} | 温度: {item['temperature']}")

        req = {
            "text": item["text"],
            "text_lang": "ja",
            "ref_audio_path": item["ref"]["path"],
            "prompt_text": item["ref"]["text"],
            "prompt_lang": item["ref"]["lang"],
            "top_k": 5,
            "top_p": item["top_p"],
            "temperature": item["temperature"],
            "text_split_method": "cut5",
            "sample_steps": 32,
            "speed_factor": 1.0
        }

        t0_infer = time.time()
        gen = tts.run(req)
        sr, chunk = next(gen)
        infer_duration = time.time() - t0_infer

        # Peak normalization to -1.0 dBFS
        if chunk.dtype == np.int16:
            chunk_f = chunk.astype(np.float32) / 32768.0
        else:
            chunk_f = chunk.astype(np.float32)

        peak = np.max(np.abs(chunk_f))
        if peak > 0:
            chunk_f = (chunk_f / peak) * 0.8912

        sf.write(out_filepath, chunk_f, sr, subtype="PCM_16")
        audio_len = len(chunk_f) / sr
        file_size_kb = os.path.getsize(out_filepath) / 1024

        print(f"   ✅ 合成成功! 音频时长: {audio_len:.2f}s | 推理耗时: {infer_duration:.2f}s | 文件大小: {file_size_kb:.1f} KB")

        generated_records.append({
            "id": dialogue_id,
            "title": item["title"],
            "category": item["category"],
            "text": item["text"],
            "translation": item["translation"],
            "ref_file": os.path.basename(item["ref"]["path"]),
            "audio_file": rel_filepath,
            "duration": round(audio_len, 2),
            "infer_time": round(infer_duration, 2),
            "size_kb": round(file_size_kb, 1),
            "note": item["note"]
        })

    # 3. Save Summary Metadata
    summary_path = os.path.join(OUTPUT_DIR, "evaluation_report.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "character": "霞之丘诗羽 (Kasumigaoka Utaha)",
            "version": "v4",
            "architecture": "GPT-SoVITS-v4 (LoRA + DPO)",
            "cv": "茅野爱衣 (Ai Kayano)",
            "total_samples": len(generated_records),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "records": generated_records
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 76)
    print(f"🎉 全部 5 条评测样本生成完毕! 报告已保存至: {summary_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()
