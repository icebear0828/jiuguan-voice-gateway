#!/usr/bin/env python3
"""
Makise Kurisu (牧瀬紅莉栖 / 助手) GPT-SoVITS v2Pro Batch Audio Generator & Evaluator
Generates 8 high-fidelity Japanese voice evaluation samples across 4 emotions (tsundere, science, gentle, daily).
Produces an interactive evaluation HTML player page and metadata report.
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

MODEL_DIR = os.path.join(WORKSPACE_DIR, "assets/models/kurisu/v2pro_master")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "assets/audio/kurisu_samples/v2pro_master")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REF_TSUNDERE = {
    "path": os.path.join(MODEL_DIR, "reference_audios/tsundere_CRS_0030.wav"),
    "text": "何が真実よ、この変態! 馬鹿なの!? 死ぬの!?",
    "lang": "ja"
}

REF_SCIENCE = {
    "path": os.path.join(MODEL_DIR, "reference_audios/science_CRS_0015.wav"),
    "text": "どうして?私、前にあなたと会ったことありますか?",
    "lang": "ja"
}

REF_GENTLE = {
    "path": os.path.join(MODEL_DIR, "reference_audios/gentle_CRS_0014.wav"),
    "text": "まるで、今にも泣き出しそうで、それにすごく辛そうでした",
    "lang": "ja"
}

REF_DAILY = {
    "path": os.path.join(MODEL_DIR, "reference_audios/daily_CRS_0038.wav"),
    "text": "こういうのは初めてなので緊張してますが、ガチガチなのは多めに見てください。 どうぞよろしく。",
    "lang": "ja"
}

EVALUATION_DIALOGUES = [
    {
        "id": "01_tsundere_coffee",
        "title": "经典傲娇 · 咖啡别误会",
        "category": "傲娇吐槽",
        "text": "べ、別にアンタのために淹れたコーヒーじゃないんだからね！勘違いしないでよ！",
        "translation": "别、别误会了，这可不是专门为你泡的咖啡！可别想歪了啊！",
        "ref": REF_TSUNDERE,
        "temperature": 1.0,
        "top_p": 1.0,
        "note": "助手殿堂级傲娇名台词，检验口是心非的高频语调起伏"
    },
    {
        "id": "02_tsundere_hentai",
        "title": "愤怒反驳 · 谁是助手啊",
        "category": "怒斥吐槽",
        "text": "誰が助手よ！私は牧瀬紅莉栖！クリスティーナでも助手でもないって言ってるでしょ、この変態！",
        "translation": "谁是助手啊！我是牧濑红莉栖！都说了我既不是克里斯蒂娜也不是助手，你这个变态！",
        "ref": REF_TSUNDERE,
        "temperature": 1.0,
        "top_p": 1.0,
        "note": "经典助手称呼反驳，检验语速加快与激烈情绪爆发下的音质清晰度"
    },
    {
        "id": "03_science_paradox",
        "title": "学术论证 · 时间跳跃悖论",
        "category": "知性科学",
        "text": "時間跳躍の可能性を否定する気はないけれど、タイムパラドックスの矛盾をどう説明するつもり？",
        "translation": "虽然我不打算否定时间跳跃的可能性，但你打算如何解释时间悖论的矛盾？",
        "ref": REF_SCIENCE,
        "temperature": 0.85,
        "top_p": 0.95,
        "note": "脑科学家知性论证，检验专业名词平稳咬字与严谨冷静语感"
    },
    {
        "id": "04_gentle_worldline",
        "title": "温柔深情 · 无论哪条世界线",
        "category": "深情羁绊",
        "text": "たとえどの世界線にいたとしても、私はきっと、あなたのことを見つけ出すから。",
        "translation": "无论身处在哪条世界线，我都一定会……找到你的。",
        "ref": REF_GENTLE,
        "temperature": 0.80,
        "top_p": 0.95,
        "note": "命运石之门催泪名台词，检验极度细腻温存的情感与气声表现"
    },
    {
        "id": "05_gentle_confession",
        "title": "动容轻语 · 闭上眼睛笨蛋",
        "category": "耳语催促",
        "text": "目を閉じて……バカ、早く閉じなさいよ。……忘れないで、私はここにいるわ。",
        "translation": "把眼睛闭上……笨蛋，快点闭上啦。……别忘记，我一直都在这里。",
        "ref": REF_GENTLE,
        "temperature": 0.85,
        "top_p": 0.95,
        "note": "高潮剧情经典告白，检验柔和耳语音、句间呼吸停顿与微颤声线"
    },
    {
        "id": "06_daily_lab_banter",
        "title": "Lab日常 · 吐槽中二病",
        "category": "日常杂谈",
        "text": "また怪しげな実験を始めて……ダル、この狂気のマッドサイエンティストをなんとかしなさいよ。",
        "translation": "又开始搞这种可疑的实验了……桶子，你想办法管管这个狂气疯狂科学家啊。",
        "ref": REF_DAILY,
        "temperature": 0.90,
        "top_p": 1.0,
        "note": "未来道具研究所日常互动，检验轻松松弛的日常吐槽语调"
    },
    {
        "id": "07_tsundere_panic",
        "title": "慌乱遮掩 · 别看我的手机",
        "category": "傲娇慌乱",
        "text": "ちょ、ちょっと！私のスマホを勝手に見ないで！別にネットの掲示板なんて見てないし！",
        "translation": "等、等等！别随便看我的手机！我才没有在看什么网络论坛呢！",
        "ref": REF_TSUNDERE,
        "temperature": 1.0,
        "top_p": 1.0,
        "note": "栗悟饭与龟波功暗号梗，检验慌乱结巴与语速急促的戏剧感"
    },
    {
        "id": "08_science_steinsgate",
        "title": "宏大决意 · 命运石之门的选择",
        "category": "宏大决意",
        "text": "因果の檻を突き破り、シュタインズ・ゲートへと到達する……それが、私たちの選択よ。",
        "translation": "突破因果的囚牢，抵达命运石之门……这就是，我们的选择。",
        "ref": REF_SCIENCE,
        "temperature": 0.85,
        "top_p": 0.95,
        "note": "长句坚定决绝气势，检验复杂意群连贯性与片假名发音纯正度"
    }
]

def main():
    print("=" * 76)
    print("🧬 牧濑红莉栖 (Makise Kurisu / 助手) GPT-SoVITS v2Pro 评测语音生成器")
    print(f"📦 模型目录: {MODEL_DIR}")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    print(f"🎯 评测样条数: {len(EVALUATION_DIALOGUES)} 条")
    print("=" * 76)

    # 1. Init pipeline
    cfg = TTS_Config()
    cfg.bert_base_path = os.path.join(SOVITS_CORE, "pretrained_models/chinese-roberta-wwm-ext-large")
    cfg.cnhuhbert_base_path = os.path.join(SOVITS_CORE, "pretrained_models/chinese-hubert-base")
    cfg.t2s_weights_path = os.path.join(MODEL_DIR, "Kurisu-e20.ckpt")
    cfg.vits_weights_path = os.path.join(MODEL_DIR, "Kurisu_e25_s9475.pth")
    cfg.device = "cpu"
    cfg.is_half = False

    print("\n[1/3] ⚙️  加载助手 Kurisu GPT-SoVITS v2Pro 模型...")
    t0_load = time.time()
    tts = TTS(cfg)
    print(f"✅ 模型加载完成! (耗时 {time.time() - t0_load:.2f}s)\n")

    # 2. Inference Loop
    print("[2/3] 🎙️  开始批量执行助手高保真日文语音合成...")
    generated_records = []

    for idx, item in enumerate(EVALUATION_DIALOGUES, 1):
        dialogue_id = item["id"]
        out_filename = f"{dialogue_id}.wav"
        out_filepath = os.path.join(OUTPUT_DIR, out_filename)
        rel_filepath = f"assets/audio/kurisu_samples/v2pro_master/{out_filename}"

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

        # Peak normalization to -1.0 dBFS (factor = 10^(-1.0/20) ~ 0.8912)
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
            "note": item["note"],
            "ref_audio": os.path.basename(item["ref"]["path"]),
            "temperature": item["temperature"],
            "audio_duration_sec": round(audio_len, 2),
            "infer_duration_sec": round(infer_duration, 2),
            "sample_rate": sr,
            "file_name": out_filename,
            "relative_path": rel_filepath,
            "file_size_kb": round(file_size_kb, 1)
        })

    # Save manifest
    manifest_path = os.path.join(OUTPUT_DIR, "index.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "Makise Kurisu (牧濑红莉栖 / 助手)",
            "model_version": "v2pro_master",
            "architecture": "GPT-SoVITS-v2Pro",
            "sample_rate": 32000,
            "total_samples": len(generated_records),
            "samples": generated_records
        }, f, ensure_ascii=False, indent=2)

    # 3. Generate Evaluation HTML Player
    print("\n[3/3] 📊 生成助手多音频听评播放器页面...")
    html_content = generate_html_player(generated_records)
    html_path = os.path.join(OUTPUT_DIR, "evaluation_player.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ 评测播放器生成完毕: {html_path}")
    print("=" * 76)
    print("🎉 全部 8 条助手日文语音评测样音生成完成！")
    print("=" * 76)


def generate_html_player(records):
    rows_html = ""
    for r in records:
        rows_html += f"""
        <div class="sample-card">
          <div class="card-header">
            <span class="badge category">{r['category']}</span>
            <h3 class="sample-title">{r['title']}</h3>
            <span class="badge duration">⏱️ {r['audio_duration_sec']}s</span>
          </div>
          <div class="dialogue-box">
            <div class="jp-text">「{r['text']}」</div>
            <div class="zh-trans">{r['translation']}</div>
          </div>
          <div class="player-wrapper">
            <audio controls preload="metadata" src="{r['file_name']}"></audio>
          </div>
          <div class="meta-footer">
            <span>🎯 评测重点: {r['note']}</span>
            <span>🎛️ 温度: {r['temperature']} | 母带: {r['ref_audio']}</span>
          </div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🧬 牧濑红莉栖 (Makise Kurisu / 助手) GPT-SoVITS 语音效果听评看板</title>
  <style>
    :root {{
      --primary: #ef4444;
      --bg: #0f172a;
      --card-bg: #1e293b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #f87171;
      --border: #334155;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Hiragino Sans GB", sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 24px;
    }}
    .container {{
      max-width: 900px;
      margin: 0 auto;
    }}
    header {{
      text-align: center;
      padding-bottom: 24px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 28px;
    }}
    h1 {{
      margin: 0 0 8px 0;
      color: var(--accent);
      font-size: 26px;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 14px;
    }}
    .sample-grid {{
      display: flex;
      flex-direction: column;
      gap: 18px;
    }}
    .sample-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px 20px;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
    }}
    .card-header {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .sample-title {{
      margin: 0;
      font-size: 17px;
      flex-grow: 1;
    }}
    .badge {{
      font-size: 12px;
      padding: 3px 8px;
      border-radius: 6px;
      font-weight: 600;
    }}
    .badge.category {{
      background: rgba(239, 68, 68, 0.2);
      color: #fca5a5;
      border: 1px solid rgba(239, 68, 68, 0.4);
    }}
    .badge.duration {{
      background: rgba(255, 255, 255, 0.08);
      color: var(--text-muted);
    }}
    .dialogue-box {{
      background: rgba(0,0,0,0.25);
      border-left: 4px solid var(--accent);
      padding: 10px 14px;
      border-radius: 0 8px 8px 0;
      margin-bottom: 12px;
    }}
    .jp-text {{
      font-size: 16px;
      line-height: 1.6;
      font-weight: 500;
      color: #e2e8f0;
    }}
    .zh-trans {{
      font-size: 13px;
      color: var(--text-muted);
      margin-top: 4px;
    }}
    .player-wrapper audio {{
      width: 100%;
      height: 38px;
      border-radius: 8px;
    }}
    .meta-footer {{
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: #64748b;
      margin-top: 8px;
      border-top: 1px dashed var(--border);
      padding-top: 6px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>🧬 牧瀬紅莉栖 (Makise Kurisu / 助手) GPT-SoVITS 试听看板</h1>
      <div class="subtitle">STEINS;GATE · CV: 今井麻美 | 模型版本: <b>v2pro_master</b> (2268句录音棚母带微调 · 20/25轮) · 32kHz 高保真还原</div>
    </header>
    <div class="sample-grid">
      {rows_html}
    </div>
  </div>
</body>
</html>
"""

if __name__ == "__main__":
    main()
