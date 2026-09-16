#!/usr/bin/env python3
"""
Rimuru Tempest (利姆鲁·特恩佩斯特) GPT-SoVITS v2Pro Batch Audio Generator & Evaluator
Generates 8 high-fidelity Japanese voice evaluation samples with peak normalization (-1.5 dBFS).
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

MODEL_DIR = os.path.join(WORKSPACE_DIR, "assets/models/rimuru/v2pro_master")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "assets/audio/rimuru_samples/v2pro_master")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REF_DAILY = {
    "path": os.path.join(MODEL_DIR, "reference_audio/01_Master_Daily.wav"),
    "text": "自己紹介がまだだったな 俺はリムル＝テンペスト",
    "lang": "ja"
}

REF_CONFIDENT = {
    "path": os.path.join(MODEL_DIR, "reference_audio/02_Master_Confident.wav"),
    "text": "その名にふさわしく、俺の役に立ってくれ！",
    "lang": "ja"
}

EVALUATION_DIALOGUES = [
    {
        "id": "01_greeting_daily",
        "title": "日常问候 · 从容和缓",
        "category": "日常基准",
        "text": "やあ！調子はどうだ？今日もテンペストは平和で、いい日になりそうだね。",
        "translation": "嗨！最近怎么样？今天魔国联邦也很和平，看起来会是个好日子呢。",
        "ref": REF_DAILY,
        "temperature": 0.70,
        "note": "检验日常基准声线、发音平稳度与空气感"
    },
    {
        "id": "02_classic_quote",
        "title": "经典台词 · 俏皮活力",
        "category": "萌王名台词",
        "text": "俺はスライムのリムル！悪いスライムじゃないよ！……って、前にも言ったっけ？わはは！",
        "translation": "我是史莱姆利姆鲁！不是坏史莱姆哦！……哎，我之前是不是也说过？哇哈哈！",
        "ref": REF_DAILY,
        "temperature": 0.72,
        "note": "检验高频轻快少女/少年中性音色与笑声语调"
    },
    {
        "id": "03_covenant_leader",
        "title": "盟主承诺 · 温柔坚定",
        "category": "情感承诺",
        "text": "仲間たちと共に、誰もが笑って暮らせる国を必ず作ってみせるさ。",
        "translation": "与同伴们一起，我一定会建立一个让所有人都能笑着生活下去的国家。",
        "ref": REF_DAILY,
        "temperature": 0.68,
        "note": "检验中低速舒缓发音、情感温度与咬字清晰度"
    },
    {
        "id": "04_combat_wrath",
        "title": "战斗威严 · 霸气护短",
        "category": "真魔王战斗",
        "text": "俺の仲間に手を出したこと、後悔させてやる！そこから一歩も動くなよ！",
        "translation": "敢对我的伙伴出手，我会让你们后悔的！在那给我一步都别动！",
        "ref": REF_CONFIDENT,
        "temperature": 0.70,
        "note": "切换威严母带Prompt，检验爆发力、压迫感与破音抑制"
    },
    {
        "id": "05_governance_rational",
        "title": "政务决断 · 理性商讨",
        "category": "内政商榷",
        "text": "よし、ドワーフ王国との交易ルートを再確認しよう。みんなの意見も聞かせてくれ。",
        "translation": "好，重新确认一下与矮人王国的贸易路线吧。大家也把意见都说来听听。",
        "ref": REF_CONFIDENT,
        "temperature": 0.68,
        "note": "检验政务理性从容声线、外来语词汇（ドワーフ）发音自然度"
    },
    {
        "id": "06_humor_slacking",
        "title": "幽默摸鱼 · 搞笑吐槽",
        "category": "日常幽默",
        "text": "ふぁ〜あ……たまには仕事をサボって、シオンの作った料理……いや、それはパスで！温泉に行こう！",
        "translation": "哈欠……偶尔翘掉工作吃吃紫苑做的料理……不对，那个还是免了吧！我们去泡温泉吧！",
        "ref": REF_DAILY,
        "temperature": 0.72,
        "note": "检验叹气（ふぁ〜あ）、紧急转折（いや、それはパスで）等丰富戏剧语气"
    },
    {
        "id": "07_emotional_memory",
        "title": "深情羁绊 · 静小姐回忆",
        "category": "深情轻语",
        "text": "シズさん、俺はちゃんとこの世界で、みんなと一緒に前に進んでるよ。",
        "translation": "静小姐，我确实在这个世界上，和大家一起好好地前进着呢。",
        "ref": REF_DAILY,
        "temperature": 0.65,
        "note": "检验耳语般的柔和低音、气声表现与细腻共鸣"
    },
    {
        "id": "08_long_speech",
        "title": "长篇剧情演说 · 联邦国宣言",
        "category": "宏大演说",
        "text": "ジュラ・テンペスト連邦国は、魔物も人間も関係なく、互いを尊重し合える場所だ。俺達の歩みは、決して止まらない！",
        "translation": "鸠拉·特恩佩斯特联邦国，是不论魔物还是人类，都能互相尊重的乐土。我们的脚步，绝不会停下！",
        "ref": REF_DAILY,
        "temperature": 0.70,
        "note": "长句复合意群切分（cut0/分句），检验长句稳定性与连贯气势"
    }
]

def main():
    print("=" * 76)
    print("👑 利姆鲁·特恩佩斯特 (Rimuru) GPT-SoVITS v2Pro 评测语音生成器")
    print(f"📦 模型目录: {MODEL_DIR}")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    print(f"🎯 评测样条数: {len(EVALUATION_DIALOGUES)} 条")
    print("=" * 76)

    # 1. Init pipeline
    cfg = TTS_Config()
    cfg.bert_base_path = os.path.join(SOVITS_CORE, "pretrained_models/chinese-roberta-wwm-ext-large")
    cfg.cnhuhbert_base_path = os.path.join(SOVITS_CORE, "pretrained_models/chinese-hubert-base")
    cfg.t2s_weights_path = os.path.join(MODEL_DIR, "Rimuru_v2Pro_Native-e10.ckpt")
    cfg.vits_weights_path = os.path.join(MODEL_DIR, "Rimuru_v2Pro_e25_s6825.pth")
    cfg.device = "cpu"
    cfg.is_half = False

    print("\n[1/3] ⚙️  加载 GPT-SoVITS v2Pro 原生神经网络模型...")
    t0_load = time.time()
    tts = TTS(cfg)
    print(f"✅ 模型加载完成! (耗时 {time.time() - t0_load:.2f}s)\n")

    # 2. Inference Loop
    print("[2/3] 🎙️  开始批量执行高保真日文语音合成...")
    generated_records = []

    for idx, item in enumerate(EVALUATION_DIALOGUES, 1):
        dialogue_id = item["id"]
        out_filename = f"{dialogue_id}.wav"
        out_filepath = os.path.join(OUTPUT_DIR, out_filename)
        rel_filepath = f"assets/audio/rimuru_samples/v2pro_master/{out_filename}"

        print(f"\n[{idx}/{len(EVALUATION_DIALOGUES)}] 🎵 生成样音: {item['title']} ({item['category']})")
        print(f"   日文台词: 「{item['text']}」")
        print(f"   中文释义: {item['translation']}")
        print(f"   母带基准: {os.path.basename(item['ref']['path'])} | 采样温度: {item['temperature']}")

        req = {
            "text": item["text"],
            "text_lang": "ja",
            "ref_audio_path": item["ref"]["path"],
            "prompt_text": item["ref"]["text"],
            "prompt_lang": item["ref"]["lang"],
            "top_k": 5,
            "top_p": 0.95,
            "temperature": item["temperature"],
            "text_split_method": "cut0",
            "sample_steps": 32,
            "speed_factor": 1.0
        }

        t0_infer = time.time()
        gen = tts.run(req)
        sr, chunk = next(gen)
        infer_duration = time.time() - t0_infer

        # Peak normalization to -1.5 dBFS (factor = 10^(-1.5/20) ~ 0.8414)
        if chunk.dtype == np.int16:
            chunk_f = chunk.astype(np.float32) / 32768.0
        else:
            chunk_f = chunk.astype(np.float32)

        peak = np.max(np.abs(chunk_f))
        if peak > 0:
            chunk_f = (chunk_f / peak) * 0.8414

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
            "model_name": "Rimuru Tempest (利姆鲁·特恩佩斯特)",
            "model_version": "v2pro_master",
            "architecture": "GPT-SoVITS-v2Pro",
            "sample_rate": 32000,
            "total_samples": len(generated_records),
            "samples": generated_records
        }, f, ensure_ascii=False, indent=2)

    # 3. Generate Evaluation HTML Player
    print("\n[3/3] 📊 生成可视化多音频听评播放器页面...")
    html_content = generate_html_player(generated_records)
    html_path = os.path.join(OUTPUT_DIR, "evaluation_player.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ 评测播放器生成完毕: {html_path}")
    print("=" * 76)
    print("🎉 全部 8 条萌王日文语音评测样音生成完成！")
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
  <title>👑 利姆鲁·特恩佩斯特 (Rimuru) GPT-SoVITS v2Pro 语音效果听评看板</title>
  <style>
    :root {{
      --primary: #3b82f6;
      --bg: #0f172a;
      --card-bg: #1e293b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
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
      background: rgba(59, 130, 246, 0.2);
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.4);
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
      <h1>👑 聖園ミカ & 萌王 (Rimuru) GPT-SoVITS 试听看板</h1>
      <div class="subtitle">模型版本: <b>v2pro_master</b> (1082句录音棚官方母带干声微调 | 10/25轮) · 32kHz 神经网络高保真还原</div>
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
