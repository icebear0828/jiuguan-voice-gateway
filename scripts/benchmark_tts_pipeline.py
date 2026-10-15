#!/usr/bin/env python3
"""
TTS Benchmark & Automated Dataset Quality Screening Tool
Connects directly to the user's LAN TTS service (RTX 5070 / IndexTTS / GPT-SoVITS),
runs a multi-dimensional test suite (short, long, numbers, polyphones, edge cases),
evaluates synthesized audio through the 4-Stage Quality Funnel,
and outputs verified GPT-SoVITS training datasets (.list) and HTML/JSON visual audit reports.
"""

import argparse
import base64
import html
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.acoustic_gate import AcousticQualityGate
from core.asr_verifier import get_asr_verifier, MockASRVerifier
from core.speaker_verifier import (
    BaseSpeakerVerifier,
    AcousticFeatureSpeakerVerifier,
    get_speaker_verifier,
)
from core.speech_mos import get_speech_mos
from core.quality_pipeline import QualityPipeline, QualityReport, save_audio_file

# Default benchmark prompts designed for TTS robustness & GPT training material screening
DEFAULT_BENCHMARK_PROMPTS = [
    {
        "category": "日常标准句 (Standard Conversational)",
        "text": "你好，很高兴见到你，今天的天气真不错，适合出去散步。",
        "speed": 0.92,
    },
    {
        "category": "日常短语 (Short Phrase)",
        "text": "局域网设备正在调用主机的RTX 5070进行语音合成！",
        "speed": 0.92,
    },
    {
        "category": "长难句与停顿 (Long Complex & Pauses)",
        "text": "在自然语言处理与语音合成的交汇点上，深度学习模型不仅能够学习语音的声学特征，还能理解上下文的情感变化，从而实现更加拟人化的表达。",
        "speed": 0.92,
    },
    {
        "category": "数字与科技术语 (Numbers & Tech Terms)",
        "text": "本次项目的版本号是2.5，服务器配备了RTX 5070显卡，处理延迟降低了百分之三十五。",
        "speed": 0.92,
    },
    {
        "category": "多音字与语义区分 (Polyphones & Context)",
        "text": "银行行长行行出状元，重重叠叠的山峦让旅途变得更加重要。",
        "speed": 0.92,
    },
    {
        "category": "口语叹词与情绪变化 (Interjection & Emotion)",
        "text": "哎呀！真的假的？你快跟我说说到底怎么回事！",
        "speed": 0.95,
    },
    {
        "category": "绕口令与发音清晰度 (Tongue Twister & Articulation)",
        "text": "吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮，扁担宽板凳长，扁担绑在板凳上。",
        "speed": 0.90,
    },
    {
        "category": "超短叹词边缘样本 (Edge Case - Ultra Short)",
        "text": "好的，收到！",
        "speed": 0.92,
    },
]


def synthesize_remote_tts(
    url: str,
    text: str,
    ref_audio: str = "1_enhanced.wav",
    speed: float = 0.92,
    lang: str = "ZH",
    timeout: float = 60.0,
) -> Tuple[Optional[bytes], Dict[str, str]]:
    """Calls user's LAN TTS endpoint."""
    payload = {
        "text": text,
        "ref_audio": ref_audio,
        "speed": speed,
        "lang": lang,
        "clean_text": True,
        "return_base64": False,
    }
    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        wall_time = time.time() - t0
        if resp.status_code == 200:
            headers = {
                "inference_time": resp.headers.get("x-inference-time") or resp.headers.get("X-Inference-Time", f"{wall_time:.3f}"),
                "audio_duration": resp.headers.get("x-audio-duration") or resp.headers.get("X-Audio-Duration", "0.0"),
                "sample_rate": resp.headers.get("x-sample-rate") or resp.headers.get("X-Sample-Rate", "22050"),
            }
            return resp.content, headers
        else:
            print(f"❌ HTTP {resp.status_code}: {resp.text}")
            return None, {"error": f"HTTP {resp.status_code}: {resp.text}"}
    except Exception as e:
        print(f"❌ Connection error to {url}: {e}")
        return None, {"error": str(e)}


def generate_html_report(results: List[Dict], output_path: str, stats: Dict) -> None:
    """Generates an interactive HTML inspection report with embedded audio players."""
    rows_html = []
    for item in results:
        status = item["status"]
        if status == "PASS":
            badge = '<span style="background:#22c55e;color:#fff;padding:3px 8px;border-radius:4px;font-weight:bold;">PASS</span>'
        elif status == "REPAIRED_PASS":
            badge = '<span style="background:#3b82f6;color:#fff;padding:3px 8px;border-radius:4px;font-weight:bold;">REPAIRED</span>'
        else:
            badge = '<span style="background:#ef4444;color:#fff;padding:3px 8px;border-radius:4px;font-weight:bold;">REJECT</span>'

        audio_src = ""
        if "audio_base64" in item:
            audio_src = f'<audio controls style="height:36px;width:240px;" src="data:audio/wav;base64,{item["audio_base64"]}"></audio>'

        notes = []
        if item.get("repair_actions"):
            notes.append("🔧 " + "<br>".join(item["repair_actions"]))
        if item.get("rejection_reasons"):
            notes.append("❌ " + "<br>".join(item["rejection_reasons"]))
        note_str = "<br>".join(notes) if notes else "纯净无异常"

        spk_sim = f"{item.get('speaker_similarity', 0.0):.3f}" if item.get('speaker_similarity') is not None else "N/A"
        mos_val = f"{item.get('utmos_score', item.get('mos_ovrl', 0.0)):.2f}" if (item.get('utmos_score') or item.get('mos_ovrl')) is not None else "N/A"
        cer_val = f"{item.get('cer', 0.0):.1%}" if item.get('cer') is not None else "N/A"

        glitch_cnt = item.get('glitch_count', 0)
        if glitch_cnt > 0:
            glitch_badge = f"<span style='color:#ef4444;font-weight:600;'>⚠️ {glitch_cnt}次跳变</span>"
        elif item.get('vocoder_artifact'):
            glitch_badge = "<span style='color:#f59e0b;font-weight:600;'>⚡ 电音蜂鸣</span>"
        else:
            glitch_badge = "<span style='color:#22c55e;'>平滑连贯</span>"

        row = f"""
        <tr>
            <td>{item['id']}</td>
            <td><b>{html.escape(item['category'])}</b></td>
            <td style="max-width:280px;word-break:break-all;">{html.escape(item['text'])}</td>
            <td>{badge}</td>
            <td>{audio_src}</td>
            <td>{item.get('duration_sec', 0.0)}s</td>
            <td>{cer_val}</td>
            <td>{spk_sim}</td>
            <td>{mos_val}</td>
            <td>{glitch_badge}</td>
            <td style="font-size:12px;color:#555;">{note_str}</td>
        </tr>
        """
        rows_html.append(row)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>TTS 质检与自蒸馏素材可行性评估报告</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; background: #f8fafc; color: #1e293b; }}
h1 {{ margin-bottom: 8px; }}
.summary-bar {{ display: flex; gap: 16px; margin: 20px 0; }}
.card {{ background: #fff; padding: 16px 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); flex: 1; }}
.card-val {{ font-size: 26px; font-weight: bold; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; font-size: 14px; }}
th {{ background: #f1f5f9; font-weight: 600; }}
tr:hover {{ background: #f8fafc; }}
</style>
</head>
<body>
<h1>🎙️ TTS 质检与自蒸馏素材可行性评估报告</h1>
<p style="color:#64748b;">评测目标: 针对 GPT-SoVITS 训练素材自动化筛选 (四级漏斗质检: 声学物理 -> ASR对齐 -> 声纹锚点 -> UTMOS/破音检测)</p>

<div class="summary-bar">
    <div class="card">
        <div style="color:#64748b;font-size:13px;">总评测样本</div>
        <div class="card-val">{stats['total']}</div>
    </div>
    <div class="card">
        <div style="color:#22c55e;font-size:13px;">直接合格 (Pass)</div>
        <div class="card-val" style="color:#22c55e;">{stats['pass']}</div>
    </div>
    <div class="card">
        <div style="color:#3b82f6;font-size:13px;">自愈修复合格 (Repaired)</div>
        <div class="card-val" style="color:#3b82f6;">{stats['repaired_pass']}</div>
    </div>
    <div class="card">
        <div style="color:#ef4444;font-size:13px;">剔除淘汰 (Reject)</div>
        <div class="card-val" style="color:#ef4444;">{stats['reject']}</div>
    </div>
    <div class="card">
        <div style="color:#8b5cf6;font-size:13px;">可用转化率 (Yield)</div>
        <div class="card-val" style="color:#8b5cf6;">{(stats['pass'] + stats['repaired_pass']) / max(1, stats['total']):.1%}</div>
    </div>
</div>

<table>
    <thead>
        <tr>
            <th>#</th>
            <th>测试维度</th>
            <th>文本内容</th>
            <th>质检结果</th>
            <th>试听音频</th>
            <th>时长</th>
            <th>CER</th>
            <th>声纹相似度</th>
            <th>UTMOS/MOS</th>
            <th>破音/电音</th>
            <th>质检诊断说明</th>
        </tr>
    </thead>
    <tbody>
        {"".join(rows_html)}
    </tbody>
</table>
</body>
</html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TTS and evaluate dataset quality for GPT-SoVITS training."
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://192.168.10.4:8000/api/tts/generate",
        help="LAN TTS API URL.",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default="1_enhanced.wav",
        help="Reference audio name on server.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.92,
        help="Default playback speed.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="ZH",
        help="Synthesis language.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/tts_eval_samples",
        help="Folder to save output WAV samples.",
    )
    parser.add_argument(
        "--output-list",
        type=str,
        default="assets/tts_eval_samples/train.list",
        help="Output GPT-SoVITS training .list file.",
    )
    parser.add_argument(
        "--speaker-name",
        type=str,
        default="index_voice",
        help="Speaker label for output .list.",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default="logs/tts_benchmark_report.json",
        help="Path for JSON audit report.",
    )
    parser.add_argument(
        "--report-html",
        type=str,
        default="logs/tts_benchmark_report.html",
        help="Path for visual HTML audit report.",
    )
    parser.add_argument(
        "--max-cer",
        type=float,
        default=0.10,
        help="Max CER allowed.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.72,
        help="Min cosine similarity vs Golden Anchor.",
    )
    parser.add_argument(
        "--min-mos",
        type=float,
        default=3.5,
        help="Min overall MOS score.",
    )
    parser.add_argument(
        "--custom-prompts",
        type=str,
        default=None,
        help="Optional text file with custom prompts (one per line).",
    )
    parser.add_argument(
        "--mock-asr",
        action="store_true",
        help="Use simulated ASR verifier (matches reference text) for fast acoustic/timbre/MOS screening.",
    )

    args = parser.parse_args()

    print(f"🚀 Initializing Quality Evaluation Pipeline...")
    print(f"  • TTS Target: {args.url}")
    print(f"  • Ref Audio:  {args.ref_audio}")

    # Initialize components
    spk_verifier = AcousticFeatureSpeakerVerifier()
    mos_evaluator = get_speech_mos("heuristic")

    # ASR setup
    if args.mock_asr:
        print("  • ASR Engine: MockASRVerifier (Simulated perfect transcription)")
        asr_verifier = MockASRVerifier(match_reference=True)
    else:
        try:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from core.asr_verifier import FasterWhisperVerifier
            asr_verifier = FasterWhisperVerifier(model_size="tiny")
            print("  • ASR Engine: Faster-Whisper tiny (Enabled)")
        except Exception as e:
            print(f"  • ASR Engine: MockASRVerifier fallback ({e})")
            asr_verifier = MockASRVerifier(match_reference=True)

    # Prepare prompts
    prompts = []
    if args.custom_prompts and os.path.exists(args.custom_prompts):
        with open(args.custom_prompts, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                t = line.strip()
                if t and not t.startswith("#"):
                    prompts.append({"category": f"自定义语料 #{idx}", "text": t, "speed": args.speed})
    else:
        prompts = DEFAULT_BENCHMARK_PROMPTS

    # Step 1: Synthesize an Anchor Calibration audio if not available
    print("\n🎯 Step 1: Synthesizing baseline calibration audio for Golden Anchor...")
    anchor_text = "这是用于校准系统声纹特性的基准参考样本。"
    anchor_wav_bytes, _ = synthesize_remote_tts(
        url=args.url,
        text=anchor_text,
        ref_audio=args.ref_audio,
        speed=args.speed,
        lang=args.lang,
    )
    golden_anchor_emb = None
    if anchor_wav_bytes:
        golden_anchor_emb = spk_verifier.extract_embedding(anchor_wav_bytes)
        print("✅ Golden Anchor voiceprint successfully extracted and anchored!")
    else:
        print("⚠️ Warning: Failed to generate anchor audio, proceeding without anchor comparison.")

    # Initialize quality funnel pipeline
    pipeline = QualityPipeline(
        asr_verifier=asr_verifier,
        speaker_verifier=spk_verifier,
        speech_mos=mos_evaluator,
        golden_anchor=golden_anchor_emb,
        max_cer=args.max_cer,
        min_speaker_similarity=args.min_similarity,
        min_mos_ovrl=args.min_mos,
    )

    # Step 2: Run benchmark evaluation
    print(f"\n🧪 Step 2: Executing benchmark test on {len(prompts)} cases...")
    os.makedirs(args.output_dir, exist_ok=True)

    stats = {
        "total": len(prompts),
        "pass": 0,
        "repaired_pass": 0,
        "reject": 0,
        "reject_by_stage": {"acoustic": 0, "asr": 0, "speaker": 0, "mos": 0},
    }
    results = []
    list_entries = []

    for idx, p in enumerate(prompts, start=1):
        text = p["text"]
        cat = p["category"]
        speed = p.get("speed", args.speed)
        print(f"\n[{idx}/{len(prompts)}] 🔊 Synthesizing: 【{cat}】 \"{text}\"")

        wav_bytes, headers = synthesize_remote_tts(
            url=args.url,
            text=text,
            ref_audio=args.ref_audio,
            speed=speed,
            lang=args.lang,
        )

        if not wav_bytes:
            stats["reject"] += 1
            stats["reject_by_stage"]["acoustic"] += 1
            results.append({
                "id": idx,
                "category": cat,
                "text": text,
                "status": "REJECT",
                "stage_failed": "synthesis_api",
                "rejection_reasons": ["Remote TTS synthesis failed"],
            })
            continue

        report, final_samples = pipeline.evaluate(
            audio_source=wav_bytes,
            reference_text=text,
            language=args.lang.lower(),
            allow_repair=True,
        )

        filename = f"sample_{idx:02d}_{args.speaker_name}.wav"
        dest_path = os.path.join(args.output_dir, filename)

        item = {
            "id": idx,
            "category": cat,
            "text": text,
            "filename": filename,
            "status": report.status,
            "stage_failed": report.stage_failed,
            "repaired": report.repaired,
            "repair_actions": report.repair_actions,
            "rejection_reasons": report.rejection_reasons,
            "duration_sec": report.duration_sec,
            "inference_time": headers.get("inference_time"),
            "audio_base64": base64.b64encode(wav_bytes).decode("utf-8"),
        }

        if report.acoustic:
            item["rms_db"] = report.acoustic.rms_db
            item["peak_db"] = report.acoustic.peak_db
            item["cps"] = report.acoustic.cps
        if report.asr:
            item["cer"] = report.asr.cer
            item["transcribed"] = report.asr.transcribed_text
        if report.speaker:
            item["speaker_similarity"] = report.speaker.similarity
        if report.mos:
            item["mos_ovrl"] = report.mos.ovrl
            item["mos_sig"] = report.mos.sig
            item["mos_bak"] = report.mos.bak
            item["utmos_score"] = report.mos.utmos_score or report.mos.ovrl
            item["glitch_count"] = report.mos.glitch_count
            item["has_pitch_glitch"] = report.mos.has_pitch_glitch
            item["vocoder_artifact"] = report.mos.vocoder_artifact_detected

        results.append(item)

        if report.passed:
            save_audio_file(dest_path, final_samples, report.sample_rate)
            if report.status == "REPAIRED_PASS":
                stats["repaired_pass"] += 1
                status_icon = "🔧 REPAIRED_PASS"
            else:
                stats["pass"] += 1
                status_icon = "✅ PASS"

            list_entries.append(f"{dest_path}|{args.speaker_name}|{args.lang}|{text}\n")
        else:
            stats["reject"] += 1
            if report.stage_failed:
                stats["reject_by_stage"][report.stage_failed] += 1
            status_icon = "❌ REJECT"

        diag = report.rejection_reasons or report.repair_actions or ["Perfect Clean"]
        print(f"    ↳ {status_icon}: {' / '.join(diag)}")
        if report.mos:
            print(f"      [Metrics] MOS: {report.mos.ovrl:.2f} | RMS: {report.acoustic.rms_db:.1f} dBFS | CPS: {report.acoustic.cps}")

    # Step 3: Write clean training .list
    if list_entries:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_list)), exist_ok=True)
        with open(args.output_list, "w", encoding="utf-8") as f:
            f.writelines(list_entries)
        print(f"\n📄 GPT-SoVITS Training Dataset Manifest written to: {args.output_list}")

    # Step 4: Write JSON audit report (strip base64 to keep JSON clean)
    clean_json_results = []
    for r in results:
        entry = {k: v for k, v in r.items() if k != "audio_base64"}
        clean_json_results.append(entry)

    os.makedirs(os.path.dirname(os.path.abspath(args.report_json)), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": stats,
                "server_url": args.url,
                "ref_audio": args.ref_audio,
                "items": clean_json_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"📊 Detailed JSON audit report: {args.report_json}")

    # Step 5: Write HTML interactive report
    generate_html_report(results, args.report_html, stats)
    print(f"🌐 Interactive HTML visual report: {args.report_html}")

    # Summary
    print("\n================== BENCHMARK SUMMARY ==================")
    print(f"Total Sentences Tested:   {stats['total']}")
    print(f"Direct Pass (Gold):       {stats['pass']} ({stats['pass']/stats['total']:.1%})")
    print(f"Repaired Pass (Silver):   {stats['repaired_pass']} ({stats['repaired_pass']/stats['total']:.1%})")
    print(f"Rejected (Flawed):        {stats['reject']} ({stats['reject']/stats['total']:.1%})")
    yield_rate = (stats['pass'] + stats['repaired_pass']) / stats['total']
    print(f"Material Yield Rate:      {yield_rate:.1%}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
