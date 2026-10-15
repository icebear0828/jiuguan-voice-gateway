#!/usr/bin/env python3
"""
Audio Dataset Screening & Self-Distillation Quality Filter CLI
Applies 4-stage funnel (Acoustic -> ASR -> Speaker Anchor -> MOS) with self-healing,
and generates clean GPT-SoVITS training datasets (.list) and detailed audit reports.
"""

import argparse
import json
import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

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


def parse_transcripts(transcript_path: str) -> Dict[str, Tuple[str, str, str]]:
    """
    Parses transcripts file.
    Supports:
    - GPT-SoVITS standard .list: wav_path|speaker_name|language|text
    - Delimited: filename|text or filename: text or filename\ttext
    Returns: dict mapping filename (basename or relpath) -> (speaker_name, language, text)
    """
    mapping = {}
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcripts file not found: {transcript_path}")

    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "|" in line:
                parts = line.split("|")
                if len(parts) >= 4:
                    wav_name = os.path.basename(parts[0].strip())
                    spk = parts[1].strip()
                    lang = parts[2].strip()
                    text = parts[3].strip()
                    mapping[wav_name] = (spk, lang, text)
                    mapping[parts[0].strip()] = (spk, lang, text)
                elif len(parts) == 2:
                    wav_name = os.path.basename(parts[0].strip())
                    mapping[wav_name] = ("", "", parts[1].strip())
                    mapping[parts[0].strip()] = ("", "", parts[1].strip())
            elif "\t" in line:
                parts = line.split("\t", 1)
                wav_name = os.path.basename(parts[0].strip())
                mapping[wav_name] = ("", "", parts[1].strip())
            elif ":" in line:
                parts = line.split(":", 1)
                wav_name = os.path.basename(parts[0].strip())
                mapping[wav_name] = ("", "", parts[1].strip())

    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Filter and curate audio datasets for GPT-SoVITS training."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing source audio .wav files.",
    )
    parser.add_argument(
        "--transcripts",
        type=str,
        default=None,
        help="Path to transcript file (.list, .txt, or .csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/cleaned_dataset",
        help="Destination directory for passed/repaired audio files.",
    )
    parser.add_argument(
        "--output-list",
        type=str,
        default=None,
        help="Path to output .list file formatted for GPT-SoVITS.",
    )
    parser.add_argument(
        "--rejected-list",
        type=str,
        default=None,
        help="Path to rejected .list file recording flawed audio samples.",
    )
    parser.add_argument(
        "--golden-anchor",
        type=str,
        default=None,
        help="Path to .npy speaker embedding anchor file or directory of pristine human wavs.",
    )
    parser.add_argument(
        "--build-anchor-from",
        type=str,
        default=None,
        help="Directory of pristine human WAVs to build and save a golden anchor .npy.",
    )
    parser.add_argument(
        "--save-anchor-to",
        type=str,
        default="assets/golden_anchor.npy",
        help="Output path when --build-anchor-from is specified.",
    )
    parser.add_argument(
        "--speaker-name",
        type=str,
        default="custom_voice",
        help="Default speaker name for .list output.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ja",
        help="Default language code ('ja', 'zh', 'en').",
    )
    parser.add_argument(
        "--max-cer",
        type=float,
        default=0.08,
        help="Max Character Error Rate allowed (default: 0.08).",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.78,
        help="Min cosine similarity vs Golden Anchor (default: 0.78).",
    )
    parser.add_argument(
        "--min-mos-ovrl",
        type=float,
        default=3.8,
        help="Min overall MOS score (default: 3.8).",
    )
    parser.add_argument(
        "--no-repair",
        action="store_true",
        help="Disable auto-healing (rejects flawed samples instead of repairing).",
    )
    parser.add_argument(
        "--mock-asr",
        action="store_true",
        help="Use mock ASR verifier for testing/CI without GPU/Faster-Whisper.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="logs/dataset_filter_report.json",
        help="Path to save audit report JSON.",
    )

    args = parser.parse_args()

    # 1. Initialize Verifiers
    spk_verifier = get_speaker_verifier("acoustic_feature")
    mos_evaluator = get_speech_mos("heuristic")

    if args.mock_asr:
        asr_verifier = MockASRVerifier(match_reference=True)
    else:
        try:
            asr_verifier = get_asr_verifier("auto")
        except Exception as e:
            print(f"⚠️ ASR verifier not available ({e}), falling back to mock ASR.")
            asr_verifier = MockASRVerifier(match_reference=True)

    # 2. Build or load Golden Anchor
    anchor_emb = None
    if args.build_anchor_from:
        anchor_wavs = glob.glob(os.path.join(args.build_anchor_from, "*.wav"))
        if not anchor_wavs:
            print(f"❌ No .wav files found in --build-anchor-from: {args.build_anchor_from}")
            sys.exit(1)
        print(f"🎙️ Building golden anchor from {len(anchor_wavs)} pristine wav files...")
        anchor_emb = spk_verifier.build_golden_anchor(anchor_wavs)
        spk_verifier.save_anchor(anchor_emb, args.save_anchor_to)
        print(f"✅ Golden anchor saved to {args.save_anchor_to}")
    elif args.golden_anchor:
        if os.path.isdir(args.golden_anchor):
            anchor_wavs = glob.glob(os.path.join(args.golden_anchor, "*.wav"))
            if anchor_wavs:
                anchor_emb = spk_verifier.build_golden_anchor(anchor_wavs)
        elif os.path.isfile(args.golden_anchor):
            anchor_emb = spk_verifier.load_anchor(args.golden_anchor)

    # 3. Setup Pipeline
    pipeline = QualityPipeline(
        asr_verifier=asr_verifier,
        speaker_verifier=spk_verifier,
        speech_mos=mos_evaluator,
        golden_anchor=anchor_emb,
        max_cer=args.max_cer,
        min_speaker_similarity=args.min_similarity,
        min_mos_ovrl=args.min_mos_ovrl,
    )

    # 4. Find audio files
    wav_files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.wav"), recursive=True))
    if not wav_files:
        print(f"❌ No .wav files found in {args.input_dir}")
        sys.exit(1)

    transcripts_map = {}
    if args.transcripts:
        transcripts_map = parse_transcripts(args.transcripts)

    print(f"🔍 Found {len(wav_files)} audio files. Starting 4-stage quality screening...")
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    passed_list_entries = []
    rejected_list_entries = []
    stats = {
        "total": len(wav_files),
        "pass": 0,
        "repaired_pass": 0,
        "reject": 0,
        "reject_by_stage": {"acoustic": 0, "asr": 0, "speaker": 0, "mos": 0},
    }

    for idx, wav_path in enumerate(wav_files, start=1):
        filename = os.path.basename(wav_path)
        spk = args.speaker_name
        lang = args.language
        text = ""

        # Lookup transcript
        if filename in transcripts_map:
            t_spk, t_lang, t_text = transcripts_map[filename]
            spk = t_spk or spk
            lang = t_lang or lang
            text = t_text
        elif wav_path in transcripts_map:
            t_spk, t_lang, t_text = transcripts_map[wav_path]
            spk = t_spk or spk
            lang = t_lang or lang
            text = t_text

        # Run pipeline
        report, processed_samples = pipeline.evaluate(
            audio_source=wav_path,
            reference_text=text if text else None,
            language=lang,
            allow_repair=not args.no_repair,
        )

        entry = {
            "filename": filename,
            "source_path": wav_path,
            "text": text,
            "status": report.status,
            "stage_failed": report.stage_failed,
            "repaired": report.repaired,
            "repair_actions": report.repair_actions,
            "rejection_reasons": report.rejection_reasons,
            "duration_sec": report.duration_sec,
        }

        if report.acoustic:
            entry["rms_db"] = report.acoustic.rms_db
            entry["peak_db"] = report.acoustic.peak_db
            entry["cps"] = report.acoustic.cps
        if report.asr:
            entry["cer"] = report.asr.cer
            entry["transcribed"] = report.asr.transcribed_text
        if report.speaker:
            entry["speaker_similarity"] = report.speaker.similarity
        if report.mos:
            entry["mos_ovrl"] = report.mos.ovrl
            entry["mos_bak"] = report.mos.bak
            entry["mos_sig"] = report.mos.sig

        results.append(entry)

        if report.passed:
            dest_wav = os.path.join(args.output_dir, filename)
            save_audio_file(dest_wav, processed_samples, report.sample_rate)

            if report.status == "REPAIRED_PASS":
                stats["repaired_pass"] += 1
                status_icon = "🔧 REPAIRED"
            else:
                stats["pass"] += 1
                status_icon = "✅ PASS"

            if text:
                passed_list_entries.append(f"{dest_wav}|{spk}|{lang}|{text}\n")
        else:
            stats["reject"] += 1
            if report.stage_failed:
                stats["reject_by_stage"][report.stage_failed] += 1
            status_icon = "❌ REJECT"

            if text:
                reasons = "; ".join(report.rejection_reasons or [f"Stage {report.stage_failed} failed"])
                rejected_list_entries.append(f"{wav_path}|{spk}|{lang}|{text}|{report.stage_failed}|{reasons}\n")

        print(
            f"[{idx}/{len(wav_files)}] {status_icon} {filename} "
            f"({' / '.join(report.rejection_reasons or report.repair_actions or ['Clean'])})"
        )

    # 5. Write output .list for GPT-SoVITS
    out_list_path = args.output_list or os.path.join(args.output_dir, "cleaned_train.list")
    if passed_list_entries:
        with open(out_list_path, "w", encoding="utf-8") as f:
            f.writelines(passed_list_entries)
        print(f"📄 Cleaned training dataset written to: {out_list_path}")

    rej_list_path = args.rejected_list or os.path.join(args.output_dir, "rejected.list")
    if rejected_list_entries:
        with open(rej_list_path, "w", encoding="utf-8") as f:
            f.writelines(rejected_list_entries)
        print(f"⚠️ Flawed/rejected dataset written to: {rej_list_path}")

    # 6. Save JSON audit report
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": stats,
                "thresholds": {
                    "max_cer": args.max_cer,
                    "min_speaker_similarity": args.min_similarity,
                    "min_mos_ovrl": args.min_mos_ovrl,
                },
                "items": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"📊 Detailed audit report saved to: {args.report}")

    print("\n================== SUMMARY ==================")
    print(f"Total Processed: {stats['total']}")
    print(f"Passed Directly: {stats['pass']} ({stats['pass']/stats['total']:.1%})")
    print(f"Repaired & Passed: {stats['repaired_pass']} ({stats['repaired_pass']/stats['total']:.1%})")
    print(f"Rejected: {stats['reject']} ({stats['reject']/stats['total']:.1%})")
    print(f"  - Acoustic Stage Rejections: {stats['reject_by_stage']['acoustic']}")
    print(f"  - ASR Alignment Rejections:  {stats['reject_by_stage']['asr']}")
    print(f"  - Speaker Timbre Drift:      {stats['reject_by_stage']['speaker']}")
    print(f"  - Speech MOS Low Quality:    {stats['reject_by_stage']['mos']}")
    print("=============================================\n")


if __name__ == "__main__":
    main()
