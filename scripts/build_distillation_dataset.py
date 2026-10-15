#!/usr/bin/env python3
"""
Self-Distillation Dataset Flywheel Builder
Constructs a balanced training dataset for GPT-SoVITS fine-tuning by dynamically
mixing Golden Human Anchor data (e.g., 30%) with passed synthetic audio (70%),
preventing model collapse, frequency degradation, and timbre drift.
"""

import argparse
import os
import random
import re
import sys
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional


def calculate_domain_richness(text: str) -> float:
    """
    Calculates domain richness bonus for vocabulary expansion:
    - Technical terms / numbers / percentages (e.g., 35%, RTX 5070)
    - Polyphones and semantic context disambiguation (行, 重, 乐, 得, 差, 调)
    - Long sentence structural complexity (>= 20 chars)
    """
    bonus = 0.0
    if re.search(r"\d+%?|[0-9]", text):
        bonus += 0.25
    if re.search(r"[a-zA-Z]{2,}", text):
        bonus += 0.20
    polyphones = ["行", "重", "乐", "得", "差", "调", "落", "模", "着", "更", "为"]
    poly_count = sum(text.count(p) for p in polyphones)
    if poly_count > 0:
        bonus += min(0.35, poly_count * 0.1)
    if len(text) >= 20:
        bonus += 0.20
    return round(bonus, 3)


def score_candidate(
    utmos: float = 4.0,
    similarity: float = 0.85,
    per: float = 0.0,
    domain_bonus: float = 0.0,
) -> float:
    """
    Calculates composite quality score for Best-of-N rejection sampling:
    Score = UTMOS + Similarity - PER + Domain_Bonus
    """
    return round(utmos + similarity - per + domain_bonus, 4)


def parse_list_file(list_path: str) -> List[Tuple[str, str, str, str, str, float]]:
    """
    Parses a GPT-SoVITS .list file with optional quality metrics:
    wav_path|speaker_name|language|text[|utmos|sim|per]
    Returns: list of (raw_line, wav_path, speaker, lang, text, composite_score)
    """
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"List file not found: {list_path}")

    items = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split("|")
            utmos = 4.0
            sim = 0.85
            per = 0.0
            if len(parts) >= 7:
                try:
                    utmos = float(parts[4])
                    sim = float(parts[5])
                    per = float(parts[6])
                except ValueError:
                    pass
                wav_path, spk, lang, text = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
            elif len(parts) >= 4:
                wav_path, spk, lang, text = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
            elif len(parts) == 2:
                wav_path, spk, lang, text = parts[0].strip(), "spk", "ja", parts[1].strip()
            else:
                continue

            domain_bonus = calculate_domain_richness(text)
            comp_score = score_candidate(utmos, sim, per, domain_bonus)
            items.append((raw, wav_path, spk, lang, text, comp_score))
    return items


def select_best_of_n(
    items: List[Tuple[str, str, str, str, str, float]],
) -> List[Tuple[str, str, str, str, str, float]]:
    """
    Performs Best-of-N Rejection Sampling on candidate synthetic audio utterances:
    Groups candidates by text and picks the top-1 highest composite scoring generation.
    """
    grouped = defaultdict(list)
    for item in items:
        text_key = item[4].strip()
        grouped[text_key].append(item)

    best_items = []
    for text_key, candidates in grouped.items():
        # Pick Pareto optimal sample
        best = max(candidates, key=lambda c: c[5])
        best_items.append(best)

    return best_items


def categorize_by_length(
    items: List[Tuple[str, str, str, str, str, float]]
) -> Dict[str, List[Tuple[str, str, str, str, str, float]]]:
    """Buckets utterances into short (<10 chars), medium (10-25 chars), and long (>25 chars)."""
    buckets = defaultdict(list)
    for item in items:
        char_len = len(item[4])
        if char_len < 10:
            buckets["short"].append(item)
        elif char_len <= 25:
            buckets["medium"].append(item)
        else:
            buckets["long"].append(item)
    return buckets


def main():
    parser = argparse.ArgumentParser(
        description="Build self-distillation training dataset by mixing Golden Anchor and synthetic audio with Best-of-N Rejection Sampling."
    )
    parser.add_argument(
        "--golden-list",
        type=str,
        required=True,
        help="Path to Golden Anchor (100%% verified human voice) .list file.",
    )
    parser.add_argument(
        "--synthetic-list",
        type=str,
        required=True,
        help="Path to quality-screened synthetic audio .list file.",
    )
    parser.add_argument(
        "--output-list",
        type=str,
        default="assets/distillation_train.list",
        help="Output path for the blended training dataset.",
    )
    parser.add_argument(
        "--human-ratio",
        type=float,
        default=0.50,
        help="Target proportion of human Golden Anchor samples (default: 0.50 for 50%% human + 50%% AI dynamic annealing).",
    )
    parser.add_argument(
        "--enable-best-of-n",
        action="store_true",
        default=True,
        help="Enable Best-of-N Rejection Sampling for duplicate text candidates (default: True).",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=None,
        help="Optional target total number of utterances in the output dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible dataset mixing.",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    print("🔄 Loading dataset manifests...")
    human_items = parse_list_file(args.golden_list)
    raw_synth_items = parse_list_file(args.synthetic_list)

    if not human_items:
        print(f"❌ Golden dataset {args.golden_list} has 0 valid entries.")
        sys.exit(1)
    if not raw_synth_items:
        print(f"❌ Synthetic dataset {args.synthetic_list} has 0 valid entries.")
        sys.exit(1)

    print(f"  • Golden Anchor (Human) entries:    {len(human_items)}")
    print(f"  • Raw Synthetic Candidates entries: {len(raw_synth_items)}")

    # Best-of-N Rejection Sampling on Synthetic Pool
    if args.enable_best_of_n:
        synth_items = select_best_of_n(raw_synth_items)
        n_filtered = len(raw_synth_items) - len(synth_items)
        if n_filtered > 0:
            print(f"  ✨ Best-of-N Rejection Sampling selected {len(synth_items)} optimal samples (filtered out {n_filtered} suboptimal candidates)")
    else:
        synth_items = raw_synth_items

    # Sort synthetic items with domain word richness priority
    synth_items.sort(key=lambda x: x[5], reverse=True)

    # Calculate optimal mixture counts
    human_ratio = max(0.10, min(0.90, args.human_ratio))
    synth_ratio = 1.0 - human_ratio

    if args.target_total is not None:
        target_human_count = int(args.target_total * human_ratio)
        target_synth_count = args.target_total - target_human_count
    else:
        # 5:5 dynamic annealing: balance human and synthetic counts
        synth_limit = len(synth_items)
        needed_human_for_all_synth = int(synth_limit * (human_ratio / synth_ratio))
        if needed_human_for_all_synth <= len(human_items):
            target_synth_count = synth_limit
            target_human_count = max(1, needed_human_for_all_synth)
        else:
            target_human_count = len(human_items)
            target_synth_count = max(1, int(target_human_count * (synth_ratio / human_ratio)))

    # Stratified length sampling for synthetic data to prevent bias towards simple short sentences
    synth_buckets = categorize_by_length(synth_items)
    selected_synth = []
    bucket_counts = {b: len(items) for b, items in synth_buckets.items()}
    total_in_buckets = sum(bucket_counts.values())

    for b, items in synth_buckets.items():
        if total_in_buckets > 0:
            quota = int(target_synth_count * (len(items) / total_in_buckets))
            # Sort within bucket by composite score (prioritize domain rich and highest quality)
            items_sorted = sorted(items, key=lambda x: x[5], reverse=True)
            selected_synth.extend(items_sorted[:quota])

    # If rounding left us short, backfill from remaining highest scoring
    remaining = [item for item in synth_items if item not in selected_synth]
    remaining.sort(key=lambda x: x[5], reverse=True)
    needed = target_synth_count - len(selected_synth)
    if needed > 0:
        selected_synth.extend(remaining[:needed])

    # Sample human items (if human pool larger than target, sample with replacement or shuffle)
    if target_human_count <= len(human_items):
        shuffled_human = human_items.copy()
        random.shuffle(shuffled_human)
        selected_human = shuffled_human[:target_human_count]
    else:
        # Oversample human anchor to satisfy ratio
        selected_human = [random.choice(human_items) for _ in range(target_human_count)]

    blended = selected_human + selected_synth
    random.shuffle(blended)

    # Write output list (extract the 4 standard fields: wav_path|spk|lang|text)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_list)), exist_ok=True)
    with open(args.output_list, "w", encoding="utf-8") as f:
        for item in blended:
            f.write(f"{item[1]}|{item[2]}|{item[3]}|{item[4]}\n")

    actual_human_ratio = len(selected_human) / len(blended) if blended else 0.0

    print("\n🎉 Self-Distillation Mixed Dataset Created!")
    print(f"  • Output file:         {args.output_list}")
    print(f"  • Total samples:       {len(blended)}")
    print(f"  • Human Golden Anchor: {len(selected_human)} ({actual_human_ratio:.1%}) [Target: {human_ratio:.1%}]")
    print(f"  • Filtered Synthetic:  {len(selected_synth)} ({1.0 - actual_human_ratio:.1%})")
    print("============================================================\n")


if __name__ == "__main__":
    main()
