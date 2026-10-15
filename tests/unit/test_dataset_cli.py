"""
Unit tests for Dataset Filtering CLI and Distillation Dataset Mixing CLI
"""

import os
import subprocess
import sys
import tempfile
import numpy as np
import pytest

from core.quality_pipeline import save_audio_file


@pytest.fixture
def temp_dataset_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_dir = os.path.join(tmp_dir, "raw_audio")
        output_dir = os.path.join(tmp_dir, "cleaned_audio")
        os.makedirs(input_dir, exist_ok=True)

        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        # Sample 1: Normal clean tone
        s1 = (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        save_audio_file(os.path.join(input_dir, "sample_clean.wav"), s1, sr)

        # Sample 2: Clipped tone
        s2 = np.ones(sr, dtype=np.float32)
        save_audio_file(os.path.join(input_dir, "sample_clipped.wav"), s2, sr)

        # Transcripts file (.list format)
        transcripts_path = os.path.join(tmp_dir, "transcripts.list")
        with open(transcripts_path, "w", encoding="utf-8") as f:
            f.write("sample_clean.wav|rimuru|ja|こんにちは\n")
            f.write("sample_clipped.wav|rimuru|ja|こんにちは\n")

        yield {
            "tmp_dir": tmp_dir,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "transcripts": transcripts_path,
        }


def test_filter_dataset_cli_execution(temp_dataset_dir):
    cmd = [
        sys.executable,
        "scripts/filter_dataset.py",
        "--input-dir",
        temp_dataset_dir["input_dir"],
        "--transcripts",
        temp_dataset_dir["transcripts"],
        "--output-dir",
        temp_dataset_dir["output_dir"],
        "--output-list",
        os.path.join(temp_dataset_dir["tmp_dir"], "cleaned.list"),
        "--report",
        os.path.join(temp_dataset_dir["tmp_dir"], "report.json"),
        "--mock-asr",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "SUMMARY" in result.stdout
    assert os.path.exists(os.path.join(temp_dataset_dir["tmp_dir"], "cleaned.list"))
    assert os.path.exists(os.path.join(temp_dataset_dir["tmp_dir"], "report.json"))

    # Clean sample should pass and be copied
    assert os.path.exists(os.path.join(temp_dataset_dir["output_dir"], "sample_clean.wav"))


def test_build_distillation_dataset_cli():
    with tempfile.TemporaryDirectory() as tmp_dir:
        golden_list = os.path.join(tmp_dir, "golden.list")
        synth_list = os.path.join(tmp_dir, "synth.list")
        output_list = os.path.join(tmp_dir, "blended.list")

        # 3 human items
        with open(golden_list, "w", encoding="utf-8") as f:
            f.write("human1.wav|spk|ja|短い\n")
            f.write("human2.wav|spk|ja|普通くらいの長さの文章です\n")
            f.write("human3.wav|spk|ja|とても長くて複雑な構文を含む台词になっていますよ\n")

        # 7 synthetic items
        with open(synth_list, "w", encoding="utf-8") as f:
            for i in range(7):
                f.write(f"synth_{i}.wav|spk|ja|合成音声のサンプルテキスト番号{i}です\n")

        cmd = [
            sys.executable,
            "scripts/build_distillation_dataset.py",
            "--golden-list",
            golden_list,
            "--synthetic-list",
            synth_list,
            "--output-list",
            output_list,
            "--human-ratio",
            "0.30",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert "Self-Distillation Mixed Dataset Created" in result.stdout
        assert os.path.exists(output_list)

        with open(output_list, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        assert len(lines) == 10
        human_lines = [l for l in lines if "human" in l]
        synth_lines = [l for l in lines if "synth" in l]
        assert len(human_lines) == 3
        assert len(synth_lines) == 7


def test_build_distillation_dataset_best_of_n_rejection_sampling():
    from scripts.build_distillation_dataset import (
        calculate_domain_richness,
        score_candidate,
        select_best_of_n,
        parse_list_file,
    )

    # 1. Domain richness check
    tech_text = "本次版本是2.5，服务器配备RTX 5070显卡，延迟降低了35%"
    plain_text = "你好呀"
    assert calculate_domain_richness(tech_text) > calculate_domain_richness(plain_text)

    # 2. Composite score check
    high_score = score_candidate(utmos=4.5, similarity=0.92, per=0.0, domain_bonus=0.3)
    low_score = score_candidate(utmos=3.2, similarity=0.75, per=0.15, domain_bonus=0.0)
    assert high_score > low_score

    # 3. Best-of-N rejection sampling selection
    with tempfile.TemporaryDirectory() as tmp_dir:
        synth_list = os.path.join(tmp_dir, "synth_candidates.list")
        golden_list = os.path.join(tmp_dir, "golden.list")
        output_list = os.path.join(tmp_dir, "blended_best.list")

        with open(golden_list, "w", encoding="utf-8") as f:
            f.write("golden_1.wav|spk|zh|这是黄金参考原声样本一\n")
            f.write("golden_2.wav|spk|zh|这是黄金参考原声样本二\n")

        # 4 attempts for the same text "笨蛋！", candidate 3 has the highest UTMOS/sim
        with open(synth_list, "w", encoding="utf-8") as f:
            f.write("cand_1.wav|spk|zh|笨蛋！|3.6|0.78|0.05\n")
            f.write("cand_2.wav|spk|zh|笨蛋！|3.8|0.82|0.02\n")
            f.write("cand_3.wav|spk|zh|笨蛋！|4.6|0.93|0.00\n")
            f.write("cand_4.wav|spk|zh|笨蛋！|3.5|0.75|0.10\n")

        cmd = [
            sys.executable,
            "scripts/build_distillation_dataset.py",
            "--golden-list",
            golden_list,
            "--synthetic-list",
            synth_list,
            "--output-list",
            output_list,
            "--human-ratio",
            "0.50",
            "--enable-best-of-n",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert "Best-of-N Rejection Sampling" in result.stdout

        with open(output_list, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        # Should only contain cand_3 for the "笨蛋！" prompt, cand_1/2/4 filtered out!
        assert any("cand_3.wav" in l for l in lines)
        assert not any("cand_1.wav" in l for l in lines)
        assert not any("cand_2.wav" in l for l in lines)
        assert not any("cand_4.wav" in l for l in lines)
