"""
Unit tests for core/asr_verifier.py
Tests CER computation, text normalization, repetition anomaly detection, and ASR verifier modes.
"""

import pytest
from core.asr_verifier import (
    normalize_text_for_cer,
    compute_levenshtein_distance,
    calculate_cer,
    detect_repetition,
    MockASRVerifier,
    get_asr_verifier,
)


@pytest.mark.unit
class TestCERComputation:
    def test_text_normalization(self):
        text = "  【こんにちは、世界！】……  "
        normalized = normalize_text_for_cer(text)
        assert normalized == "こんにちは世界"

    def test_levenshtein_exact_and_diff(self):
        assert compute_levenshtein_distance("abc", "abc") == 0
        assert compute_levenshtein_distance("abc", "abd") == 1
        assert compute_levenshtein_distance("abc", "") == 3

    def test_cer_exact_match(self):
        cer = calculate_cer("俺はスライムのリムル！", "俺はスライムのリムル！")
        assert cer == 0.0

    def test_cer_small_difference(self):
        # 1 char diff in 10 chars -> 10% CER
        ref = "1234567890"
        hyp = "12345x7890"
        cer = calculate_cer(ref, hyp)
        assert cer == 0.10

    def test_cer_threshold_check(self):
        # 1 substitution out of 20 chars -> 5% CER <= 8% threshold
        ref = "ジュラの大森林に仇なす者は許さん覚悟はいいか"
        hyp = "ジュラの大森林に仇なす者は許さん覚悟はいいよ"
        cer = calculate_cer(ref, hyp)
        assert cer <= 0.08


@pytest.mark.unit
class TestRepetitionDetection:
    def test_no_repetition_normal(self):
        text = "今日はテンペストは平和で、いい日になりそうだね。"
        has_repeat, detail = detect_repetition(text, max_repeat=2)
        assert has_repeat is False
        assert detail is None

    def test_repetition_loop_detected(self):
        # "悪いスライム" repeats 4 times
        stuck_loop = "俺はスライム悪いスライム悪いスライム悪いスライム悪いスライム"
        has_repeat, detail = detect_repetition(stuck_loop, max_repeat=2)
        assert has_repeat is True
        assert "repeated 4 times" in detail
        assert "スライム" in detail


@pytest.mark.unit
class TestMockASRVerifier:
    def test_mock_transcribe_and_verify(self):
        verifier = MockASRVerifier()
        ref = "やあ調子はどうだ"
        # Simulated accurate output
        res = verifier.verify_with_simulation(b"dummy_wav", ref, simulated_transcript=ref)
        assert res.passed is True
        assert res.cer == 0.0
        assert len(res.violations) == 0

    def test_mock_verify_cer_failure(self):
        verifier = MockASRVerifier()
        ref = "やあ調子はどうだ"
        # Simulated highly inaccurate transcript
        res = verifier.verify_with_simulation(
            b"dummy_wav", ref, simulated_transcript="全然違う言葉", max_cer=0.08
        )
        assert res.passed is False
        assert res.cer > 0.08
        assert any("exceeded allowed threshold" in v for v in res.violations)

    def test_mock_verify_repetition_failure(self):
        verifier = MockASRVerifier()
        ref = "俺はスライム"
        looping_output = "俺はスライムスライムスライムスライム"
        res = verifier.verify_with_simulation(
            b"dummy_wav", ref, simulated_transcript=looping_output
        )
        assert res.passed is False
        assert res.repetition_detected is True

    def test_get_asr_verifier_factory(self):
        v = get_asr_verifier("mock")
        assert isinstance(v, MockASRVerifier)


@pytest.mark.unit
class TestStage2Enhancements:
    def test_spoken_text_normalization_numbers_and_percent(self):
        verifier = MockASRVerifier()
        ref = "RTX 5070 降价 35%"
        asr_hyp = "RTX五千零七十降价百分之三十五"

        res = verifier.verify_with_simulation(
            b"dummy_wav", ref, simulated_transcript=asr_hyp, language="zh"
        )
        assert res.passed is True
        assert res.cer == 0.0

    def test_phoneme_error_rate_g2p_chinese_and_japanese(self):
        from core.asr_verifier import text_to_phonemes, calculate_per
        # Chinese G2P phoneme extraction
        ph_zh = text_to_phonemes("你好世界", language="zh")
        assert len(ph_zh) == 4
        assert ph_zh[0] == "ni3"

        # Japanese G2P phoneme extraction via pyopenjtalk
        ph_ja = text_to_phonemes("こんにちは世界", language="ja")
        assert len(ph_ja) >= 5
        assert "k" in ph_ja or "ko" in ph_ja or "こ" in ph_ja

        # PER detects phoneme discrepancy
        per_diff = calculate_per("银行行长", "银行校长", language="zh")
        assert per_diff > 0.0

    def test_differential_repetition_script_exemption(self):
        verifier = MockASRVerifier()
        # Script naturally has 3 consecutive repeats of '笨蛋'
        script_ref = "笨蛋笨蛋笨蛋！你到底在想什么？"

        # 1. Normal delivery of the script (3 repeats) should NOT be flagged
        res_normal = verifier.verify_with_simulation(
            b"dummy_wav", script_ref, simulated_transcript="笨蛋笨蛋笨蛋！你到底在想什么？"
        )
        assert res_normal.passed is True
        assert res_normal.repetition_detected is False

        # 2. Infinite hallucination loop (7 repeats) MUST be flagged
        res_loop = verifier.verify_with_simulation(
            b"dummy_wav", script_ref, simulated_transcript="笨蛋笨蛋笨蛋笨蛋笨蛋笨蛋笨蛋！"
        )
        assert res_loop.passed is False
        assert res_loop.repetition_detected is True

    def test_pure_interjection_exemption(self):
        from core.asr_verifier import is_pure_interjection
        assert is_pure_interjection("唔……") is True
        assert is_pure_interjection("啊！") is True
        assert is_pure_interjection("へー！？") is True
        assert is_pure_interjection("今天的天气真好") is False

        # Verify interjection exemption in verifier
        verifier = MockASRVerifier()
        res = verifier.verify_with_simulation(
            b"dummy_wav", "唔……！", simulated_transcript="呜…", language="zh"
        )
        assert res.passed is True
        assert res.is_interjection_exempt is True

    def test_trailing_drone_detection(self):
        from core.asr_verifier import detect_trailing_drone
        import numpy as np

        sr = 16000
        # 1.0s normal speech-like variance + 0.6s static drone sine wave
        t_speech = np.linspace(0, 1.0, int(sr * 1.0))
        speech = (np.sin(2 * np.pi * 200 * t_speech) * np.linspace(0.1, 0.5, len(t_speech))).astype(np.float32)

        t_drone = np.linspace(0, 0.6, int(sr * 0.6))
        # Perfectly flat drone tone with near 0 variance
        drone = (0.4 * np.sin(2 * np.pi * 300 * t_drone)).astype(np.float32)

        audio_with_drone = np.concatenate([speech, drone])
        has_drone, detail = detect_trailing_drone(audio_with_drone, sample_rate=sr, tail_ms=500.0)
        assert has_drone is True
        assert "Trailing drone detected" in detail

