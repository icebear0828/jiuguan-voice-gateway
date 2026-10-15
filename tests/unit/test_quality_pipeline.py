"""
Unit tests for 4-Stage Quality Funnel Pipeline and Self-Healing
"""

import os
import tempfile
import numpy as np
import pytest

from core.acoustic_gate import AcousticQualityGate, load_audio_samples
from core.asr_verifier import MockASRVerifier
from core.speaker_verifier import MockSpeakerVerifier
from core.speech_mos import MockSpeechMOS
from core.quality_pipeline import (
    QualityPipeline,
    QualityReport,
    trim_silence,
    normalize_loudness,
    save_audio_file,
)


def generate_tone(sr: int = 32000, duration: float = 1.0, freq: float = 300.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestSelfHealingUtilities:
    def test_trim_silence_strips_padding(self):
        sr = 16000
        # 0.5s silence + 1.0s tone + 0.5s silence
        silence = np.zeros(int(sr * 0.5), dtype=np.float32)
        tone = generate_tone(sr=sr, duration=1.0, amp=0.5)
        audio = np.concatenate([silence, tone, silence])

        trimmed, was_trimmed = trim_silence(audio, sr, threshold_db=-45.0, pad_sec=0.05)
        assert was_trimmed is True
        assert len(trimmed) < len(audio)
        # Should be approximately 1.0s + 2*0.05s padding ~ 1.1s
        assert 0.9 <= len(trimmed) / sr <= 1.3

    def test_trim_silence_noop_on_already_trimmed(self):
        sr = 16000
        tone = generate_tone(sr=sr, duration=1.0, amp=0.5)
        _, was_trimmed = trim_silence(tone, sr)
        assert was_trimmed is False

    def test_normalize_loudness_attenuates_clipping(self):
        samples = np.ones(1000, dtype=np.float32)  # 0 dBFS
        normed, was_normed = normalize_loudness(samples, target_peak_db=-1.0)
        assert was_normed is True
        assert np.max(np.abs(normed)) <= 10 ** (-1.0 / 20.0) + 1e-5

    def test_normalize_loudness_boosts_soft_audio(self):
        # Soft audio with peak around -30 dBFS
        samples = generate_tone(amp=0.03)
        normed, was_normed = normalize_loudness(samples, target_rms_db=-20.0)
        assert was_normed is True
        assert np.max(np.abs(normed)) > np.max(np.abs(samples))

    def test_save_and_load_audio_file(self):
        sr = 16000
        samples = generate_tone(sr=sr, duration=0.5)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            save_audio_file(tmp_path, samples, sr)
            loaded_samples, loaded_sr = load_audio_samples(tmp_path)
            assert loaded_sr == sr
            assert len(loaded_samples) == len(samples)
            # Floating point accuracy check for 16-bit PCM quantization
            np.testing.assert_allclose(samples, loaded_samples, atol=1e-3)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestQualityPipelineFunnel:
    @pytest.fixture
    def standard_pipeline(self):
        return QualityPipeline(
            acoustic_gate=AcousticQualityGate(min_duration_sec=0.2),
            asr_verifier=MockASRVerifier(mock_transcription="こんにちは"),
            speaker_verifier=MockSpeakerVerifier(fixed_similarity=0.85),
            speech_mos=MockSpeechMOS(sig=4.2, bak=4.3, ovrl=4.2),
            golden_anchor=np.zeros(192),
            max_cer=0.08,
            min_speaker_similarity=0.78,
            min_mos_ovrl=3.8,
        )

    def test_clean_audio_passes_all_stages(self, standard_pipeline):
        audio = generate_tone(duration=1.0)
        report, _ = standard_pipeline.evaluate(
            audio, reference_text="こんにちは", language="ja"
        )
        assert report.passed is True
        assert report.status == "PASS"
        assert report.repaired is False
        assert report.stage_failed is None
        assert report.acoustic.passed is True
        assert report.asr.passed is True
        assert report.speaker.passed is True
        assert report.mos.passed is True

    def test_silence_padding_is_repaired_and_passes(self, standard_pipeline):
        sr = 16000
        # 1.0s silence + 0.6s tone + 1.0s silence
        silence = np.zeros(int(sr * 1.0), dtype=np.float32)
        tone = generate_tone(sr=sr, duration=0.6)
        audio = np.concatenate([silence, tone, silence])

        report, repaired = standard_pipeline.evaluate(
            audio, reference_text="こんにちは", allow_repair=True, sample_rate=sr
        )
        assert report.passed is True
        assert report.status == "REPAIRED_PASS"
        assert report.repaired is True
        assert len(report.repair_actions) > 0
        assert len(repaired) < len(audio)

    def test_unrecoverable_silence_rejected_at_acoustic_stage(self, standard_pipeline):
        sr = 16000
        silence = np.zeros(sr * 2, dtype=np.float32)

        report, _ = standard_pipeline.evaluate(silence, sample_rate=sr)
        assert report.passed is False
        assert report.status == "REJECT"
        assert report.stage_failed == "acoustic"
        assert len(report.rejection_reasons) > 0
        # Should short-circuit and not run ASR / Speaker / MOS
        assert report.asr is None
        assert report.speaker is None
        assert report.mos is None

    def test_asr_mismatch_rejected_at_asr_stage(self, standard_pipeline):
        audio = generate_tone(duration=1.0)
        # Reference text differs completely from mock transcription "こんにちは"
        report, _ = standard_pipeline.evaluate(
            audio, reference_text="全然違う文章です", language="ja"
        )
        assert report.passed is False
        assert report.status == "REJECT"
        assert report.stage_failed == "asr"
        assert any("CER" in r for r in report.rejection_reasons)
        # Should short-circuit: speaker and MOS not evaluated
        assert report.speaker is None
        assert report.mos is None

    def test_timbre_drift_rejected_at_speaker_stage(self):
        pipeline = QualityPipeline(
            acoustic_gate=AcousticQualityGate(min_duration_sec=0.2),
            asr_verifier=MockASRVerifier(mock_transcription="こんにちは"),
            speaker_verifier=MockSpeakerVerifier(fixed_similarity=0.62),  # < 0.78
            speech_mos=MockSpeechMOS(),
            golden_anchor=np.zeros(192),
            min_speaker_similarity=0.78,
        )
        audio = generate_tone(duration=1.0)
        report, _ = pipeline.evaluate(audio, reference_text="こんにちは")

        assert report.passed is False
        assert report.status == "REJECT"
        assert report.stage_failed == "speaker"
        assert any("timbre drift" in r for r in report.rejection_reasons)
        assert report.mos is None

    def test_poor_speech_quality_rejected_at_mos_stage(self):
        pipeline = QualityPipeline(
            acoustic_gate=AcousticQualityGate(min_duration_sec=0.2),
            asr_verifier=MockASRVerifier(mock_transcription="こんにちは"),
            speaker_verifier=MockSpeakerVerifier(fixed_similarity=0.90),
            speech_mos=MockSpeechMOS(sig=3.0, bak=3.0, ovrl=3.1),  # < 3.8
            golden_anchor=np.zeros(192),
            min_mos_ovrl=3.8,
        )
        audio = generate_tone(duration=1.0)
        report, _ = pipeline.evaluate(audio, reference_text="こんにちは")

        assert report.passed is False
        assert report.status == "REJECT"
        assert report.stage_failed == "mos"
        assert any("MOS" in r for r in report.rejection_reasons)

    def test_normalize_loudness_whisper_exemption(self):
        # Soft whisper audio (amp=0.01)
        samples = generate_tone(amp=0.01)
        orig_peak = np.max(np.abs(samples))

        # Standard mode would boost soft audio
        normed_std, std_changed = normalize_loudness(samples, is_whisper=False)
        assert std_changed is True
        assert np.max(np.abs(normed_std)) > orig_peak

        # Whisper mode must NOT boost soft audio (exempt)
        normed_wsp, wsp_changed = normalize_loudness(samples, is_whisper=True)
        assert wsp_changed is False
        assert np.max(np.abs(normed_wsp)) == orig_peak

    def test_pipeline_whisper_audio_passes_without_gain_explosion(self, standard_pipeline):
        # Soft audio with whisper tag
        audio = generate_tone(duration=1.0, amp=0.01)
        report, final_audio = standard_pipeline.evaluate(
            audio, reference_text="[whisper] こんにちは", language="ja"
        )
        assert report.passed is True
        # Amplitude should not be blown up into clipping
        assert np.max(np.abs(final_audio)) < 0.1

