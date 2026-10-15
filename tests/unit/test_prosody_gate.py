"""
Unit tests for Stage 3.5 Prosody & Emotion Expressiveness Gate
"""

import numpy as np
import pytest

from core.prosody_gate import (
    ProsodyQualityGate,
    MockProsodyGate,
    extract_expected_emotion,
    extract_pitch_contour,
    check_prosody_quality,
)
from core.quality_pipeline import (
    QualityPipeline,
    QualityReport,
)
from core.acoustic_gate import AcousticQualityGate
from core.asr_verifier import MockASRVerifier
from core.speaker_verifier import MockSpeakerVerifier
from core.speech_mos import MockSpeechMOS


def generate_flat_monotone(sr: int = 16000, duration: float = 1.0, freq: float = 220.0, amp: float = 0.5) -> np.ndarray:
    """Generates pure sine wave with 0.0 Hz pitch standard deviation (completely flat monotone)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def generate_expressive_speech(sr: int = 16000, duration: float = 1.2, base_freq: float = 240.0, vibrato_hz: float = 5.0, depth_hz: float = 40.0) -> np.ndarray:
    """Generates synthetic speech with natural intonation pitch sweep and vibrato."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Instantaneous frequency: sweeps up and down
    f_t = base_freq + depth_hz * np.sin(2 * np.pi * vibrato_hz * t) + 20.0 * np.cos(np.pi * t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / sr
    # Add second harmonic and natural amplitude envelope
    env = np.sin(np.pi * t / duration) ** 0.5
    signal = 0.6 * np.sin(phase) + 0.3 * np.sin(2 * phase)
    return (signal * env).astype(np.float32)


class TestProsodyEmotionExtraction:
    def test_extract_emotion_tags(self):
        assert extract_expected_emotion("[tsundere] 哼，我才不是特意来找你的呢！") == "tsundere"
        assert extract_expected_emotion("(傲娇) 笨蛋！") == "tsundere"
        assert extract_expected_emotion("[angry] 闭嘴！") == "angry"
        assert extract_expected_emotion("[whisper] 嘘……小声点") == "whisper"
        assert extract_expected_emotion("[happy] 太棒啦！") == "happy"
        assert extract_expected_emotion("今天天气真好") is None


class TestProsodyGatePitchAndExpressiveness:
    def test_extract_pitch_contour(self):
        sr = 16000
        audio = generate_flat_monotone(sr=sr, duration=0.8, freq=300.0)
        f0_series, rms_series = extract_pitch_contour(audio, sr=sr)
        assert len(f0_series) > 0
        voiced = f0_series[f0_series > 0]
        assert len(voiced) > 0
        # Pitch should be close to 300Hz
        assert abs(float(np.mean(voiced)) - 300.0) < 15.0

    def test_monotone_reading_rejected(self):
        sr = 16000
        # Completely flat 220Hz tone (Pitch Std is ~0Hz < 12Hz threshold)
        audio = generate_flat_monotone(sr=sr, duration=1.0, freq=220.0)
        gate = ProsodyQualityGate(min_pitch_std=12.0)
        report = gate.evaluate(audio, sample_rate=sr)

        assert report.passed is False
        assert report.pitch_std < 5.0
        assert any("Monotone robotic pitch detected" in v for v in report.violations)

    def test_expressive_speech_passes(self):
        sr = 16000
        audio = generate_expressive_speech(sr=sr, duration=1.2, base_freq=240.0, depth_hz=35.0)
        gate = ProsodyQualityGate(min_pitch_std=12.0, min_semitone_range=3.0)
        report = gate.evaluate(audio, sample_rate=sr)

        assert report.passed is True
        assert report.pitch_std >= 12.0
        assert report.pitch_range_semitones >= 3.0
        assert len(report.violations) == 0

    def test_emotion_mismatch_interception(self):
        sr = 16000
        # Audio is a flat robotic monotone (detected as neutral)
        flat_audio = generate_flat_monotone(sr=sr, duration=1.0, freq=200.0)
        gate = ProsodyQualityGate(min_pitch_std=12.0, enforce_emotion_alignment=True)

        # Prompt requires intense emotion [tsundere], but audio is flat monotone neutral
        report = gate.evaluate(
            flat_audio,
            reference_text="[tsundere] 哼，我才不在乎呢！",
            sample_rate=sr,
        )

        assert report.passed is False
        assert any("Emotion expressiveness mismatch" in v or "Monotone" in v for v in report.violations)


class TestPipelineProsodyOrchestration:
    def test_pipeline_fails_at_prosody_stage(self):
        pipeline = QualityPipeline(
            acoustic_gate=AcousticQualityGate(min_duration_sec=0.2),
            asr_verifier=MockASRVerifier(mock_transcription="テスト"),
            speaker_verifier=MockSpeakerVerifier(fixed_similarity=0.90),
            prosody_gate=MockProsodyGate(passed=False, violations=["Pitch standard deviation too low"]),
            speech_mos=MockSpeechMOS(sig=4.0, bak=4.0, ovrl=4.0),
            golden_anchor=np.zeros(128),
        )

        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

        report, _ = pipeline.evaluate(audio, reference_text="テスト", sample_rate=sr)

        assert report.passed is False
        assert report.status == "REJECT"
        assert report.stage_failed == "prosody"
        assert "Pitch standard deviation too low" in report.rejection_reasons
        # Short-circuit: Stage 4 MOS should not be evaluated
        assert report.mos is None

    def test_pipeline_passes_all_stages_including_prosody(self):
        pipeline = QualityPipeline(
            acoustic_gate=AcousticQualityGate(min_duration_sec=0.2),
            asr_verifier=MockASRVerifier(mock_transcription="テスト"),
            speaker_verifier=MockSpeakerVerifier(fixed_similarity=0.90),
            prosody_gate=MockProsodyGate(passed=True, pitch_std=26.0),
            speech_mos=MockSpeechMOS(sig=4.0, bak=4.0, ovrl=4.0),
            golden_anchor=np.zeros(128),
        )

        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

        report, _ = pipeline.evaluate(audio, reference_text="テスト", sample_rate=sr)

        assert report.passed is True
        assert report.status == "PASS"
        assert report.prosody is not None
        assert report.prosody.passed is True
        assert report.mos is not None
        assert report.mos.passed is True
