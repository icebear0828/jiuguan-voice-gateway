"""
Unit tests for core/acoustic_gate.py
Testing silence detection, clipping detection, duration bounds, speech rate (CPS), and error handling.
"""

import io
import math
import wave
import pytest
import numpy as np

from core.acoustic_gate import (
    AcousticQualityGate,
    check_acoustic_quality,
    assert_acoustic_quality,
    AcousticReport,
    load_audio_samples,
)


def _generate_wav_bytes(
    duration: float = 2.0,
    sample_rate: int = 16000,
    freq: float = 440.0,
    amplitude: float = 0.5,
    clip: bool = False,
    silent: bool = False,
) -> bytes:
    """Helper to synthesize PCM WAV bytes for testing."""
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    if silent:
        samples = np.zeros(n_samples, dtype=np.float32)
    else:
        samples = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        if clip:
            # Overdrive samples to exceed 1.0 and clip
            samples = np.clip(samples * 10.0, -1.0, 1.0)

    pcm_int16 = (samples * 32767.0).astype(np.int16)

    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return bio.getvalue()


@pytest.mark.unit
class TestAcousticGate:
    def test_normal_audio_passes(self):
        wav_bytes = _generate_wav_bytes(duration=2.5, amplitude=0.4)
        report = check_acoustic_quality(wav_bytes, reference_text="你好世界，这是一段正常语速的测试语音")
        assert report.passed is True
        assert report.is_silent is False
        assert report.is_clipping is False
        assert report.duration_sec == 2.5
        assert len(report.violations) == 0

    def test_silence_detection(self):
        wav_bytes = _generate_wav_bytes(duration=2.0, silent=True)
        report = check_acoustic_quality(wav_bytes)
        assert report.passed is False
        assert report.is_silent is True
        assert any("Silence detected" in v for v in report.violations)

    def test_clipping_detection(self):
        wav_bytes = _generate_wav_bytes(duration=1.5, amplitude=1.5, clip=True)
        report = check_acoustic_quality(wav_bytes)
        assert report.passed is False
        assert report.is_clipping is True
        assert any("clipping" in v.lower() for v in report.violations)

    def test_duration_too_short(self):
        wav_bytes = _generate_wav_bytes(duration=0.2, amplitude=0.3)
        report = check_acoustic_quality(wav_bytes, min_duration_sec=0.5)
        assert report.passed is False
        assert any("too short" in v for v in report.violations)

    def test_duration_too_long(self):
        wav_bytes = _generate_wav_bytes(duration=10.0, amplitude=0.3)
        report = check_acoustic_quality(wav_bytes, max_duration_sec=5.0)
        assert report.passed is False
        assert any("too long" in v for v in report.violations)

    def test_cps_hallucination_loop_detection(self):
        # 3 characters stretched over 8 seconds -> CPS < 0.8
        wav_bytes = _generate_wav_bytes(duration=8.0, amplitude=0.3)
        report = check_acoustic_quality(wav_bytes, reference_text="你好呀")
        assert report.passed is False
        assert any("abnormally slow" in v for v in report.violations)

    def test_cps_truncation_detection(self):
        # 40 characters squashed into 1 second -> CPS > 16.0
        text = "这是一个非常长非常长非常长非常长非常长的台词文本，绝对不可能在一秒内说完"
        wav_bytes = _generate_wav_bytes(duration=1.0, amplitude=0.3)
        report = check_acoustic_quality(wav_bytes, reference_text=text)
        assert report.passed is False
        assert any("abnormally fast" in v for v in report.violations)

    def test_empty_audio_bytes(self):
        gate = AcousticQualityGate()
        # Empty WAV structure
        bio = io.BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"")
        report = gate.evaluate(bio.getvalue())
        assert report.passed is False
        assert report.duration_sec == 0.0

    def test_assert_acoustic_quality_raises(self):
        silent_wav = _generate_wav_bytes(duration=1.0, silent=True)
        with pytest.raises(AssertionError) as exc_info:
            assert_acoustic_quality(silent_wav)
        assert "Silence detected" in str(exc_info.value)


@pytest.mark.unit
class TestStage1Enhancements:
    def test_lufs_and_true_peak_calculation(self):
        wav_bytes = _generate_wav_bytes(duration=1.5, amplitude=0.5)
        report = check_acoustic_quality(wav_bytes)
        assert report.lufs is not None
        assert -30.0 <= report.lufs <= -5.0
        assert report.true_peak_db is not None
        # Peak around 0.5 is approx -6 dBTP
        assert -8.0 <= report.true_peak_db <= -4.0

    def test_whisper_tag_detection_and_silence_tolerance(self):
        # Generate soft audio (amplitude 0.003 -> RMS approx -53 dBFS)
        # Normally fails default -45 dBFS threshold
        soft_wav = _generate_wav_bytes(duration=1.5, amplitude=0.003)

        # 1. Standard text should fail silence check
        report_normal = check_acoustic_quality(soft_wav, reference_text="普通说话的声音")
        assert report_normal.passed is False
        assert report_normal.is_whisper is False
        assert any("Silence detected" in v for v in report_normal.violations)

        # 2. Whisper tagged text should be exempted
        report_whisper = check_acoustic_quality(soft_wav, reference_text="[whisper] 嘘…不要出声")
        assert report_whisper.is_whisper is True
        assert report_whisper.is_silent is False

    def test_vad_internal_silence_detection(self):
        sr = 16000
        # Construct 1.0s tone + 0.8s internal silence + 1.0s tone (pause 800ms > 600ms limit)
        s1 = (0.4 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr))).astype(np.float32)
        sil = np.zeros(int(sr * 0.8), dtype=np.float32)
        s2 = (0.4 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr))).astype(np.float32)
        audio_with_pause = np.concatenate([s1, sil, s2])

        gate = AcousticQualityGate(max_internal_silence_sec=0.60)
        report = gate.evaluate(audio_with_pause, sample_rate=sr)

        assert report.max_internal_silence_sec >= 0.75
        assert report.passed is False
        assert any("Internal pause too long" in v for v in report.violations)

        # Clean audio with acceptable 200ms pause should pass
        sil_short = np.zeros(int(sr * 0.2), dtype=np.float32)
        audio_normal = np.concatenate([s1, sil_short, s2])
        report_clean = gate.evaluate(audio_normal, sample_rate=sr)
        assert report_clean.max_internal_silence_sec <= 0.30
        assert not any("Internal pause too long" in v for v in report_clean.violations)

    def test_net_cps_speech_rate_accuracy(self):
        sr = 16000
        # 1.0s speech + 1.0s pause + 1.0s speech = 3.0s total, 2.0s voiced
        s1 = (0.4 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr))).astype(np.float32)
        sil = np.zeros(int(sr * 1.0), dtype=np.float32)
        s2 = (0.4 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr))).astype(np.float32)
        audio = np.concatenate([s1, sil, s2])

        # 10 characters over 2.0s voiced duration -> Net CPS = 5.0 CPS
        gate = AcousticQualityGate(max_internal_silence_sec=2.0)
        report = gate.evaluate(audio, reference_text="今天的天气真是不错啊", sample_rate=sr)
        assert report.net_cps is not None
        assert 4.5 <= report.net_cps <= 5.5

    def test_trim_silence_cosine_fade_smoothness(self):
        from core.quality_pipeline import trim_silence
        sr = 16000
        # 0.5s silence + 1.0s full tone + 0.5s silence
        sil = np.zeros(int(sr * 0.5), dtype=np.float32)
        tone = np.ones(int(sr * 1.0), dtype=np.float32) * 0.5
        audio = np.concatenate([sil, tone, sil])

        trimmed, was_trimmed = trim_silence(audio, sr=sr, fade_ms=20.0)
        assert was_trimmed is True
        # First sample should be smoothly attenuated close to 0 due to cosine fade
        assert abs(trimmed[0]) < 0.05
        assert abs(trimmed[-1]) < 0.05

