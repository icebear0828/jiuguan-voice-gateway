"""
Unit tests for Speech MOS (Perception Quality) Module
"""

import numpy as np
import pytest

from core.speech_mos import (
    BaseSpeechMOS,
    MockSpeechMOS,
    AcousticHeuristicMOS,
    get_speech_mos,
)


class TestAcousticHeuristicMOS:
    @pytest.fixture
    def evaluator(self):
        return AcousticHeuristicMOS()

    def test_clean_audio_scores_high(self, evaluator):
        sr = 16000
        t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
        # Speech-like audio with fundamental and harmonics
        samples = (
            0.4 * np.sin(2 * np.pi * 200 * t)
            + 0.2 * np.sin(2 * np.pi * 400 * t)
            + 0.1 * np.sin(2 * np.pi * 600 * t)
        ).astype(np.float32)

        sig, bak, ovrl = evaluator.score(samples, sample_rate=sr)
        assert 1.0 <= sig <= 5.0
        assert 1.0 <= bak <= 5.0
        assert 1.0 <= ovrl <= 5.0
        assert ovrl >= 3.5

    def test_verify_passes_clean_speech(self, evaluator):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        samples = (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

        res = evaluator.verify(samples, sample_rate=sr, min_ovrl=3.0, min_bak=3.0)
        assert res.passed is True
        assert len(res.violations) == 0

    def test_verify_detects_severe_clipping(self, evaluator):
        sr = 16000
        samples = np.ones(sr, dtype=np.float32)  # Completely clipped flat line

        res = evaluator.verify(samples, sample_rate=sr, min_ovrl=4.0)
        assert res.passed is False
        assert any("MOS" in v for v in res.violations)


class TestMockSpeechMOS:
    def test_mock_speech_mos_scores(self):
        mos = MockSpeechMOS(sig=4.5, bak=4.2, ovrl=4.3)
        res = mos.verify(np.zeros(100), min_ovrl=4.0, min_bak=4.0)
        assert res.passed is True
        assert res.ovrl == 4.3
        assert res.bak == 4.2

    def test_mock_speech_mos_failure(self):
        mos = MockSpeechMOS(sig=2.5, bak=2.0, ovrl=2.3)
        res = mos.verify(np.zeros(100), min_ovrl=3.8, min_bak=4.0)
        assert res.passed is False
        assert len(res.violations) == 2

    def test_get_speech_mos_factory(self):
        mock_mos = get_speech_mos("mock")
        assert isinstance(mock_mos, MockSpeechMOS)

        heur_mos = get_speech_mos("heuristic")
        assert isinstance(heur_mos, AcousticHeuristicMOS)

        utmos = get_speech_mos("utmos")
        from core.speech_mos import UTMOSVerifier
        assert isinstance(utmos, UTMOSVerifier)


class TestStage4UTMOSAndGlitchDetection:
    def test_detect_pitch_glitches_clean_speech(self):
        from core.speech_mos import detect_pitch_glitches

        sr = 16000
        # Smooth pitch vibrato: no sudden 10+ semitone jumps
        t = np.linspace(0, 1.0, sr, endpoint=False)
        f_t = 220.0 + 15.0 * np.sin(2 * np.pi * 5.0 * t)
        phase = 2 * np.pi * np.cumsum(f_t) / sr
        audio = (0.5 * np.sin(phase)).astype(np.float32)

        cnt, glitches = detect_pitch_glitches(audio, sr=sr)
        assert cnt == 0
        assert len(glitches) == 0

    def test_detect_pitch_glitches_with_jump(self):
        from core.speech_mos import detect_pitch_glitches

        sr = 16000
        # 0.4s at 150Hz then abrupt jump to 480Hz (> 19 semitone jump)
        t1 = np.linspace(0, 0.4, int(sr * 0.4), endpoint=False)
        t2 = np.linspace(0, 0.4, int(sr * 0.4), endpoint=False)
        audio1 = 0.5 * np.sin(2 * np.pi * 150.0 * t1)
        audio2 = 0.5 * np.sin(2 * np.pi * 480.0 * t2)
        glitchy_audio = np.concatenate([audio1, audio2]).astype(np.float32)

        cnt, glitches = detect_pitch_glitches(glitchy_audio, sr=sr, max_semitone_jump=10.0, max_hz_jump=100.0)
        assert cnt >= 1
        assert any("Pitch glitch" in g for g in glitches)

    def test_detect_vocoder_artifacts_metallic_hiss(self):
        from core.speech_mos import detect_vocoder_artifacts

        sr = 32000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        # Speech band is weak, high frequency 9kHz metallic whine is dominant
        speech_band = 0.1 * np.sin(2 * np.pi * 400.0 * t)
        metallic_whine = 0.8 * np.sin(2 * np.pi * 9500.0 * t)
        noisy_audio = (speech_band + metallic_whine).astype(np.float32)

        has_artifact, score, diags = detect_vocoder_artifacts(noisy_audio, sr=sr)
        assert has_artifact is True
        assert len(diags) > 0

    def test_utmos_verifier_verification(self):
        from core.speech_mos import UTMOSVerifier

        verifier = UTMOSVerifier()
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        clean_audio = (0.4 * np.sin(2 * np.pi * 260.0 * t)).astype(np.float32)

        res_clean = verifier.verify(clean_audio, sample_rate=sr, min_ovrl=3.0, min_bak=3.0)
        assert res_clean.passed is True
        assert res_clean.utmos_score is not None
        assert res_clean.glitch_count == 0

        # Glitchy audio should fail UTMOS verification
        t1 = np.linspace(0, 0.4, int(sr * 0.4), endpoint=False)
        t2 = np.linspace(0, 0.4, int(sr * 0.4), endpoint=False)
        audio1 = 0.5 * np.sin(2 * np.pi * 140.0 * t1)
        audio2 = 0.5 * np.sin(2 * np.pi * 460.0 * t2)
        glitchy_audio = np.concatenate([audio1, audio2]).astype(np.float32)

        res_glitch = verifier.verify(glitchy_audio, sample_rate=sr)
        assert res_glitch.has_pitch_glitch is True
        assert res_glitch.passed is False
        assert any("Pitch glitch" in v for v in res_glitch.violations)
