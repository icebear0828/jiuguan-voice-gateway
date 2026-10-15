"""
Unit tests for Speaker Verification & Timbre Consistency Module
"""

import os
import tempfile
import numpy as np
import pytest

from core.speaker_verifier import (
    cosine_similarity,
    normalize_l2,
    BaseSpeakerVerifier,
    MockSpeakerVerifier,
    AcousticFeatureSpeakerVerifier,
    get_speaker_verifier,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sim = cosine_similarity(v, v)
        assert pytest.approx(sim, 0.0001) == 1.0

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        sim = cosine_similarity(v1, v2)
        assert pytest.approx(sim, 0.0001) == 0.0

    def test_opposite_vectors(self):
        v1 = np.array([1.0, 2.0], dtype=np.float32)
        v2 = np.array([-1.0, -2.0], dtype=np.float32)
        sim = cosine_similarity(v1, v2)
        assert pytest.approx(sim, 0.0001) == -1.0

    def test_zero_vector(self):
        v1 = np.zeros(5, dtype=np.float32)
        v2 = np.ones(5, dtype=np.float32)
        sim = cosine_similarity(v1, v2)
        assert sim == 0.0

    def test_normalize_l2(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        normed = normalize_l2(v)
        assert pytest.approx(float(np.linalg.norm(normed)), 0.0001) == 1.0


class TestAcousticFeatureSpeakerVerifier:
    @pytest.fixture
    def verifier(self):
        return AcousticFeatureSpeakerVerifier(embedding_dim=128)

    def test_extract_embedding_unit_norm(self, verifier):
        # Generate 1 sec of synthetic harmonic audio (speech-like tones)
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        samples = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        emb = verifier.extract_embedding(samples, sample_rate=sr)
        assert len(emb) == 128
        assert pytest.approx(float(np.linalg.norm(emb)), 0.001) == 1.0

    def test_identical_audio_high_similarity(self, verifier):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        samples = (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

        emb1 = verifier.extract_embedding(samples, sample_rate=sr)
        emb2 = verifier.extract_embedding(samples, sample_rate=sr)

        sim = verifier.compute_similarity(emb1, emb2)
        assert pytest.approx(sim, 0.001) == 1.0

    def test_verify_passes_when_similar(self, verifier):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        samples = (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        anchor = verifier.extract_embedding(samples, sample_rate=sr)

        res = verifier.verify(samples, anchor=anchor, min_similarity=0.8)
        assert res.passed is True
        assert res.similarity >= 0.8
        assert len(res.violations) == 0

    def test_verify_fails_when_timbre_drifts(self, verifier):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio_anchor = (0.5 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        audio_drift = (0.5 * np.random.RandomState(42).randn(sr)).astype(np.float32)

        anchor = verifier.extract_embedding(audio_anchor, sample_rate=sr)
        res = verifier.verify(audio_drift, anchor=anchor, min_similarity=0.90)

        assert res.passed is False
        assert len(res.violations) > 0
        assert "timbre drift detected" in res.violations[0]

    def test_build_golden_anchor_and_io(self, verifier):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        s1 = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        s2 = (0.5 * np.sin(2 * np.pi * 205 * t)).astype(np.float32)

        anchor = verifier.build_golden_anchor([s1, s2])
        assert len(anchor) == 128
        assert pytest.approx(float(np.linalg.norm(anchor)), 0.001) == 1.0

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            verifier.save_anchor(anchor, tmp_path)
            loaded = verifier.load_anchor(tmp_path)
            np.testing.assert_allclose(anchor, loaded, atol=1e-5)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestMockSpeakerVerifier:
    def test_mock_verifier_fixed_similarity(self):
        verifier = MockSpeakerVerifier(fixed_similarity=0.85)
        res = verifier.verify(np.zeros(100), anchor=np.zeros(192), min_similarity=0.78)
        assert res.passed is True
        assert res.similarity == 0.85

    def test_get_speaker_verifier_factory(self):
        mock_v = get_speaker_verifier("mock")
        assert isinstance(mock_v, MockSpeakerVerifier)

        feat_v = get_speaker_verifier("acoustic_feature")
        assert isinstance(feat_v, AcousticFeatureSpeakerVerifier)


class TestStage3MultiPrototypeAndAdaptiveThreshold:
    def test_duration_adaptive_threshold(self):
        from core.speaker_verifier import get_duration_adaptive_threshold

        assert get_duration_adaptive_threshold(2.0, base_threshold=0.78, min_threshold=0.70) == 0.78
        assert get_duration_adaptive_threshold(1.5, base_threshold=0.78, min_threshold=0.70) == 0.78
        assert get_duration_adaptive_threshold(0.5, base_threshold=0.78, min_threshold=0.70) == 0.70
        assert get_duration_adaptive_threshold(0.8, base_threshold=0.78, min_threshold=0.70) == 0.70
        mid = get_duration_adaptive_threshold(1.15, base_threshold=0.78, min_threshold=0.70)
        assert 0.73 <= mid <= 0.75

    def test_multi_prototype_max_pooling(self):
        verifier = AcousticFeatureSpeakerVerifier(embedding_dim=64)
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)

        # Prototype 1: 200Hz base
        s_normal = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        # Prototype 2: 400Hz base (e.g. shout/angry)
        s_angry = (0.5 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)

        emb_normal = verifier.extract_embedding(s_normal, sample_rate=sr)
        emb_angry = verifier.extract_embedding(s_angry, sample_rate=sr)

        prototype_anchors = {
            "Anchor_Normal": emb_normal,
            "Anchor_Angry": emb_angry,
        }

        # Test audio close to angry
        s_test_angry = (0.5 * np.sin(2 * np.pi * 405 * t)).astype(np.float32)
        res = verifier.verify(s_test_angry, anchor=prototype_anchors, min_similarity=0.85)

        assert res.passed is True
        assert res.matched_prototype == "Anchor_Angry"
        assert res.similarity > 0.90

    def test_short_audio_adaptive_threshold_in_verify(self):
        verifier = MockSpeakerVerifier(fixed_similarity=0.73)
        # Short audio (0.7s) -> threshold drops to 0.70, so 0.73 should PASS
        short_audio = np.zeros(int(16000 * 0.7), dtype=np.float32)
        res_short = verifier.verify(short_audio, anchor=np.zeros(128), min_similarity=0.78, sample_rate=16000)
        assert res_short.passed is True
        assert res_short.effective_min_similarity == 0.70

        # Long audio (2.0s) -> threshold stays 0.78, so 0.73 should REJECT
        long_audio = np.zeros(int(16000 * 2.0), dtype=np.float32)
        res_long = verifier.verify(long_audio, anchor=np.zeros(128), min_similarity=0.78, sample_rate=16000)
        assert res_long.passed is False
        assert res_long.effective_min_similarity == 0.78
