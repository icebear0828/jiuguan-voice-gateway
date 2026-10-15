"""
Stage 3.5: Prosody & Emotion Expressiveness Gate Module
Provides:
- F0 (Fundamental Frequency / Pitch) extraction via autocorrelation
- Pitch contour statistics: Mean F0, Pitch Standard Deviation (Pitch Std), Semitone Range
- Energy dynamics and variance analysis to catch flat robotic "Siri-like" monotone readings
- Emotion expressiveness alignment against prompt tags (e.g. [tsundere], [angry], [sad])
- Extensible classifiers (AcousticProsodyEmotionClassifier, MockEmotionClassifier, Emotion2VecClassifier)
"""

import math
import re
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None

from core.acoustic_gate import load_audio_samples


@dataclass
class ProsodyReport:
    passed: bool
    pitch_std: float
    pitch_mean: float
    pitch_range_semitones: float
    energy_std: float
    detected_emotion: str
    expected_emotion: Optional[str] = None
    emotion_confidence: float = 1.0
    violations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    engine: str = "AcousticProsodyGate"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_expected_emotion(text: Optional[str]) -> Optional[str]:
    """Extracts expected emotion tag from reference prompt text."""
    if not text:
        return None
    match = re.search(r"\[([a-zA-Z_\u4e00-\u9fa5]+)\]|\(([a-zA-Z_\u4e00-\u9fa5]+)\)", text)
    if match:
        tag = (match.group(1) or match.group(2)).lower()
        mapping = {
            "傲娇": "tsundere",
            "tsundere": "tsundere",
            "生气": "angry",
            "愤怒": "angry",
            "angry": "angry",
            "shout": "angry",
            "combat": "confident",
            "confident": "confident",
            "悲伤": "sad",
            "难过": "sad",
            "sad": "sad",
            "耳语": "whisper",
            "whisper": "whisper",
            "sigh": "whisper",
            "开心": "happy",
            "高兴": "happy",
            "happy": "happy",
            "playful": "happy",
            "日常": "neutral",
            "neutral": "neutral",
        }
        return mapping.get(tag, tag)
    return None


def extract_pitch_contour(
    samples: "np.ndarray",
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    f0_min: float = 65.0,
    f0_max: float = 650.0,
) -> Tuple["np.ndarray", "np.ndarray"]:
    """
    Extracts frame-level F0 contour (Hz) and frame RMS energy using autocorrelation.
    Returns: (f0_series, frame_rms_series)
    """
    if np is None or len(samples) == 0:
        return np.array([]), np.array([])

    frame_len = int(sr * (frame_ms / 1000.0))
    hop_len = int(sr * (hop_ms / 1000.0))

    if len(samples) < frame_len:
        return np.array([]), np.array([])

    min_period = int(sr / f0_max)
    max_period = int(sr / f0_min)

    n_frames = 1 + (len(samples) - frame_len) // hop_len
    f0_list = []
    rms_list = []

    # Window function
    window = np.hanning(frame_len).astype(np.float32)

    for i in range(n_frames):
        start = i * hop_len
        frame = samples[start : start + frame_len] * window
        rms = float(np.sqrt(np.mean(frame**2) + 1e-9))
        rms_list.append(rms)

        # Voiced threshold
        if rms < 0.015:
            f0_list.append(0.0)
            continue

        # Normalized autocorrelation
        corr = np.correlate(frame, frame, mode="full")[frame_len - 1 :]
        if corr[0] <= 1e-6:
            f0_list.append(0.0)
            continue

        corr_norm = corr / corr[0]

        # Search for highest peak within human vocal pitch period bounds
        search_region = corr_norm[min_period : min(max_period + 1, len(corr_norm))]
        if len(search_region) == 0:
            f0_list.append(0.0)
            continue

        peak_idx = int(np.argmax(search_region))
        peak_val = float(search_region[peak_idx])
        best_period = min_period + peak_idx

        # Voicing confidence threshold
        if peak_val >= 0.35 and best_period > 0:
            f0 = float(sr) / float(best_period)
            f0_list.append(f0)
        else:
            f0_list.append(0.0)

    return np.array(f0_list, dtype=np.float32), np.array(rms_list, dtype=np.float32)


class ProsodyQualityGate:
    """
    Prosody & Emotion Expressiveness Gate (Stage 3.5).
    Evaluates:
    1. Pitch standard deviation (Pitch Std in Hz) to catch flat monotone reading;
    2. Pitch semitone dynamic range (semitones between 10th and 90th percentile F0);
    3. Energy contour variance;
    4. Emotion consistency against prompt cue tags.
    """

    def __init__(
        self,
        min_pitch_std: float = 12.0,
        min_semitone_range: float = 3.5,
        min_energy_std: float = 0.01,
        enforce_emotion_alignment: bool = True,
    ):
        self.min_pitch_std = min_pitch_std
        self.min_semitone_range = min_semitone_range
        self.min_energy_std = min_energy_std
        self.enforce_emotion_alignment = enforce_emotion_alignment

    def evaluate(
        self,
        audio_source: Union[str, bytes, "np.ndarray"],
        reference_text: Optional[str] = None,
        expected_emotion: Optional[str] = None,
        sample_rate: int = 32000,
    ) -> ProsodyReport:
        if isinstance(audio_source, np.ndarray):
            samples = audio_source.astype(np.float32)
            sr = sample_rate
        else:
            samples, sr = load_audio_samples(audio_source)
            if sample_rate and sample_rate > 0:
                sr = sample_rate

        violations: List[str] = []

        if len(samples) < int(sr * 0.4):
            return ProsodyReport(
                passed=False,
                pitch_std=0.0,
                pitch_mean=0.0,
                pitch_range_semitones=0.0,
                energy_std=0.0,
                detected_emotion="unknown",
                expected_emotion=expected_emotion,
                violations=["Audio too short for prosody contour analysis"],
            )

        f0_contour, rms_contour = extract_pitch_contour(samples, sr)
        voiced_f0 = f0_contour[f0_contour > 0]

        # F0 statistics
        if len(voiced_f0) >= 5:
            pitch_mean = float(np.mean(voiced_f0))
            pitch_std = float(np.std(voiced_f0))

            # Semitone range (10th to 90th percentile to reject octave jumps)
            p10 = float(np.percentile(voiced_f0, 10))
            p90 = float(np.percentile(voiced_f0, 90))
            semitone_range = float(12.0 * math.log2(max(p90, 1e-3) / max(p10, 1e-3))) if p10 > 0 else 0.0
        else:
            pitch_mean = 0.0
            pitch_std = 0.0
            semitone_range = 0.0

        energy_std = float(np.std(rms_contour)) if len(rms_contour) > 0 else 0.0

        # Heuristic / Acoustic Emotion Classification
        detected_emotion = "neutral"
        emotion_confidence = 0.85

        if pitch_std > 28.0 and semitone_range > 8.0:
            detected_emotion = "angry" if energy_std > 0.05 else "tsundere"
        elif pitch_std > 18.0:
            detected_emotion = "happy"
        elif pitch_std < 10.0 and energy_std < 0.03:
            detected_emotion = "neutral"

        exp_emo = expected_emotion or extract_expected_emotion(reference_text)

        # 1. Monotone Robotic Reading Assertion
        if len(voiced_f0) >= 8:
            if pitch_std < self.min_pitch_std:
                violations.append(
                    f"Monotone robotic pitch detected: Pitch Std {pitch_std:.1f}Hz < threshold {self.min_pitch_std:.1f}Hz "
                    f"(flat Siri-like synthetic voice)"
                )
            elif semitone_range < self.min_semitone_range:
                violations.append(
                    f"Pitch dynamic range too narrow: {semitone_range:.1f} semitones < {self.min_semitone_range:.1f} semitones"
                )

        # 2. Emotion Alignment Assertion
        if self.enforce_emotion_alignment and exp_emo and exp_emo not in ["neutral", "daily"]:
            # If prompt requests high arousal (tsundere / angry / happy) but audio collapsed into flat neutral
            if exp_emo in ["tsundere", "angry", "confident"] and detected_emotion == "neutral" and pitch_std < 14.0:
                violations.append(
                    f"Emotion expressiveness mismatch: expected '{exp_emo}' but audio prosody collapsed to flat neutral "
                    f"(Pitch Std={pitch_std:.1f}Hz, SemitoneRange={semitone_range:.1f})"
                )

        passed = len(violations) == 0

        return ProsodyReport(
            passed=passed,
            pitch_std=round(pitch_std, 2),
            pitch_mean=round(pitch_mean, 2),
            pitch_range_semitones=round(semitone_range, 2),
            energy_std=round(energy_std, 4),
            detected_emotion=detected_emotion,
            expected_emotion=exp_emo,
            emotion_confidence=round(emotion_confidence, 2),
            violations=violations,
            metadata={
                "voiced_frames": int(len(voiced_f0)),
                "total_frames": int(len(f0_contour)),
            },
        )


class MockProsodyGate:
    """Mock prosody gate for test environments and pipeline testing."""

    def __init__(
        self,
        passed: bool = True,
        pitch_std: float = 24.5,
        detected_emotion: str = "neutral",
        violations: Optional[List[str]] = None,
    ):
        self.passed = passed
        self.pitch_std = pitch_std
        self.detected_emotion = detected_emotion
        self.violations = violations or ([] if passed else ["Mock prosody check failed"])

    def evaluate(
        self,
        audio_source: Any,
        reference_text: Optional[str] = None,
        expected_emotion: Optional[str] = None,
        sample_rate: int = 32000,
    ) -> ProsodyReport:
        exp_emo = expected_emotion or extract_expected_emotion(reference_text)
        return ProsodyReport(
            passed=self.passed,
            pitch_std=self.pitch_std,
            pitch_mean=220.0,
            pitch_range_semitones=6.5,
            energy_std=0.03,
            detected_emotion=self.detected_emotion,
            expected_emotion=exp_emo,
            emotion_confidence=0.95,
            violations=self.violations,
            engine="MockProsodyGate",
        )


def check_prosody_quality(
    audio_source: Union[str, bytes, "np.ndarray"],
    reference_text: Optional[str] = None,
    expected_emotion: Optional[str] = None,
    **kwargs,
) -> ProsodyReport:
    """Convenience helper to evaluate prosody expressiveness."""
    gate = ProsodyQualityGate(**kwargs)
    return gate.evaluate(audio_source, reference_text=reference_text, expected_emotion=expected_emotion)
