"""
4-Stage Quality Funnel Pipeline for Audio Dataset Curation & Self-Distillation Filtering
Orchestrates:
1. Acoustic Physics Gate (RMS, Clipping, Duration, CPS) + Tiered Self-Healing (Silence trim, Normalization)
2. ASR Alignment Gate (CER, N-Gram repetition loops, Word truncation)
3. Speaker Verification Gate (Cosine similarity against Golden Anchor)
4. Neural/Psychoacoustic Speech MOS Gate (SIG, BAK, OVRL)
"""

import io
import math
import os
import wave
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any, Union, Tuple, Literal

try:
    import numpy as np
except ImportError:
    np = None

try:
    import soundfile as sf
except ImportError:
    sf = None

from core.acoustic_gate import (
    AcousticQualityGate,
    AcousticReport,
    load_audio_samples,
    calculate_lufs,
    calculate_true_peak,
    is_whisper_text,
)
from core.asr_verifier import (
    BaseASRVerifier,
    ASRVerificationResult,
    MockASRVerifier,
)
from core.speaker_verifier import (
    BaseSpeakerVerifier,
    SpeakerVerificationResult,
    AcousticFeatureSpeakerVerifier,
)
from core.prosody_gate import (
    ProsodyQualityGate,
    ProsodyReport,
    MockProsodyGate,
)
from core.speech_mos import (
    BaseSpeechMOS,
    MOSResult,
    AcousticHeuristicMOS,
)


@dataclass
class QualityReport:
    status: Literal["PASS", "REPAIRED_PASS", "REJECT"]
    passed: bool
    repaired: bool
    stage_failed: Optional[str] = None
    repair_actions: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)
    duration_sec: float = 0.0
    sample_rate: int = 32000
    acoustic: Optional[AcousticReport] = None
    asr: Optional[ASRVerificationResult] = None
    speaker: Optional[SpeakerVerificationResult] = None
    prosody: Optional[ProsodyReport] = None
    mos: Optional[MOSResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.acoustic:
            d["acoustic"] = self.acoustic.to_dict()
        if self.asr:
            d["asr"] = self.asr.to_dict()
        if self.speaker:
            d["speaker"] = self.speaker.to_dict()
        if self.prosody:
            d["prosody"] = self.prosody.to_dict()
        if self.mos:
            d["mos"] = self.mos.to_dict()
        return d


def trim_silence(
    samples: "np.ndarray",
    sr: int,
    threshold_db: float = -45.0,
    pad_sec: float = 0.12,
    fade_ms: float = 20.0,
) -> Tuple["np.ndarray", bool]:
    """
    Trims leading and trailing silence below threshold_db while retaining pad_sec comfort padding,
    and applies a smooth cosine fade-in/fade-out (fade_ms) to eliminate click noise and clipping.
    Returns (trimmed_samples, was_trimmed).
    """
    if np is None or len(samples) == 0:
        return samples, False

    frame_len = max(1, int(sr * 0.01))  # 10ms frames
    n_frames = len(samples) // frame_len
    if n_frames < 2:
        return samples, False

    frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms_vals = np.sqrt(np.mean(frames**2, axis=1) + 1e-9)
    db_vals = 20 * np.log10(rms_vals + 1e-9)

    active_indices = np.where(db_vals >= threshold_db)[0]
    if len(active_indices) == 0:
        # All silent, cannot trim safely
        return samples, False

    first_frame = active_indices[0]
    last_frame = active_indices[-1]

    pad_frames = int(pad_sec / 0.01)
    start_frame = max(0, first_frame - pad_frames)
    end_frame = min(n_frames, last_frame + 1 + pad_frames)

    start_sample = start_frame * frame_len
    end_sample = min(len(samples), end_frame * frame_len)

    # Only trim if we cut off at least 50ms of audio
    if start_sample > int(sr * 0.05) or (len(samples) - end_sample) > int(sr * 0.05):
        trimmed = samples[start_sample:end_sample].copy()

        # Apply smooth cosine fade-in and fade-out to prevent click noise
        fade_samples = int(sr * (fade_ms / 1000.0))
        if len(trimmed) > fade_samples * 2 and fade_samples > 0:
            t_in = np.linspace(0.0, np.pi, fade_samples, endpoint=False)
            fade_in = (0.5 * (1.0 - np.cos(t_in))).astype(np.float32)
            trimmed[:fade_samples] *= fade_in

            t_out = np.linspace(0.0, np.pi, fade_samples, endpoint=False)
            fade_out = (0.5 * (1.0 + np.cos(t_out))).astype(np.float32)
            trimmed[-fade_samples:] *= fade_out

        return trimmed, True

    return samples, False


def normalize_loudness(
    samples: "np.ndarray",
    sr: int = 32000,
    target_lufs: float = -18.0,
    max_true_peak_db: float = -1.0,
    lufs_acceptable_range: Tuple[float, float] = (-23.0, -14.0),
    is_whisper: bool = False,
    target_peak_db: Optional[float] = None,
    target_rms_db: Optional[float] = None,
) -> Tuple["np.ndarray", bool]:
    """
    Normalizes loudness using Integrated LUFS and True Peak constraints with emotion cue awareness.
    - If is_whisper is True: exempt from aggressive volume boosting; only attenuates if true peak > max_true_peak_db.
    - If standard speech: checks if loudness falls outside lufs_acceptable_range or exceeds max_true_peak_db.
      If outside, aligns towards target_lufs (-18 ~ -16 LUFS) while keeping True Peak <= max_true_peak_db.
    Returns (normalized_samples, was_normalized).
    """
    if np is None or len(samples) == 0:
        return samples, False

    if target_peak_db is not None:
        max_true_peak_db = target_peak_db

    true_peak_amp, current_peak_db = calculate_true_peak(samples)
    if true_peak_amp < 1e-6:
        return samples, False

    if is_whisper:
        # Whisper exemption: never boost weak whisper signals, only protect against clipping
        if current_peak_db > max_true_peak_db:
            gain_db = max_true_peak_db - current_peak_db
            gain_linear = 10.0 ** (gain_db / 20.0)
            norm_samples = np.clip(samples * gain_linear, -1.0, 1.0).astype(np.float32)
            return norm_samples, True
        return samples, False

    current_lufs = calculate_lufs(samples, sr)
    min_lufs = lufs_acceptable_range[0] if isinstance(lufs_acceptable_range, (tuple, list)) else -25.0

    needs_attenuation = current_peak_db > max_true_peak_db
    needs_boost = current_lufs < min_lufs

    if not needs_attenuation and not needs_boost:
        # Audio is within healthy dynamics and headroom zone (no clipping and not silent/too soft)
        return samples, False

    desired_boost = target_lufs - current_lufs
    max_allowed_boost = max_true_peak_db - current_peak_db
    gain_db = min(desired_boost, max_allowed_boost)

    if abs(gain_db) >= 0.5:
        gain_linear = 10.0 ** (gain_db / 20.0)
        norm_samples = np.clip(samples * gain_linear, -1.0, 1.0).astype(np.float32)
        return norm_samples, True

    return samples, False


def save_audio_file(file_path: str, samples: "np.ndarray", sample_rate: int) -> None:
    """Saves float32 audio samples as 16-bit PCM WAV."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    if sf is not None:
        sf.write(file_path, samples, sample_rate, subtype="PCM_16")
    else:
        # Standard wave fallback
        int16_samples = np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)
        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int16_samples.tobytes())


class QualityPipeline:
    """
    Unified 4-stage funnel quality evaluation pipeline for GPT-SoVITS training datasets.
    """

    def __init__(
        self,
        acoustic_gate: Optional[AcousticQualityGate] = None,
        asr_verifier: Optional[BaseASRVerifier] = None,
        speaker_verifier: Optional[BaseSpeakerVerifier] = None,
        prosody_gate: Optional[Union[ProsodyQualityGate, MockProsodyGate]] = None,
        speech_mos: Optional[BaseSpeechMOS] = None,
        golden_anchor: Optional[Union["np.ndarray", str, Dict[str, Any]]] = None,
        max_cer: float = 0.08,
        min_speaker_similarity: float = 0.78,
        min_mos_ovrl: float = 3.8,
        min_mos_bak: float = 4.0,
        enable_prosody_gate: bool = False,
    ):
        self.acoustic_gate = acoustic_gate or AcousticQualityGate()
        self.asr_verifier = asr_verifier
        self.speaker_verifier = speaker_verifier or AcousticFeatureSpeakerVerifier()
        if prosody_gate is not None:
            self.prosody_gate = prosody_gate
        elif enable_prosody_gate:
            self.prosody_gate = ProsodyQualityGate()
        else:
            self.prosody_gate = None
        self.speech_mos = speech_mos or AcousticHeuristicMOS()
        self.golden_anchor = golden_anchor

        self.max_cer = max_cer
        self.min_speaker_similarity = min_speaker_similarity
        self.min_mos_ovrl = min_mos_ovrl
        self.min_mos_bak = min_mos_bak

    def evaluate(
        self,
        audio_source: Union[str, bytes, "np.ndarray"],
        reference_text: Optional[str] = None,
        language: str = "ja",
        golden_anchor: Optional[Union["np.ndarray", str, Dict[str, Any]]] = None,
        allow_repair: bool = True,
        sample_rate: Optional[int] = None,
    ) -> Tuple[QualityReport, "np.ndarray"]:
        """
        Runs the full 4-stage funnel evaluation.
        Returns: (QualityReport, final_samples)
        """
        if isinstance(audio_source, np.ndarray):
            samples = audio_source.copy()
            sr = sample_rate or 32000
        else:
            samples, sr = load_audio_samples(audio_source)

        repair_actions: List[str] = []
        anchor = golden_anchor if golden_anchor is not None else self.golden_anchor

        # ----------------------------------------------------
        # Stage 1: Level 1 Acoustic Physics Gate & Self-Healing
        # ----------------------------------------------------
        ac_report = self.acoustic_gate.evaluate(
            samples, reference_text=reference_text, sample_rate=sr
        )
        was_repaired = False

        is_whisper = is_whisper_text(reference_text)

        if allow_repair:
            # Self-healing: clean padding silence and normalize volume to optimal level
            trimmed, trimmed_done = trim_silence(samples, sr)
            if trimmed_done:
                samples = trimmed
                repair_actions.append("Trimmed excessive leading/trailing silence with cosine fade")

            normed, norm_done = normalize_loudness(samples, sr=sr, is_whisper=is_whisper)
            if norm_done:
                samples = normed
                repair_actions.append("Normalized loudness (LUFS) and peak headroom")

            if trimmed_done or norm_done:
                was_repaired = True
                # Re-evaluate acoustic metrics on repaired samples
                ac_report = self.acoustic_gate.evaluate(
                    samples, reference_text=reference_text, sample_rate=sr
                )

        duration = round(len(samples) / float(sr), 3) if sr > 0 else 0.0

        if not ac_report.passed:
            return (
                QualityReport(
                    status="REJECT",
                    passed=False,
                    repaired=was_repaired,
                    stage_failed="acoustic",
                    repair_actions=repair_actions,
                    rejection_reasons=ac_report.violations,
                    duration_sec=duration,
                    sample_rate=sr,
                    acoustic=ac_report,
                ),
                samples,
            )

        # ----------------------------------------------------
        # Stage 2: Level 2 ASR Alignment & Repetition Gate
        # ----------------------------------------------------
        asr_report = None
        if self.asr_verifier is not None and reference_text:
            asr_report = self.asr_verifier.verify(
                samples,
                reference_text=reference_text,
                language=language,
                max_cer=self.max_cer,
                sample_rate=sr,
            )
            if not asr_report.passed:
                return (
                    QualityReport(
                        status="REJECT",
                        passed=False,
                        repaired=was_repaired,
                        stage_failed="asr",
                        repair_actions=repair_actions,
                        rejection_reasons=asr_report.violations,
                        duration_sec=duration,
                        sample_rate=sr,
                        acoustic=ac_report,
                        asr=asr_report,
                    ),
                    samples,
                )

        # ----------------------------------------------------
        # Stage 3: Level 3 Speaker Timbre Consistency Gate
        # ----------------------------------------------------
        spk_report = None
        if self.speaker_verifier is not None and anchor is not None:
            spk_report = self.speaker_verifier.verify(
                samples,
                anchor=anchor,
                min_similarity=self.min_speaker_similarity,
            )
            if not spk_report.passed:
                return (
                    QualityReport(
                        status="REJECT",
                        passed=False,
                        repaired=was_repaired,
                        stage_failed="speaker",
                        repair_actions=repair_actions,
                        rejection_reasons=spk_report.violations,
                        duration_sec=duration,
                        sample_rate=sr,
                        acoustic=ac_report,
                        asr=asr_report,
                        speaker=spk_report,
                    ),
                    samples,
                )

        # ----------------------------------------------------
        # Stage 3.5: Level 3.5 Prosody & Emotion Expressiveness Gate
        # ----------------------------------------------------
        prosody_report = None
        if self.prosody_gate is not None:
            prosody_report = self.prosody_gate.evaluate(
                samples,
                reference_text=reference_text,
                sample_rate=sr,
            )
            if not prosody_report.passed:
                return (
                    QualityReport(
                        status="REJECT",
                        passed=False,
                        repaired=was_repaired,
                        stage_failed="prosody",
                        repair_actions=repair_actions,
                        rejection_reasons=prosody_report.violations,
                        duration_sec=duration,
                        sample_rate=sr,
                        acoustic=ac_report,
                        asr=asr_report,
                        speaker=spk_report,
                        prosody=prosody_report,
                    ),
                    samples,
                )

        # ----------------------------------------------------
        # Stage 4: Level 4 Neural / Psychoacoustic Speech MOS
        # ----------------------------------------------------
        mos_report = None
        if self.speech_mos is not None:
            mos_report = self.speech_mos.verify(
                samples,
                sample_rate=sr,
                min_ovrl=self.min_mos_ovrl,
                min_bak=self.min_mos_bak,
            )
            if not mos_report.passed:
                return (
                    QualityReport(
                        status="REJECT",
                        passed=False,
                        repaired=was_repaired,
                        stage_failed="mos",
                        repair_actions=repair_actions,
                        rejection_reasons=mos_report.violations,
                        duration_sec=duration,
                        sample_rate=sr,
                        acoustic=ac_report,
                        asr=asr_report,
                        speaker=spk_report,
                        prosody=prosody_report,
                        mos=mos_report,
                    ),
                    samples,
                )

        final_status: Literal["PASS", "REPAIRED_PASS", "REJECT"] = (
            "REPAIRED_PASS" if was_repaired else "PASS"
        )

        return (
            QualityReport(
                status=final_status,
                passed=True,
                repaired=was_repaired,
                stage_failed=None,
                repair_actions=repair_actions,
                rejection_reasons=[],
                duration_sec=duration,
                sample_rate=sr,
                acoustic=ac_report,
                asr=asr_report,
                speaker=spk_report,
                prosody=prosody_report,
                mos=mos_report,
            ),
            samples,
        )
