"""
Acoustic Quality Gate Module
Provides objective acoustic quality assertions:
- Silence Detection (RMS energy threshold)
- Clipping / Distortion Detection (dBFS peak, consecutive clipping samples)
- Duration & Speech Rate Bounds (CPS - Characters Per Second to catch truncation or repetition loop)
"""

import io
import math
import re
import wave
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None

try:
    from scipy import signal
except ImportError:
    signal = None


@dataclass
class AcousticReport:
    passed: bool
    duration_sec: float
    rms_db: float
    peak_db: float
    is_silent: bool
    is_clipping: bool
    cps: Optional[float] = None
    clipping_ratio: float = 0.0
    violations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Stage 1 Enhanced Fields
    lufs: Optional[float] = None
    true_peak_db: Optional[float] = None
    max_internal_silence_sec: float = 0.0
    voiced_duration_sec: float = 0.0
    net_cps: Optional[float] = None
    is_whisper: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_audio_samples(audio_source: Union[str, bytes]) -> Tuple["np.ndarray", int]:
    """
    Loads audio from a file path or raw bytes into a 1D float32 numpy array in range [-1.0, 1.0].
    Returns (samples, sample_rate).
    """
    if np is None:
        raise ImportError("numpy is required for acoustic quality analysis")

    if isinstance(audio_source, np.ndarray):
        data = audio_source.astype(np.float32)
        sr = 32000
    elif isinstance(audio_source, (bytes, bytearray)):
        bio = io.BytesIO(audio_source)
        if sf is not None:
            try:
                data, sr = sf.read(bio, dtype="float32")
            except Exception:
                bio.seek(0)
                data, sr = _load_with_wave(bio)
        else:
            data, sr = _load_with_wave(bio)
    elif isinstance(audio_source, str):
        if sf is not None:
            try:
                data, sr = sf.read(audio_source, dtype="float32")
            except Exception:
                with open(audio_source, "rb") as f:
                    data, sr = _load_with_wave(f)
        else:
            with open(audio_source, "rb") as f:
                data, sr = _load_with_wave(f)
    else:
        raise TypeError(f"Unsupported audio source type: {type(audio_source)}")

    # Convert multi-channel to mono by averaging channels
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    return data, sr


def _load_with_wave(file_like) -> Tuple["np.ndarray", int]:
    with wave.open(file_like, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if sampwidth == 2:  # 16-bit PCM
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:  # 8-bit unsigned PCM
        samples = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 4:  # 32-bit float or int
        samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples, sr


def is_whisper_text(text: Optional[str]) -> bool:
    """Checks if the reference text contains whisper, sigh, or soft speech cue tags."""
    if not text:
        return False
    whisper_pattern = (
        r"\[(whisper|sigh|gasp|whimper|panting|耳语|叹气|气声)\]|"
        r"\((whisper|sigh|gasp|whimper|panting|耳语|叹气|气声)\)"
    )
    return bool(re.search(whisper_pattern, text, re.IGNORECASE))


def calculate_lufs(samples: "np.ndarray", sample_rate: int) -> float:
    """
    Computes Integrated LUFS (EBU R128 / ITU-R BS.1770-4).
    Falls back gracefully to RMS-based approximation if pyloudnorm is unavailable or fails.
    """
    if len(samples) == 0:
        return -100.0

    if pyln is not None:
        try:
            meter = pyln.Meter(sample_rate)
            lufs = meter.integrated_loudness(samples)
            if not math.isnan(lufs) and not math.isinf(lufs):
                return float(lufs)
        except Exception:
            pass

    # Fallback to RMS-based LUFS approximation: RMS dBFS - 0.691 dB
    rms_amp = float(np.sqrt(np.mean(samples**2) + 1e-9))
    return float(20.0 * math.log10(rms_amp + 1e-9) - 0.691)


def calculate_true_peak(samples: "np.ndarray") -> Tuple[float, float]:
    """
    Computes True Peak amplitude and dBTP (decibels relative to Full Scale True Peak).
    Uses 4x oversampling when scipy.signal is available according to ITU-R BS.1770.
    Returns (true_peak_linear, true_peak_dbtp).
    """
    if len(samples) == 0:
        return 0.0, -100.0

    if signal is not None and len(samples) >= 32:
        try:
            # 4x polyphase oversampling for inter-sample true peak detection
            resampled = signal.resample_poly(samples, 4, 1)
            peak_amp = float(np.max(np.abs(resampled)))
        except Exception:
            peak_amp = float(np.max(np.abs(samples)))
    else:
        peak_amp = float(np.max(np.abs(samples)))

    peak_db = 20.0 * math.log10(max(peak_amp, 1e-9))
    return peak_amp, peak_db


def analyze_vad(
    samples: "np.ndarray",
    sample_rate: int,
    frame_ms: float = 20.0,
    threshold_db: float = -42.0,
    min_speech_ms: float = 60.0,
    min_silence_ms: float = 80.0,
) -> Tuple[List[Tuple[float, float]], float, float]:
    """
    Performs frame-level adaptive VAD on audio to detect voiced intervals and internal pauses.
    Returns:
    - speech_intervals: List of (start_sec, end_sec) voiced segments
    - voiced_duration_sec: Total duration of non-silent speech
    - max_internal_silence_sec: Maximum pause length between consecutive speech segments
    """
    if np is None or len(samples) == 0:
        return [], 0.0, 0.0

    frame_len = max(1, int(sample_rate * (frame_ms / 1000.0)))
    n_frames = len(samples) // frame_len
    total_dur = len(samples) / float(sample_rate)

    if n_frames < 2:
        return [(0.0, total_dur)], total_dur, 0.0

    frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms_vals = np.sqrt(np.mean(frames**2, axis=1) + 1e-9)
    db_vals = 20.0 * np.log10(rms_vals + 1e-9)
    is_speech = db_vals >= threshold_db

    min_speech_frames = max(1, int(min_speech_ms / frame_ms))
    min_silence_frames = max(1, int(min_silence_ms / frame_ms))

    raw_intervals: List[Tuple[float, float]] = []
    in_speech = False
    start_f = 0

    for i, active in enumerate(is_speech):
        if active and not in_speech:
            in_speech = True
            start_f = i
        elif not active and in_speech:
            in_speech = False
            if (i - start_f) >= min_speech_frames:
                raw_intervals.append(
                    (round(start_f * frame_ms / 1000.0, 3), round(i * frame_ms / 1000.0, 3))
                )
    if in_speech and (n_frames - start_f) >= min_speech_frames:
        raw_intervals.append(
            (round(start_f * frame_ms / 1000.0, 3), round(n_frames * frame_ms / 1000.0, 3))
        )

    if not raw_intervals:
        return [], 0.0, 0.0

    # Merge intervals separated by less than min_silence_ms
    merged: List[List[float]] = []
    min_silence_sec = min_silence_ms / 1000.0
    for start, end in raw_intervals:
        if not merged:
            merged.append([start, end])
        else:
            if start - merged[-1][1] < min_silence_sec:
                merged[-1][1] = end
            else:
                merged.append([start, end])

    speech_intervals = [(round(s, 3), round(e, 3)) for s, e in merged]
    voiced_duration = sum(e - s for s, e in speech_intervals)

    max_internal_silence = 0.0
    for i in range(len(speech_intervals) - 1):
        pause = speech_intervals[i + 1][0] - speech_intervals[i][1]
        if pause > max_internal_silence:
            max_internal_silence = pause

    return speech_intervals, round(voiced_duration, 3), round(max_internal_silence, 3)


class AcousticQualityGate:
    def __init__(
        self,
        min_rms_db: float = -45.0,
        min_duration_sec: float = 0.4,
        max_duration_sec: float = 60.0,
        clipping_threshold_db: float = -0.1,
        max_clipping_ratio: float = 0.001,
        max_consecutive_clipping: int = 5,
        min_cps: float = 0.8,
        max_cps: float = 16.0,
        # Batch 1 Enhanced Parameters
        target_lufs_min: float = -28.0,
        target_lufs_max: float = -12.0,
        max_true_peak_db: float = -1.0,
        max_internal_silence_sec: float = 0.60,
        min_net_cps: Optional[float] = None,
        max_net_cps: Optional[float] = None,
        whisper_min_rms_db: float = -60.0,
        enable_vad: bool = True,
    ):
        """
        Thresholds for acoustic gating:
        :param min_rms_db: Audio with RMS below this is flagged as silent (-45 dBFS).
        :param min_duration_sec: Minimum acceptable duration for any utterance.
        :param max_duration_sec: Maximum acceptable duration (to catch endless generation).
        :param clipping_threshold_db: Peak amplitude above this (-0.1 dBFS) counts towards clipping.
        :param max_clipping_ratio: Maximum proportion of clipped samples allowed (0.1%).
        :param max_consecutive_clipping: Consecutive samples >= threshold indicating hard clipping.
        :param min_cps: Min rough character-per-second (0.8 CPS).
        :param max_cps: Max rough character-per-second (16.0 CPS).
        :param target_lufs_min: Minimum acceptable integrated loudness (-28.0 LUFS).
        :param target_lufs_max: Maximum acceptable integrated loudness (-12.0 LUFS).
        :param max_true_peak_db: True peak ceiling (-1.0 dBTP).
        :param max_internal_silence_sec: Maximum intra-sentence non-speech pause (0.60s / 600ms).
        :param min_net_cps: Minimum speech rate over voiced segments (2.0 CPS).
        :param max_net_cps: Maximum speech rate over voiced segments (10.0 CPS).
        :param whisper_min_rms_db: Generous silence floor for whisper/sigh samples (-60.0 dBFS).
        :param enable_vad: Whether to perform local VAD and pause/net-CPS analysis.
        """
        self.min_rms_db = min_rms_db
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.clipping_threshold_db = clipping_threshold_db
        self.max_clipping_ratio = max_clipping_ratio
        self.max_consecutive_clipping = max_consecutive_clipping
        self.min_cps = min_cps
        self.max_cps = max_cps
        self.target_lufs_min = target_lufs_min
        self.target_lufs_max = target_lufs_max
        self.max_true_peak_db = max_true_peak_db
        self.max_internal_silence_sec = max_internal_silence_sec
        self.min_net_cps = min_net_cps if min_net_cps is not None else min(2.0, min_cps)
        self.max_net_cps = max_net_cps if max_net_cps is not None else max(10.0, max_cps)
        self.whisper_min_rms_db = whisper_min_rms_db
        self.enable_vad = enable_vad

    def evaluate(
        self,
        audio_source: Union[str, bytes, "np.ndarray"],
        reference_text: Optional[str] = None,
        custom_min_duration: Optional[float] = None,
        custom_max_duration: Optional[float] = None,
        sample_rate: Optional[int] = None,
    ) -> AcousticReport:
        samples, sr = load_audio_samples(audio_source)
        if sample_rate is not None and sample_rate > 0:
            sr = sample_rate
        total_samples = len(samples)
        duration = float(total_samples) / float(sr) if sr > 0 else 0.0

        violations: List[str] = []

        if total_samples == 0 or duration == 0.0:
            return AcousticReport(
                passed=False,
                duration_sec=0.0,
                rms_db=-100.0,
                peak_db=-100.0,
                is_silent=True,
                is_clipping=False,
                cps=0.0,
                clipping_ratio=0.0,
                violations=["Empty audio: duration is 0 seconds"],
                metadata={"total_samples": 0, "sample_rate": sr},
                lufs=-100.0,
                true_peak_db=-100.0,
            )

        # Detect whisper / sigh tags in reference text
        is_whisper = is_whisper_text(reference_text)

        # 1. RMS Energy, Peak dBFS, True Peak and LUFS calculation
        abs_samples = np.abs(samples)
        peak_amp = float(np.max(abs_samples))
        peak_db = 20.0 * math.log10(max(peak_amp, 1e-9))

        rms_amp = float(np.sqrt(np.mean(samples**2)))
        rms_db = 20.0 * math.log10(max(rms_amp, 1e-9))

        # True Peak computation (dBTP)
        _, true_peak_db = calculate_true_peak(samples)

        # LUFS integrated loudness
        lufs_val = calculate_lufs(samples, sr)

        # Silence Check (with whisper tolerance)
        effective_min_rms = self.whisper_min_rms_db if is_whisper else self.min_rms_db
        is_silent = rms_db < effective_min_rms
        if is_silent:
            violations.append(
                f"Silence detected: RMS energy {rms_db:.2f} dBFS is below threshold {effective_min_rms} dBFS"
                f"{' (whisper mode)' if is_whisper else ''}"
            )

        # 2. Duration Bound Check
        min_dur = custom_min_duration if custom_min_duration is not None else self.min_duration_sec
        max_dur = custom_max_duration if custom_max_duration is not None else self.max_duration_sec

        if duration < min_dur:
            violations.append(f"Audio duration too short ({duration:.2f}s < {min_dur:.2f}s)")
        elif duration > max_dur:
            violations.append(f"Audio duration too long ({duration:.2f}s > {max_dur:.2f}s)")

        # 3. Clipping & True Peak Detection
        clip_linear = 10.0 ** (self.clipping_threshold_db / 20.0)
        clipped_mask = abs_samples >= clip_linear
        clipping_count = int(np.sum(clipped_mask))
        clipping_ratio = float(clipping_count) / float(total_samples)

        has_consecutive_clip = False
        if clipping_count >= self.max_consecutive_clipping:
            diffs = np.diff(np.pad(clipped_mask.astype(int), (1, 1), 'constant'))
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            run_lengths = ends - starts
            if len(run_lengths) > 0 and np.max(run_lengths) >= self.max_consecutive_clipping:
                has_consecutive_clip = True

        is_clipping = (clipping_ratio > self.max_clipping_ratio) or has_consecutive_clip
        if is_clipping:
            violations.append(
                f"Audio clipping/overload detected: peak={peak_db:.2f} dBFS (true_peak={true_peak_db:.2f} dBTP), "
                f"clipping_ratio={clipping_ratio * 100:.3f}%, "
                f"consecutive_clipped_samples={'yes' if has_consecutive_clip else 'no'}"
            )

        # 4. VAD Analysis, Internal Silence & Speech Rate (Net CPS)
        speech_intervals: List[Tuple[float, float]] = []
        voiced_duration = duration
        max_internal_silence = 0.0
        net_cps: Optional[float] = None
        cps: Optional[float] = None

        if self.enable_vad and total_samples > 0:
            speech_intervals, voiced_duration, max_internal_silence = analyze_vad(
                samples, sr, threshold_db=effective_min_rms + 3.0
            )
            # Check internal pauses between speech
            if max_internal_silence > self.max_internal_silence_sec:
                violations.append(
                    f"Internal pause too long: max non-speech gap {max_internal_silence * 1000:.0f}ms "
                    f"exceeds threshold {self.max_internal_silence_sec * 1000:.0f}ms (potential stutter/stalling)"
                )

        if reference_text:
            # Strip control/emotion prompt tags e.g. [whisper], (sigh), <emotion>
            stripped_text = re.sub(r"\[.*?\]|\(.*?\)|<.*?>|（.*?）|【.*?】", "", reference_text)
            clean_chars = [c for c in stripped_text if not c.isspace() and c not in "，。！？,.!?:;\"'‘’“”~…"]
            char_count = len(clean_chars) or len(stripped_text.strip())
            if char_count > 0:
                # Traditional raw CPS
                cps = float(char_count) / max(duration, 0.01)
                if cps < self.min_cps:
                    violations.append(
                        f"Speech rate abnormally slow / potential hallucination loop: "
                        f"CPS={cps:.2f} chars/sec (min {self.min_cps}) for {char_count} chars in {duration:.2f}s"
                    )
                elif cps > self.max_cps:
                    violations.append(
                        f"Speech rate abnormally fast / potential swallowing/truncation: "
                        f"CPS={cps:.2f} chars/sec (max {self.max_cps}) for {char_count} chars in {duration:.2f}s"
                    )

                # Net CPS over voiced segments
                if self.enable_vad and voiced_duration > 0:
                    net_cps = float(char_count) / max(voiced_duration, 0.01)
                    if net_cps < self.min_net_cps:
                        violations.append(
                            f"Net speech rate abnormally slow: net CPS={net_cps:.2f} chars/sec (min {self.min_net_cps}) "
                            f"on voiced duration {voiced_duration:.2f}s"
                        )
                    elif net_cps > self.max_net_cps:
                        violations.append(
                            f"Net speech rate abnormally fast: net CPS={net_cps:.2f} chars/sec (max {self.max_net_cps}) "
                            f"on voiced duration {voiced_duration:.2f}s"
                        )

        passed = len(violations) == 0

        return AcousticReport(
            passed=passed,
            duration_sec=round(duration, 3),
            rms_db=round(rms_db, 2),
            peak_db=round(peak_db, 2),
            is_silent=is_silent,
            is_clipping=is_clipping,
            cps=round(cps, 2) if cps is not None else None,
            clipping_ratio=round(clipping_ratio, 5),
            violations=violations,
            metadata={
                "sample_rate": sr,
                "total_samples": total_samples,
                "peak_amplitude": round(peak_amp, 4),
                "speech_intervals": speech_intervals,
            },
            lufs=round(lufs_val, 2),
            true_peak_db=round(true_peak_db, 2),
            max_internal_silence_sec=max_internal_silence,
            voiced_duration_sec=voiced_duration,
            net_cps=round(net_cps, 2) if net_cps is not None else None,
            is_whisper=is_whisper,
        )


def check_acoustic_quality(
    audio_source: Union[str, bytes],
    reference_text: Optional[str] = None,
    **kwargs,
) -> AcousticReport:
    """Convenience helper to evaluate audio against default or customized thresholds."""
    gate = AcousticQualityGate(**kwargs)
    return gate.evaluate(audio_source, reference_text=reference_text)


def assert_acoustic_quality(
    audio_source: Union[str, bytes],
    reference_text: Optional[str] = None,
    **kwargs,
) -> AcousticReport:
    """Convenience helper that evaluates quality and raises AssertionError if violations exist."""
    report = check_acoustic_quality(audio_source, reference_text=reference_text, **kwargs)
    if not report.passed:
        raise AssertionError(f"Acoustic quality gate failed: {'; '.join(report.violations)} (Report: {report.to_dict()})")
    return report
