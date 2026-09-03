from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import signal, stats


SLEEP_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}

SPINDLE_BAND = (11.0, 16.0)
SLOW_WAVE_BAND = (0.5, 4.0)


def hjorth_parameters(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if x.size < 3 or np.nanstd(x) == 0:
        return np.nan, np.nan
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x = np.nanvar(x)
    var_dx = np.nanvar(dx)
    var_ddx = np.nanvar(ddx)
    mobility = np.sqrt(var_dx / var_x) if var_x > 0 else np.nan
    mobility_dx = np.sqrt(var_ddx / var_dx) if var_dx > 0 else np.nan
    complexity = mobility_dx / mobility if mobility and mobility > 0 else np.nan
    return mobility, complexity


def spectral_features(x: np.ndarray, sfreq: float, prefix: str) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    if x.size < int(sfreq) or np.nanstd(x) == 0:
        return {f"{prefix}_{name}_power": np.nan for name in SLEEP_BANDS}

    freqs, psd = signal.welch(
        x,
        fs=sfreq,
        nperseg=min(len(x), int(4 * sfreq)),
        noverlap=None,
        scaling="density",
    )
    total_mask = (freqs >= 0.5) & (freqs <= 30.0)
    total_power = _trapezoid(psd[total_mask], freqs[total_mask])
    out = {f"{prefix}_total_power_0p5_30": float(total_power)}
    for band, (lo, hi) in SLEEP_BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        power = _trapezoid(psd[mask], freqs[mask])
        out[f"{prefix}_{band}_power"] = float(power)
        out[f"{prefix}_{band}_relative_power"] = float(power / total_power) if total_power > 0 else np.nan

    out[f"{prefix}_theta_alpha_ratio"] = _safe_ratio(out[f"{prefix}_theta_power"], out[f"{prefix}_alpha_power"])
    out[f"{prefix}_delta_beta_ratio"] = _safe_ratio(out[f"{prefix}_delta_power"], out[f"{prefix}_beta_power"])
    out[f"{prefix}_delta_sigma_ratio"] = _safe_ratio(out[f"{prefix}_delta_power"], out[f"{prefix}_sigma_power"])
    out[f"{prefix}_spectral_entropy"] = spectral_entropy_from_psd(psd[total_mask])
    return out


def time_domain_features(x: np.ndarray, prefix: str) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    mobility, complexity = hjorth_parameters(x)
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_iqr": float(stats.iqr(x)),
        f"{prefix}_skew": float(stats.skew(x)) if x.size > 2 else np.nan,
        f"{prefix}_kurtosis": float(stats.kurtosis(x)) if x.size > 3 else np.nan,
        f"{prefix}_zero_crossings": float(np.sum(np.diff(np.signbit(x - np.mean(x))) != 0)),
        f"{prefix}_hjorth_mobility": float(mobility),
        f"{prefix}_hjorth_complexity": float(complexity),
        f"{prefix}_permutation_entropy": permutation_entropy(x),
    }


def advanced_event_features(x: np.ndarray, sfreq: float, prefix: str, include_sample_entropy: bool = False) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < int(2 * sfreq) or np.nanstd(x) == 0:
        out = {
            f"{prefix}_spindle_density": np.nan,
            f"{prefix}_spindle_mean_envelope": np.nan,
            f"{prefix}_slowwave_density": np.nan,
            f"{prefix}_slowwave_mean_envelope": np.nan,
        }
        if include_sample_entropy:
            out[f"{prefix}_sample_entropy"] = np.nan
        return out

    out = {}
    out.update(_band_event_features(x, sfreq, prefix, "spindle", SPINDLE_BAND, min_duration=0.5, max_duration=3.0))
    out.update(_band_event_features(x, sfreq, prefix, "slowwave", SLOW_WAVE_BAND, min_duration=0.25, max_duration=2.0))
    if include_sample_entropy:
        out[f"{prefix}_sample_entropy"] = sample_entropy(x)
    return out


def extract_epoch_features(data: np.ndarray, sfreq: float, channel_names: list[str]) -> dict[str, float]:
    features: dict[str, float] = {}
    for idx, ch_name in enumerate(channel_names):
        clean_name = (
            ch_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace(".", "p")
        )
        x = data[idx]
        x = signal.detrend(x)
        features.update(time_domain_features(x, clean_name))
        features.update(spectral_features(x, sfreq, clean_name))
        features.update(advanced_event_features(x, sfreq, clean_name))
    return features


def spectral_entropy_from_psd(psd: np.ndarray) -> float:
    psd = np.asarray(psd, dtype=float)
    psd = psd[np.isfinite(psd) & (psd > 0)]
    if psd.size <= 1:
        return np.nan
    probabilities = psd / psd.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy / np.log2(probabilities.size))


def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n_patterns = x.size - delay * (order - 1)
    if n_patterns <= 1:
        return np.nan
    embedded = np.column_stack([x[idx * delay : idx * delay + n_patterns] for idx in range(order)])
    patterns = np.argsort(embedded, axis=1)
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy / np.log2(math.factorial(order)))


def sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None, max_points: int = 500) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size > max_points:
        idx = np.linspace(0, x.size - 1, max_points).astype(int)
        x = x[idx]
    if x.size < m + 2 or np.std(x) == 0:
        return np.nan
    tolerance = 0.2 * np.std(x) if r is None else r
    a = _template_match_count(x, m + 1, tolerance)
    b = _template_match_count(x, m, tolerance)
    if a == 0 or b == 0:
        return np.nan
    return float(-np.log(a / b))


def _template_match_count(x: np.ndarray, m: int, tolerance: float) -> int:
    templates = np.array([x[i : i + m] for i in range(x.size - m + 1)])
    count = 0
    for idx in range(len(templates) - 1):
        distances = np.max(np.abs(templates[idx + 1 :] - templates[idx]), axis=1)
        count += int(np.sum(distances <= tolerance))
    return count


def _band_event_features(
    x: np.ndarray,
    sfreq: float,
    prefix: str,
    event_name: str,
    band: tuple[float, float],
    min_duration: float,
    max_duration: float,
) -> dict[str, float]:
    filtered = _bandpass(x, sfreq, band[0], band[1])
    envelope = np.abs(signal.hilbert(filtered))
    threshold = np.nanmean(envelope) + 2.0 * np.nanstd(envelope)
    event_count = _count_threshold_events(envelope > threshold, sfreq, min_duration, max_duration)
    duration_minutes = x.size / sfreq / 60.0
    return {
        f"{prefix}_{event_name}_density": float(event_count / duration_minutes) if duration_minutes > 0 else np.nan,
        f"{prefix}_{event_name}_mean_envelope": float(np.nanmean(envelope)),
    }


def _bandpass(x: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    high = min(high, sfreq / 2.0 - 0.1)
    if low <= 0 or high <= low:
        return np.full_like(x, np.nan, dtype=float)
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return signal.sosfiltfilt(sos, x)


def _count_threshold_events(mask: np.ndarray, sfreq: float, min_duration: float, max_duration: float) -> int:
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(np.diff(padded.astype(int)))
    starts, stops = changes[0::2], changes[1::2]
    durations = (stops - starts) / sfreq
    return int(np.sum((durations >= min_duration) & (durations <= max_duration)))


def aggregate_subject_features(records: pd.DataFrame, group_col: str, target_cols: list[str]) -> pd.DataFrame:
    numeric_cols = records.select_dtypes(include=[np.number]).columns.difference(target_cols)
    grouped = records.groupby(group_col, dropna=False)
    parts = []
    for stat_name, func in [("mean", "mean"), ("median", "median"), ("std", "std"), ("min", "min"), ("max", "max")]:
        stat = getattr(grouped[numeric_cols], func)()
        stat.columns = [f"{c}_{stat_name}" for c in stat.columns]
        parts.append(stat)
    out = pd.concat(parts, axis=1).reset_index()
    return out


def _safe_ratio(num: float, den: float) -> float:
    if den is None or not np.isfinite(den) or den == 0:
        return np.nan
    return float(num / den)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    integrator = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(integrator(y, x))
