#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

from run_dreem_h5_pipeline import (
    aggregate_subject_features,
    estimate_sfreq,
    find_hypnogram,
    parse_hypnogram,
    patient_from_path,
    record_summary,
    spectral_features,
    time_domain_features,
)


SPINDLE_BAND = (11.0, 16.0)
SLOW_WAVE_BAND = (0.5, 4.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract advanced Dreem H5 EEG features: entropy, spindle density and slow-wave density."
    )
    parser.add_argument("--data-root", required=True, help="Shared read-only Dreem __dl folder.")
    parser.add_argument("--scratch-out", required=True, help="Writable output folder for per-record epoch features.")
    parser.add_argument("--home-out", required=True, help="Writable summary output folder.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional test limit.")
    parser.add_argument("--channels", nargs="+", default=["eeg1", "eeg2", "eeg3", "eeg4"])
    parser.add_argument("--signal-version", choices=["filtered", "raw"], default="filtered")
    parser.add_argument("--epoch-seconds", type=int, default=30)
    parser.add_argument(
        "--include-sample-entropy",
        action="store_true",
        help="Also compute sample entropy. This is slow at full-dataset scale, so leave off for the main run.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    scratch_out = Path(args.scratch_out).expanduser().resolve()
    home_out = Path(args.home_out).expanduser().resolve()
    epoch_out = scratch_out / "advanced_epoch_features_by_record"
    scratch_out.mkdir(parents=True, exist_ok=True)
    home_out.mkdir(parents=True, exist_ok=True)
    epoch_out.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise SystemExit(f"Data root does not exist: {data_root}")

    h5_files = [p for p in sorted(data_root.rglob("*.h5")) if "ignore_test" not in p.parts]
    if args.max_records is not None:
        h5_files = h5_files[: args.max_records]

    failures = []
    record_rows = []
    epoch_feature_files = []

    for h5_path in tqdm(h5_files, desc="Advanced Dreem H5 records"):
        try:
            hypnogram_path = find_hypnogram(h5_path)
            if hypnogram_path is None:
                raise FileNotFoundError("No *_hypnogram.txt found in record folder.")
            features = extract_advanced_record_features(
                h5_path=h5_path,
                hypnogram_path=hypnogram_path,
                channels=args.channels,
                signal_version=args.signal_version,
                epoch_seconds=args.epoch_seconds,
                include_sample_entropy=args.include_sample_entropy,
            )
            if features.empty:
                raise ValueError("No features extracted.")

            patient_id = patient_from_path(h5_path)
            recording_id = h5_path.stem
            out_file = epoch_out / f"{patient_id}__{recording_id}.parquet"
            features.to_parquet(out_file, index=False)
            epoch_feature_files.append(out_file)
            record_rows.append(record_summary(features, patient_id, recording_id, h5_path, hypnogram_path))
        except Exception as exc:
            failures.append(
                {
                    "h5_path": str(h5_path),
                    "patient_id": patient_from_path(h5_path),
                    "recording_id": h5_path.stem,
                    "error": str(exc),
                }
            )

    pd.DataFrame(failures).to_csv(home_out / "dreem_h5_advanced_failures.csv", index=False)

    if not record_rows:
        raise SystemExit("No Dreem records were processed. See dreem_h5_advanced_failures.csv.")

    record_features = pd.DataFrame(record_rows)
    record_features.to_csv(home_out / "dreem_h5_advanced_record_features.csv", index=False)

    subject_features = aggregate_subject_features(record_features)
    subject_features.to_csv(home_out / "dreem_h5_advanced_subject_features.csv", index=False)

    feature_families = feature_family_counts(subject_features)
    pd.DataFrame([feature_families]).to_csv(home_out / "dreem_h5_advanced_feature_family_counts.csv", index=False)

    run_summary = {
        "data_root_read_only": str(data_root),
        "scratch_output": str(scratch_out),
        "home_output": str(home_out),
        "h5_files_seen": len(h5_files),
        "records_processed": len(record_rows),
        "records_failed": len(failures),
        "subjects_processed": int(record_features["patient_id"].nunique()),
        "channels": args.channels,
        "signal_version": args.signal_version,
        "include_sample_entropy": bool(args.include_sample_entropy),
        "epoch_feature_files": len(epoch_feature_files),
        **feature_families,
    }
    (home_out / "dreem_h5_advanced_run_summary.json").write_text(json.dumps(run_summary, indent=2))
    pd.DataFrame([run_summary]).to_csv(home_out / "dreem_h5_advanced_run_summary.csv", index=False)
    print(json.dumps(run_summary, indent=2))


def extract_advanced_record_features(
    h5_path: Path,
    hypnogram_path: Path,
    channels: list[str],
    signal_version: str,
    epoch_seconds: int,
    include_sample_entropy: bool,
) -> pd.DataFrame:
    hyp = parse_hypnogram(hypnogram_path)
    if hyp.empty:
        return pd.DataFrame()

    with h5py.File(h5_path, "r") as h5:
        sfreq = estimate_sfreq(h5)
        epoch_len = int(round(epoch_seconds * sfreq))
        available = [ch for ch in channels if ch in h5 and signal_version in h5[ch]]
        if not available:
            raise ValueError(f"None of requested channels available: {channels}")
        max_epochs_by_signal = min(h5[ch][signal_version].shape[0] // epoch_len for ch in available)
        n_epochs = min(len(hyp), max_epochs_by_signal)

        rows = []
        for epoch_idx in range(n_epochs):
            row = {
                "patient_id": patient_from_path(h5_path),
                "recording_id": h5_path.stem,
                "epoch_index": epoch_idx,
                "label": hyp.iloc[epoch_idx]["label"],
                "raw_stage": hyp.iloc[epoch_idx]["raw_stage"],
            }
            start = epoch_idx * epoch_len
            stop = start + epoch_len
            for channel in available:
                x = np.asarray(h5[channel][signal_version][start:stop], dtype=float)
                x = signal.detrend(x)
                row.update(time_domain_features(x, channel))
                row.update(spectral_features(x, sfreq, channel))
                row.update(entropy_features(x, sfreq, channel, include_sample_entropy))
                row.update(event_detector_features(x, sfreq, channel))
            rows.append(row)
    return pd.DataFrame(rows)


def entropy_features(x: np.ndarray, sfreq: float, prefix: str, include_sample_entropy: bool) -> dict[str, float]:
    freqs, psd = signal.welch(x, fs=sfreq, nperseg=min(len(x), int(round(4 * sfreq))), scaling="density")
    total_mask = (freqs >= 0.5) & (freqs <= 30.0)
    out = {
        f"{prefix}_spectral_entropy": spectral_entropy_from_psd(psd[total_mask]),
        f"{prefix}_permutation_entropy": permutation_entropy(x),
    }
    if include_sample_entropy:
        out[f"{prefix}_sample_entropy"] = sample_entropy(x)
    return out


def event_detector_features(x: np.ndarray, sfreq: float, prefix: str) -> dict[str, float]:
    out = {}
    out.update(_band_event_features(x, sfreq, prefix, "spindle", SPINDLE_BAND, min_duration=0.5, max_duration=3.0))
    out.update(_band_event_features(x, sfreq, prefix, "slowwave", SLOW_WAVE_BAND, min_duration=0.25, max_duration=2.0))
    return out


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
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < int(2 * sfreq) or np.nanstd(x) == 0:
        return {
            f"{prefix}_{event_name}_density": np.nan,
            f"{prefix}_{event_name}_mean_envelope": np.nan,
            f"{prefix}_{event_name}_max_envelope": np.nan,
        }
    filtered = _bandpass(x, sfreq, band[0], band[1])
    envelope = np.abs(signal.hilbert(filtered))
    threshold = np.nanmean(envelope) + 2.0 * np.nanstd(envelope)
    event_count = _count_threshold_events(envelope > threshold, sfreq, min_duration, max_duration)
    duration_minutes = x.size / sfreq / 60.0
    return {
        f"{prefix}_{event_name}_density": float(event_count / duration_minutes) if duration_minutes > 0 else np.nan,
        f"{prefix}_{event_name}_mean_envelope": float(np.nanmean(envelope)),
        f"{prefix}_{event_name}_max_envelope": float(np.nanmax(envelope)),
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


def feature_family_counts(df: pd.DataFrame) -> dict[str, int]:
    cols = [c.lower() for c in df.columns]
    return {
        "entropy_feature_count": sum("entropy" in c for c in cols),
        "spindle_feature_count": sum("spindle" in c for c in cols),
        "slowwave_feature_count": sum("slowwave" in c for c in cols),
        "total_subject_feature_columns": len(df.columns),
    }


if __name__ == "__main__":
    main()
