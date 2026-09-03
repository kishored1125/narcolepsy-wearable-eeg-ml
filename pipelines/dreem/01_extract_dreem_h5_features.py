#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import signal, stats
from tqdm import tqdm


LABEL_MAP = {
    "SLEEP-S0": "Wake",
    "SLEEP-S1": "N1",
    "SLEEP-S2": "N2",
    "SLEEP-S3": "N3",
    "SLEEP-REM": "REM",
}

SLEEP_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract raw Dreem H5 EEG features without modifying source data.")
    parser.add_argument("--data-root", required=True, help="Shared read-only Dreem __dl folder.")
    parser.add_argument("--scratch-out", required=True, help="Writable output folder for per-record features.")
    parser.add_argument("--home-out", required=True, help="Writable summary output folder for compact tables.")
    parser.add_argument("--max-records", type=int, default=None, help="Limit records for testing. Omit for full run.")
    parser.add_argument("--channels", nargs="+", default=["eeg1", "eeg2", "eeg3", "eeg4"], help="H5 EEG channel groups to use.")
    parser.add_argument("--signal-version", choices=["filtered", "raw"], default="filtered")
    parser.add_argument("--epoch-seconds", type=int, default=30)
    parser.add_argument("--save-epoch-csv", action="store_true", help="Also save combined epoch CSV to home output. Can be large.")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    scratch_out = Path(args.scratch_out).expanduser().resolve()
    home_out = Path(args.home_out).expanduser().resolve()
    epoch_out = scratch_out / "epoch_features_by_record"
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

    for h5_path in tqdm(h5_files, desc="Dreem H5 records"):
        try:
            hypnogram_path = find_hypnogram(h5_path)
            if hypnogram_path is None:
                raise FileNotFoundError("No *_hypnogram.txt found in record folder.")
            features = extract_record_features(
                h5_path=h5_path,
                hypnogram_path=hypnogram_path,
                channels=args.channels,
                signal_version=args.signal_version,
                epoch_seconds=args.epoch_seconds,
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

    failures_df = pd.DataFrame(failures)
    failures_df.to_csv(home_out / "dreem_h5_failures.csv", index=False)

    if not record_rows:
        raise SystemExit("No Dreem records were processed. See dreem_h5_failures.csv.")

    record_features = pd.DataFrame(record_rows)
    record_features.to_csv(home_out / "dreem_h5_record_features.csv", index=False)

    subject_features = aggregate_subject_features(record_features)
    subject_features.to_csv(home_out / "dreem_h5_subject_features.csv", index=False)

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
        "epoch_feature_files": len(epoch_feature_files),
    }
    (home_out / "dreem_h5_run_summary.json").write_text(json.dumps(run_summary, indent=2))
    pd.DataFrame([run_summary]).to_csv(home_out / "dreem_h5_run_summary.csv", index=False)

    if args.save_epoch_csv:
        combined = pd.concat([pd.read_parquet(p) for p in epoch_feature_files], ignore_index=True)
        combined.to_csv(home_out / "dreem_h5_epoch_features.csv", index=False)

    print(json.dumps(run_summary, indent=2))


def find_hypnogram(h5_path: Path) -> Path | None:
    hypnograms = sorted(h5_path.parent.glob("*_hypnogram.txt"))
    return hypnograms[0] if hypnograms else None


def patient_from_path(path: Path) -> str | None:
    for part in path.parts:
        if re.fullmatch(r"narcorev_\d+", part):
            return part
    return None


def parse_hypnogram(path: Path) -> pd.DataFrame:
    rows = []
    in_table = False
    with path.open(errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Sleep Stage"):
                in_table = True
                continue
            if not in_table:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            raw_stage, time_text, event, duration = parts[:4]
            label = LABEL_MAP.get(raw_stage)
            if label is None:
                continue
            rows.append(
                {
                    "raw_stage": raw_stage,
                    "label": label,
                    "clock_time": time_text,
                    "event": event,
                    "duration_seconds": float(duration),
                }
            )
    return pd.DataFrame(rows)


def extract_record_features(
    h5_path: Path,
    hypnogram_path: Path,
    channels: list[str],
    signal_version: str,
    epoch_seconds: int,
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
            for ch in available:
                x = np.asarray(h5[ch][signal_version][start:stop], dtype=float)
                x = signal.detrend(x)
                row.update(time_domain_features(x, ch))
                row.update(spectral_features(x, sfreq, ch))
            rows.append(row)
    return pd.DataFrame(rows)


def estimate_sfreq(h5: h5py.File) -> float:
    if "eeg_timestamps" in h5 and len(h5["eeg_timestamps"]) > 100:
        ts = h5["eeg_timestamps"][: min(10000, len(h5["eeg_timestamps"]))]
        dt = np.median(np.diff(ts))
        if np.isfinite(dt) and dt > 0:
            return float(1.0 / dt)
    return 250.0


def hjorth_parameters(x: np.ndarray) -> tuple[float, float]:
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
    return float(mobility), float(complexity)


def time_domain_features(x: np.ndarray, prefix: str) -> dict[str, float]:
    mobility, complexity = hjorth_parameters(x)
    return {
        f"{prefix}_mean": float(np.nanmean(x)),
        f"{prefix}_std": float(np.nanstd(x)),
        f"{prefix}_iqr": float(stats.iqr(x, nan_policy="omit")),
        f"{prefix}_skew": float(stats.skew(x, nan_policy="omit")) if x.size > 2 else np.nan,
        f"{prefix}_kurtosis": float(stats.kurtosis(x, nan_policy="omit")) if x.size > 3 else np.nan,
        f"{prefix}_zero_crossings": float(np.sum(np.diff(np.signbit(x - np.nanmean(x))) != 0)),
        f"{prefix}_hjorth_mobility": mobility,
        f"{prefix}_hjorth_complexity": complexity,
    }


def spectral_features(x: np.ndarray, sfreq: float, prefix: str) -> dict[str, float]:
    freqs, psd = signal.welch(
        x,
        fs=sfreq,
        nperseg=min(len(x), int(round(4 * sfreq))),
        scaling="density",
    )
    total_mask = (freqs >= 0.5) & (freqs <= 30.0)
    total_power = trapezoid(psd[total_mask], freqs[total_mask])
    out = {f"{prefix}_total_power_0p5_30": total_power}
    for band, (lo, hi) in SLEEP_BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        power = trapezoid(psd[mask], freqs[mask])
        out[f"{prefix}_{band}_power"] = power
        out[f"{prefix}_{band}_relative_power"] = power / total_power if total_power > 0 else np.nan
    out[f"{prefix}_theta_alpha_ratio"] = safe_ratio(out[f"{prefix}_theta_power"], out[f"{prefix}_alpha_power"])
    out[f"{prefix}_delta_beta_ratio"] = safe_ratio(out[f"{prefix}_delta_power"], out[f"{prefix}_beta_power"])
    out[f"{prefix}_delta_sigma_ratio"] = safe_ratio(out[f"{prefix}_delta_power"], out[f"{prefix}_sigma_power"])
    return out


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b != 0 else np.nan


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    integrator = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(integrator(y, x))


def record_summary(features: pd.DataFrame, patient_id: str | None, recording_id: str, h5_path: Path, hypnogram_path: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "patient_id": patient_id,
        "recording_id": recording_id,
        "h5_path": str(h5_path),
        "hypnogram_path": str(hypnogram_path),
        "n_epochs": len(features),
    }
    counts = features["label"].value_counts()
    sleep_epochs = int(counts.drop(labels=["Wake"], errors="ignore").sum())
    row["wake_epochs"] = int(counts.get("Wake", 0))
    row["sleep_epochs"] = sleep_epochs
    row["tst_minutes_from_hypnogram"] = sleep_epochs * 0.5
    row["sleep_efficiency_epoch_ratio"] = sleep_epochs / len(features) if len(features) else np.nan
    for label in ["Wake", "N1", "N2", "N3", "REM"]:
        row[f"{label}_epochs"] = int(counts.get(label, 0))
        row[f"{label}_percentage"] = int(counts.get(label, 0)) / sleep_epochs if sleep_epochs > 0 and label != "Wake" else np.nan

    numeric_cols = features.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in {"epoch_index"}]
    grouped_stats = features[numeric_cols].agg(["mean", "std", "median"])
    for col in grouped_stats.columns:
        for stat in grouped_stats.index:
            row[f"{col}_{stat}"] = grouped_stats.loc[stat, col]
    return row


def aggregate_subject_features(record_features: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = record_features.select_dtypes(include=[np.number]).columns
    grouped = record_features.groupby("patient_id", dropna=False)[numeric_cols]
    parts = []
    for stat in ["mean", "median", "std", "min", "max"]:
        values = getattr(grouped, stat)()
        values.columns = [f"{c}_{stat}" for c in values.columns]
        parts.append(values)
    return pd.concat(parts, axis=1).reset_index()


if __name__ == "__main__":
    main()
