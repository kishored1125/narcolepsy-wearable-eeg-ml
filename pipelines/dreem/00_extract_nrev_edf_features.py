#!/usr/bin/env python3
"""Extract Narcolepsy Revolution Dreem EDF features.

This script mirrors the Dreem H5 extraction workflow, but reads EDF files
instead of H5 files. It is intentionally self-contained so it can be copied to
the project environment and executed without importing local project
modules.

Expected data structure:
    __dl/
      narcorev_01/
        <recording_uuid>/
          <recording_uuid>.edf
          <participant>_hypnogram.txt

The data root is treated as read-only. All outputs are written only to
the user-supplied output folders.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats
from tqdm import tqdm

try:
    import mne
except ModuleNotFoundError as exc:
    raise SystemExit("mne is required for EDF processing. Install with: python -m pip install mne") from exc


LABEL_MAP = {
    "SLEEP-S0": "Wake",
    "SLEEP-S1": "N1",
    "SLEEP-S2": "N2",
    "SLEEP-S3": "N3",
    "SLEEP-REM": "REM",
    "Sleep stage W": "Wake",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "W": "Wake",
    "N1": "N1",
    "N2": "N2",
    "N3": "N3",
    "R": "REM",
    "REM": "REM",
}

STAGE_ORDER = ["Wake", "N1", "N2", "N3", "REM"]

SLEEP_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}

DEFAULT_CHANNEL_PATTERNS = ["F7", "F8", "O1", "O2", "FP1", "EEG", "E1", "E2"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract NRev Dreem EDF EEG features without modifying source data.")
    parser.add_argument("--data-root", required=True, help="Shared read-only Narcolepsy Revolution __dl folder.")
    parser.add_argument("--scratch-out", required=True, help="Writable output folder for epoch-level parquet files.")
    parser.add_argument("--home-out", required=True, help="Writable summary output folder for record/subject CSV files.")
    parser.add_argument("--max-records", type=int, default=None, help="Limit EDF records for a test run. Omit for full run.")
    parser.add_argument("--epoch-seconds", type=int, default=30)
    parser.add_argument("--max-channels", type=int, default=4, help="Maximum EEG channels to use.")
    parser.add_argument(
        "--channel-patterns",
        nargs="+",
        default=DEFAULT_CHANNEL_PATTERNS,
        help="Preferred channel name patterns, in priority order.",
    )
    parser.add_argument(
        "--channel-mode",
        choices=["per_channel", "average", "both"],
        default="per_channel",
        help="Extract features separately per channel, from the averaged channel, or both.",
    )
    parser.add_argument("--save-epoch-csv", action="store_true", help="Also save a combined epoch CSV. This can be large.")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    scratch_out = Path(args.scratch_out).expanduser().resolve()
    home_out = Path(args.home_out).expanduser().resolve()
    assert_not_inside_data_root(scratch_out, data_root, "--scratch-out")
    assert_not_inside_data_root(home_out, data_root, "--home-out")

    epoch_out = scratch_out / "nrev_edf_epoch_features_by_record"
    epoch_out.mkdir(parents=True, exist_ok=True)
    home_out.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise SystemExit(f"Data root does not exist: {data_root}")

    edf_files = discover_edf_files(data_root)
    if args.max_records is not None:
        edf_files = edf_files[: args.max_records]

    failures: list[dict[str, object]] = []
    record_rows: list[dict[str, object]] = []
    epoch_feature_files: list[Path] = []

    for edf_path in tqdm(edf_files, desc="NRev EDF records"):
        try:
            hypnogram_path = find_hypnogram(edf_path)
            if hypnogram_path is None:
                raise FileNotFoundError("No *_hypnogram.txt found in record folder.")
            epoch_df, channels = extract_record_features(
                edf_path=edf_path,
                hypnogram_path=hypnogram_path,
                epoch_seconds=args.epoch_seconds,
                max_channels=args.max_channels,
                channel_patterns=args.channel_patterns,
                channel_mode=args.channel_mode,
            )
            if epoch_df.empty:
                raise ValueError("No features extracted.")

            patient_id = patient_from_path(edf_path)
            recording_id = edf_path.stem
            out_file = epoch_out / f"{patient_id}__{recording_id}.parquet"
            epoch_df.to_parquet(out_file, index=False)
            epoch_feature_files.append(out_file)
            record_rows.append(record_summary(epoch_df, patient_id, recording_id, edf_path, hypnogram_path, channels))
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "edf_path": str(edf_path),
                    "patient_id": patient_from_path(edf_path),
                    "recording_id": edf_path.stem,
                    "error": str(exc),
                }
            )

    failures_df = pd.DataFrame(failures)
    failures_df.to_csv(home_out / "nrev_edf_failures.csv", index=False)

    if not record_rows:
        raise SystemExit("No NRev EDF records were processed. See nrev_edf_failures.csv.")

    record_features = pd.DataFrame(record_rows)
    record_features.to_csv(home_out / "nrev_edf_record_features.csv", index=False)

    subject_features = aggregate_subject_features(record_features)
    subject_features.to_csv(home_out / "nrev_edf_subject_features.csv", index=False)

    channel_inventory = record_features[["patient_id", "recording_id", "channels", "n_channels"]].copy()
    channel_inventory.to_csv(home_out / "nrev_edf_channel_inventory.csv", index=False)

    summary = {
        "data_root_read_only": str(data_root),
        "scratch_output": str(scratch_out),
        "home_output": str(home_out),
        "edf_files_seen": len(edf_files),
        "records_processed": len(record_rows),
        "records_failed": len(failures),
        "subjects_processed": int(record_features["patient_id"].nunique()),
        "channel_mode": args.channel_mode,
        "max_channels": args.max_channels,
        "epoch_seconds": args.epoch_seconds,
        "epoch_feature_files": len(epoch_feature_files),
    }
    (home_out / "nrev_edf_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(home_out / "nrev_edf_run_summary.csv", index=False)

    if args.save_epoch_csv:
        combined = pd.concat([pd.read_parquet(path) for path in epoch_feature_files], ignore_index=True)
        combined.to_csv(home_out / "nrev_edf_epoch_features.csv", index=False)

    print(json.dumps(summary, indent=2))


def discover_edf_files(data_root: Path) -> list[Path]:
    files = []
    for path in sorted(data_root.rglob("*.edf")):
        if path.name.startswith("._"):
            continue
        if "ignore_test" in path.parts:
            continue
        if patient_from_path(path) is None:
            continue
        files.append(path)
    return files


def assert_not_inside_data_root(path: Path, data_root: Path, name: str) -> None:
    try:
        path.resolve().relative_to(data_root.resolve())
    except ValueError:
        return
    raise SystemExit(f"Refusing to write {name} inside data root: {path}")


def find_hypnogram(edf_path: Path) -> Path | None:
    hypnograms = sorted(edf_path.parent.glob("*_hypnogram.txt"))
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
            label = LABEL_MAP.get(raw_stage) or LABEL_MAP.get(event)
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
    edf_path: Path,
    hypnogram_path: Path,
    epoch_seconds: int,
    max_channels: int,
    channel_patterns: list[str],
    channel_mode: str,
) -> tuple[pd.DataFrame, list[str]]:
    hyp = parse_hypnogram(hypnogram_path)
    if hyp.empty:
        return pd.DataFrame(), []

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    channels = choose_channels(raw, channel_patterns, max_channels)
    if not channels:
        raise ValueError(f"No usable EEG channels found in {edf_path.name}: {raw.ch_names}")

    raw.pick(channels)
    raw.filter(0.5, 30.0, fir_design="firwin", verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    epoch_len = int(round(epoch_seconds * sfreq))
    n_epochs = min(len(hyp), raw.n_times // epoch_len)
    if n_epochs == 0:
        return pd.DataFrame(), channels

    rows = []
    for epoch_idx in range(n_epochs):
        label = hyp.iloc[epoch_idx]["label"]
        if label not in STAGE_ORDER:
            continue
        start = epoch_idx * epoch_len
        stop = start + epoch_len
        data = raw.get_data(start=start, stop=stop)
        features = extract_epoch_features(data, sfreq, raw.ch_names, channel_mode)
        features.update(
            {
                "patient_id": patient_from_path(edf_path),
                "recording_id": edf_path.stem,
                "epoch_index": epoch_idx,
                "label": label,
                "raw_stage": hyp.iloc[epoch_idx]["raw_stage"],
            }
        )
        rows.append(features)
    return pd.DataFrame(rows), channels


def choose_channels(raw, channel_patterns: list[str], max_channels: int) -> list[str]:
    ch_names = list(raw.ch_names)
    selected: list[str] = []
    for pattern in channel_patterns:
        for ch in ch_names:
            if ch in selected:
                continue
            if pattern.lower() in ch.lower():
                selected.append(ch)
                if len(selected) >= max_channels:
                    return selected

    eeg_like = []
    try:
        eeg_idx = mne.pick_types(raw.info, eeg=True, eog=False, emg=False, ecg=False, exclude=[])
        eeg_like = [ch_names[idx] for idx in eeg_idx]
    except Exception:  # noqa: BLE001
        eeg_like = []
    for ch in eeg_like:
        if ch not in selected:
            selected.append(ch)
            if len(selected) >= max_channels:
                return selected

    return selected[:max_channels] if selected else ch_names[:max_channels]


def extract_epoch_features(data: np.ndarray, sfreq: float, channel_names: list[str], channel_mode: str) -> dict[str, float]:
    features: dict[str, float] = {}
    if channel_mode in {"per_channel", "both"}:
        for idx, ch_name in enumerate(channel_names):
            prefix = clean_channel_name(ch_name)
            x = signal.detrend(np.asarray(data[idx], dtype=float))
            features.update(time_domain_features(x, prefix))
            features.update(spectral_features(x, sfreq, prefix))
    if channel_mode in {"average", "both"}:
        avg = signal.detrend(np.nanmean(np.asarray(data, dtype=float), axis=0))
        features.update(time_domain_features(avg, "eeg_avg"))
        features.update(spectral_features(avg, sfreq, "eeg_avg"))
    return features


def clean_channel_name(name: str) -> str:
    return (
        str(name)
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "p")
        .replace("(", "")
        .replace(")", "")
    )


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
        f"{prefix}_hjorth_mobility": mobility,
        f"{prefix}_hjorth_complexity": complexity,
    }


def spectral_features(x: np.ndarray, sfreq: float, prefix: str) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    if x.size < int(sfreq) or np.nanstd(x) == 0:
        out = {f"{prefix}_total_power_0p5_30": np.nan}
        for band in SLEEP_BANDS:
            out[f"{prefix}_{band}_power"] = np.nan
            out[f"{prefix}_{band}_relative_power"] = np.nan
        out[f"{prefix}_theta_alpha_ratio"] = np.nan
        out[f"{prefix}_delta_beta_ratio"] = np.nan
        out[f"{prefix}_delta_sigma_ratio"] = np.nan
        return out

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


def record_summary(
    epoch_df: pd.DataFrame,
    patient_id: str | None,
    recording_id: str,
    edf_path: Path,
    hypnogram_path: Path,
    channels: list[str],
) -> dict[str, object]:
    row: dict[str, object] = {
        "patient_id": patient_id,
        "recording_id": recording_id,
        "edf_path": str(edf_path),
        "hypnogram_path": str(hypnogram_path),
        "n_epochs": len(epoch_df),
        "n_channels": len(channels),
        "channels": ";".join(channels),
    }
    counts = epoch_df["label"].value_counts()
    sleep_epochs = int(counts.drop(labels=["Wake"], errors="ignore").sum())
    row["wake_epochs"] = int(counts.get("Wake", 0))
    row["sleep_epochs"] = sleep_epochs
    row["sleep_efficiency_epoch_ratio"] = sleep_epochs / len(epoch_df) if len(epoch_df) else np.nan
    for label in STAGE_ORDER:
        row[f"{label}_epochs"] = int(counts.get(label, 0))
        row[f"{label}_percentage"] = int(counts.get(label, 0)) / sleep_epochs if sleep_epochs and label != "Wake" else np.nan

    numeric_cols = [c for c in epoch_df.select_dtypes(include=[np.number]).columns if c != "epoch_index"]
    overall = epoch_df[numeric_cols].agg(["mean", "std", "median"])
    for stat_name in overall.index:
        for col, value in overall.loc[stat_name].items():
            row[f"{col}_{stat_name}"] = value

    for stage_name, stage_df in epoch_df.groupby("label"):
        stage_numeric = stage_df[numeric_cols]
        stage_mean = stage_numeric.mean()
        for col, value in stage_mean.items():
            row[f"{stage_name.lower()}__{col}_mean"] = value
    return row


def aggregate_subject_features(record_features: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = record_features.select_dtypes(include=[np.number]).columns.difference(["epoch_index"])
    grouped = record_features.groupby("patient_id", dropna=False)
    parts = []
    for stat_name, func in [("mean", "mean"), ("median", "median"), ("std", "std"), ("min", "min"), ("max", "max")]:
        stat = getattr(grouped[numeric_cols], func)()
        stat.columns = [f"{col}_{stat_name}" for col in stat.columns]
        parts.append(stat)
    out = pd.concat(parts, axis=1).reset_index()
    out["nrev_record_count"] = grouped.size().values
    return out


if __name__ == "__main__":
    main()
