#!/usr/bin/env python3
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


SLEEP_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}
PREFERRED_EEG_CHANNELS = ["F3", "F4", "C3", "C4", "O1", "O2", "E1", "E2"]
STAGE_ORDER = ["Wake", "N1", "N2", "N3", "REM"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CNC EDF EEG features for T1 narcolepsy vs control.")
    parser.add_argument("--data-root", required=True, help="Shared read-only CNC folder.")
    parser.add_argument("--metadata", default=None, help="CNC metadata CSV/XLSX containing ID, Cohort and Diagnosis.")
    parser.add_argument("--scratch-out", required=True, help="Writable output directory.")
    parser.add_argument("--home-out", required=True, help="Writable summary output directory.")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--epoch-seconds", type=int, default=30)
    parser.add_argument("--max-channels", type=int, default=4)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    scratch = Path(args.scratch_out).expanduser().resolve()
    home = Path(args.home_out).expanduser().resolve()
    assert_not_inside_data_root(scratch, data_root, "--scratch-out")
    assert_not_inside_data_root(home, data_root, "--home-out")
    epoch_out = scratch / "cnc_epoch_features_by_record"
    epoch_out.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(Path(args.metadata).expanduser().resolve() if args.metadata else None)
    records = discover_records(data_root, metadata)
    if args.max_records:
        records = records[: args.max_records]

    record_rows = []
    failures = []
    for rec in tqdm(records, desc="CNC EDF records"):
        try:
            record_rows.append(process_record(rec, epoch_out, args.epoch_seconds, args.max_channels))
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "subject_id": rec["subject_id"],
                    "file_id": rec["file_id"],
                    "edf_path": str(rec["edf_path"]),
                    "csv_path": str(rec["csv_path"]),
                    "error": str(exc),
                }
            )

    record_df = pd.DataFrame(record_rows)
    failures_df = pd.DataFrame(failures)
    record_df.to_csv(home / "cnc_edf_record_features.csv", index=False)
    failures_df.to_csv(home / "cnc_edf_failures.csv", index=False)

    if not record_df.empty:
        subject_df = aggregate_subject_features(
            record_df,
            group_col="subject_id",
            target_cols=["subject_id", "file_id", "recording_id", "label", "diagnosis", "diagnosis_binary"],
        )
        keep_meta = record_df[
            ["subject_id", "file_id", "diagnosis", "label", "diagnosis_binary", "metadata_source"]
        ].drop_duplicates("subject_id")
        subject_df = subject_df.merge(keep_meta, on="subject_id", how="left")
        subject_df.to_csv(home / "cnc_edf_subject_features.csv", index=False)

    inventory = pd.DataFrame(records)
    if not inventory.empty:
        inventory.assign(edf_path=inventory["edf_path"].astype(str), csv_path=inventory["csv_path"].astype(str)).to_csv(
            home / "cnc_matched_file_inventory.csv", index=False
        )

    missing = metadata_missing_files(metadata, records)
    missing.to_csv(home / "cnc_metadata_without_matched_files.csv", index=False)
    unmatched = unmatched_files(data_root, records)
    unmatched.to_csv(home / "cnc_unmatched_files.csv", index=False)

    summary = {
        "data_root_read_only": str(data_root),
        "metadata": str(args.metadata) if args.metadata else "prefix-inferred fallback",
        "scratch_output": str(scratch),
        "home_output": str(home),
        "candidate_records": len(records),
        "records_processed": len(record_df),
        "records_failed": len(failures_df),
        "subjects_processed": int(record_df["subject_id"].nunique()) if not record_df.empty else 0,
        "control_subjects_processed": int((record_df["diagnosis_binary"] == 0).sum()) if not record_df.empty else 0,
        "t1_narcolepsy_subjects_processed": int((record_df["diagnosis_binary"] == 1).sum()) if not record_df.empty else 0,
        "epoch_feature_files": len(list(epoch_out.glob("*.parquet"))),
    }
    pd.DataFrame([summary]).to_csv(home / "cnc_edf_run_summary.csv", index=False)
    (home / "cnc_edf_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def assert_not_inside_data_root(path: Path, data_root: Path, name: str) -> None:
    try:
        path.resolve().relative_to(data_root.resolve())
    except ValueError:
        return
    raise SystemExit(f"Refusing to write {name} inside data root: {path}")


def load_metadata(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["subject_id", "diagnosis", "label", "diagnosis_binary", "metadata_source"])
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if "Cohort" in df.columns:
        df = df[df["Cohort"].astype(str).str.upper().eq("CNC")].copy()
    if "ID" not in df.columns or "Diagnosis" not in df.columns:
        raise ValueError("Metadata must contain ID and Diagnosis columns.")
    out = pd.DataFrame()
    out["subject_id"] = df["ID"].astype(str).str.upper().str.strip()
    out["diagnosis"] = df["Diagnosis"].astype(str).str.replace("'", "", regex=False).str.strip()
    out["label"] = out["diagnosis"].map(
        {
            "NON-NARCOLEPSY CONTROL": "control",
            "T1 NARCOLEPSY": "t1_narcolepsy",
        }
    )
    out["diagnosis_binary"] = out["label"].map({"control": 0, "t1_narcolepsy": 1})
    out["metadata_source"] = str(path)
    return out[out["label"].notna()].drop_duplicates("subject_id")


def discover_records(data_root: Path, metadata: pd.DataFrame) -> list[dict[str, object]]:
    label_map = metadata.set_index("subject_id").to_dict("index") if not metadata.empty else {}
    records = []
    for edf_path in sorted(data_root.glob("*.edf")):
        if edf_path.name.startswith("._"):
            continue
        file_id = edf_path.stem.replace("-nsrr", "")
        subject_id = normalise_file_id(file_id)
        csv_path = edf_path.with_suffix(".csv")
        if not csv_path.exists():
            continue
        meta = label_map.get(subject_id)
        if meta is None:
            meta = infer_label_from_subject_id(subject_id)
        if meta is None:
            continue
        records.append(
            {
                "subject_id": subject_id,
                "file_id": file_id,
                "recording_id": edf_path.stem,
                "edf_path": edf_path,
                "csv_path": csv_path,
                "diagnosis": meta["diagnosis"],
                "label": meta["label"],
                "diagnosis_binary": int(meta["diagnosis_binary"]),
                "metadata_source": meta["metadata_source"],
            }
        )
    return records


def normalise_file_id(file_id: str) -> str:
    value = file_id.upper().strip()
    match = re.fullmatch(r"(CHC|CHP)(\d{3})", value)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return value


def infer_label_from_subject_id(subject_id: str) -> dict[str, object] | None:
    if re.fullmatch(r"CHC\d{3}", subject_id):
        return {
            "diagnosis": "NON-NARCOLEPSY CONTROL",
            "label": "control",
            "diagnosis_binary": 0,
            "metadata_source": "inferred_from_CHC_prefix",
        }
    if re.fullmatch(r"CHP\d{3}", subject_id):
        return {
            "diagnosis": "T1 NARCOLEPSY",
            "label": "t1_narcolepsy",
            "diagnosis_binary": 1,
            "metadata_source": "inferred_from_CHP_prefix",
        }
    return None


def process_record(rec: dict[str, object], epoch_out: Path, epoch_seconds: int, max_channels: int) -> dict[str, object]:
    hyp = read_annotation_csv(Path(rec["csv_path"]), epoch_seconds)
    raw = mne.io.read_raw_edf(Path(rec["edf_path"]), preload=True, verbose="ERROR")
    picks = choose_channels(raw.ch_names, max_channels)
    if not picks:
        raise ValueError(f"No usable EEG channels found: {raw.ch_names}")
    raw.pick(picks)
    raw.filter(0.5, 30.0, fir_design="firwin", verbose="ERROR")

    sfreq = float(raw.info["sfreq"])
    epoch_len = int(round(epoch_seconds * sfreq))
    n_epochs = min(len(hyp), raw.n_times // epoch_len)
    if n_epochs == 0:
        raise ValueError("No aligned 30-second sleep-stage epochs found.")

    rows = []
    for idx in range(n_epochs):
        label = hyp.loc[idx, "label"]
        if label not in STAGE_ORDER:
            continue
        start = idx * epoch_len
        stop = start + epoch_len
        data = raw.get_data(start=start, stop=stop)
        features = extract_epoch_features(data, sfreq, raw.ch_names)
        features.update(
            {
                "subject_id": rec["subject_id"],
                "file_id": rec["file_id"],
                "recording_id": rec["recording_id"],
                "epoch_index": idx,
                "label": label,
                "diagnosis_binary": rec["diagnosis_binary"],
            }
        )
        rows.append(features)
    epoch_df = pd.DataFrame(rows)
    if epoch_df.empty:
        raise ValueError("No features extracted after stage filtering.")

    out_file = epoch_out / f"{rec['subject_id']}__{rec['recording_id']}.parquet"
    epoch_df.to_parquet(out_file, index=False)
    return summarize_record(epoch_df, rec, picks)


def read_annotation_csv(path: Path, epoch_seconds: int) -> pd.DataFrame:
    ann = pd.read_csv(path)
    required = {"onset", "duration", "description"}
    if not required.issubset(set(ann.columns)):
        raise ValueError(f"Annotation CSV must contain columns {required}; found {list(ann.columns)}")
    ann = ann.sort_values("onset").reset_index(drop=True)
    labels: list[str] = []
    for row in ann.itertuples(index=False):
        label = parse_stage_description(str(row.description))
        if label is None:
            continue
        duration = float(row.duration)
        repeats = max(1, int(round(duration / epoch_seconds)))
        labels.extend([label] * repeats)
    return pd.DataFrame({"label": labels})


def parse_stage_description(description: str) -> str | None:
    text = description.lower()
    if "stage 1" in text or " n1" in text or text.endswith("n1"):
        return "N1"
    if "stage 2" in text or " n2" in text or text.endswith("n2"):
        return "N2"
    if "stage 3" in text or "stage 4" in text or " n3" in text or text.endswith("n3"):
        return "N3"
    if "rem" in text or "stage r" in text:
        return "REM"
    if "wake" in text or "stage w" in text:
        return "Wake"
    return None


def choose_channels(ch_names: list[str], max_channels: int) -> list[str]:
    selected = []
    for wanted in PREFERRED_EEG_CHANNELS:
        for ch in ch_names:
            if ch.upper() == wanted and ch not in selected:
                selected.append(ch)
                break
    return selected[:max_channels]


def extract_epoch_features(data: np.ndarray, sfreq: float, channel_names: list[str]) -> dict[str, float]:
    features: dict[str, float] = {}
    for idx, ch_name in enumerate(channel_names):
        clean_name = clean_channel_name(ch_name)
        x = np.asarray(data[idx], dtype=float)
        x = signal.detrend(x)
        features.update(time_domain_features(x, clean_name))
        features.update(spectral_features(x, sfreq, clean_name))
    return features


def clean_channel_name(name: str) -> str:
    return str(name).lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "p")


def time_domain_features(x: np.ndarray, prefix: str) -> dict[str, float]:
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
    }


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


def spectral_features(x: np.ndarray, sfreq: float, prefix: str) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    if x.size < int(sfreq) or np.nanstd(x) == 0:
        return empty_spectral(prefix)
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


def empty_spectral(prefix: str) -> dict[str, float]:
    out = {f"{prefix}_total_power_0p5_30": np.nan}
    for band in SLEEP_BANDS:
        out[f"{prefix}_{band}_power"] = np.nan
        out[f"{prefix}_{band}_relative_power"] = np.nan
    out[f"{prefix}_theta_alpha_ratio"] = np.nan
    out[f"{prefix}_delta_beta_ratio"] = np.nan
    out[f"{prefix}_delta_sigma_ratio"] = np.nan
    return out


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b != 0 else np.nan


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    integrator = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(integrator(y, x))


def summarize_record(epoch_df: pd.DataFrame, rec: dict[str, object], channels: list[str]) -> dict[str, object]:
    row: dict[str, object] = {
        "subject_id": rec["subject_id"],
        "file_id": rec["file_id"],
        "recording_id": rec["recording_id"],
        "diagnosis": rec["diagnosis"],
        "label": rec["label"],
        "diagnosis_binary": rec["diagnosis_binary"],
        "metadata_source": rec["metadata_source"],
        "n_epochs": len(epoch_df),
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
    numeric_cols = [
        c
        for c in epoch_df.select_dtypes(include=[np.number]).columns
        if c not in {"epoch_index", "diagnosis_binary"}
    ]
    stats_df = epoch_df[numeric_cols].agg(["mean", "std", "median"])
    for stat in stats_df.index:
        for col, value in stats_df.loc[stat].items():
            row[f"{col}_{stat}"] = value
    return row


def aggregate_subject_features(records: pd.DataFrame, group_col: str, target_cols: list[str]) -> pd.DataFrame:
    numeric_cols = records.select_dtypes(include=[np.number]).columns.difference(target_cols)
    grouped = records.groupby(group_col, dropna=False)
    parts = []
    for stat_name, func in [("mean", "mean"), ("median", "median"), ("std", "std"), ("min", "min"), ("max", "max")]:
        stat = getattr(grouped[numeric_cols], func)()
        stat.columns = [f"{c}_{stat_name}" for c in stat.columns]
        parts.append(stat)
    return pd.concat(parts, axis=1).reset_index()


def metadata_missing_files(metadata: pd.DataFrame, records: list[dict[str, object]]) -> pd.DataFrame:
    if metadata.empty:
        return pd.DataFrame()
    matched = {str(r["subject_id"]) for r in records}
    return metadata[~metadata["subject_id"].isin(matched)].copy()


def unmatched_files(data_root: Path, records: list[dict[str, object]]) -> pd.DataFrame:
    matched_edfs = {Path(r["edf_path"]).name for r in records}
    rows = []
    for edf_path in sorted(data_root.glob("*.edf")):
        if edf_path.name.startswith("._"):
            continue
        if edf_path.name not in matched_edfs:
            rows.append(
                {
                    "file_name": edf_path.name,
                    "file_id": edf_path.stem.replace("-nsrr", ""),
                    "normalised_id": normalise_file_id(edf_path.stem.replace("-nsrr", "")),
                    "reason": "no metadata label or missing annotation csv",
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
