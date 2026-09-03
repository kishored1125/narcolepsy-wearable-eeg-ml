#!/usr/bin/env python
from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats
from tqdm import tqdm

try:
    import mne
except ModuleNotFoundError as exc:
    raise SystemExit("mne is required for EDF processing. Install with: python -m pip install mne") from exc

STAGE_MAP = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
DEFAULT_CHANNEL_PATTERNS = ["F7", "F8", "O1", "O2", "EEG"]
SLEEP_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Simons Dreem EDF EEG features .")
    parser.add_argument("--data-root", required=True, help="Shared read-only simons_sleep folder containing _meta/ and SP*/ participant folders.")
    parser.add_argument("--metadata", default=None, help="Metadata CSV. If omitted, auto-detects *_participant_data_diag.csv under data-root/_meta.")
    parser.add_argument("--scratch-out", required=True)
    parser.add_argument("--home-out", required=True)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--max-records-per-subject", type=int, default=None)
    parser.add_argument("--control-only", action="store_true", help="Process only asd=False participants.")
    parser.add_argument("--epoch-seconds", type=int, default=30)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    metadata_path = Path(args.metadata).expanduser().resolve() if args.metadata else find_metadata(data_root)
    scratch = Path(args.scratch_out).expanduser().resolve()
    home = Path(args.home_out).expanduser().resolve()
    assert_not_inside_data_root(scratch, data_root, "--scratch-out")
    assert_not_inside_data_root(home, data_root, "--home-out")
    epoch_out = scratch / "simons_epoch_features_by_record"
    epoch_out.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    if args.control_only:
        metadata = metadata[metadata["asd"] == False].copy()
    wanted = set(metadata["subject_sp_id"].astype(str))
    participants = [p for p in discover_participants(data_root) if participant_id(p) in wanted]
    if args.max_subjects:
        participants = participants[: args.max_subjects]

    failures = []
    record_rows = []
    for participant in tqdm(participants, desc="Simons participants"):
        try:
            recs = process_participant(participant, epoch_out, args.epoch_seconds, args.max_records_per_subject)
            record_rows.extend(recs)
        except Exception as exc:
            failures.append({"subject_sp_id": participant_id(participant), "path": str(participant), "error": str(exc)})

    record_df = pd.DataFrame(record_rows)
    failures_df = pd.DataFrame(failures)
    record_df.to_csv(home / "simons_edf_record_features.csv", index=False)
    failures_df.to_csv(home / "simons_edf_failures.csv", index=False)

    if not record_df.empty:
        subject_df = aggregate_subject_features(record_df, group_col="subject_sp_id", target_cols=["subject_sp_id", "recording_id"])
        subject_df = subject_df.merge(metadata, on="subject_sp_id", how="left")
        subject_df.to_csv(home / "simons_edf_subject_features.csv", index=False)

    summary = {
        "participants_seen": len(participants),
        "data_root_read_only": str(data_root),
        "metadata": str(metadata_path),
        "scratch_output": str(scratch),
        "home_output": str(home),
        "records_processed": len(record_df),
        "participants_processed": int(record_df["subject_sp_id"].nunique()) if not record_df.empty else 0,
        "failures": len(failures_df),
        "epoch_feature_files": len(list(epoch_out.glob("*.parquet"))),
    }
    pd.DataFrame([summary]).to_csv(home / "simons_edf_run_summary.csv", index=False)
    (home / "simons_edf_run_summary.json").write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")


def discover_participants(root: Path) -> list[Path]:
    zips = sorted(root.glob("SP*.zip")) + sorted(root.glob("sp*.zip"))
    folders = sorted(p for p in list(root.glob("SP*")) + list(root.glob("sp*")) if p.is_dir() and p.name != "_meta")
    return zips + folders


def find_metadata(root: Path) -> Path:
    candidates = sorted((root / "_meta").glob("*participant*diag*.csv")) + sorted((root / "_meta").glob("*.csv"))
    if not candidates:
        raise SystemExit(f"No metadata CSV found under {root / '_meta'}. Pass --metadata explicitly.")
    return candidates[0].resolve()


def assert_not_inside_data_root(path: Path, data_root: Path, name: str) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(data_root)
    except ValueError:
        return
    raise SystemExit(f"Refusing to write {name} inside data root: {resolved}")


def participant_id(path: Path) -> str:
    return path.stem if path.suffix.lower() == ".zip" else path.name


def process_participant(participant: Path, epoch_out: Path, epoch_seconds: int, max_records: int | None) -> list[dict[str, object]]:
    if participant.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            return process_zip_participant(participant, tmpdir, epoch_out, epoch_seconds, max_records)
    return process_folder_participant(participant, epoch_out, epoch_seconds, max_records)


def process_zip_participant(zip_path: Path, tmpdir: Path, epoch_out: Path, epoch_seconds: int, max_records: int | None) -> list[dict[str, object]]:
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        edfs = sorted(n for n in zf.namelist() if "/dreem/edf/" in n and n.endswith(".edf"))
        if max_records:
            edfs = edfs[:max_records]
        for edf_name in edfs:
            hypno_name = edf_name.replace("/edf/dreem_eeg_", "/hypno/dreem_hypno_").replace(".edf", ".csv")
            if hypno_name not in zf.namelist():
                continue
            edf_path = tmpdir / Path(edf_name).name
            hypno_path = tmpdir / Path(hypno_name).name
            zf.extract(edf_name, tmpdir)
            zf.extract(hypno_name, tmpdir)
            extracted_edf = tmpdir / edf_name
            extracted_hypno = tmpdir / hypno_name
            rows.append(process_record(extracted_edf, extracted_hypno, participant_id(zip_path), epoch_out, epoch_seconds))
    return rows


def process_folder_participant(folder: Path, epoch_out: Path, epoch_seconds: int, max_records: int | None) -> list[dict[str, object]]:
    rows = []
    edfs = sorted((folder / "dreem" / "edf").glob("*.edf"))
    if max_records:
        edfs = edfs[:max_records]
    for edf_path in edfs:
        hypno_path = folder / "dreem" / "hypno" / edf_path.name.replace("dreem_eeg_", "dreem_hypno_").replace(".edf", ".csv")
        if hypno_path.exists():
            rows.append(process_record(edf_path, hypno_path, participant_id(folder), epoch_out, epoch_seconds))
    return rows


def process_record(edf_path: Path, hypno_path: Path, subject_sp_id: str, epoch_out: Path, epoch_seconds: int) -> dict[str, object]:
    hyp = pd.read_csv(hypno_path)
    hyp["label"] = hyp["Sleep Stage"].map(STAGE_MAP)
    hyp = hyp[hyp["label"].notna()].reset_index(drop=True)

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    picks = choose_channels(raw.ch_names)
    if not picks:
        raise ValueError(f"No usable EEG channels found in {edf_path.name}: {raw.ch_names}")
    raw.pick(picks)
    raw.filter(0.5, 30.0, fir_design="firwin", verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    epoch_len = int(round(epoch_seconds * sfreq))
    n_epochs = min(len(hyp), raw.n_times // epoch_len)

    rows = []
    for idx in range(n_epochs):
        start = idx * epoch_len
        stop = start + epoch_len
        data = raw.get_data(start=start, stop=stop)
        features = extract_epoch_features(data, sfreq, raw.ch_names)
        features.update({"subject_sp_id": subject_sp_id, "recording_id": edf_path.stem, "epoch_index": idx, "label": hyp.loc[idx, "label"]})
        rows.append(features)

    epoch_df = pd.DataFrame(rows)
    out_file = epoch_out / f"{subject_sp_id}__{edf_path.stem}.parquet"
    epoch_df.to_parquet(out_file, index=False)
    return summarize_record(epoch_df, subject_sp_id, edf_path.stem, picks)


def choose_channels(ch_names: list[str]) -> list[str]:
    selected = [ch for ch in ch_names if any(pattern.lower() in ch.lower() for pattern in DEFAULT_CHANNEL_PATTERNS)]
    return selected[:4] if selected else ch_names[:4]


def aggregate_subject_features(records: pd.DataFrame, group_col: str, target_cols: list[str]) -> pd.DataFrame:
    numeric_cols = records.select_dtypes(include=[np.number]).columns.difference(target_cols)
    grouped = records.groupby(group_col, dropna=False)
    parts = []
    for stat_name, func in [("mean", "mean"), ("median", "median"), ("std", "std"), ("min", "min"), ("max", "max")]:
        stat = getattr(grouped[numeric_cols], func)()
        stat.columns = [f"{c}_{stat_name}" for c in stat.columns]
        parts.append(stat)
    return pd.concat(parts, axis=1).reset_index()


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


def summarize_record(epoch_df: pd.DataFrame, subject_sp_id: str, recording_id: str, channels: list[str]) -> dict[str, object]:
    row: dict[str, object] = {"subject_sp_id": subject_sp_id, "recording_id": recording_id, "n_epochs": len(epoch_df), "channels": ";".join(channels)}
    counts = epoch_df["label"].value_counts()
    sleep_epochs = int(counts.drop(labels=["Wake"], errors="ignore").sum())
    row["wake_epochs"] = int(counts.get("Wake", 0))
    row["sleep_epochs"] = sleep_epochs
    row["sleep_efficiency_epoch_ratio"] = sleep_epochs / len(epoch_df) if len(epoch_df) else np.nan
    for label in ["Wake", "N1", "N2", "N3", "REM"]:
        row[f"{label}_epochs"] = int(counts.get(label, 0))
        row[f"{label}_percentage"] = int(counts.get(label, 0)) / sleep_epochs if sleep_epochs and label != "Wake" else np.nan
    numeric_cols = [c for c in epoch_df.select_dtypes(include=[np.number]).columns if c != "epoch_index"]
    stats = epoch_df[numeric_cols].agg(["mean", "std", "median"])
    for stat in stats.index:
        for col, value in stats.loc[stat].items():
            row[f"{col}_{stat}"] = value
    return row


if __name__ == "__main__":
    main()
