#!/usr/bin/env python
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ENDPOINT_MAP = {
    "TRT": "record_duration",
    "TST": "tst",
    "SE": "sleep_efficiency",
    "SOL": "sol",
    "LPS": "lps",
    "WASO": "waso",
    "RL": "rem_latency",
    "NREM": "nrem_duration",
    "N1": "n1_duration",
    "N2": "n2_duration",
    "N3": "n3_duration",
    "REM": "rem_duration",
    "p_NREM": "nrem_percentage",
    "p_N1": "n1_percentage",
    "p_N2": "n2_percentage",
    "p_N3": "n3_percentage",
    "p_REM": "rem_percentage",
    "RR": "average_respiration_rate",
    "RRN1": "respiration_rate_n1",
    "RRN2": "respiration_rate_n2",
    "RRN3": "respiration_rate_n3",
    "RRREM": "respiration_rate_rem",
    "RRW": "respiration_rate_wake",
    "QUAL": "record_quality_index",
}

META_KEEP = ["subject_sp_id", "sex", "asd", "age_at_registration_years", "family_sf_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Simons Dreem report subject features.")
    parser.add_argument("--data-root", required=True, help="Shared read-only simons_sleep folder containing _meta/ and SP*/ participant folders.")
    parser.add_argument("--metadata", default=None, help="Metadata CSV. If omitted, auto-detects *_participant_data_diag.csv under data-root/_meta.")
    parser.add_argument("--out-dir", default=None, help="Output folder. Defaults to this experiment outputs/report_outputs.")
    parser.add_argument("--max-subjects", type=int, default=None)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    data_root = Path(args.data_root).expanduser().resolve()
    metadata_path = Path(args.metadata).expanduser().resolve() if args.metadata else find_metadata(data_root)
    out = Path(args.out_dir).expanduser().resolve() if args.out_dir else here / "outputs" / "report_outputs"
    assert_not_inside_data_root(out, data_root, "--out-dir")
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    records = []
    participants = discover_participants(data_root)
    if args.max_subjects:
        participants = participants[: args.max_subjects]

    for participant in participants:
        records.extend(load_report_records(participant))

    record_df = pd.DataFrame(records)
    record_df.to_csv(tables / "simons_report_record_features.csv", index=False)

    if record_df.empty:
        raise SystemExit("No Simons report CSV files were found.")

    subject_features = aggregate_subject_features(record_df)
    subject_features = subject_features.merge(metadata[META_KEEP], left_on="subject_sp_id", right_on="subject_sp_id", how="left")
    subject_features["external_group"] = np.where(subject_features["asd"] == False, "simons_asd_false_control", "simons_asd_true")
    subject_features["sex_numeric"] = subject_features["sex"].map({"Female": 0, "Male": 1}).astype(float)
    subject_features = subject_features.rename(columns={"age_at_registration_years": "age"})
    subject_features.to_csv(tables / "simons_report_subject_features.csv", index=False)

    summary = {
        "participants_discovered": len(participants),
        "data_root_read_only": str(data_root),
        "metadata": str(metadata_path),
        "output_dir": str(out),
        "participants_with_reports": int(record_df["subject_sp_id"].nunique()),
        "report_records": len(record_df),
        "subjects_after_metadata_merge": len(subject_features),
        "asd_true_subjects": int((subject_features["asd"] == True).sum()),
        "asd_false_subjects": int((subject_features["asd"] == False).sum()),
        "features": int(len(subject_features.columns)),
    }
    pd.DataFrame([summary]).to_csv(tables / "simons_report_pipeline_summary.csv", index=False)
    (out / "summary.md").write_text(
        "# Simons Report Pipeline\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in summary.items())
        + "\n",
        encoding="utf-8",
    )


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


def load_report_records(participant_path: Path) -> list[dict[str, object]]:
    if participant_path.suffix.lower() == ".zip":
        return load_report_records_from_zip(participant_path)
    return load_report_records_from_folder(participant_path)


def load_report_records_from_zip(zip_path: Path) -> list[dict[str, object]]:
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if "/dreem_reports/" in n and n.endswith(".csv")]
        for name in names:
            with zf.open(name) as handle:
                df = pd.read_csv(handle)
            rows.append(report_csv_to_record(df, name))
    return rows


def load_report_records_from_folder(folder: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(folder.glob("dreem/dreem_reports/*.csv")):
        rows.append(report_csv_to_record(pd.read_csv(path), str(path)))
    return rows


def report_csv_to_record(df: pd.DataFrame, source: str) -> dict[str, object]:
    if df.empty:
        return {}
    subject_sp_id = Path(source).parts[0] if not source.startswith("/") else next((p for p in Path(source).parts if p.startswith("SP")), None)
    row: dict[str, object] = {
        "subject_sp_id": subject_sp_id,
        "record_source": source,
        "record_datetime": df["REC_DATE_TIME"].iloc[0] if "REC_DATE_TIME" in df else pd.NA,
        "off_head": pd.to_numeric(df["OFFHEAD"], errors="coerce").dropna().iloc[0] if "OFFHEAD" in df and df["OFFHEAD"].notna().any() else np.nan,
    }
    for _, item in df.iterrows():
        endpoint = str(item.get("ENDPOINT", "")).strip()
        feature = ENDPOINT_MAP.get(endpoint, endpoint.lower())
        value = pd.to_numeric(item.get("VALUE"), errors="coerce")
        qi = pd.to_numeric(item.get("QI_INDEX"), errors="coerce")
        row[feature] = value
        if pd.notna(qi):
            row[f"confidence_{feature}"] = qi
    return row


def aggregate_subject_features(records: pd.DataFrame) -> pd.DataFrame:
    protected = {"subject_sp_id", "record_source", "record_datetime"}
    numeric_cols = [c for c in records.columns if c not in protected and pd.api.types.is_numeric_dtype(records[c])]
    grouped = records.groupby("subject_sp_id", dropna=False)
    pieces = []
    for stat in ["mean", "median", "std", "min", "max"]:
        frame = getattr(grouped[numeric_cols], stat)()
        frame.columns = [f"{c}_{stat}" for c in frame.columns]
        pieces.append(frame)
    out = pd.concat(pieces, axis=1).reset_index()
    out["simons_record_count"] = grouped.size().to_numpy()
    return out


if __name__ == "__main__":
    main()
