from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from diss_eeg.features import aggregate_subject_features


EXCLUDE_DIAGNOSES = {"WITHDRAWN"}
NARCOLEPSY_DIAGNOSES = {"NT1", "NT2"}


def parse_report_csv(path: Path) -> dict[str, object]:
    row: dict[str, object] = {"report_path": str(path)}
    row["patient_id"] = _patient_from_path(path)
    row["recording_id"] = path.parent.name
    with path.open(errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or "," not in line:
                continue
            key, value = line.split(",", 1)
            key = key.strip()
            value = value.strip()
            row[key] = _coerce_value(value)
    return row


def load_report_table(nrdreem_dir: Path) -> pd.DataFrame:
    rows = [parse_report_csv(path) for path in sorted(nrdreem_dir.glob("narcorev_*/*/*_report.csv"))]
    return pd.DataFrame(rows)


def load_diagnosis_table(sample_dir: Path) -> pd.DataFrame:
    conv = pd.read_excel(sample_dir / "NR_ID_conv_dreem.xlsx", sheet_name="conv")
    conv["patient_id"] = conv["BeaconID"].astype(str).str.extract(r"(narcorev_\d+)")
    diag = pd.read_excel(sample_dir / "NRev.xlsx", sheet_name="diag")
    merged = conv.merge(diag, on="NRID", how="left")
    merged = merged[["patient_id", "NRID", "Sex", "Age at time of study", "Diagnosis"]]
    merged = merged.rename(
        columns={
            "Sex": "sex",
            "Age at time of study": "age",
            "Diagnosis": "diagnosis",
        }
    )
    merged["diagnosis"] = merged["diagnosis"].astype("string")
    return merged


def build_subject_table(reports: pd.DataFrame, diagnosis: pd.DataFrame) -> pd.DataFrame:
    merged = reports.merge(diagnosis, on="patient_id", how="left")
    merged = merged[~merged["diagnosis"].isin(EXCLUDE_DIAGNOSES)]
    merged = merged[merged["diagnosis"].notna()]
    merged["binary_target"] = np.where(merged["diagnosis"].isin(NARCOLEPSY_DIAGNOSES), "narcolepsy", "comparison")

    protected_cols = ["patient_id", "recording_id", "diagnosis", "binary_target", "sex", "age"]
    subject_features = aggregate_subject_features(merged, group_col="patient_id", target_cols=protected_cols)
    meta = (
        merged[["patient_id", "NRID", "sex", "age", "diagnosis", "binary_target"]]
        .drop_duplicates("patient_id")
        .reset_index(drop=True)
    )
    out = meta.merge(subject_features, on="patient_id", how="left")
    out["sex"] = out["sex"].map({"F": 0, "M": 1}).astype(float)
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    return out


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = {
        "patient_id",
        "NRID",
        "diagnosis",
        "binary_target",
        "report_path",
        "recording_id",
        "record",
        "device",
        "user",
    }
    return [c for c in df.columns if c not in blocked and pd.api.types.is_numeric_dtype(df[c])]


def _patient_from_path(path: Path) -> str | None:
    for part in path.parts:
        if re.match(r"narcorev_\d+", part):
            return part
    return None


def _coerce_value(value: str) -> object:
    if value == "":
        return np.nan
    try:
        return float(value)
    except ValueError:
        return value

