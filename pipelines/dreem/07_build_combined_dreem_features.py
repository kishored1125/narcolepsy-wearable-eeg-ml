#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT / "src"))

from diss_eeg.nrdreem_reports import NARCOLEPSY_DIAGNOSES, build_subject_table
from diss_eeg.pipeline_utils import ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Dreem report, H5 and combined subject feature tables.")
    parser.add_argument("--report-records", default=str(PROJECT / "dreem_nrev" / "outputs" / "loading_outputs" / "report_record_table.csv"))
    parser.add_argument("--diagnosis", default=str(PROJECT / "dreem_nrev" / "outputs" / "loading_outputs" / "diagnosis_mapping.csv"))
    parser.add_argument("--h5-subject-features", default=str(PROJECT / "dreem_nrev" / "outputs" / "h5_subject_features" / "dreem_h5_subject_features.csv"))
    args = parser.parse_args()

    report_out = PROJECT / "dreem_nrev" / "outputs" / "feature_outputs" / "report_features"
    h5_out = PROJECT / "dreem_nrev" / "outputs" / "feature_outputs" / "h5_features"
    ensure_dirs(report_out, h5_out)

    reports = pd.read_csv(args.report_records)
    diagnosis = pd.read_csv(args.diagnosis)

    report_subjects = build_subject_table(reports, diagnosis)
    report_subjects.to_csv(report_out / "report_subject_features.csv", index=False)

    h5 = pd.read_csv(args.h5_subject_features)
    h5_labelled = h5.merge(diagnosis, on="patient_id", how="left")
    h5_labelled = h5_labelled[h5_labelled["diagnosis"].notna()].copy()
    h5_labelled = h5_labelled[h5_labelled["diagnosis"] != "WITHDRAWN"].copy()
    h5_labelled["binary_target"] = np.where(h5_labelled["diagnosis"].isin(NARCOLEPSY_DIAGNOSES), "narcolepsy", "comparison")
    h5_labelled["sex"] = h5_labelled["sex"].map({"F": 0, "M": 1}).astype(float)
    h5_labelled["age"] = pd.to_numeric(h5_labelled["age"], errors="coerce")
    h5_labelled.to_csv(h5_out / "h5_subject_features_labelled.csv", index=False)

    report_prefixed = prefix_features(report_subjects, prefix="report")
    h5_prefixed = prefix_features(h5_labelled, prefix="h5")
    combined = report_prefixed.merge(
        h5_prefixed[["patient_id"] + [c for c in h5_prefixed.columns if c.startswith("h5__")]],
        on="patient_id",
        how="inner",
    )
    combined.to_csv(PROJECT / "dreem_nrev" / "outputs" / "feature_outputs" / "combined_subject_features.csv", index=False)

    pd.DataFrame(
        [
            {
                "report_subjects": len(report_subjects),
                "h5_labelled_subjects": len(h5_labelled),
                "combined_subjects": len(combined),
                "combined_features": len([c for c in combined.columns if "__" in c]),
            }
        ]
    ).to_csv(PROJECT / "dreem_nrev" / "outputs" / "feature_outputs" / "feature_engineering_summary.csv", index=False)
    print("Dreem feature engineering outputs saved.")


def prefix_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    meta = {"patient_id", "NRID", "sex", "age", "diagnosis", "binary_target"}
    rename = {c: f"{prefix}__{c}" for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])}
    return df.rename(columns=rename)


if __name__ == "__main__":
    main()
