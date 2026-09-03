#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT / "src"))

from diss_eeg.pipeline_utils import clean_feature_matrix, ensure_dirs, numeric_features, repeated_cv_model_comparison


DATASETS = {
    "report_only": PROJECT / "dreem_nrev" / "outputs" / "feature_outputs" / "report_features" / "report_subject_features.csv",
    "h5_only": PROJECT / "dreem_nrev" / "outputs" / "feature_outputs" / "h5_features" / "h5_subject_features_labelled.csv",
    "combined_report_h5": PROJECT / "dreem_nrev" / "outputs" / "feature_outputs" / "combined_subject_features.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Dreem diagnostic models for report, H5 and combined feature sets.")
    parser.add_argument("--max-selected-features", type=int, default=50)
    parser.add_argument("--cv-repeats", type=int, default=10)
    args = parser.parse_args()

    out = PROJECT / "dreem_nrev" / "outputs" / "model_outputs"
    ensure_dirs(out)

    all_metrics = []
    all_summaries = []
    for name, path in DATASETS.items():
        df = pd.read_csv(path)
        blocked = {"patient_id", "NRID", "sex", "age", "diagnosis", "binary_target"}
        feature_cols = numeric_features(df, blocked)
        X, feature_cols = clean_feature_matrix(df, feature_cols)
        y = df["binary_target"].reset_index(drop=True)
        k = min(args.max_selected_features, X.shape[1])
        metrics, fold_metrics = repeated_cv_model_comparison(
            X,
            y,
            k_features=k,
            positive_label="narcolepsy",
            n_splits=5,
            n_repeats=args.cv_repeats,
        )
        metrics.insert(0, "dataset", name)
        fold_metrics.insert(0, "dataset", name)
        metrics.to_csv(out / f"{name}_model_metrics.csv", index=False)
        fold_metrics.to_csv(out / f"{name}_fold_metrics.csv", index=False)
        pd.DataFrame({"feature": feature_cols}).to_csv(out / f"{name}_features_used.csv", index=False)
        all_metrics.append(metrics)
        all_summaries.append(
            {
                "dataset": name,
                "subjects": len(df),
                "narcolepsy_subjects": int((y == "narcolepsy").sum()),
                "comparison_subjects": int((y == "comparison").sum()),
                "features_after_qc": len(feature_cols),
                "selected_features_per_fold": k,
                "best_model": metrics.iloc[0]["model"],
                "best_balanced_accuracy_mean": metrics.iloc[0]["balanced_accuracy_mean"],
                "best_macro_f1_mean": metrics.iloc[0]["macro_f1_mean"],
                "best_roc_auc_mean": metrics.iloc[0].get("roc_auc_mean"),
            }
        )

    pd.concat(all_metrics, ignore_index=True).to_csv(out / "all_model_metrics.csv", index=False)
    pd.DataFrame(all_summaries).to_csv(out / "all_dataset_summaries.csv", index=False)
    print(f"Dreem model outputs saved to {out}")


if __name__ == "__main__":
    main()
