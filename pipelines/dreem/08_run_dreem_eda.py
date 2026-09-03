#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT = Path(__file__).resolve().parents[2]


def main() -> None:
    out = PROJECT / "dreem_nrev" / "outputs" / "eda_outputs"
    tables = out / "tables"
    figures = out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    report = pd.read_csv(PROJECT / "dreem_nrev" / "outputs" / "feature_outputs" / "report_features" / "report_subject_features.csv")
    h5 = pd.read_csv(PROJECT / "dreem_nrev" / "outputs" / "feature_outputs" / "h5_features" / "h5_subject_features_labelled.csv")
    combined = pd.read_csv(PROJECT / "dreem_nrev" / "outputs" / "feature_outputs" / "combined_subject_features.csv")

    summary = pd.DataFrame(
        [
            {"dataset": "report_subject_features", "subjects": len(report), "features": feature_count(report), "missing_values": int(report.isna().sum().sum())},
            {"dataset": "h5_subject_features", "subjects": len(h5), "features": feature_count(h5), "missing_values": int(h5.isna().sum().sum())},
            {"dataset": "combined_subject_features", "subjects": len(combined), "features": feature_count(combined), "missing_values": int(combined.isna().sum().sum())},
        ]
    )
    summary.to_csv(tables / "dataset_summary.csv", index=False)

    diagnosis = combined["diagnosis"].value_counts().rename_axis("diagnosis").reset_index(name="subjects")
    diagnosis["percentage"] = diagnosis["subjects"] / diagnosis["subjects"].sum()
    diagnosis.to_csv(tables / "diagnosis_distribution.csv", index=False)
    plot_bar(diagnosis, "diagnosis", "subjects", figures / "diagnosis_distribution.png", "Dreem Diagnosis Distribution")

    target = combined["binary_target"].value_counts().rename_axis("binary_target").reset_index(name="subjects")
    target["percentage"] = target["subjects"] / target["subjects"].sum()
    target.to_csv(tables / "binary_target_distribution.csv", index=False)
    plot_bar(target, "binary_target", "subjects", figures / "binary_target_distribution.png", "Dreem Binary Target Distribution")

    combined[["patient_id", "age", "sex", "diagnosis", "binary_target"]].to_csv(tables / "demographics.csv", index=False)
    plt.figure(figsize=(8, 5))
    sns.histplot(data=combined, x="age", hue="binary_target", bins=12)
    plt.title("Dreem Age Distribution by Target")
    plt.tight_layout()
    plt.savefig(figures / "age_distribution.png", dpi=180)
    plt.close()

    sleep_cols = [c for c in ["sleep_efficiency_epoch_ratio_mean", "REM_percentage_mean", "N1_percentage_mean", "N2_percentage_mean", "N3_percentage_mean"] if c in h5.columns]
    if sleep_cols:
        sleep_summary = h5.groupby("binary_target")[sleep_cols].agg(["mean", "std", "median"]).reset_index()
        sleep_summary.columns = ["_".join([x for x in col if x]) for col in sleep_summary.columns.to_flat_index()]
        sleep_summary.to_csv(tables / "h5_sleep_architecture_by_target.csv", index=False)
        plot_df = h5[["binary_target"] + sleep_cols].melt(id_vars="binary_target", var_name="feature", value_name="value")
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=plot_df, x="feature", y="value", hue="binary_target", showfliers=False)
        plt.title("Dreem H5 Sleep Architecture by Target")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(figures / "h5_sleep_architecture_by_target.png", dpi=180)
        plt.close()

    missingness(combined).to_csv(tables / "combined_missingness.csv", index=False)
    print(f"Dreem EDA outputs saved to {out}")


def feature_count(df: pd.DataFrame) -> int:
    meta = {"patient_id", "NRID", "sex", "age", "diagnosis", "binary_target"}
    return len([c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])])


def missingness(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"column": df.columns, "missing_count": df.isna().sum().to_numpy(), "missing_percentage": df.isna().mean().to_numpy()}
    ).sort_values("missing_percentage", ascending=False)


def plot_bar(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x=x, y=y)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
