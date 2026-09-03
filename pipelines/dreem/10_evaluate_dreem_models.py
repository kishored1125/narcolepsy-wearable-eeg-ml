#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT / "src"))

from diss_eeg.pipeline_utils import (
    clean_feature_matrix,
    confusion_frame,
    ensure_dirs,
    model_pipelines,
    numeric_features,
    save_confusion_plot,
    save_importance_plot,
    selected_feature_importance,
    single_cv_predictions,
)


DATASETS = {
    "report_only": PROJECT / "dreem_nrev" / "outputs" / "feature_outputs" / "report_features" / "report_subject_features.csv",
    "h5_only": PROJECT / "dreem_nrev" / "outputs" / "feature_outputs" / "h5_features" / "h5_subject_features_labelled.csv",
    "combined_report_h5": PROJECT / "dreem_nrev" / "outputs" / "feature_outputs" / "combined_subject_features.csv",
}


def main() -> None:
    tables = PROJECT / "dreem_nrev" / "outputs" / "evaluation_outputs" / "tables"
    figures = PROJECT / "dreem_nrev" / "outputs" / "evaluation_outputs" / "figures"
    ensure_dirs(tables, figures)

    model_summary = pd.read_csv(PROJECT / "dreem_nrev" / "outputs" / "model_outputs" / "all_dataset_summaries.csv")
    all_rows = []
    for dataset, path in DATASETS.items():
        df = pd.read_csv(path)
        row = model_summary[model_summary["dataset"] == dataset].iloc[0]
        best_model = row["best_model"]
        blocked = {"patient_id", "NRID", "sex", "age", "diagnosis", "binary_target"}
        feature_cols = numeric_features(df, blocked)
        X, feature_cols = clean_feature_matrix(df, feature_cols)
        y = df["binary_target"].reset_index(drop=True)
        k = min(50, X.shape[1])
        model = model_pipelines(k_features=k)[best_model]

        pred = single_cv_predictions(X, y, model, n_splits=5)
        cm = confusion_frame(y, pred, labels=["comparison", "narcolepsy"])
        cm.to_csv(tables / f"{dataset}_confusion_matrix.csv")
        save_confusion_plot(cm, figures / f"{dataset}_confusion_matrix.png", f"{dataset}: {best_model}")

        importance = selected_feature_importance(model, X, y, feature_cols)
        importance.to_csv(tables / f"{dataset}_feature_importance.csv", index=False)
        save_importance_plot(importance, figures / f"{dataset}_feature_importance.png", f"{dataset}: Feature Importance")
        all_rows.append(row.to_dict())

    best = model_summary.sort_values("best_balanced_accuracy_mean", ascending=False).iloc[0]
    write_interpretation(best, model_summary, tables.parent / "interpretation.md")
    save_comparison_plot(model_summary, figures / "feature_set_comparison.png")
    pd.DataFrame(all_rows).to_csv(tables / "evaluation_summary.csv", index=False)
    print(f"Dreem evaluation outputs saved to {tables.parent}")


def save_comparison_plot(summary: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(9, 5))
    sns.barplot(data=summary, x="dataset", y="best_balanced_accuracy_mean", hue="best_model")
    plt.ylim(0, 1)
    plt.title("Dreem Feature Set Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_interpretation(best: pd.Series, summary: pd.DataFrame, path: Path) -> None:
    lines = []
    for row in summary.itertuples(index=False):
        lines.append(
            f"- `{row.dataset}`: `{row.best_model}`, balanced accuracy {row.best_balanced_accuracy_mean:.3f}, "
            f"macro F1 {row.best_macro_f1_mean:.3f}, ROC-AUC {row.best_roc_auc_mean:.3f}."
        )
    path.write_text(
        "# Dreem Diagnostic Evaluation\n\n"
        "This folder contains the final NRDREEM diagnostic evaluation. Three feature sets are compared: "
        "report-only, H5-only and combined report+H5 features.\n\n"
        + "\n".join(lines)
        + "\n\n"
        f"The best current setting is `{best.dataset}` with `{best.best_model}`. "
        "The results should be interpreted as exploratory because the labelled narcolepsy class is small.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
