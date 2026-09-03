#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from diss_eeg.nrdreem_reports import NARCOLEPSY_DIAGNOSES, load_diagnosis_table
from diss_eeg.paths import FIGURE_DIR, TABLE_DIR, ensure_output_dirs
from diss_eeg.plotting import save_barplot, save_confusion_matrix, save_feature_importance


BLOCKED_FEATURE_COLUMNS = {
    "patient_id",
    "NRID",
    "sex",
    "diagnosis",
    "binary_target",
}


def build_models(k_features: int, random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("select", SelectKBest(score_func=f_classif, k=k_features)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("select", SelectKBest(score_func=f_classif, k=k_features)),
                (
                    "model",
                    SVC(
                        kernel="linear",
                        class_weight="balanced",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("select", SelectKBest(score_func=f_classif, k=k_features)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        class_weight="balanced",
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("select", SelectKBest(score_func=f_classif, k=k_features)),
                ("model", GradientBoostingClassifier(random_state=random_state)),
            ]
        ),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Model narcolepsy from raw Dreem H5-derived subject features.")
    parser.add_argument(
        "--h5-output-dir",
        default=str(project_root / "dreem_nrev" / "outputs" / "h5_subject_features"),
        help="Directory containing dreem_h5_subject_features.csv from the H5 feature extraction run.",
    )
    parser.add_argument(
        "--sample-dir",
        default=str(Path(__file__).resolve().parents[2] / "narcolepsy_dreem"),
        help="Directory containing NR_ID_conv_dreem.xlsx and NRev.xlsx.",
    )
    parser.add_argument("--max-selected-features", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    ensure_output_dirs()
    out_dir = project_root / "dreem_nrev" / "outputs" / "dreem_h5_final"
    table_dir = out_dir / "tables"
    figure_dir = out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    h5_output_dir = Path(args.h5_output_dir)
    subjects = pd.read_csv(h5_output_dir / "dreem_h5_subject_features.csv")
    record_features = pd.read_csv(h5_output_dir / "dreem_h5_record_features.csv")
    failures = pd.read_csv(h5_output_dir / "dreem_h5_failures.csv")
    with (h5_output_dir / "dreem_h5_run_summary.json").open() as handle:
        extraction_summary = json.load(handle)

    diagnosis = load_diagnosis_table(Path(args.sample_dir))
    labelled = subjects.merge(diagnosis, on="patient_id", how="left")
    labelled = labelled[labelled["diagnosis"].notna()].copy()
    labelled = labelled[labelled["diagnosis"] != "WITHDRAWN"].copy()
    labelled["binary_target"] = np.where(
        labelled["diagnosis"].isin(NARCOLEPSY_DIAGNOSES),
        "narcolepsy",
        "comparison",
    )
    labelled["sex"] = labelled["sex"].map({"F": 0, "M": 1}).astype(float)
    labelled["age"] = pd.to_numeric(labelled["age"], errors="coerce")

    numeric_cols = [
        c
        for c in labelled.columns
        if c not in BLOCKED_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(labelled[c])
    ]
    feature_table = labelled[numeric_cols].replace([np.inf, -np.inf], np.nan)
    enough_data_cols = [c for c in feature_table.columns if feature_table[c].notna().mean() >= 0.70]
    feature_table = feature_table[enough_data_cols]
    variable_cols = [c for c in feature_table.columns if feature_table[c].nunique(dropna=True) > 1]
    feature_table = feature_table[variable_cols]

    X = feature_table
    y = labelled["binary_target"].reset_index(drop=True)
    model_table = labelled[["patient_id", "NRID", "sex", "age", "diagnosis", "binary_target"]].copy()
    model_table = pd.concat([model_table.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    model_table.to_csv(table_dir / "dreem_h5_subject_model_table.csv", index=False)

    diag_counts = labelled["diagnosis"].value_counts().reset_index()
    diag_counts.columns = ["diagnosis", "participants"]
    diag_counts.to_csv(table_dir / "dreem_h5_diagnosis_counts.csv", index=False)
    save_barplot(
        diag_counts,
        x="diagnosis",
        y="participants",
        path=figure_dir / "dreem_h5_diagnosis_counts.png",
        title="Dreem H5 Diagnosis Counts",
    )

    target_counts = labelled["binary_target"].value_counts().reset_index()
    target_counts.columns = ["binary_target", "participants"]
    target_counts.to_csv(table_dir / "dreem_h5_binary_target_counts.csv", index=False)

    min_class = int(y.value_counts().min())
    n_splits = max(2, min(5, min_class))
    selected_k = min(args.max_selected_features, X.shape[1])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)

    metrics_rows = []
    predictions = {}
    fitted_models = {}
    for name, model in build_models(selected_k, random_state=args.random_state).items():
        estimator = clone(model)
        y_pred = cross_val_predict(estimator, X, y, cv=cv, method="predict")
        y_proba = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")
        fitted = clone(model).fit(X, y)
        fitted_models[name] = fitted
        predictions[name] = y_pred

        positive_idx = list(fitted.classes_).index("narcolepsy")
        auc = roc_auc_score(y == "narcolepsy", y_proba[:, positive_idx])
        row = {
            "model": name,
            "accuracy": accuracy_score(y, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "macro_f1": f1_score(y, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y, y_pred, average="weighted", zero_division=0),
            "macro_precision": precision_score(y, y_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y, y_pred, average="macro", zero_division=0),
            "sensitivity_narcolepsy": recall_score(y == "narcolepsy", y_pred == "narcolepsy", zero_division=0),
            "specificity_comparison": recall_score(y != "narcolepsy", y_pred != "narcolepsy", zero_division=0),
            "roc_auc": auc,
        }
        metrics_rows.append(row)

    metrics = pd.DataFrame(metrics_rows).sort_values("balanced_accuracy", ascending=False)
    metrics.to_csv(table_dir / "dreem_h5_model_metrics.csv", index=False)
    best_name = str(metrics.iloc[0]["model"])

    labels = ["comparison", "narcolepsy"]
    cm = confusion_matrix(y, predictions[best_name], labels=labels)
    cm_frame = pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
    cm_frame.to_csv(table_dir / "dreem_h5_confusion_matrix.csv")
    save_confusion_matrix(cm_frame, figure_dir / "dreem_h5_confusion_matrix.png", f"Dreem H5 Confusion Matrix: {best_name}")

    importance = model_feature_importance(fitted_models[best_name], list(X.columns), top_n=40)
    importance.to_csv(table_dir / "dreem_h5_feature_importance.csv", index=False)
    save_feature_importance(
        importance,
        figure_dir / "dreem_h5_feature_importance.png",
        f"Dreem H5 Feature Importance: {best_name}",
    )

    summary = {
        "h5_files_seen": extraction_summary.get("h5_files_seen"),
        "h5_records_processed": extraction_summary.get("records_processed"),
        "h5_records_failed": extraction_summary.get("records_failed"),
        "h5_subjects_processed": extraction_summary.get("subjects_processed"),
        "record_feature_rows_loaded": len(record_features),
        "failed_record_rows_loaded": len(failures),
        "subjects_with_usable_diagnosis": len(labelled),
        "narcolepsy_subjects": int((y == "narcolepsy").sum()),
        "other_hypersomnia_subjects": int((y == "comparison").sum()),
        "candidate_feature_count_after_qc": int(X.shape[1]),
        "selected_features_per_model_fold": int(selected_k),
        "cv_splits": int(n_splits),
        "best_model": best_name,
        "best_balanced_accuracy": float(metrics.iloc[0]["balanced_accuracy"]),
        "best_macro_f1": float(metrics.iloc[0]["macro_f1"]),
        "best_roc_auc": float(metrics.iloc[0]["roc_auc"]),
    }
    pd.DataFrame([summary]).to_csv(table_dir / "dreem_h5_run_summary.csv", index=False)
    with (table_dir / "dreem_h5_run_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    write_interpretation(out_dir / "interpretation.md", summary, metrics, importance)
    print("Dreem H5 diagnostic summary:", summary)


def model_feature_importance(model: Pipeline, feature_names: list[str], top_n: int = 40) -> pd.DataFrame:
    selector = model.named_steps["select"]
    selected_names = np.array(feature_names)[selector.get_support()]
    final = model.named_steps["model"]
    if hasattr(final, "feature_importances_"):
        values = final.feature_importances_
    elif hasattr(final, "coef_"):
        values = np.mean(np.abs(final.coef_), axis=0)
    else:
        values = selector.scores_[selector.get_support()]
    return (
        pd.DataFrame({"feature": selected_names, "importance": values})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def write_interpretation(path: Path, summary: dict[str, object], metrics: pd.DataFrame, importance: pd.DataFrame) -> None:
    best = metrics.iloc[0]
    top_features = "\n".join(
        f"- {row.feature}: {row.importance:.4g}" for row in importance.head(10).itertuples(index=False)
    )
    path.write_text(
        f"""# Dreem H5 Diagnostic Modelling Interpretation

## Dataset

The Dreem H5 extraction processed {summary['h5_records_processed']} recordings from {summary['h5_subjects_processed']} subjects, with {summary['h5_records_failed']} recordings failing quality/feature extraction. The diagnostic model used {summary['subjects_with_usable_diagnosis']} labelled subjects: {summary['narcolepsy_subjects']} narcolepsy (`NT1`/`NT2`) and {summary['other_hypersomnia_subjects']} other-hypersomnia participants.

## Validation

The model is evaluated at subject level using stratified {summary['cv_splits']}-fold cross-validation. Feature selection is performed inside each model pipeline, so the validation folds are not used to choose the retained features. This is important because the H5 feature table is high-dimensional relative to the number of participants.

## Best Preliminary Model

- Model: {summary['best_model']}
- Balanced accuracy: {best['balanced_accuracy']:.3f}
- Macro F1: {best['macro_f1']:.3f}
- ROC-AUC: {best['roc_auc']:.3f}
- Narcolepsy sensitivity: {best['sensitivity_narcolepsy']:.3f}
- Comparison specificity: {best['specificity_comparison']:.3f}

## Most Influential Features

{top_features}

## Dissertation Interpretation

These results are best described as a preliminary subject-level diagnostic model from wearable EEG-derived features. The sample size is small and class imbalance is present, so the key value is methodological: the project now demonstrates the complete path from raw Dreem H5 files, hypnogram alignment, epoch-level EEG feature engineering, subject-level aggregation, cross-validated model comparison and interpretable feature ranking.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
