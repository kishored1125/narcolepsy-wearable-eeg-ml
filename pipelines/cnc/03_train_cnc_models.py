#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


META_COLUMNS = {
    "subject_id",
    "file_id",
    "recording_id",
    "diagnosis",
    "label",
    "diagnosis_binary",
    "metadata_source",
    "nrevid",
    "Diagnosis",
    "diagnosis_clean",
    "DQ0602",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CNC T1 narcolepsy vs control models.")
    parser.add_argument("--features", required=True, help="CNC subject feature CSV.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    features_path = Path(args.features).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_path)
    if "diagnosis_binary" not in df.columns:
        raise ValueError("Expected diagnosis_binary target column.")
    y = df["diagnosis_binary"].astype(int)
    feature_cols = select_feature_columns(df)
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    n_splits = min(args.n_splits, int(y.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = build_models()

    metrics_rows = []
    prediction_frames = []
    for name, model in models.items():
        y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
        y_score = get_scores(model, X, y, cv)
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        metrics_rows.append(
            {
                "dataset": "cnc_edf",
                "model": name,
                "n_subjects": len(df),
                "n_controls": int((y == 0).sum()),
                "n_t1_narcolepsy": int((y == 1).sum()),
                "n_features_before_qc": len(feature_cols),
                "balanced_accuracy": balanced_accuracy_score(y, y_pred),
                "macro_f1": f1_score(y, y_pred, average="macro"),
                "roc_auc": roc_auc_score(y, y_score),
                "sensitivity_t1_narcolepsy": recall_score(y, y_pred, pos_label=1),
                "specificity_control": recall_score(y, y_pred, pos_label=0),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "subject_id": df["subject_id"],
                    "true_label": y,
                    "predicted_label": y_pred,
                    "predicted_probability_t1": y_score,
                    "model": name,
                }
            )
        )

    metrics = pd.DataFrame(metrics_rows).sort_values(["balanced_accuracy", "macro_f1", "roc_auc"], ascending=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics.to_csv(tables / "cnc_edf_model_metrics.csv", index=False)
    predictions.to_csv(tables / "cnc_edf_cross_validated_predictions.csv", index=False)

    best_model_name = str(metrics.iloc[0]["model"])
    best_pipeline = models[best_model_name]
    best_pipeline.fit(X, y)
    importance = extract_feature_importance(best_pipeline, feature_cols, best_model_name)
    importance.to_csv(tables / "cnc_edf_feature_importance.csv", index=False)

    dataset_summary = pd.DataFrame(
        [
            {
                "dataset": "cnc_edf",
                "input_file": str(features_path),
                "subjects": len(df),
                "controls": int((y == 0).sum()),
                "t1_narcolepsy": int((y == 1).sum()),
                "features_used": len(feature_cols),
                "cv_splits": n_splits,
                "best_model": best_model_name,
                "best_balanced_accuracy": float(metrics.iloc[0]["balanced_accuracy"]),
                "best_macro_f1": float(metrics.iloc[0]["macro_f1"]),
                "best_roc_auc": float(metrics.iloc[0]["roc_auc"]),
            }
        ]
    )
    dataset_summary.to_csv(tables / "cnc_edf_dataset_summary.csv", index=False)
    (out_dir / "summary.md").write_text(
        "\n".join(
            [
                "# CNC EDF Model Evaluation",
                "",
                f"- Subjects: {len(df)}",
                f"- Controls: {int((y == 0).sum())}",
                f"- T1 narcolepsy: {int((y == 1).sum())}",
                f"- Features used: {len(feature_cols)}",
                f"- Best model: {best_model_name}",
                f"- Best balanced accuracy: {metrics.iloc[0]['balanced_accuracy']:.3f}",
                f"- Best macro F1: {metrics.iloc[0]['macro_f1']:.3f}",
                f"- Best ROC-AUC: {metrics.iloc[0]['roc_auc']:.3f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(dataset_summary.iloc[0].to_dict(), indent=2))


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [
        c
        for c in numeric
        if c not in META_COLUMNS
        and not c.endswith("_binary")
        and "diagnosis" not in c.lower()
        and "label" not in c.lower()
        and "target" not in c.lower()
    ]
    missingness = df[feature_cols].isna().mean()
    feature_cols = [c for c in feature_cols if missingness[c] < 0.4]
    nunique = df[feature_cols].nunique(dropna=True)
    feature_cols = [c for c in feature_cols if nunique[c] > 1]
    return feature_cols


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "support_vector_machine": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
    }


def get_scores(model: Pipeline, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> np.ndarray:
    if hasattr(model[-1], "predict_proba"):
        return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    decision = cross_val_predict(model, X, y, cv=cv, method="decision_function")
    return (decision - decision.min()) / (decision.max() - decision.min())


def extract_feature_importance(model: Pipeline, feature_cols: list[str], model_name: str) -> pd.DataFrame:
    final = model[-1]
    if hasattr(final, "feature_importances_"):
        values = final.feature_importances_
    elif hasattr(final, "coef_"):
        values = np.abs(final.coef_).ravel()
    else:
        return pd.DataFrame(columns=["model", "feature", "importance"])
    out = pd.DataFrame({"model": model_name, "feature": feature_cols, "importance": values})
    return out.sort_values("importance", ascending=False)


if __name__ == "__main__":
    main()
