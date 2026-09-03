#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


META_COLUMNS = {"patient_id", "NRID", "sex", "age", "diagnosis", "binary_target"}
NARCOLEPSY_DIAGNOSES = {"NT1", "NT2", "NARCOLEPSY", "NARCOLEPSY TYPE 1", "NARCOLEPSY TYPE 2"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate advanced Dreem H5 features and optionally create true SHAP outputs.")
    parser.add_argument("--advanced-subject-csv", required=True, help="dreem_h5_advanced_subject_features.csv")
    parser.add_argument("--label-csv", required=True, help="CSV with patient_id and diagnosis.")
    parser.add_argument("--out-dir", required=True, help="Writable output folder.")
    parser.add_argument("--max-selected-features", type=int, default=80)
    parser.add_argument("--run-shap", action="store_true", help="Run shap.TreeExplainer if shap is installed.")
    args = parser.parse_args()

    subject_csv = Path(args.advanced_subject_csv).expanduser().resolve()
    label_csv = Path(args.label_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    tables = out_dir / "tables"
    figures = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    subjects = pd.read_csv(subject_csv)
    labels = load_labels(label_csv)
    labelled = subjects.merge(labels, on="patient_id", how="inner")
    labelled.to_csv(tables / "dreem_advanced_subject_features_labelled.csv", index=False)

    feature_cols = [c for c in labelled.columns if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(labelled[c])]
    X, clean_cols = clean_matrix(labelled[feature_cols])
    y = labelled["binary_target"].reset_index(drop=True)
    pd.DataFrame({"feature": clean_cols}).to_csv(tables / "dreem_advanced_features_after_qc.csv", index=False)

    k = min(args.max_selected_features, X.shape[1])
    models = build_models(k)
    n_splits = int(min(5, y.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    metric_rows = []
    predictions = {}
    probabilities = {}
    fitted_models = {}
    for name, model in models.items():
        pred = cross_val_predict(clone(model), X, y, cv=cv, method="predict")
        proba = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")
        fitted = clone(model).fit(X, y)
        pos_idx = list(fitted.classes_).index("narcolepsy")
        row = metric_row(y, pred, proba[:, pos_idx])
        row["model"] = name
        metric_rows.append(row)
        predictions[name] = pred
        probabilities[name] = proba[:, pos_idx]
        fitted_models[name] = fitted

    metrics = pd.DataFrame(metric_rows).sort_values("balanced_accuracy", ascending=False)
    metrics.to_csv(tables / "dreem_advanced_model_metrics.csv", index=False)
    best_model_name = str(metrics.iloc[0]["model"])
    best_model = fitted_models[best_model_name]

    cm = pd.DataFrame(
        confusion_matrix(y, predictions[best_model_name], labels=["comparison", "narcolepsy"]),
        index=["true_comparison", "true_narcolepsy"],
        columns=["pred_comparison", "pred_narcolepsy"],
    )
    cm.to_csv(tables / "dreem_advanced_confusion_matrix.csv")
    save_confusion_plot(cm, figures / "dreem_advanced_confusion_matrix.png", f"Advanced Features: {best_model_name}")

    importance = selected_importance(best_model, clean_cols)
    importance.to_csv(tables / "dreem_advanced_feature_importance.csv", index=False)
    save_importance_plot(importance, figures / "dreem_advanced_feature_importance.png", "Advanced Feature Importance")

    shap_status = "not_requested"
    if args.run_shap:
        shap_status = run_true_shap(best_model, X, clean_cols, tables, figures)

    summary = {
        "advanced_subject_csv_read_only": str(subject_csv),
        "label_csv_read_only": str(label_csv),
        "output_dir": str(out_dir),
        "subjects": int(len(labelled)),
        "narcolepsy_subjects": int((y == "narcolepsy").sum()),
        "comparison_subjects": int((y == "comparison").sum()),
        "raw_feature_count": int(len(feature_cols)),
        "features_after_qc": int(len(clean_cols)),
        "selected_features": int(k),
        "cv_folds": int(n_splits),
        "best_model": best_model_name,
        "best_balanced_accuracy": float(metrics.iloc[0]["balanced_accuracy"]),
        "best_macro_f1": float(metrics.iloc[0]["macro_f1"]),
        "best_roc_auc": float(metrics.iloc[0]["roc_auc"]),
        "shap_status": shap_status,
    }
    pd.DataFrame([summary]).to_csv(tables / "dreem_advanced_model_summary.csv", index=False)
    (out_dir / "dreem_advanced_model_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "summary.md").write_text(
        "# Dreem Advanced Feature Experiment\n\n"
        f"- Subjects: {summary['subjects']}\n"
        f"- Narcolepsy subjects: {summary['narcolepsy_subjects']}\n"
        f"- Comparison subjects: {summary['comparison_subjects']}\n"
        f"- Features after QC: {summary['features_after_qc']}\n"
        f"- Best model: {summary['best_model']}\n"
        f"- Balanced accuracy: {summary['best_balanced_accuracy']:.3f}\n"
        f"- Macro F1: {summary['best_macro_f1']:.3f}\n"
        f"- ROC-AUC: {summary['best_roc_auc']:.3f}\n"
        f"- SHAP status: {shap_status}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    diagnosis = df["diagnosis"].astype(str).str.upper().str.strip()
    valid = df["patient_id"].notna() & df["diagnosis"].notna() & ~diagnosis.isin({"", "NAN", "WITHDRAWN"})
    out = df.loc[valid, ["patient_id", "NRID", "sex", "age", "diagnosis"]].copy()
    diagnosis = out["diagnosis"].astype(str).str.upper().str.strip()
    out["binary_target"] = np.where(
        diagnosis.isin(NARCOLEPSY_DIAGNOSES) | diagnosis.str.contains("NARCOLEPSY", na=False),
        "narcolepsy",
        "comparison",
    )
    out["sex"] = out["sex"].map({"F": 0, "M": 1}).astype(float)
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    return out


def clean_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    X = frame.replace([np.inf, -np.inf], np.nan)
    keep = X.columns[X.notna().mean() >= 0.7]
    X = X[keep]
    variable = X.columns[X.nunique(dropna=True) > 1]
    X = X[variable]
    return X.reset_index(drop=True), list(X.columns)


def build_models(k: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("select", SelectKBest(f_classif, k=k)),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("select", SelectKBest(f_classif, k=k)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def metric_row(y_true: pd.Series, pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true == "narcolepsy", proba)),
        "sensitivity": float(recall_score(y_true == "narcolepsy", pred == "narcolepsy", zero_division=0)),
        "specificity": float(recall_score(y_true == "comparison", pred == "comparison", zero_division=0)),
    }


def selected_importance(model: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    selector = model.named_steps["select"]
    selected = list(np.asarray(feature_cols)[selector.get_support()])
    final = model.named_steps["model"]
    if hasattr(final, "feature_importances_"):
        values = final.feature_importances_
    elif hasattr(final, "coef_"):
        values = np.mean(np.abs(final.coef_), axis=0)
    else:
        values = selector.scores_[selector.get_support()]
    return pd.DataFrame({"feature": selected, "importance": values}).sort_values("importance", ascending=False)


def run_true_shap(model: Pipeline, X: pd.DataFrame, feature_cols: list[str], tables: Path, figures: Path) -> str:
    try:
        import shap
    except ModuleNotFoundError:
        (tables.parent / "SHAP_NOT_RUN.md").write_text(
            "shap is not installed in this environment. Install with: python -m pip install shap\n",
            encoding="utf-8",
        )
        return "shap_not_installed"

    if "random_forest" not in model.named_steps["model"].__class__.__name__.lower():
        return "skipped_best_model_not_tree"

    selected_features = list(np.asarray(feature_cols)[model.named_steps["select"].get_support()])
    transformed = model.named_steps["imputer"].transform(X)
    transformed = model.named_steps["select"].transform(transformed)
    final = model.named_steps["model"]
    explainer = shap.TreeExplainer(final)
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        class_index = list(final.classes_).index("narcolepsy")
        values = shap_values[class_index]
    else:
        values = shap_values
        if values.ndim == 3:
            class_index = list(final.classes_).index("narcolepsy")
            values = values[:, :, class_index]

    shap_importance = (
        pd.DataFrame({"feature": selected_features, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_importance.to_csv(tables / "dreem_advanced_true_shap_importance.csv", index=False)

    shap.summary_plot(values, pd.DataFrame(transformed, columns=selected_features), show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(figures / "dreem_advanced_true_shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()
    return "completed"


def save_confusion_plot(cm: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(cm.columns)), labels=cm.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(cm.index)), labels=cm.index)
    for (row, col), value in np.ndenumerate(cm.values):
        ax.text(col, row, str(value), ha="center", va="center")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_importance_plot(importance: pd.DataFrame, path: Path, title: str) -> None:
    plot_df = importance.sort_values("importance", ascending=True).tail(25)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(plot_df["feature"], plot_df["importance"])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
