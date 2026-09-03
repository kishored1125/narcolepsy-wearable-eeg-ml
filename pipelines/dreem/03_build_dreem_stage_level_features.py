#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


META_COLUMNS = {
    "patient_id",
    "NRID",
    "sex",
    "age",
    "diagnosis",
    "binary_target",
}
NARCOLEPSY_DIAGNOSES = {
    "NARCOLEPSY",
    "NARCOLEPSY TYPE 1",
    "NARCOLEPSY TYPE 2",
    "NT1",
    "NT2",
}
STAGE_ORDER = ["Wake", "N1", "N2", "N3", "REM"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and evaluate Dreem H5 EEG features aggregated separately by sleep stage."
    )
    parser.add_argument("--epoch-dir", required=True, help="Folder containing epoch_features_by_record/*.parquet.")
    parser.add_argument("--label-csv", required=True, help="CSV containing patient_id and diagnosis or binary_target.")
    parser.add_argument("--out-dir", required=True, help="Writable output folder. Use scratch/home, not shared data.")
    parser.add_argument("--target", choices=["narcolepsy_vs_other", "nt1_vs_other", "nt2_vs_other"], default="narcolepsy_vs_other")
    parser.add_argument("--max-files", type=int, default=None, help="Optional small test limit.")
    args = parser.parse_args()

    epoch_dir = Path(args.epoch_dir).expanduser().resolve()
    label_csv = Path(args.label_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    tables_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(epoch_dir.glob("*.parquet"))
    if args.max_files is not None:
        parquet_files = parquet_files[: args.max_files]
    if not parquet_files:
        raise SystemExit(f"No parquet files found in: {epoch_dir}")
    if not label_csv.exists():
        raise SystemExit(f"Label CSV not found: {label_csv}")

    epoch_df = pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)
    required_epoch = {"patient_id", "recording_id", "epoch_index", "label"}
    missing_epoch = sorted(required_epoch - set(epoch_df.columns))
    if missing_epoch:
        raise SystemExit(f"Epoch parquet files are missing required columns: {missing_epoch}")

    labels = load_labels(label_csv, args.target)
    stage_subject = build_stage_subject_features(epoch_df)
    labelled = stage_subject.merge(labels, on="patient_id", how="inner")
    labelled = labelled[labelled["binary_target"].isin(["narcolepsy", "comparison"])].copy()
    labelled.to_csv(tables_dir / "dreem_stage_level_subject_features.csv", index=False)

    stage_counts = build_stage_counts(epoch_df)
    stage_counts.to_csv(tables_dir / "dreem_stage_epoch_counts.csv", index=False)

    feature_cols = select_numeric_features(labelled)
    X, clean_cols = clean_matrix(labelled[feature_cols])
    y = labelled["binary_target"].reset_index(drop=True)
    pd.DataFrame({"feature": clean_cols}).to_csv(tables_dir / "dreem_stage_features_after_qc.csv", index=False)

    metrics = evaluate_model(X, y)
    metrics.update(
        {
            "target": args.target,
            "epoch_files": len(parquet_files),
            "epochs": int(len(epoch_df)),
            "labelled_subjects": int(len(labelled)),
            "narcolepsy_subjects": int((y == "narcolepsy").sum()),
            "comparison_subjects": int((y == "comparison").sum()),
            "raw_stage_features": int(len(feature_cols)),
            "features_after_qc": int(len(clean_cols)),
        }
    )
    pd.DataFrame([metrics]).to_csv(tables_dir / "dreem_stage_level_model_summary.csv", index=False)

    summary = {
        "epoch_dir_read_only": str(epoch_dir),
        "label_csv_read_only": str(label_csv),
        "output_dir": str(out_dir),
        **metrics,
    }
    (out_dir / "dreem_stage_level_run_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "summary.md").write_text(
        "# Dreem Stage-Level Feature Experiment\n\n"
        f"- Target: {args.target}\n"
        f"- Epoch parquet files read: {len(parquet_files)}\n"
        f"- Labelled subjects: {metrics['labelled_subjects']}\n"
        f"- Narcolepsy subjects: {metrics['narcolepsy_subjects']}\n"
        f"- Comparison subjects: {metrics['comparison_subjects']}\n"
        f"- Features after QC: {metrics['features_after_qc']}\n"
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.3f}\n"
        f"- Macro F1: {metrics['macro_f1']:.3f}\n"
        f"- ROC-AUC: {metrics['roc_auc']:.3f}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def load_labels(path: Path, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "patient_id" not in df.columns:
        raise SystemExit("Label CSV must contain a patient_id column.")
    df = df[df["patient_id"].notna()].copy()
    out = df[["patient_id"]].copy()

    if target == "narcolepsy_vs_other" and "binary_target" in df.columns:
        out["binary_target"] = df["binary_target"].astype(str).str.lower()
        out["binary_target"] = out["binary_target"].replace({"narcolepsy": "narcolepsy", "comparison": "comparison"})
        return out[["patient_id", "binary_target"]]

    if "diagnosis" not in df.columns:
        raise SystemExit("Label CSV must contain diagnosis, or binary_target for narcolepsy_vs_other.")

    diagnosis = df["diagnosis"].astype(str).str.upper().str.strip()
    valid = df["diagnosis"].notna() & ~diagnosis.isin({"", "NAN", "WITHDRAWN"})
    out = out[valid].copy()
    diagnosis = diagnosis[valid]
    if target == "nt1_vs_other":
        positive = diagnosis.isin({"NT1", "NARCOLEPSY TYPE 1", "NARCOLEPSY_TYPE_1", "NARCOLEPSY 1"})
    elif target == "nt2_vs_other":
        positive = diagnosis.isin({"NT2", "NARCOLEPSY TYPE 2", "NARCOLEPSY_TYPE_2", "NARCOLEPSY 2"})
    else:
        positive = diagnosis.isin(NARCOLEPSY_DIAGNOSES) | diagnosis.str.contains("NARCOLEPSY", na=False)
    out["binary_target"] = np.where(positive, "narcolepsy", "comparison")
    return out[["patient_id", "binary_target"]]


def build_stage_subject_features(epoch_df: pd.DataFrame) -> pd.DataFrame:
    blocked = {"epoch_index"}
    numeric_cols = [
        col
        for col in epoch_df.select_dtypes(include=[np.number]).columns
        if col not in blocked and not col.endswith("_seconds")
    ]
    grouped = epoch_df.groupby(["patient_id", "label"], dropna=False)[numeric_cols]
    pieces = []
    for stat in ["mean", "std", "median", "min", "max"]:
        frame = getattr(grouped, stat)().reset_index()
        frame = frame.pivot(index="patient_id", columns="label", values=numeric_cols)
        frame.columns = [f"{stage}__{feature}__{stat}" for feature, stage in frame.columns]
        pieces.append(frame)
    return pd.concat(pieces, axis=1).reset_index()


def build_stage_counts(epoch_df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        epoch_df.groupby(["patient_id", "label"])
        .size()
        .rename("epochs")
        .reset_index()
    )
    totals = counts.groupby("patient_id")["epochs"].transform("sum")
    counts["epoch_fraction"] = counts["epochs"] / totals
    counts["label"] = pd.Categorical(counts["label"], categories=STAGE_ORDER, ordered=True)
    return counts.sort_values(["patient_id", "label"])


def select_numeric_features(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in META_COLUMNS]
    return [c for c in cols if not c.lower().endswith("_target")]


def clean_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    X = frame.replace([np.inf, -np.inf], np.nan)
    keep = X.columns[X.notna().mean() >= 0.6]
    X = X[keep]
    variable = X.columns[X.nunique(dropna=True) > 1]
    X = X[variable]
    return X.reset_index(drop=True), list(X.columns)


def evaluate_model(X: pd.DataFrame, y: pd.Series) -> dict[str, float | int]:
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise SystemExit("Only one class is present after labelling; cannot train classifier.")
    n_splits = int(min(5, class_counts.min()))
    if n_splits < 2:
        raise SystemExit("At least two subjects are needed in each class for cross-validation.")

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("select", SelectKBest(f_classif, k=min(80, X.shape[1]))),
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
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pred = cross_val_predict(clone(model), X, y, cv=cv, method="predict")
    proba = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")
    fitted = clone(model).fit(X, y)
    pos_idx = list(fitted.classes_).index("narcolepsy")
    cm = confusion_matrix(y, pred, labels=["comparison", "narcolepsy"])
    return {
        "cv_folds": n_splits,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y == "narcolepsy", proba[:, pos_idx])),
        "sensitivity": float(recall_score(y == "narcolepsy", pred == "narcolepsy", zero_division=0)),
        "specificity": float(recall_score(y == "comparison", pred == "comparison", zero_division=0)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


if __name__ == "__main__":
    main()
