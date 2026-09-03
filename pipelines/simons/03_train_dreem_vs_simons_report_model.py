#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.append(str(PROJECT / "src"))
from diss_eeg.pipeline_utils import clean_feature_matrix, numeric_features


META = {"subject_id", "source_dataset", "diagnosis", "binary_target", "age", "sex_numeric", "subject_sp_id", "patient_id", "NRID", "sex", "external_group", "asd", "family_sf_id"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare narcolepsy subjects against Simons ASD-negative external controls using common report features.")
    parser.add_argument(
        "--simons-report-subjects",
        default=str(PROJECT / "simons_ssp" / "outputs" / "report_outputs" / "tables" / "simons_report_subject_features.csv"),
    )
    parser.add_argument(
        "--narcolepsy-report-subjects",
        default=str(PROJECT / "dreem_nrev" / "outputs" / "feature_outputs" / "report_features" / "report_subject_features.csv"),
    )
    parser.add_argument("--out-dir", default=str(PROJECT / "simons_ssp" / "outputs" / "comparison_outputs"))
    args = parser.parse_args()

    out = Path(args.out_dir)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    simons = pd.read_csv(args.simons_report_subjects)
    narco = pd.read_csv(args.narcolepsy_report_subjects)

    simons_controls = simons[simons["asd"] == False].copy()
    narco_cases = narco[narco["diagnosis"].isin(["NT1", "NT2"])].copy()

    simons_ready = prepare_simons(simons_controls)
    narco_ready = prepare_narco(narco_cases)
    common = sorted(set(feature_cols(narco_ready)) & set(feature_cols(simons_ready)))

    combined = pd.concat(
        [
            narco_ready[["subject_id", "source_dataset", "diagnosis", "binary_target", "age", "sex_numeric"] + common],
            simons_ready[["subject_id", "source_dataset", "diagnosis", "binary_target", "age", "sex_numeric"] + common],
        ],
        ignore_index=True,
    )
    combined.to_csv(tables / "narcolepsy_vs_simons_report_common_features.csv", index=False)

    summary = {
        "narcolepsy_subjects": len(narco_ready),
        "simons_asd_false_controls": len(simons_ready),
        "common_features": len(common),
    }

    if len(narco_ready) < 3 or len(simons_ready) < 3 or len(common) < 2:
        pd.DataFrame([summary | {"status": "not_enough_data_for_model"}]).to_csv(tables / "narcolepsy_vs_simons_report_summary.csv", index=False)
        (out / "summary.md").write_text(
            "# Narcolepsy vs Simons Report Comparison\n\n"
            "Model was not run because there are not enough local Simons ASD-negative controls or common features yet.\n\n"
            + "\n".join(f"- {k}: {v}" for k, v in summary.items())
            + "\n",
            encoding="utf-8",
        )
        return

    results = evaluate(combined, common)
    results.to_csv(tables / "narcolepsy_vs_simons_report_model_metrics.csv", index=False)
    best = results.iloc[0].to_dict()
    pd.DataFrame([summary | best]).to_csv(tables / "narcolepsy_vs_simons_report_summary.csv", index=False)
    (out / "summary.md").write_text(
        "# Narcolepsy vs Simons Report Comparison\n\n"
        f"Narcolepsy subjects: {summary['narcolepsy_subjects']}\n\n"
        f"Simons ASD-negative controls: {summary['simons_asd_false_controls']}\n\n"
        f"Common report features: {summary['common_features']}\n\n"
        f"Best model: {best['model']}\n\n"
        f"Balanced accuracy: {best['balanced_accuracy']:.3f}\n\n"
        f"Macro F1: {best['macro_f1']:.3f}\n\n"
        f"ROC-AUC: {best['roc_auc']:.3f}\n",
        encoding="utf-8",
    )


def prepare_simons(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["subject_id"] = out["subject_sp_id"]
    out["source_dataset"] = "simons"
    out["diagnosis"] = "simons_asd_false"
    out["binary_target"] = "external_control"
    return out


def prepare_narco(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["subject_id"] = out["patient_id"]
    out["source_dataset"] = "narcolepsy_dreem"
    out["binary_target"] = "narcolepsy"
    out["sex_numeric"] = out["sex"]
    return out


def feature_cols(df: pd.DataFrame) -> list[str]:
    return numeric_features(df, META)


def evaluate(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X, cols = clean_feature_matrix(df, cols)
    y = df["binary_target"].reset_index(drop=True)
    splits = max(2, min(5, int(y.value_counts().min())))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    rows = []
    for name, model in models(min(50, X.shape[1])).items():
        pred = cross_val_predict(clone(model), X, y, cv=cv, method="predict")
        proba = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")
        fitted = clone(model).fit(X, y)
        pos_idx = list(fitted.classes_).index("narcolepsy")
        rows.append(
            {
                "model": name,
                "balanced_accuracy": balanced_accuracy_score(y, pred),
                "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
                "sensitivity_narcolepsy": recall_score(y == "narcolepsy", pred == "narcolepsy", zero_division=0),
                "specificity_external_control": recall_score(y != "narcolepsy", pred != "narcolepsy", zero_division=0),
                "roc_auc": roc_auc_score(y == "narcolepsy", proba[:, pos_idx]),
                "features_after_qc": len(cols),
                "cv_splits": splits,
            }
        )
    return pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)


def models(k: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("select", SelectKBest(f_classif, k=k)),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("select", SelectKBest(f_classif, k=k)),
                ("model", SVC(kernel="linear", class_weight="balanced", probability=True, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("select", SelectKBest(f_classif, k=k)),
                ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=1)),
            ]
        ),
    }


if __name__ == "__main__":
    main()
