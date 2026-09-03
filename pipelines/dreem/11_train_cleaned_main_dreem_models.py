#!/usr/bin/env python3
"""Train cleaned main Narcolepsy Revolution Dreem diagnostic models.

The earlier main result used the full combined H5 + report table. This script
adds a model-audit analysis that removes report confidence, quality,
off-head, scorable, respiration and recording-duration variables, then compares
the cleaned report/H5 feature sets at subject level.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


META_COLUMNS = {"patient_id", "NRID", "sex", "age", "diagnosis", "binary_target"}
ARTEFACT_RE = re.compile(
    r"(confidence|quality|off_head|scorable|record_duration|respiration|respiratory)",
    flags=re.IGNORECASE,
)
REPORT_SLEEP_RE = re.compile(
    r"(tst|sleep_efficiency|sol|lps|waso|wake_duration|rem_latency|"
    r"n1_|n2_|n3_|nrem_|rem_|percentage|duration|awakenings|nr_of_shifts|micro_arousal)",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--combined-features",
        type=Path,
        default=Path("dreem_nrev/outputs/feature_outputs/combined_subject_features.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("final_results/outputs/model_audits"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else project_root() / path


def make_dirs(out_dir: Path) -> tuple[Path, Path, Path]:
    tables = out_dir / "tables"
    figures = out_dir / "figures"
    docs = out_dir / "docs"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    return tables, figures, docs


def get_feature_sets(columns: list[str]) -> dict[str, list[str]]:
    h5 = [c for c in columns if c.startswith("h5__")]
    report = [c for c in columns if c.startswith("report__")]
    report_no_artifact = [c for c in report if not ARTEFACT_RE.search(c)]
    report_sleep = [c for c in report_no_artifact if REPORT_SLEEP_RE.search(c)]
    return {
        "full_combined_no_demographics": [c for c in columns if c not in META_COLUMNS],
        "h5_only": h5,
        "report_sleep_architecture_clean": report_sleep,
        "combined_h5_report_sleep_clean": h5 + report_sleep,
        "combined_h5_report_artifacts_removed": h5 + report_no_artifact,
    }


def clean_X(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    X = df[features].apply(pd.to_numeric, errors="coerce")
    cols = [c for c in X.columns if X[c].notna().any() and X[c].nunique(dropna=True) > 1]
    return X[cols], cols


def models(random_state: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold()),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=800,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold()),
                ("model", GradientBoostingClassifier(random_state=random_state)),
            ]
        ),
    }


def score_model(model: Pipeline, X: pd.DataFrame, y: np.ndarray, folds: int) -> tuple[np.ndarray, np.ndarray]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    pred = cross_val_predict(model, X, y, cv=cv, method="predict")
    if hasattr(model[-1], "predict_proba"):
        score = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    else:
        score = cross_val_predict(model, X, y, cv=cv, method="decision_function")
    return pred, score


def metric_row(feature_set: str, model: str, y: np.ndarray, pred: np.ndarray, score: np.ndarray, n_features: int) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "feature_set": feature_set,
        "model": model,
        "features_used": n_features,
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, score),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def kept_features(fitted: Pipeline, features: list[str]) -> list[str]:
    mask = fitted.named_steps["variance"].get_support()
    return list(np.array(features)[mask])


def importance_table(fitted: Pipeline, features: list[str]) -> pd.DataFrame:
    kept = kept_features(fitted, features)
    clf = fitted.named_steps["model"]
    if hasattr(clf, "feature_importances_"):
        return pd.DataFrame({"feature": kept, "importance": clf.feature_importances_}).sort_values("importance", ascending=False)
    if hasattr(clf, "coef_"):
        coef = clf.coef_.ravel()
        return pd.DataFrame({"feature": kept, "coefficient": coef, "importance": np.abs(coef)}).sort_values("importance", ascending=False)
    return pd.DataFrame({"feature": kept})


def plot_bar(df: pd.DataFrame, value: str, title: str, out_file: Path, n: int = 20) -> None:
    top = df.head(n).iloc[::-1]
    plt.figure(figsize=(9.5, 7))
    plt.barh(top["feature"], top[value])
    plt.xlabel(value.replace("_", " ").title())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close()


def plot_metrics(metrics: pd.DataFrame, out_file: Path) -> None:
    best = metrics.groupby("feature_set", as_index=False).head(1)
    x = np.arange(len(best))
    width = 0.24
    labels = best["feature_set"].str.replace("_", "\n")
    plt.figure(figsize=(11, 5.8))
    for i, metric in enumerate(["balanced_accuracy", "macro_f1", "roc_auc"]):
        plt.bar(x + (i - 1) * width, best[metric], width, label=metric.replace("_", " ").title())
    plt.xticks(x, labels, fontsize=8)
    plt.ylim(0, 1.05)
    plt.ylabel("Cross-validated score")
    plt.title("Narcolepsy Revolution main model: full versus cleaned features")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close()


def plot_confusion(row: pd.Series, out_file: Path) -> None:
    cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=int)
    plt.figure(figsize=(4.8, 4.2))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], ["Other hypersomnia", "NT1/NT2"], rotation=20, ha="right")
    plt.yticks([0, 1], ["Other hypersomnia", "NT1/NT2"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=13)
    plt.title("Clean main NRev confusion matrix")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close()


def artifact_summary(df: pd.DataFrame, full_importance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in full_importance["feature"].head(100):
        if not ARTEFACT_RE.search(feature):
            continue
        values = pd.to_numeric(df[feature], errors="coerce")
        pos = values[df["binary_target"] == "narcolepsy"]
        neg = values[df["binary_target"] == "comparison"]
        pooled = values.std(ddof=0)
        rows.append(
            {
                "feature": feature,
                "narcolepsy_mean": pos.mean(),
                "other_hypersomnia_mean": neg.mean(),
                "absolute_standardised_difference": abs((pos.mean() - neg.mean()) / pooled) if pooled else np.nan,
            }
        )
    return pd.DataFrame(rows).merge(full_importance, on="feature", how="left")


def main() -> None:
    args = parse_args()
    combined_path = resolve(args.combined_features)
    out_dir = resolve(args.out_dir)
    tables, figures, docs = make_dirs(out_dir)

    df = pd.read_csv(combined_path)
    df = df[df["binary_target"].isin(["comparison", "narcolepsy"])].copy()
    y = (df["binary_target"] == "narcolepsy").astype(int).to_numpy()
    folds = min(args.cv_folds, int(np.bincount(y).min()))

    metrics = []
    fitted = {}
    for set_name, features in get_feature_sets(list(df.columns)).items():
        X, usable = clean_X(df, features)
        for model_name, model in models(args.random_state).items():
            pred, score = score_model(model, X, y, folds)
            metrics.append(metric_row(set_name, model_name, y, pred, score, len(usable)))
            fitted[(set_name, model_name)] = (model.fit(X, y), X, usable, pred, score)

    metrics_df = pd.DataFrame(metrics).sort_values(["feature_set", "balanced_accuracy", "roc_auc"], ascending=[True, False, False])
    metrics_df.to_csv(tables / "nrev_cleaned_main_model_metrics.csv", index=False)
    best_by_set = metrics_df.groupby("feature_set", as_index=False).head(1)
    best_by_set.to_csv(tables / "nrev_cleaned_main_best_model_by_feature_set.csv", index=False)

    full_rf, _, full_features, _, _ = fitted[("full_combined_no_demographics", "random_forest")]
    full_imp = importance_table(full_rf, full_features).reset_index(drop=True)
    full_imp.to_csv(tables / "nrev_full_combined_random_forest_feature_importance_audit.csv", index=False)
    artifact_summary(df, full_imp).to_csv(tables / "nrev_main_model_artifact_feature_audit.csv", index=False)

    clean_rows = metrics_df[metrics_df["feature_set"].isin(["combined_h5_report_sleep_clean", "combined_h5_report_artifacts_removed"])]
    best_clean = clean_rows.sort_values(["balanced_accuracy", "roc_auc"], ascending=False).iloc[0]
    clean_key = (best_clean["feature_set"], best_clean["model"])
    clean_model, clean_matrix, clean_features, _, _ = fitted[clean_key]
    clean_imp = importance_table(clean_model, clean_features).reset_index(drop=True)
    clean_imp.to_csv(tables / "nrev_cleaned_main_best_model_feature_importance.csv", index=False)

    permutation_candidates = list(clean_imp["feature"].head(150))
    permutation_matrix = clean_matrix[permutation_candidates]
    permutation_model = models(args.random_state)[str(best_clean["model"])].fit(permutation_matrix, y)
    perm = permutation_importance(
        permutation_model,
        permutation_matrix,
        y,
        n_repeats=50,
        random_state=args.random_state,
        scoring="balanced_accuracy",
    )
    perm_df = pd.DataFrame(
        {
            "feature": permutation_candidates,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
            "candidate_selection": "top_150_by_clean_model_importance",
        }
    ).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(tables / "nrev_cleaned_main_permutation_importance.csv", index=False)

    plot_metrics(metrics_df, figures / "nrev_main_full_vs_cleaned_metrics.png")
    plot_confusion(best_clean, figures / "nrev_cleaned_main_confusion_matrix.png")
    plot_bar(full_imp, "importance", "Main NRev full combined RF feature importance audit", figures / "nrev_full_combined_artifact_importance_audit.png")
    plot_bar(clean_imp, "importance", "Main NRev cleaned feature importance", figures / "nrev_cleaned_main_feature_importance.png")
    plot_bar(perm_df.rename(columns={"importance_mean": "importance"}), "importance", "Main NRev cleaned permutation importance", figures / "nrev_cleaned_main_permutation_importance.png")

    diagnosis_counts = df.groupby(["diagnosis", "binary_target"]).size().reset_index(name="subjects").sort_values("subjects", ascending=False)
    diagnosis_counts.to_csv(tables / "nrev_main_diagnosis_counts.csv", index=False)
    plt.figure(figsize=(8, 4.8))
    plt.bar(diagnosis_counts["diagnosis"], diagnosis_counts["subjects"])
    plt.ylabel("Subjects")
    plt.title("Narcolepsy Revolution modelling cohort diagnoses")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "nrev_main_diagnosis_counts.png", dpi=220)
    plt.close()

    summary = {
        "combined_features": str(combined_path),
        "output_dir": str(out_dir),
        "subjects": int(len(df)),
        "narcolepsy_subjects": int(y.sum()),
        "other_hypersomnia_subjects": int((1 - y).sum()),
        "cv_folds": int(folds),
        "best_clean_feature_set": best_clean["feature_set"],
        "best_clean_model": best_clean["model"],
        "best_clean_balanced_accuracy": float(best_clean["balanced_accuracy"]),
        "best_clean_macro_f1": float(best_clean["macro_f1"]),
        "best_clean_roc_auc": float(best_clean["roc_auc"]),
        "best_clean_sensitivity": float(best_clean["sensitivity"]),
        "best_clean_specificity": float(best_clean["specificity"]),
    }
    (tables / "nrev_cleaned_main_run_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([summary]).to_csv(tables / "nrev_cleaned_main_run_summary.csv", index=False)
    (docs / "nrev_cleaned_main_model_summary.md").write_text(
        f"""# Cleaned Main Narcolepsy Revolution Model

The cleaned analysis removes likely report artefact variables before modelling the Narcolepsy Revolution cohort.

- Subjects: {summary['subjects']} ({summary['narcolepsy_subjects']} NT1/NT2 and {summary['other_hypersomnia_subjects']} other hypersomnia/comparison).
- Best cleaned model: {summary['best_clean_model']} using `{summary['best_clean_feature_set']}`.
- Metrics: balanced accuracy {summary['best_clean_balanced_accuracy']:.3f}, macro F1 {summary['best_clean_macro_f1']:.3f}, ROC-AUC {summary['best_clean_roc_auc']:.3f}, sensitivity {summary['best_clean_sensitivity']:.3f}, specificity {summary['best_clean_specificity']:.3f}.

Use this alongside the original tuned combined model to discuss whether performance depends on physiological sleep features or report quality/confidence fields.
"""
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
