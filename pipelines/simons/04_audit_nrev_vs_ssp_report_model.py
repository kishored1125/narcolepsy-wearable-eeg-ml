#!/usr/bin/env python3
"""Audit Narcolepsy Revolution vs SSP Dreem report-feature classification.

This script focuses on the same-device comparison between Narcolepsy
Revolution Dreem participants and SSP Dreem healthy/control participants. It
keeps the original cross-dataset feature set for audit, then re-runs the model
after removing likely non-sleep/domain artefact variables such as report
confidence, quality, off-head and recording-duration features.
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
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


META_COLUMNS = {"subject_id", "source_dataset", "diagnosis", "binary_target", "age", "sex_numeric"}
POSITIVE_LABEL = "narcolepsy"
NEGATIVE_LABEL = "external_control"
CLASS_ORDER = [NEGATIVE_LABEL, POSITIVE_LABEL]

NON_SLEEP_ARTEFACT_RE = re.compile(
    r"(quality|confidence|off_head|scorable|record_duration|respiration|respiratory)",
    flags=re.IGNORECASE,
)
SLEEP_ARCHITECTURE_RE = re.compile(
    r"(tst|sleep_efficiency|sol|lps|waso|wake_duration|rem_latency|"
    r"n1_|n2_|n3_|nrem_|rem_|percentage|duration|awakenings|nr_of_shifts|micro_arousal)",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--common-features",
        type=Path,
        default=Path("simons_ssp/outputs/comparison_outputs/tables/narcolepsy_vs_simons_report_common_features.csv"),
        help="CSV containing common NRev and SSP report features.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("final_results/outputs/model_audits"),
        help="Output directory for corrected comparison assets.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return project_root() / path


def ensure_dirs(out_dir: Path) -> tuple[Path, Path, Path]:
    tables = out_dir / "tables"
    figures = out_dir / "figures"
    docs = out_dir / "docs"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    return tables, figures, docs


def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    candidate = [c for c in columns if c not in META_COLUMNS]
    no_artifact = [c for c in candidate if not NON_SLEEP_ARTEFACT_RE.search(c)]
    sleep_architecture = [c for c in no_artifact if SLEEP_ARCHITECTURE_RE.search(c)]
    conservative_core = [
        c
        for c in sleep_architecture
        if not re.search(r"(wake_duration|micro_arousal)", c, flags=re.IGNORECASE)
    ]
    return {
        "original_common_report_features": candidate,
        "artifact_removed_report_features": no_artifact,
        "clean_sleep_architecture_features": sleep_architecture,
        "conservative_sleep_architecture_core": conservative_core,
    }


def clean_matrix(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    X = df[features].apply(pd.to_numeric, errors="coerce")
    usable = [c for c in X.columns if X[c].notna().any()]
    X = X[usable]
    nunique = X.nunique(dropna=True)
    usable = [c for c in X.columns if nunique[c] > 1]
    return X[usable], usable


def build_models(random_state: int) -> dict[str, Pipeline]:
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


def metric_row(feature_set: str, model_name: str, y_true: np.ndarray, pred: np.ndarray, score: np.ndarray, n_features: int) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "feature_set": feature_set,
        "model": model_name,
        "features_used": n_features,
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, score),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def cv_predictions(model: Pipeline, X: pd.DataFrame, y: np.ndarray, folds: int) -> tuple[np.ndarray, np.ndarray]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    pred = cross_val_predict(model, X, y, cv=cv, method="predict")
    if hasattr(model[-1], "predict_proba"):
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    else:
        proba = cross_val_predict(model, X, y, cv=cv, method="decision_function")
    return pred, proba


def selected_features_after_variance(model: Pipeline, features: list[str]) -> list[str]:
    if "variance" not in model.named_steps:
        return features
    mask = model.named_steps["variance"].get_support()
    return list(np.array(features)[mask])


def tree_importance(model: Pipeline, features: list[str]) -> pd.DataFrame:
    kept = selected_features_after_variance(model, features)
    clf = model.named_steps["model"]
    if not hasattr(clf, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    return (
        pd.DataFrame({"feature": kept, "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def coefficients(model: Pipeline, features: list[str]) -> pd.DataFrame:
    kept = selected_features_after_variance(model, features)
    clf = model.named_steps["model"]
    if not hasattr(clf, "coef_"):
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])
    coefs = clf.coef_.ravel()
    return (
        pd.DataFrame({"feature": kept, "coefficient": coefs, "abs_coefficient": np.abs(coefs)})
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def threshold_audit(y: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.unique(np.quantile(score, np.linspace(0, 1, 101))):
        pred = (score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": threshold,
                "balanced_accuracy": balanced_accuracy_score(y, pred),
                "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
                "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
                "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return pd.DataFrame(rows).sort_values(["balanced_accuracy", "macro_f1"], ascending=False)


def artefact_audit(df: pd.DataFrame, features: list[str], original_importance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = df.groupby("binary_target")
    for feature in features:
        if not NON_SLEEP_ARTEFACT_RE.search(feature):
            continue
        values = pd.to_numeric(df[feature], errors="coerce")
        if values.notna().sum() < 4:
            continue
        nrev = pd.to_numeric(grouped.get_group(POSITIVE_LABEL)[feature], errors="coerce")
        ssp = pd.to_numeric(grouped.get_group(NEGATIVE_LABEL)[feature], errors="coerce")
        pooled = values.std(ddof=0)
        effect = (nrev.mean() - ssp.mean()) / pooled if pooled and np.isfinite(pooled) else np.nan
        rows.append(
            {
                "feature": feature,
                "nrev_mean": nrev.mean(),
                "ssp_mean": ssp.mean(),
                "absolute_standardised_difference": abs(effect) if np.isfinite(effect) else np.nan,
            }
        )
    audit = pd.DataFrame(rows)
    if audit.empty:
        return audit
    audit = audit.merge(original_importance, on="feature", how="left")
    return audit.sort_values(
        ["importance", "absolute_standardised_difference"], ascending=False, na_position="last"
    ).reset_index(drop=True)


def plot_metrics(metrics: pd.DataFrame, out_file: Path) -> None:
    best = (
        metrics.sort_values(["feature_set", "balanced_accuracy"], ascending=[True, False])
        .groupby("feature_set", as_index=False)
        .head(1)
    )
    labels = best["feature_set"].str.replace("_", "\n")
    x = np.arange(len(best))
    width = 0.24
    plt.figure(figsize=(11, 5.8))
    for i, metric in enumerate(["balanced_accuracy", "macro_f1", "roc_auc"]):
        plt.bar(x + (i - 1) * width, best[metric], width, label=metric.replace("_", " ").title())
    plt.xticks(x, labels, fontsize=8)
    plt.ylim(0, 1.05)
    plt.ylabel("Cross-validated score")
    plt.title("NRev vs SSP: original report features compared with cleaned feature sets")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close()


def plot_confusion(row: pd.Series, out_file: Path) -> None:
    cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=float)
    plt.figure(figsize=(4.8, 4.2))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], ["SSP control", "NRev narcolepsy"], rotation=20, ha="right")
    plt.yticks([0, 1], ["SSP control", "NRev narcolepsy"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=13)
    plt.title("Clean NRev-vs-SSP confusion matrix")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close()


def plot_top_bars(df: pd.DataFrame, value_col: str, title: str, out_file: Path, n: int = 20) -> None:
    top = df.head(n).iloc[::-1]
    plt.figure(figsize=(9.5, 7))
    plt.barh(top["feature"], top[value_col])
    plt.xlabel(value_col.replace("_", " ").title())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close()


def plot_feature_boxplots(df: pd.DataFrame, features: list[str], out_file: Path, title: str = "Feature distributions") -> None:
    chosen = [f for f in features if f in df.columns][:6]
    if not chosen:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8))
    axes = axes.ravel()
    groups = [NEGATIVE_LABEL, POSITIVE_LABEL]
    for ax, feature in zip(axes, chosen):
        data = [pd.to_numeric(df.loc[df["binary_target"] == g, feature], errors="coerce").dropna() for g in groups]
        ax.boxplot(data, tick_labels=["SSP", "NRev"], showfliers=False)
        ax.set_title(feature, fontsize=9)
    for ax in axes[len(chosen):]:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close()


def plot_cohort_counts(df: pd.DataFrame, out_file: Path) -> pd.DataFrame:
    counts = (
        df.assign(cohort_label=df["diagnosis"].replace({"simons_asd_false": "SSP healthy/control"}))
        .groupby(["source_dataset", "cohort_label"])
        .size()
        .reset_index(name="subjects")
        .sort_values(["source_dataset", "subjects"], ascending=[True, False])
    )
    labels = counts["cohort_label"]
    plt.figure(figsize=(8, 4.8))
    plt.bar(labels, counts["subjects"], color=["#4C78A8", "#F58518", "#54A24B"])
    plt.ylabel("Subjects")
    plt.title("Main NRev-vs-SSP comparison cohort composition")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close()
    return counts


def plot_hypnogram_concept(out_file: Path) -> None:
    stages = ["Wake", "N1", "N2", "N3", "REM"]
    stage_to_y = {s: i for i, s in enumerate(stages)}
    sequence = (
        ["Wake"] * 6 + ["N1"] * 2 + ["N2"] * 8 + ["N3"] * 7 + ["N2"] * 6 + ["REM"] * 5
        + ["Wake"] * 2 + ["N1"] * 1 + ["N2"] * 8 + ["N3"] * 3 + ["N2"] * 5 + ["REM"] * 7
        + ["Wake"] * 2 + ["N2"] * 6 + ["REM"] * 8
    )
    x = np.arange(len(sequence)) * 0.5
    y = [stage_to_y[s] for s in sequence]
    plt.figure(figsize=(10, 4))
    plt.step(x, y, where="post", linewidth=2)
    plt.yticks(range(len(stages)), stages)
    plt.gca().invert_yaxis()
    plt.xlabel("Hours from recording start")
    plt.ylabel("Sleep stage")
    plt.title("Thirty-second sleep-stage structure used for stage-aware features")
    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close()


def demographic_baseline(df: pd.DataFrame, y: np.ndarray, folds: int) -> pd.DataFrame:
    rows = []
    for features in [["age"], ["sex_numeric"], ["age", "sex_numeric"]]:
        available = [f for f in features if f in df.columns]
        if not available:
            continue
        X = df[available].apply(pd.to_numeric, errors="coerce")
        model = Pipeline(
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
        )
        pred, score = cv_predictions(model, X, y, folds)
        rows.append(metric_row("demographic_baseline", "+".join(available), y, pred, score, len(available)))
    return pd.DataFrame(rows)


def demographic_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["source_dataset", "binary_target"])
        .agg(
            subjects=("subject_id", "count"),
            age_mean=("age", "mean"),
            age_sd=("age", "std"),
            age_min=("age", "min"),
            age_median=("age", "median"),
            age_max=("age", "max"),
            female_or_zero_count=("sex_numeric", lambda s: int((s == 0).sum())),
            male_or_one_count=("sex_numeric", lambda s: int((s == 1).sum())),
        )
        .reset_index()
    )


def plot_age_distribution(df: pd.DataFrame, out_file: Path) -> None:
    plt.figure(figsize=(7, 4.8))
    groups = [
        pd.to_numeric(df.loc[df["binary_target"] == NEGATIVE_LABEL, "age"], errors="coerce").dropna(),
        pd.to_numeric(df.loc[df["binary_target"] == POSITIVE_LABEL, "age"], errors="coerce").dropna(),
    ]
    plt.boxplot(groups, tick_labels=["SSP control", "NRev narcolepsy"], showfliers=False)
    for i, values in enumerate(groups, start=1):
        x = np.random.default_rng(42 + i).normal(i, 0.035, size=len(values))
        plt.scatter(x, values, alpha=0.65, s=22)
    plt.ylabel("Age")
    plt.title("NRev-vs-SSP age distribution audit")
    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    common_path = resolve_path(args.common_features)
    out_dir = resolve_path(args.out_dir)
    tables_dir, figures_dir, docs_dir = ensure_dirs(out_dir)

    df = pd.read_csv(common_path)
    df = df[df["binary_target"].isin(CLASS_ORDER)].copy()
    y = (df["binary_target"] == POSITIVE_LABEL).astype(int).to_numpy()
    folds = min(args.cv_folds, int(np.bincount(y).min()))
    if folds < 2:
        raise ValueError("Not enough positive/negative cases for cross-validation.")

    metrics = []
    fitted_models: dict[tuple[str, str], tuple[Pipeline, pd.DataFrame, list[str], np.ndarray, np.ndarray]] = {}
    sets = feature_sets(list(df.columns))
    for set_name, feature_list in sets.items():
        X, usable = clean_matrix(df, feature_list)
        if X.empty:
            continue
        for model_name, model in build_models(args.random_state).items():
            pred, score = cv_predictions(model, X, y, folds)
            metrics.append(metric_row(set_name, model_name, y, pred, score, len(usable)))
            fitted = model.fit(X, y)
            fitted_models[(set_name, model_name)] = (fitted, X, usable, pred, score)

    metrics_df = pd.DataFrame(metrics).sort_values(
        ["feature_set", "balanced_accuracy", "roc_auc"], ascending=[True, False, False]
    )
    metrics_df.to_csv(tables_dir / "nrev_vs_ssp_corrected_model_metrics.csv", index=False)

    best_by_set = metrics_df.groupby("feature_set", as_index=False).head(1)
    best_by_set.to_csv(tables_dir / "nrev_vs_ssp_best_model_by_feature_set.csv", index=False)

    original_key = ("original_common_report_features", "random_forest")
    clean_candidates = metrics_df[metrics_df["feature_set"].str.contains("sleep_architecture")]
    clean_best = clean_candidates.sort_values(["balanced_accuracy", "roc_auc"], ascending=False).iloc[0]
    clean_key = (clean_best["feature_set"], clean_best["model"])

    original_model, original_X, original_features, _, _ = fitted_models[original_key]
    original_importance = tree_importance(original_model, original_features)
    original_importance.to_csv(tables_dir / "nrev_vs_ssp_original_random_forest_feature_importance.csv", index=False)

    clean_model, clean_X, clean_features, clean_pred, clean_score = fitted_models[clean_key]
    if clean_best["model"] == "random_forest":
        clean_importance = tree_importance(clean_model, clean_features)
    else:
        clean_importance = coefficients(clean_model, clean_features).rename(columns={"abs_coefficient": "importance"})
        clean_importance = clean_importance[["feature", "importance", "coefficient"]]
    clean_importance.to_csv(tables_dir / "nrev_vs_ssp_clean_best_model_feature_importance.csv", index=False)

    perm = permutation_importance(
        clean_model,
        clean_X,
        y,
        n_repeats=100,
        random_state=args.random_state,
        scoring="balanced_accuracy",
    )
    perm_df = (
        pd.DataFrame(
            {
                "feature": clean_features,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    perm_df.to_csv(tables_dir / "nrev_vs_ssp_clean_permutation_importance.csv", index=False)

    audit_df = artefact_audit(df, original_features, original_importance)
    audit_df.to_csv(tables_dir / "nrev_vs_ssp_artifact_feature_audit.csv", index=False)

    threshold_df = threshold_audit(y, clean_score)
    threshold_df.to_csv(tables_dir / "nrev_vs_ssp_clean_threshold_audit.csv", index=False)

    demo_metrics = demographic_baseline(df, y, folds)
    demo_metrics.to_csv(tables_dir / "nrev_vs_ssp_demographic_baseline_metrics.csv", index=False)
    demographic_summary(df).to_csv(tables_dir / "nrev_vs_ssp_demographic_summary.csv", index=False)

    cohort_counts = plot_cohort_counts(df, figures_dir / "nrev_vs_ssp_cohort_diagnosis_counts.png")
    cohort_counts.to_csv(tables_dir / "nrev_vs_ssp_cohort_diagnosis_counts.csv", index=False)

    plot_metrics(metrics_df, figures_dir / "nrev_vs_ssp_original_vs_cleaned_metrics.png")
    plot_confusion(clean_best, figures_dir / "nrev_vs_ssp_clean_confusion_matrix.png")
    plot_top_bars(
        original_importance,
        "importance",
        "Original NRev-vs-SSP RF importances: artefact audit",
        figures_dir / "nrev_vs_ssp_original_artifact_importance.png",
    )
    plot_top_bars(
        clean_importance,
        "importance",
        "Clean NRev-vs-SSP feature importance",
        figures_dir / "nrev_vs_ssp_clean_feature_importance.png",
    )
    if not perm_df.empty:
        plot_top_bars(
            perm_df.rename(columns={"importance_mean": "importance"}),
            "importance",
            "Clean NRev-vs-SSP permutation importance",
            figures_dir / "nrev_vs_ssp_clean_permutation_importance.png",
        )
    quality_features = [f for f in original_importance["feature"].head(10) if f in df.columns]
    plot_feature_boxplots(
        df,
        quality_features[:6],
        figures_dir / "nrev_vs_ssp_artifact_feature_distributions.png",
        title="Report-quality and artefact feature distributions",
    )
    plot_feature_boxplots(
        df,
        list(clean_importance["feature"].head(6)),
        figures_dir / "nrev_vs_ssp_clean_top_feature_distributions.png",
        title="Clean sleep-architecture feature distributions",
    )
    plot_hypnogram_concept(figures_dir / "sleep_architecture_hypnogram_concept.png")
    plot_age_distribution(df, figures_dir / "nrev_vs_ssp_age_distribution_audit.png")

    run_summary = {
        "common_features": str(common_path),
        "output_dir": str(out_dir),
        "subjects": int(len(df)),
        "nrev_narcolepsy_subjects": int((df["binary_target"] == POSITIVE_LABEL).sum()),
        "ssp_control_subjects": int((df["binary_target"] == NEGATIVE_LABEL).sum()),
        "cv_folds": int(folds),
        "feature_sets_tested": {k: len(v) for k, v in sets.items()},
        "best_clean_feature_set": clean_best["feature_set"],
        "best_clean_model": clean_best["model"],
        "best_clean_balanced_accuracy": float(clean_best["balanced_accuracy"]),
        "best_clean_macro_f1": float(clean_best["macro_f1"]),
        "best_clean_roc_auc": float(clean_best["roc_auc"]),
        "best_clean_sensitivity": float(clean_best["sensitivity"]),
        "best_clean_specificity": float(clean_best["specificity"]),
        "age_only_roc_auc": float(demo_metrics.loc[demo_metrics["model"] == "age", "roc_auc"].iloc[0]) if not demo_metrics.empty and (demo_metrics["model"] == "age").any() else None,
    }
    (tables_dir / "nrev_vs_ssp_corrected_run_summary.json").write_text(json.dumps(run_summary, indent=2))
    pd.DataFrame([run_summary]).to_csv(tables_dir / "nrev_vs_ssp_corrected_run_summary.csv", index=False)

    summary_md = f"""# NRev-vs-SSP Corrected Comparison Audit

This output audits the original Narcolepsy Revolution versus SSP Dreem report-feature comparison and re-runs it after removing likely non-sleep artefact variables.

- Subjects: {len(df)} total ({run_summary['nrev_narcolepsy_subjects']} Narcolepsy Revolution narcolepsy, {run_summary['ssp_control_subjects']} SSP healthy/control).
- Best cleaned model: {clean_best['model']} using `{clean_best['feature_set']}`.
- Best cleaned metrics: balanced accuracy {clean_best['balanced_accuracy']:.3f}, macro F1 {clean_best['macro_f1']:.3f}, ROC-AUC {clean_best['roc_auc']:.3f}, sensitivity {clean_best['sensitivity']:.3f}, specificity {clean_best['specificity']:.3f}.
- The original feature set is retained only as an artefact audit because quality/confidence/report-format variables dominate the random-forest feature importance.

Use `tables/nrev_vs_ssp_corrected_model_metrics.csv` and the figures in `figures/` for the revised report section.
"""
    (docs_dir / "nrev_vs_ssp_corrected_comparison_summary.md").write_text(summary_md)
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
