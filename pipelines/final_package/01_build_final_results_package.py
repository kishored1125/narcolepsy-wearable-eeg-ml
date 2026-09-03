#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = PROJECT_ROOT / "final_results" / "outputs"
TABLES = OUT_ROOT / "tables"
FIGURES = OUT_ROOT / "figures"
DOCS = OUT_ROOT / "docs"


def main() -> None:
    for folder in [TABLES, FIGURES, DOCS]:
        folder.mkdir(parents=True, exist_ok=True)

    master = build_master_results()
    master.to_csv(TABLES / "final_master_results_table.csv", index=False)

    detailed = build_detailed_model_metrics()
    detailed.to_csv(TABLES / "final_model_metrics_detailed.csv", index=False)

    datasets = build_dataset_comparison()
    datasets.to_csv(TABLES / "final_dataset_comparison_table.csv", index=False)

    feature_families = build_feature_family_audit()
    feature_families.to_csv(TABLES / "final_feature_family_audit.csv", index=False)

    sleep_stages = build_sleep_stage_appendix()
    sleep_stages.to_csv(TABLES / "final_sleep_stage_appendix_table.csv", index=False)

    uncertainty = build_uncertainty_summary()
    uncertainty.to_csv(TABLES / "final_metric_uncertainty_summary.csv", index=False)

    cnc_perm = build_cnc_permutation_importance()
    cnc_perm.to_csv(TABLES / "final_cnc_permutation_importance.csv", index=False)

    top_features = build_top_feature_summary()
    top_features.to_csv(TABLES / "final_top_feature_summary.csv", index=False)

    validation_audit, excluded_cols = build_validation_and_leakage_audit()
    validation_audit.to_csv(TABLES / "final_validation_and_leakage_audit.csv", index=False)
    excluded_cols.to_csv(TABLES / "final_excluded_target_like_columns.csv", index=False)

    status = build_completion_status()
    status.to_csv(TABLES / "final_completion_status.csv", index=False)

    generate_figures(master, datasets, feature_families, sleep_stages, top_features)
    write_docs(master, datasets, feature_families, sleep_stages, uncertainty, validation_audit, status)

    index = pd.DataFrame(
        [
            {"output_type": "table", "path": str(p.relative_to(PROJECT_ROOT))}
            for p in sorted(TABLES.glob("*"))
        ]
        + [
            {"output_type": "figure", "path": str(p.relative_to(PROJECT_ROOT))}
            for p in sorted(FIGURES.glob("*"))
        ]
        + [
            {"output_type": "document", "path": str(p.relative_to(PROJECT_ROOT))}
            for p in sorted(DOCS.glob("*"))
        ]
    )
    index.to_csv(OUT_ROOT / "final_package_index.csv", index=False)
    print(json.dumps({"final_package": str(OUT_ROOT), "outputs": len(index)}, indent=2))


def read_csv(rel: str) -> pd.DataFrame:
    path = PROJECT_ROOT / rel
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(rel: str) -> dict:
    path = PROJECT_ROOT / rel
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def first_notna(*values):
    for value in values:
        if pd.notna(value):
            return value
    return np.nan


def build_master_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    eval_summary = read_csv("dreem_nrev/outputs/evaluation_outputs/tables/evaluation_summary.csv")
    for row in eval_summary.to_dict("records"):
        rows.append(
            {
                "priority": 10,
                "dataset": "Dreem",
                "experiment": f"{row['dataset']} baseline",
                "target": "Narcolepsy vs comparison",
                "feature_set": row["dataset"],
                "best_model": row["best_model"],
                "subjects": row["subjects"],
                "positive_subjects": row["narcolepsy_subjects"],
                "negative_subjects": row["comparison_subjects"],
                "features_used": row["features_after_qc"],
                "balanced_accuracy": row["best_balanced_accuracy_mean"],
                "macro_f1": row["best_macro_f1_mean"],
                "roc_auc": row["best_roc_auc_mean"],
                "sensitivity": np.nan,
                "specificity": np.nan,
                "interpretation_role": "main Dreem feature-set comparison",
                "notes": "Subject-level cross-validation on the main wearable EEG/report dataset.",
            }
        )

    tuned = read_csv("dreem_nrev/improvements/hyperparameter_tuning/outputs/tables/hyperparameter_tuning_results.csv")
    if not tuned.empty:
        best = tuned.sort_values(["balanced_accuracy", "macro_f1", "roc_auc"], ascending=False).iloc[0]
        rows.append(
            {
                "priority": 1,
                "dataset": "Dreem",
                "experiment": "hyperparameter tuned combined model",
                "target": "Narcolepsy vs comparison",
                "feature_set": "combined_report_h5",
                "best_model": best["model"],
                "subjects": 47,
                "positive_subjects": 11,
                "negative_subjects": 36,
                "features_used": best["features_after_qc"],
                "balanced_accuracy": best["balanced_accuracy"],
                "macro_f1": best["macro_f1"],
                "roc_auc": best["roc_auc"],
                "sensitivity": best.get("sensitivity_narcolepsy", np.nan),
                "specificity": best.get("specificity_comparison", np.nan),
                "interpretation_role": "best main analysis model",
                "notes": "Primary tuned result for the main Dreem dataset.",
            }
        )

    channels = read_csv("dreem_nrev/improvements/channel_strategy/outputs/tables/channel_strategy_results.csv")
    if not channels.empty:
        best = channels.sort_values(["balanced_accuracy", "macro_f1", "roc_auc"], ascending=False).iloc[0]
        rows.append(
            {
                "priority": 40,
                "dataset": "Dreem",
                "experiment": "channel strategy",
                "target": "Narcolepsy vs comparison",
                "feature_set": best["strategy"],
                "best_model": "logistic/random forest comparison",
                "subjects": best["subjects"],
                "positive_subjects": 11,
                "negative_subjects": 36,
                "features_used": best["features_after_qc"],
                "balanced_accuracy": best["balanced_accuracy"],
                "macro_f1": best["macro_f1"],
                "roc_auc": best["roc_auc"],
                "sensitivity": best.get("sensitivity_narcolepsy", np.nan),
                "specificity": best.get("specificity_comparison", np.nan),
                "interpretation_role": "wearable channel sensitivity analysis",
                "notes": f"Best channel/aggregation strategy was {best['strategy']}.",
            }
        )

    diag = read_csv("dreem_nrev/improvements/diagnosis_specific/outputs/tables/diagnosis_specific_summary.csv")
    for row in diag.to_dict("records"):
        rows.append(
            {
                "priority": 50,
                "dataset": "Dreem",
                "experiment": "diagnosis-specific exploratory model",
                "target": row["target"],
                "feature_set": "combined_report_h5",
                "best_model": row["model"],
                "subjects": row["positive_subjects"] + row["other_subjects"],
                "positive_subjects": row["positive_subjects"],
                "negative_subjects": row["other_subjects"],
                "features_used": row["features_after_qc"],
                "balanced_accuracy": row["balanced_accuracy"],
                "macro_f1": row["macro_f1"],
                "roc_auc": row["roc_auc"],
                "sensitivity": first_notna(row.get("sensitivity_NT1", np.nan), row.get("sensitivity_NT2", np.nan)),
                "specificity": row["specificity_all_other"],
                "interpretation_role": "exploratory subtype analysis",
                "notes": "Small subtype sample; interpret cautiously.",
            }
        )

    for stage in load_stage_level_summaries():
        target_label = stage["target"].replace("_", " ")
        rows.append(
            {
                "priority": 45 if stage["target"] == "narcolepsy_vs_other" else 55,
                "dataset": "Dreem",
                "experiment": "stage-level H5 EEG aggregation",
                "target": target_label,
                "feature_set": "H5 EEG features aggregated within Wake/N1/N2/N3/REM",
                "best_model": "random_forest",
                "subjects": stage["labelled_subjects"],
                "positive_subjects": stage["narcolepsy_subjects"],
                "negative_subjects": stage["comparison_subjects"],
                "features_used": stage["features_after_qc"],
                "balanced_accuracy": stage["balanced_accuracy"],
                "macro_f1": stage["macro_f1"],
                "roc_auc": stage["roc_auc"],
                "sensitivity": stage["sensitivity"],
                "specificity": stage["specificity"],
                "interpretation_role": "sleep-stage-aware feature experiment",
                "notes": "Uses 30-second H5 epochs summarised separately by scored sleep stage; subtype rows are exploratory.",
            }
        )

    advanced_summary = read_csv("dreem_nrev/improvements/advanced_h5_features/model_outputs/tables/dreem_advanced_model_summary.csv")
    if not advanced_summary.empty:
        best = advanced_summary.iloc[0]
        advanced_metrics = read_csv("dreem_nrev/improvements/advanced_h5_features/model_outputs/tables/dreem_advanced_model_metrics.csv")
        best_metric = advanced_metrics[advanced_metrics["model"] == best["best_model"]].iloc[0] if not advanced_metrics.empty else {}
        rows.append(
            {
                "priority": 46,
                "dataset": "Dreem",
                "experiment": "advanced H5 EEG feature experiment",
                "target": "Narcolepsy vs comparison",
                "feature_set": "H5 entropy, spindle, slow-wave and spectral/time features",
                "best_model": best["best_model"],
                "subjects": best["subjects"],
                "positive_subjects": best["narcolepsy_subjects"],
                "negative_subjects": best["comparison_subjects"],
                "features_used": best["features_after_qc"],
                "balanced_accuracy": best["best_balanced_accuracy"],
                "macro_f1": best["best_macro_f1"],
                "roc_auc": best["best_roc_auc"],
                "sensitivity": best_metric.get("sensitivity", np.nan),
                "specificity": best_metric.get("specificity", np.nan),
                "interpretation_role": "advanced EEG feature ablation/extension",
                "notes": "Full-cohort advanced H5 extraction completed; performance did not improve over simpler H5/combined features.",
            }
        )

    cnc = read_csv("cnc/outputs/evaluation_outputs/tables/cnc_edf_model_metrics.csv")
    if not cnc.empty:
        best = cnc.sort_values(["balanced_accuracy", "macro_f1", "roc_auc"], ascending=False).iloc[0]
        rows.append(
            {
                "priority": 2,
                "dataset": "CNC",
                "experiment": "external PSG EDF narcolepsy/control",
                "target": "T1 narcolepsy vs non-narcolepsy control",
                "feature_set": "EDF spectral/time/sleep-architecture",
                "best_model": best["model"],
                "subjects": best["n_subjects"],
                "positive_subjects": best["n_t1_narcolepsy"],
                "negative_subjects": best["n_controls"],
                "features_used": best["n_features_before_qc"],
                "balanced_accuracy": best["balanced_accuracy"],
                "macro_f1": best["macro_f1"],
                "roc_auc": best["roc_auc"],
                "sensitivity": best["sensitivity_t1_narcolepsy"],
                "specificity": best["specificity_control"],
                "interpretation_role": "external narcolepsy/control validation-style experiment",
                "notes": "PSG EDF cohort; not wearable Dreem, but directly narcolepsy/control.",
            }
        )

    simons = read_csv("simons_ssp/outputs/comparison_outputs/tables/narcolepsy_vs_simons_report_model_metrics.csv")
    if not simons.empty:
        best = simons.sort_values(["balanced_accuracy", "macro_f1", "roc_auc"], ascending=False).iloc[0]
        rows.append(
            {
                "priority": 60,
                "dataset": "NRev + SSP",
                "experiment": "same-device external comparison",
                "target": "NRev narcolepsy vs SSP healthy/control",
                "feature_set": "common report features",
                "best_model": best["model"],
                "subjects": 106,
                "positive_subjects": 11,
                "negative_subjects": 95,
                "features_used": best["features_after_qc"],
                "balanced_accuracy": best["balanced_accuracy"],
                "macro_f1": best["macro_f1"],
                "roc_auc": best["roc_auc"],
                "sensitivity": best["sensitivity_narcolepsy"],
                "specificity": best["specificity_external_control"],
                "interpretation_role": "domain-shift sensitivity analysis",
                "notes": "Original common-feature result was audited with a cleaned sleep-architecture feature set.",
            }
        )

    phys = read_csv("physionet_sleep_edf/outputs/model_outputs/model_metrics.csv")
    if not phys.empty:
        best = phys.sort_values(["balanced_accuracy", "macro_f1"], ascending=False).iloc[0]
        rows.append(
            {
                "priority": 80,
                "dataset": "PhysioNet Sleep-EDF",
                "experiment": "reference sleep-stage baseline",
                "target": "sleep-stage classification/reference",
                "feature_set": "EDF spectral/time features",
                "best_model": best["model"],
                "subjects": np.nan,
                "positive_subjects": np.nan,
                "negative_subjects": np.nan,
                "features_used": np.nan,
                "balanced_accuracy": best["balanced_accuracy"],
                "macro_f1": best["macro_f1"],
                "roc_auc": np.nan,
                "sensitivity": np.nan,
                "specificity": np.nan,
                "interpretation_role": "method development/reference dataset",
                "notes": "Used to validate EDF loading, epoching, and feature extraction workflow.",
            }
        )

    return pd.DataFrame(rows).sort_values(["priority", "dataset", "experiment"]).drop(columns=["priority"])


def build_detailed_model_metrics() -> pd.DataFrame:
    rows = []
    dreem = read_csv("dreem_nrev/outputs/model_outputs/all_model_metrics.csv")
    for row in dreem.to_dict("records"):
        rows.append(normalize_metric_row("Dreem", row["dataset"], "Narcolepsy vs comparison", row))

    cnc = read_csv("cnc/outputs/evaluation_outputs/tables/cnc_edf_model_metrics.csv")
    for row in cnc.to_dict("records"):
        rows.append(
            {
                "dataset": "CNC",
                "experiment": "cnc_edf",
                "target": "T1 narcolepsy vs control",
                "model": row["model"],
                "balanced_accuracy_mean": row["balanced_accuracy"],
                "balanced_accuracy_std": np.nan,
                "macro_f1_mean": row["macro_f1"],
                "macro_f1_std": np.nan,
                "roc_auc_mean": row["roc_auc"],
                "roc_auc_std": np.nan,
                "sensitivity_mean": row["sensitivity_t1_narcolepsy"],
                "sensitivity_std": np.nan,
                "specificity_mean": row["specificity_control"],
                "specificity_std": np.nan,
            }
        )

    for stage in load_stage_level_summaries():
        rows.append(
            {
                "dataset": "Dreem",
                "experiment": "stage_level_h5",
                "target": stage["target"].replace("_", " "),
                "model": "random_forest",
                "balanced_accuracy_mean": stage["balanced_accuracy"],
                "balanced_accuracy_std": np.nan,
                "macro_f1_mean": stage["macro_f1"],
                "macro_f1_std": np.nan,
                "roc_auc_mean": stage["roc_auc"],
                "roc_auc_std": np.nan,
                "sensitivity_mean": stage["sensitivity"],
                "sensitivity_std": np.nan,
                "specificity_mean": stage["specificity"],
                "specificity_std": np.nan,
            }
        )

    advanced = read_csv("dreem_nrev/improvements/advanced_h5_features/model_outputs/tables/dreem_advanced_model_metrics.csv")
    for row in advanced.to_dict("records"):
        rows.append(
            {
                "dataset": "Dreem",
                "experiment": "advanced_h5_features",
                "target": "Narcolepsy vs comparison",
                "model": row["model"],
                "balanced_accuracy_mean": row["balanced_accuracy"],
                "balanced_accuracy_std": np.nan,
                "macro_f1_mean": row["macro_f1"],
                "macro_f1_std": np.nan,
                "roc_auc_mean": row["roc_auc"],
                "roc_auc_std": np.nan,
                "sensitivity_mean": row["sensitivity"],
                "sensitivity_std": np.nan,
                "specificity_mean": row["specificity"],
                "specificity_std": np.nan,
            }
        )

    phys = read_csv("physionet_sleep_edf/outputs/model_outputs/model_metrics.csv")
    for row in phys.to_dict("records"):
        rows.append(
            {
                "dataset": "PhysioNet",
                "experiment": "sleep_edf_reference",
                "target": "sleep-stage classification/reference",
                "model": row["model"],
                "balanced_accuracy_mean": row["balanced_accuracy"],
                "balanced_accuracy_std": np.nan,
                "macro_f1_mean": row["macro_f1"],
                "macro_f1_std": np.nan,
                "roc_auc_mean": np.nan,
                "roc_auc_std": np.nan,
                "sensitivity_mean": np.nan,
                "sensitivity_std": np.nan,
                "specificity_mean": np.nan,
                "specificity_std": np.nan,
            }
        )
    return pd.DataFrame(rows)


def load_stage_level_summaries() -> list[dict]:
    roots = [
        "dreem_nrev/improvements/stage_level_features/outputs/dreem_stage_level_outputs/dreem_stage_level_run_summary.json",
        "dreem_nrev/improvements/stage_level_features/outputs/dreem_stage_level_outputs_nt1/dreem_stage_level_run_summary.json",
        "dreem_nrev/improvements/stage_level_features/outputs/dreem_stage_level_outputs_nt2/dreem_stage_level_run_summary.json",
        "dreem_nrev/improvements/stage_level_features/outputs/narcolepsy_vs_other/dreem_stage_level_run_summary.json",
        "dreem_nrev/improvements/stage_level_features/outputs/nt1_vs_other/dreem_stage_level_run_summary.json",
        "dreem_nrev/improvements/stage_level_features/outputs/nt2_vs_other/dreem_stage_level_run_summary.json",
    ]
    seen = set()
    summaries = []
    for rel in roots:
        data = read_json(rel)
        if not data:
            continue
        target = data.get("target", rel)
        if target in seen:
            continue
        seen.add(target)
        summaries.append(data)
    order = {"narcolepsy_vs_other": 0, "nt1_vs_other": 1, "nt2_vs_other": 2}
    return sorted(summaries, key=lambda item: order.get(item.get("target"), 99))


def normalize_metric_row(dataset: str, experiment: str, target: str, row: dict) -> dict:
    return {
        "dataset": dataset,
        "experiment": experiment,
        "target": target,
        "model": row["model"],
        "balanced_accuracy_mean": row.get("balanced_accuracy_mean", np.nan),
        "balanced_accuracy_std": row.get("balanced_accuracy_std", np.nan),
        "macro_f1_mean": row.get("macro_f1_mean", np.nan),
        "macro_f1_std": row.get("macro_f1_std", np.nan),
        "roc_auc_mean": row.get("roc_auc_mean", np.nan),
        "roc_auc_std": row.get("roc_auc_std", np.nan),
        "sensitivity_mean": row.get("sensitivity_narcolepsy_mean", np.nan),
        "sensitivity_std": row.get("sensitivity_narcolepsy_std", np.nan),
        "specificity_mean": row.get("specificity_not_positive_mean", row.get("specificity_comparison_mean", np.nan)),
        "specificity_std": row.get("specificity_not_positive_std", row.get("specificity_comparison_std", np.nan)),
    }


def build_dataset_comparison() -> pd.DataFrame:
    rows = [
        {
            "dataset": "Dreem/narcolepsy",
            "role": "main analysis dataset",
            "format": "H5 wearable EEG plus report CSV features",
            "subjects_processed": get_summary_value("dreem_nrev/outputs/evaluation_outputs/tables/evaluation_summary.csv", "subjects", 47),
            "records_processed": get_summary_value("dreem_nrev/outputs/h5_subject_features/dreem_h5_run_summary.csv", "records_processed", 363),
            "positive_group": "narcolepsy",
            "negative_or_comparison_group": "hypersomnia/comparison groups",
            "main_limitations": "Small number of narcolepsy subjects; wearable channel naming differs from PSG cohorts.",
        },
        {
            "dataset": "Dreem stage-level H5",
            "role": "sleep-stage-aware Dreem feature experiment",
            "format": "H5 wearable EEG epoch parquet features aggregated within Wake/N1/N2/N3/REM",
            "subjects_processed": get_stage_summary_value("narcolepsy_vs_other", "labelled_subjects", 49),
            "records_processed": get_stage_summary_value("narcolepsy_vs_other", "epoch_files", 363),
            "positive_group": "narcolepsy, NT1, or NT2 depending on target",
            "negative_or_comparison_group": "all other labelled Dreem participants",
            "main_limitations": "High-dimensional relative to sample size; NT2 has only 3 positive subjects.",
        },
        {
            "dataset": "Dreem advanced H5",
            "role": "advanced EEG feature experiment",
            "format": "H5 wearable EEG with entropy, spindle, slow-wave, spectral and time-domain features",
            "subjects_processed": get_summary_value("dreem_nrev/improvements/advanced_h5_features/model_outputs/tables/dreem_advanced_model_summary.csv", "subjects", 47),
            "records_processed": get_summary_value("dreem_nrev/improvements/advanced_h5_features/outputs/dreem_h5_advanced_run_summary.csv", "records_processed", 363),
            "positive_group": "narcolepsy",
            "negative_or_comparison_group": "hypersomnia/comparison groups",
            "main_limitations": "Very high-dimensional relative to sample size; advanced features did not improve classification performance.",
        },
        {
            "dataset": "CNC",
            "role": "external narcolepsy/control PSG experiment",
            "format": "EDF PSG plus CSV sleep-stage annotations",
            "subjects_processed": 78,
            "records_processed": 78,
            "positive_group": "T1 narcolepsy",
            "negative_or_comparison_group": "non-narcolepsy control",
            "main_limitations": "Only 23 of 56 official controls had matched EDF+CSV files; CHC2 files lacked metadata mapping.",
        },
        {
            "dataset": "Simons",
            "role": "same-device external comparison",
            "format": "Dreem report CSV and EDF",
            "subjects_processed": 95,
            "records_processed": 100,
            "positive_group": "not used as positive class",
            "negative_or_comparison_group": "ASD-negative controls",
            "main_limitations": "Different population and study context; useful for sensitivity, not primary narcolepsy evidence.",
        },
        {
            "dataset": "PhysioNet Sleep-EDF",
            "role": "public reference dataset",
            "format": "EDF PSG plus hypnogram",
            "subjects_processed": np.nan,
            "records_processed": np.nan,
            "positive_group": "sleep-stage labels",
            "negative_or_comparison_group": "sleep-stage labels",
            "main_limitations": "Practice/reference dataset, not a narcolepsy diagnostic cohort.",
        },
    ]
    return pd.DataFrame(rows)


def get_summary_value(rel: str, col: str, default):
    df = read_csv(rel)
    if df.empty or col not in df.columns:
        return default
    return df[col].iloc[0]


def get_stage_summary_value(target: str, key: str, default):
    for summary in load_stage_level_summaries():
        if summary.get("target") == target:
            return summary.get(key, default)
    return default


def build_feature_family_audit() -> pd.DataFrame:
    rows = []
    family = read_csv("dreem_nrev/outputs/feature_outputs/feature_family_summary.csv")
    if not family.empty:
        for row in family.to_dict("records"):
            rows.append(
                {
                    "dataset": "Dreem combined",
                    "source": row.get("source", "combined"),
                    "feature_family": row.get("family", row.get("feature_family", "unknown")),
                    "feature_count": row.get("feature_count", np.nan),
                }
            )

    for dataset, rel in [
        ("CNC EDF", "cnc/outputs/edf_outputs/cnc_edf_subject_features_with_official_metadata.csv"),
        ("Simons EDF", "simons_ssp/outputs/edf_outputs_50x2/simons_edf_subject_features.csv"),
        ("Dreem stage-level H5", "dreem_nrev/improvements/stage_level_features/outputs/dreem_stage_level_outputs/tables/dreem_stage_level_subject_features.csv"),
        ("Dreem advanced H5", "dreem_nrev/improvements/advanced_h5_features/outputs/dreem_h5_advanced_subject_features.csv"),
    ]:
        df = read_csv(rel)
        if not df.empty:
            for fam, count in count_feature_families(df).items():
                rows.append({"dataset": dataset, "source": "engineered_features", "feature_family": fam, "feature_count": count})

    return pd.DataFrame(rows, columns=["dataset", "source", "feature_family", "feature_count"])


def count_feature_families(df: pd.DataFrame) -> dict[str, int]:
    meta_terms = ["subject", "recording", "diagnosis", "label", "metadata", "file_id", "nrevid", "sex", "age", "asd", "dq0602"]
    counts = {
        "sleep_architecture": 0,
        "spectral_power": 0,
        "relative_power_or_ratio": 0,
        "entropy_or_nonlinear": 0,
        "spindle_slowwave_events": 0,
        "time_domain": 0,
        "other_numeric": 0,
    }
    for col in df.select_dtypes(include=[np.number]).columns:
        lower = col.lower()
        if any(term in lower for term in meta_terms):
            continue
        family_text = lower
        if "__" in col and col.split("__", 1)[0] in ["Wake", "N1", "N2", "N3", "REM"]:
            family_text = col.split("__", 1)[1].lower()
        if (
            any(col == f"{stage}_epochs" or col == f"{stage}_percentage" for stage in ["Wake", "N1", "N2", "N3", "REM"])
            or "sleep_efficiency" in lower
            or "n_epochs" in lower
        ):
            counts["sleep_architecture"] += 1
        elif "entropy" in family_text:
            counts["entropy_or_nonlinear"] += 1
        elif "spindle" in family_text or "slowwave" in family_text:
            counts["spindle_slowwave_events"] += 1
        elif "relative_power" in family_text or "ratio" in family_text:
            counts["relative_power_or_ratio"] += 1
        elif "power" in family_text:
            counts["spectral_power"] += 1
        elif any(term in family_text for term in ["mean", "std", "iqr", "skew", "kurtosis", "zero_crossings", "hjorth"]):
            counts["time_domain"] += 1
        else:
            counts["other_numeric"] += 1
    return counts


def build_sleep_stage_appendix() -> pd.DataFrame:
    rows = []

    dreem = read_csv("dreem_nrev/outputs/eda_outputs/tables/dreem_h5_sleep_architecture_by_target.csv")
    if not dreem.empty:
        rows.extend(flatten_stage_table("Dreem H5", dreem))

    stage_counts = read_csv("dreem_nrev/improvements/stage_level_features/outputs/dreem_stage_level_outputs/tables/dreem_stage_epoch_counts.csv")
    if not stage_counts.empty:
        total = stage_counts.groupby("label")["epochs"].sum()
        denom = total.sum()
        for stage, epochs in total.items():
            rows.append(
                {
                    "dataset": "Dreem H5 stage-level experiment",
                    "group": "overall",
                    "stage_or_metric": stage,
                    "value": int(epochs),
                    "percentage": float(epochs / denom) if denom else np.nan,
                }
            )

    cnc = read_csv("cnc/outputs/edf_outputs/cnc_edf_record_features.csv")
    if not cnc.empty:
        rows.extend(record_stage_rows("CNC EDF", cnc, "label"))

    simons = read_csv("simons_ssp/outputs/edf_outputs_50x2/simons_edf_record_features.csv")
    if not simons.empty:
        rows.extend(record_stage_rows("Simons EDF controls", simons, "external_group" if "external_group" in simons.columns else None))

    phys = read_csv("physionet_sleep_edf/outputs/eda_outputs/tables/physionet_sleep_stage_counts.csv")
    if not phys.empty:
        rows.extend(flatten_stage_table("PhysioNet", phys))

    return pd.DataFrame(rows)


def flatten_stage_table(dataset: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    group_cols = [c for c in df.columns if c.lower() in {"target", "label", "diagnosis", "dataset", "group", "sleep_stage"}]
    for _, r in df.iterrows():
        group = "overall"
        if group_cols:
            group = " | ".join(str(r[c]) for c in group_cols if c in r.index)
        for col, val in r.items():
            if col in group_cols:
                continue
            if any(stage.lower() in col.lower() for stage in ["wake", "n1", "n2", "n3", "rem"]):
                rows.append({"dataset": dataset, "group": group, "stage_or_metric": col, "value": val})
    return rows


def record_stage_rows(dataset: str, df: pd.DataFrame, group_col: str | None) -> list[dict]:
    rows = []
    stage_cols = [c for c in df.columns if c in [f"{s}_epochs" for s in ["Wake", "N1", "N2", "N3", "REM"]]]
    if not stage_cols:
        return rows
    groups = [(None, df)] if not group_col or group_col not in df.columns else list(df.groupby(group_col))
    for group, part in groups:
        total = part[stage_cols].sum()
        denom = total.sum()
        for col, val in total.items():
            rows.append(
                {
                    "dataset": dataset,
                    "group": "overall" if group is None else group,
                    "stage_or_metric": col.replace("_epochs", ""),
                    "value": int(val),
                    "percentage": float(val / denom) if denom else np.nan,
                }
            )
    return rows


def build_uncertainty_summary() -> pd.DataFrame:
    rows = []
    fold = read_csv("dreem_nrev/outputs/model_outputs/all_fold_metrics.csv")
    if not fold.empty:
        best_lookup = read_csv("dreem_nrev/outputs/evaluation_outputs/tables/evaluation_summary.csv")
        for best in best_lookup.to_dict("records"):
            part = fold[(fold["dataset"] == best["dataset"]) & (fold["model"] == best["best_model"])]
            rows.extend(fold_uncertainty_rows("Dreem", best["dataset"], best["best_model"], part))

    cnc_pred = read_csv("cnc/outputs/evaluation_outputs/tables/cnc_edf_cross_validated_predictions.csv")
    cnc_metrics = read_csv("cnc/outputs/evaluation_outputs/tables/cnc_edf_model_metrics.csv")
    if not cnc_pred.empty and not cnc_metrics.empty:
        best_model = cnc_metrics.sort_values(["balanced_accuracy", "macro_f1", "roc_auc"], ascending=False).iloc[0]["model"]
        part = cnc_pred[cnc_pred["model"] == best_model]
        rows.extend(bootstrap_uncertainty_rows("CNC", "cnc_edf", best_model, part))
    return pd.DataFrame(rows)


def fold_uncertainty_rows(dataset: str, experiment: str, model: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    for metric in ["balanced_accuracy", "macro_f1", "roc_auc", "sensitivity_narcolepsy", "specificity_comparison"]:
        if metric not in df.columns:
            continue
        values = df[metric].dropna()
        if values.empty:
            continue
        rows.append(
            {
                "dataset": dataset,
                "experiment": experiment,
                "model": model,
                "metric": metric,
                "estimate": values.mean(),
                "std": values.std(ddof=1),
                "lower_95": values.quantile(0.025),
                "upper_95": values.quantile(0.975),
                "method": "fold distribution",
                "n": len(values),
            }
        )
    return rows


def bootstrap_uncertainty_rows(dataset: str, experiment: str, model: str, df: pd.DataFrame, n_boot: int = 2000) -> list[dict]:
    rng = np.random.default_rng(42)
    y = df["true_label"].astype(int).to_numpy()
    pred = df["predicted_label"].astype(int).to_numpy()
    score = df["predicted_probability_t1"].astype(float).to_numpy()
    metrics = {"balanced_accuracy": [], "macro_f1": [], "roc_auc": [], "sensitivity": [], "specificity": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        metrics["balanced_accuracy"].append(balanced_accuracy_score(y[idx], pred[idx]))
        metrics["macro_f1"].append(f1_score(y[idx], pred[idx], average="macro"))
        metrics["roc_auc"].append(roc_auc_score(y[idx], score[idx]))
        metrics["sensitivity"].append(recall_score(y[idx], pred[idx], pos_label=1))
        metrics["specificity"].append(recall_score(y[idx], pred[idx], pos_label=0))
    return [
        {
            "dataset": dataset,
            "experiment": experiment,
            "model": model,
            "metric": metric,
            "estimate": np.mean(values),
            "std": np.std(values, ddof=1),
            "lower_95": np.quantile(values, 0.025),
            "upper_95": np.quantile(values, 0.975),
            "method": "subject bootstrap over cross-validated predictions",
            "n": len(values),
        }
        for metric, values in metrics.items()
        if values
    ]


def build_top_feature_summary() -> pd.DataFrame:
    sources = [
        ("Dreem report only", "dreem_nrev/outputs/evaluation_outputs/tables/report_only_feature_importance.csv"),
        ("Dreem H5 only", "dreem_nrev/outputs/evaluation_outputs/tables/h5_only_feature_importance.csv"),
        ("Dreem combined", "dreem_nrev/outputs/evaluation_outputs/tables/combined_report_h5_feature_importance.csv"),
        ("Dreem combined permutation", "outputs/final_diagnostic_analysis/tables/combined_report_h5_permutation_importance.csv"),
        ("Dreem advanced H5", "dreem_nrev/improvements/advanced_h5_features/model_outputs/tables/dreem_advanced_feature_importance.csv"),
        ("CNC EDF permutation", "final_results/outputs/tables/final_cnc_permutation_importance.csv"),
        ("CNC EDF", "cnc/outputs/evaluation_outputs/tables/cnc_edf_feature_importance.csv"),
    ]
    rows = []
    for source, rel in sources:
        df = read_csv(rel)
        if df.empty:
            continue
        feature_col = "feature" if "feature" in df.columns else df.columns[0]
        importance_col = None
        for candidate in ["importance", "mean_importance", "permutation_importance_mean", "importance_mean"]:
            if candidate in df.columns:
                importance_col = candidate
                break
        if importance_col is None:
            numeric = df.select_dtypes(include=[np.number]).columns
            importance_col = numeric[0] if len(numeric) else None
        if importance_col is None:
            continue
        part = df[[feature_col, importance_col]].rename(columns={feature_col: "feature", importance_col: "importance"})
        part = part.sort_values("importance", ascending=False)
        positive = part[part["importance"] > 0]
        part = (positive if not positive.empty else part).head(25)
        part.insert(0, "source", source)
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["source", "feature", "importance"])


def build_cnc_permutation_importance() -> pd.DataFrame:
    df = read_csv("cnc/outputs/edf_outputs/cnc_edf_subject_features_with_official_metadata.csv")
    if df.empty or "diagnosis_binary" not in df.columns:
        return pd.DataFrame(columns=["feature", "importance", "importance_std"])
    y = df["diagnosis_binary"].astype(int)
    feature_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if not is_leakage_or_metadata_column(c)
    ]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    missingness = X.isna().mean()
    feature_cols = [c for c in feature_cols if missingness[c] < 0.4 and X[c].nunique(dropna=True) > 1]
    X = X[feature_cols]
    if not feature_cols:
        return pd.DataFrame(columns=["feature", "importance", "importance_std"])

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=150,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(X, y)
    internal = model.named_steps["model"].feature_importances_
    ranked = pd.Series(internal, index=feature_cols).sort_values(ascending=False)
    selected_cols = ranked.head(min(150, len(ranked))).index.tolist()
    X_selected = X[selected_cols]
    model.fit(X_selected, y)
    result = permutation_importance(
        model,
        X_selected,
        y,
        scoring="balanced_accuracy",
        n_repeats=6,
        random_state=42,
        n_jobs=1,
    )
    out = pd.DataFrame(
        {
            "feature": selected_cols,
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
            "candidate_selection": "top_150_by_random_forest_internal_importance",
        }
    )
    return out.sort_values("importance", ascending=False)


def is_leakage_or_metadata_column(col: str) -> bool:
    lower = col.lower()
    blocked_exact = {"diagnosis_binary", "dq0602"}
    if lower in blocked_exact:
        return True
    return any(
        term in lower
        for term in ["diagnosis", "label", "target", "subject_id", "recording_id", "metadata", "file_id", "nrevid"]
    )


def build_validation_and_leakage_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = [
        {
            "area": "Dreem main models",
            "status": "implemented",
            "evidence": "Subject-level cross-validation and fold metrics are stored in dreem_nrev/outputs/model_outputs/all_fold_metrics.csv.",
            "risk": "Small positive class; report uncertainty and avoid overclaiming.",
        },
        {
            "area": "CNC leakage check",
            "status": "fixed",
            "evidence": "Initial target-derived diagnosis_binary aggregate features were detected and excluded in run_cnc_model_evaluation.py.",
            "risk": "CNC labels remain class-imbalanced after matched-file filtering.",
        },
        {
            "area": "Simons external control",
            "status": "interpreted cautiously",
            "evidence": "Perfect report-feature separation is marked as domain-shift sensitivity evidence, not primary diagnostic performance.",
            "risk": "Dataset-source artefacts may dominate.",
        },
        {
            "area": "Dataset merging",
            "status": "not used as main claim",
            "evidence": "Datasets differ by device, cohort, channel montage and labels; treated as separate experiments.",
            "risk": "Naive merging could learn dataset identity.",
        },
    ]
    excluded = []
    for name, rel in [
        ("CNC EDF", "cnc/outputs/edf_outputs/cnc_edf_subject_features_with_official_metadata.csv"),
        ("Dreem combined", "dreem_nrev/outputs/feature_outputs/combined_subject_features.csv"),
    ]:
        df = read_csv(rel)
        if df.empty:
            continue
        for col in df.columns:
            lower = col.lower()
            if any(term in lower for term in ["diagnosis", "label", "target", "subject_id", "recording_id", "metadata", "file_id", "nrevid"]):
                excluded.append({"dataset": name, "column": col, "reason": "identifier, target, label, or metadata leakage risk"})
    return pd.DataFrame(audit), pd.DataFrame(excluded)


def build_completion_status() -> pd.DataFrame:
    checks = [
        {
            "area": "Main Dreem wearable EEG/report modelling",
            "status": "complete",
            "evidence": "Dreem report-only, H5-only and combined models are included in final_master_results_table.csv.",
            "next_action": "Use as the main Narcolepsy Revolution result.",
        },
        {
            "area": "Dreem hyperparameter tuning",
            "status": "complete",
            "evidence": "Best tuned random forest result is included as the top main Dreem model.",
            "next_action": "Report as the primary tuned model.",
        },
        {
            "area": "Dreem channel strategy",
            "status": "complete",
            "evidence": "Best single-channel/aggregation result is included in the final master table.",
            "next_action": "Discuss as wearable-channel sensitivity analysis.",
        },
        {
            "area": "Dreem diagnosis-specific NT1/NT2",
            "status": "complete but exploratory",
            "evidence": "NT1-vs-all and NT2-vs-all outputs are included; NT2 has only 3 positive subjects.",
            "next_action": "Use cautiously in discussion, not as a central claim.",
        },
        {
            "area": "Dreem stage-level epoch aggregation",
            "status": "complete",
            "evidence": "H5 epoch-level features were aggregated into stage-level narcolepsy, NT1 and NT2 experiment outputs.",
            "next_action": "Use as a sleep-stage-aware feature experiment; present NT2 cautiously because there are only 3 positives.",
        },
        {
            "area": "Dreem advanced H5 EEG features",
            "status": "complete",
            "evidence": "Full advanced H5 extraction processed 363 records and 49 subjects; advanced model outputs are included.",
            "next_action": "Negative/extension experiment: entropy, spindle and slow-wave features were implemented but did not outperform simpler H5 or combined features.",
        },
        {
            "area": "CNC EDF narcolepsy/control experiment",
            "status": "complete",
            "evidence": "78 EDF+CSV records processed with zero failures; corrected non-leaky random forest result is included.",
            "next_action": "Use as an external PSG narcolepsy/control experiment.",
        },
        {
            "area": "Simons external-control experiment",
            "status": "complete",
            "evidence": "Report and EDF outputs are present; report comparison is included as sensitivity analysis.",
            "next_action": "Discuss as external-control/domain-shift experiment only.",
        },
        {
            "area": "PhysioNet reference workflow",
            "status": "complete",
            "evidence": "Sleep-EDF baseline metrics and EDA outputs are present.",
            "next_action": "Use to explain development and validation of EDF processing workflow.",
        },
        {
            "area": "Permutation importance",
            "status": "complete for Dreem; feature importance complete for CNC",
            "evidence": "Dreem permutation importance exists in final_diagnostic_analysis; CNC random forest importance is included.",
            "next_action": "Optional: add CNC permutation importance if more interpretability is needed.",
        },
        {
            "area": "XGBoost benchmark",
            "status": "optional not run",
            "evidence": "xgboost is not installed in the local environment. Gradient boosting is already included as a fallback benchmark.",
            "next_action": "Optional only: install xgboost and rerun final model comparison.",
        },
    ]
    return pd.DataFrame(checks)


def generate_figures(master: pd.DataFrame, datasets: pd.DataFrame, feature_families: pd.DataFrame, sleep_stages: pd.DataFrame, top_features: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_master_metric(master, "balanced_accuracy", "Balanced Accuracy", "final_model_comparison_balanced_accuracy.png")
    plot_master_metric(master, "roc_auc", "ROC-AUC", "final_model_comparison_roc_auc.png")
    plot_dataset_subjects(datasets)
    plot_feature_families(feature_families)
    plot_top_features(top_features)
    plot_sleep_stages(sleep_stages)
    plot_confusion_matrix("Dreem combined", "dreem_nrev/outputs/evaluation_outputs/tables/combined_report_h5_confusion_matrix.csv", "final_dreem_combined_confusion_matrix.png")
    plot_cnc_confusion_matrix()
    plot_pipeline_diagram()


def plot_master_metric(master: pd.DataFrame, metric: str, label: str, filename: str) -> None:
    df = master.dropna(subset=[metric]).copy()
    df["name"] = df.apply(short_result_label, axis=1)
    df = df.sort_values(metric)
    fig, ax = plt.subplots(figsize=(12, max(5, len(df) * 0.55)))
    colors = ["#4C78A8" if "Dreem" in d else "#59A14F" if "CNC" in d else "#F28E2B" if "Simons" in d else "#9C755F" for d in df["dataset"]]
    ax.barh(df["name"], df[metric], color=colors)
    ax.set_xlabel(label)
    ax.set_xlim(0, 1.05)
    ax.set_title(f"Final Experiment Comparison: {label}")
    for i, v in enumerate(df[metric]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=200)
    plt.close(fig)


def short_result_label(row: pd.Series) -> str:
    dataset = str(row["dataset"])
    experiment = str(row["experiment"])
    feature = str(row.get("feature_set", ""))
    if dataset == "Dreem" and "hyperparameter" in experiment:
        return "Dreem tuned\ncombined"
    if dataset == "Dreem" and "baseline" in experiment:
        return f"Dreem\n{feature}"
    if dataset == "Dreem" and "channel" in experiment:
        return "Dreem channel\nbest eeg4"
    if dataset == "Dreem" and "diagnosis-specific" in experiment:
        return f"Dreem subtype\n{row['target']}"
    if dataset == "Dreem" and "stage-level" in experiment:
        return f"Dreem stage\n{row['target']}"
    if dataset == "Dreem" and "advanced H5" in experiment:
        return "Dreem advanced\nH5 features"
    if dataset == "CNC":
        return "CNC EDF\nT1 vs control"
    if "Simons" in dataset:
        return "Dreem vs Simons\nsensitivity"
    if "PhysioNet" in dataset:
        return "PhysioNet\nreference"
    return f"{dataset}\n{experiment}"


def plot_dataset_subjects(datasets: pd.DataFrame) -> None:
    df = datasets.dropna(subset=["subjects_processed"]).copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(df["dataset"], df["subjects_processed"], color=["#4C78A8", "#59A14F", "#F28E2B", "#9C755F"][: len(df)])
    ax.set_ylabel("Subjects processed")
    ax.set_title("Dataset Roles and Processed Subject Counts")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "final_dataset_subject_counts.png", dpi=200)
    plt.close(fig)


def plot_feature_families(feature_families: pd.DataFrame) -> None:
    if feature_families.empty:
        return
    df = feature_families.copy()
    fam_col = "feature_family" if "feature_family" in df.columns else df.columns[1]
    count_col = "feature_count" if "feature_count" in df.columns else [c for c in df.columns if "count" in c.lower()][0]
    pivot = df.pivot_table(index=fam_col, columns="dataset", values=count_col, aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Feature count")
    ax.set_xlabel("Feature family")
    ax.set_title("Engineered Feature Families Across Datasets")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURES / "final_feature_family_counts.png", dpi=200)
    plt.close(fig)


def plot_top_features(top_features: pd.DataFrame) -> None:
    if top_features.empty:
        return
    chosen_source = "CNC EDF" if "CNC EDF" in set(top_features["source"]) else top_features["source"].iloc[0]
    df = top_features[top_features["source"] == chosen_source].head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["feature"], df["importance"], color="#4C78A8")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top Features: {chosen_source}")
    fig.tight_layout()
    fig.savefig(FIGURES / "final_top_feature_importance.png", dpi=200)
    plt.close(fig)


def plot_sleep_stages(sleep_stages: pd.DataFrame) -> None:
    if sleep_stages.empty or "percentage" not in sleep_stages.columns:
        return
    df = sleep_stages.dropna(subset=["percentage"])
    df = df[df["stage_or_metric"].isin(["Wake", "N1", "N2", "N3", "REM"])]
    if df.empty:
        return
    pivot = df.pivot_table(index=["dataset", "group"], columns="stage_or_metric", values="percentage", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(11, max(4, len(pivot) * 0.45)))
    pivot[["Wake", "N1", "N2", "N3", "REM"]].plot(kind="barh", stacked=True, ax=ax)
    ax.set_xlabel("Proportion of labelled epochs")
    ax.set_title("Sleep-Stage Distribution by Dataset and Group")
    fig.tight_layout()
    fig.savefig(FIGURES / "final_sleep_stage_distribution.png", dpi=200)
    plt.close(fig)


def plot_confusion_matrix(title: str, rel: str, filename: str) -> None:
    df = read_csv(rel)
    if df.empty:
        return
    values = df.select_dtypes(include=[np.number]).to_numpy()
    if values.size != 4:
        values = values[:2, :2]
    values = values.reshape(2, 2)
    draw_confusion(values, title, filename)


def plot_cnc_confusion_matrix() -> None:
    metrics = read_csv("cnc/outputs/evaluation_outputs/tables/cnc_edf_model_metrics.csv")
    if metrics.empty:
        return
    best = metrics.sort_values(["balanced_accuracy", "macro_f1", "roc_auc"], ascending=False).iloc[0]
    values = np.array([[best["tn"], best["fp"]], [best["fn"], best["tp"]]])
    draw_confusion(values, "CNC EDF Random Forest", "final_cnc_confusion_matrix.png")


def draw_confusion(values: np.ndarray, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(values, cmap="Blues")
    ax.set_xticks([0, 1], ["Predicted negative", "Predicted positive"])
    ax.set_yticks([0, 1], ["Actual negative", "Actual positive"])
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(values[i, j]), ha="center", va="center", color="black", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=200)
    plt.close(fig)


def plot_pipeline_diagram() -> None:
    stages = [
        "Datasets\nDreem, CNC,\nSimons, PhysioNet",
        "Preprocessing\nread EDF/H5,\nfilter EEG",
        "30-second\nepoching",
        "Feature\nengineering",
        "Subject-level\naggregation",
        "ML models\nLR, SVM, RF,\nboosting",
        "Evaluation\nBA, F1, AUC,\nSe, Sp",
        "Interpretation\nimportance,\naudit",
    ]
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")
    x = np.linspace(0.05, 0.95, len(stages))
    for i, (xi, text) in enumerate(zip(x, stages)):
        ax.text(xi, 0.5, text, ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.35", fc="#EAF2F8", ec="#4C78A8"))
        if i < len(stages) - 1:
            ax.annotate("", xy=(x[i + 1] - 0.045, 0.5), xytext=(xi + 0.045, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    fig.tight_layout()
    fig.savefig(FIGURES / "final_methodology_pipeline.png", dpi=200)
    plt.close(fig)


def write_docs(master: pd.DataFrame, datasets: pd.DataFrame, feature_families: pd.DataFrame, sleep_stages: pd.DataFrame, uncertainty: pd.DataFrame, validation_audit: pd.DataFrame, status: pd.DataFrame) -> None:
    best_main = master[master["interpretation_role"] == "best main analysis model"].iloc[0]
    cnc = master[master["dataset"] == "CNC"].iloc[0]
    text = f"""# Final Implementation Summary

This folder consolidates the implementation results for the project before final reporting.

## Main Aim

The project investigates whether engineered EEG and sleep features can support machine-learning detection of narcolepsy and related hypersomnia disorders.

## Dataset Roles

{markdown_table(datasets)}

## Main Results

The main Dreem wearable/report model is the tuned combined model:

- Model: {best_main['best_model']}
- Balanced accuracy: {best_main['balanced_accuracy']:.3f}
- Macro F1: {best_main['macro_f1']:.3f}
- ROC-AUC: {best_main['roc_auc']:.3f}

The strongest external narcolepsy/control experiment is CNC EDF:

- Model: {cnc['best_model']}
- Balanced accuracy: {cnc['balanced_accuracy']:.3f}
- Macro F1: {cnc['macro_f1']:.3f}
- ROC-AUC: {cnc['roc_auc']:.3f}
- Sensitivity: {cnc['sensitivity']:.3f}
- Specificity: {cnc['specificity']:.3f}

## Important Interpretation

Dreem remains the main analysis dataset because it is the wearable EEG dataset aligned with the project title. CNC is a strong external PSG EDF experiment and should be discussed separately. Simons is a sensitivity experiment and should not be overclaimed because cohort and dataset-source effects are likely.

## Generated Outputs

The consolidated outputs contain:

- integrated result tables
- dataset comparison tables
- feature-family audit tables
- sleep-stage appendix tables
- uncertainty summaries
- validation and leakage audit files
- figures for model comparison, feature importance, cohort composition and validation checks

## Completion Status

{markdown_table(status)}
"""
    (DOCS / "implementation_summary.md").write_text(text, encoding="utf-8")

    audit_text = "# Validation And Leakage Audit\n\n" + markdown_table(validation_audit) + "\n"
    (DOCS / "validation_and_leakage_audit.md").write_text(audit_text, encoding="utf-8")

    readme = """# Final Dissertation Package

This directory is the final results consolidation area. It does not replace the original experiment folders; it collects the key results, figures, and audit material used for final analysis reporting.

Run:

```bash
python pipelines/final_package/01_build_final_results_package.py
```

Important outputs:

- `outputs/tables/final_master_results_table.csv`
- `outputs/tables/final_dataset_comparison_table.csv`
- `outputs/tables/final_metric_uncertainty_summary.csv`
- `outputs/tables/final_validation_and_leakage_audit.csv`
- `outputs/tables/final_completion_status.csv`
- `outputs/figures/final_methodology_pipeline.png`
- `outputs/figures/final_model_comparison_balanced_accuracy.png`
- `outputs/docs/implementation_summary.md`
"""
    (PROJECT_ROOT / "final_results" / "README.md").write_text(readme, encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    display = df.fillna("").astype(str)
    cols = list(display.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in display.iterrows():
        values = [row[c].replace("\n", " ").replace("|", "/") for c in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
