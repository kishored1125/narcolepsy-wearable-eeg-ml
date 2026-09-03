#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from diss_eeg.modeling import best_model_name, confusion_matrix_frame, evaluate_models, feature_importance_table
from diss_eeg.paths import FIGURE_DIR, ROOT_DIR, TABLE_DIR, ensure_output_dirs
from diss_eeg.physionet import discover_sleep_edf_pairs, extract_physionet_features, trim_wake_edges
from diss_eeg.plotting import save_barplot, save_confusion_matrix, save_feature_importance


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PhysioNet Sleep-EDF EEG feature and ML pipeline.")
    parser.add_argument("--physionet-dir", default=str(ROOT_DIR / "PhysioNet"))
    parser.add_argument("--max-recordings", type=int, default=8)
    parser.add_argument("--max-epochs-per-recording", type=int, default=None)
    parser.add_argument("--subset", choices=["sleep-cassette", "sleep-telemetry", "both"], default="sleep-cassette")
    args = parser.parse_args()

    ensure_output_dirs()
    subsets = ("sleep-cassette", "sleep-telemetry") if args.subset == "both" else (args.subset,)
    pairs = discover_sleep_edf_pairs(Path(args.physionet_dir), subsets=subsets)
    if args.max_recordings:
        pairs = pairs.head(args.max_recordings)
    pairs.to_csv(TABLE_DIR / "physionet_recording_pairs.csv", index=False)

    feature_parts = []
    failures = []
    for row in pairs.itertuples(index=False):
        try:
            df = extract_physionet_features(
                Path(row.psg_path),
                Path(row.hypnogram_path),
                subject_id=row.subject_id,
                recording_id=row.recording_id,
                max_epochs_per_recording=args.max_epochs_per_recording,
            )
            feature_parts.append(df)
            print(f"Processed {row.recording_id}: {len(df)} epochs")
        except Exception as exc:
            failures.append({"recording_id": row.recording_id, "error": str(exc)})
            print(f"FAILED {row.recording_id}: {exc}")

    if failures:
        pd.DataFrame(failures).to_csv(TABLE_DIR / "physionet_failures.csv", index=False)
    if not feature_parts:
        raise SystemExit("No PhysioNet features extracted.")

    features = pd.concat(feature_parts, ignore_index=True)
    features = trim_wake_edges(features)
    features.to_csv(TABLE_DIR / "physionet_epoch_features.csv", index=False)

    label_counts = features["label"].value_counts().reset_index()
    label_counts.columns = ["label", "epochs"]
    label_counts.to_csv(TABLE_DIR / "physionet_label_counts.csv", index=False)
    save_barplot(label_counts, x="label", y="epochs", path=FIGURE_DIR / "physionet_label_counts.png", title="PhysioNet Sleep Stage Counts")

    skip_cols = {"subject_id", "recording_id", "epoch_index", "label", "raw_label"}
    feature_cols = [c for c in features.columns if c not in skip_cols]
    X = features[feature_cols]
    y = features["label"]
    groups = features["subject_id"]
    metrics, preds, fitted = evaluate_models(X, y, groups=groups, n_splits=5)
    metrics.to_csv(TABLE_DIR / "physionet_model_metrics.csv", index=False)

    best = best_model_name(metrics)
    cm = confusion_matrix_frame(y, preds[best])
    cm.to_csv(TABLE_DIR / "physionet_confusion_matrix.csv")
    save_confusion_matrix(cm, FIGURE_DIR / "physionet_confusion_matrix.png", f"PhysioNet Confusion Matrix: {best}")

    importance = feature_importance_table(fitted[best], feature_cols)
    importance.to_csv(TABLE_DIR / "physionet_feature_importance.csv", index=False)
    save_feature_importance(importance, FIGURE_DIR / "physionet_feature_importance.png", f"PhysioNet Feature Importance: {best}")

    summary = {
        "recordings_attempted": len(pairs),
        "recordings_failed": len(failures),
        "epochs_after_trimming": len(features),
        "subjects": features["subject_id"].nunique(),
        "best_model": best,
        "best_balanced_accuracy": float(metrics.iloc[0]["balanced_accuracy"]),
        "best_macro_f1": float(metrics.iloc[0]["macro_f1"]),
    }
    pd.DataFrame([summary]).to_csv(TABLE_DIR / "physionet_run_summary.csv", index=False)
    print("PhysioNet summary:", summary)


if __name__ == "__main__":
    main()

