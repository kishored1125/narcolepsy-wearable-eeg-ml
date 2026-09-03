#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT / "src"))

from diss_eeg.physionet import discover_sleep_edf_pairs, extract_physionet_features
from diss_eeg.pipeline_utils import ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect 3-4 real Sleep-EDF EDF samples and save readable outputs.")
    parser.add_argument("--physionet-dir", default=str(PROJECT.parent / "PhysioNet"))
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--signal-seconds", type=int, default=60)
    parser.add_argument("--feature-epochs", type=int, default=80)
    args = parser.parse_args()

    out = PROJECT / "physionet_sleep_edf" / "outputs" / "edf_sample_outputs"
    tables = out / "tables"
    figures = out / "figures"
    ensure_dirs(tables, figures)

    pairs = discover_sleep_edf_pairs(Path(args.physionet_dir), subsets=("sleep-cassette",)).head(args.n_samples)
    pairs.to_csv(tables / "sample_recording_pairs.csv", index=False)

    all_metadata = []
    all_annotations = []
    all_features = []
    all_signal_samples = []
    for row in pairs.itertuples(index=False):
        print(f"Inspecting {row.recording_id}")
        psg_path = Path(row.psg_path)
        hypnogram_path = Path(row.hypnogram_path)
        raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="ERROR")
        annotations = mne.read_annotations(hypnogram_path)

        all_metadata.extend(recording_metadata(raw, row.recording_id, row.subject_id, psg_path, hypnogram_path, annotations))
        annotation_df = annotation_sample(annotations, row.recording_id)
        all_annotations.append(annotation_df)

        raw_eeg = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
        eeg_channels = [ch for ch in ("EEG Fpz-Cz", "EEG Pz-Oz") if ch in raw_eeg.ch_names]
        raw_eeg.pick(eeg_channels)
        signal_df = signal_sample(raw_eeg, row.recording_id, eeg_channels, seconds=max(args.signal_seconds, 300))
        all_signal_samples.append(signal_df.head(2000))

        plot_signal_sample(signal_df, eeg_channels, figures / f"{row.recording_id}_signal_first_{args.signal_seconds}_seconds.png", args.signal_seconds)
        plot_hypnogram(annotation_df, figures / f"{row.recording_id}_hypnogram.png")

        features = extract_physionet_features(
            psg_path,
            hypnogram_path,
            subject_id=row.subject_id,
            recording_id=row.recording_id,
            max_epochs_per_recording=args.feature_epochs,
        )
        all_features.append(features)
        plot_relative_power_features(features, figures / f"{row.recording_id}_relative_power_features.png")

    metadata = pd.DataFrame(all_metadata)
    metadata.to_csv(tables / "edf_channel_metadata_samples.csv", index=False)
    pd.concat(all_annotations, ignore_index=True).to_csv(tables / "edf_hypnogram_annotation_samples.csv", index=False)
    pd.concat(all_signal_samples, ignore_index=True).to_csv(tables / "edf_signal_first_2000_points_samples.csv", index=False)
    feature_table = pd.concat(all_features, ignore_index=True)
    feature_table.to_csv(tables / "edf_30s_epoch_feature_samples.csv", index=False)

    feature_cols = [c for c in feature_table.columns if c.startswith("eeg_fpz_cz_") and "relative_power" in c]
    feature_table[["recording_id", "epoch_index", "label", "raw_label"] + feature_cols[:5]].to_csv(
        tables / "edf_relative_power_feature_samples.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "sample_recordings": len(pairs),
                "subjects": pairs["subject_id"].nunique(),
                "feature_epochs_saved": len(feature_table),
                "signal_points_per_recording_saved": 2000,
                "output_folder": str(out),
            }
        ]
    )
    summary.to_csv(tables / "edf_sample_summary.csv", index=False)
    write_readme(out, pairs)
    print(f"EDF sample outputs saved to {out}")


def recording_metadata(raw, recording_id: str, subject_id: str, psg_path: Path, hypnogram_path: Path, annotations) -> list[dict[str, object]]:
    rows = []
    for idx, channel in enumerate(raw.ch_names):
        rows.append(
            {
                "recording_id": recording_id,
                "subject_id": subject_id,
                "psg_file": str(psg_path),
                "hypnogram_file": str(hypnogram_path),
                "channel": channel,
                "mne_channel_type": raw.get_channel_types(picks=[idx])[0],
                "sampling_frequency_hz": raw.info["sfreq"],
                "duration_hours": raw.n_times / raw.info["sfreq"] / 3600,
                "annotation_count": len(annotations),
                "annotation_labels": "; ".join(sorted(set(annotations.description))),
                "unit": str(raw._orig_units.get(channel, "unknown")) if hasattr(raw, "_orig_units") else "unknown",
            }
        )
    return rows


def annotation_sample(annotations, recording_id: str, max_rows: int = 80) -> pd.DataFrame:
    rows = []
    for onset, duration, label in zip(annotations.onset, annotations.duration, annotations.description):
        rows.append(
            {
                "recording_id": recording_id,
                "onset_seconds": onset,
                "duration_seconds": duration,
                "raw_label": label,
            }
        )
    return pd.DataFrame(rows).head(max_rows)


def signal_sample(raw, recording_id: str, channels: list[str], seconds: int) -> pd.DataFrame:
    sfreq = float(raw.info["sfreq"])
    data = raw.get_data(start=0, stop=int(seconds * sfreq))
    time = np.arange(data.shape[1]) / sfreq
    out = pd.DataFrame({"recording_id": recording_id, "time_seconds": time})
    for idx, channel in enumerate(channels):
        out[channel] = data[idx]
    return out


def plot_signal_sample(df: pd.DataFrame, channels: list[str], path: Path, seconds: int) -> None:
    plot_df = df[df["time_seconds"] <= seconds]
    plt.figure(figsize=(12, 5))
    for idx, channel in enumerate(channels):
        plt.plot(plot_df["time_seconds"], plot_df[channel] * 1e6 + idx * 250, label=channel)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (microvolts, offset for display)")
    plt.title(f"{df['recording_id'].iloc[0]} EDF EEG Signal Sample")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_hypnogram(df: pd.DataFrame, path: Path) -> None:
    stage_map = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,
        "Sleep stage R": 4,
        "Sleep stage ?": np.nan,
        "Movement time": np.nan,
    }
    plot_df = df.copy()
    plot_df["stage_code"] = plot_df["raw_label"].map(stage_map)
    plot_df["onset_hours"] = plot_df["onset_seconds"] / 3600
    plt.figure(figsize=(12, 4))
    plt.step(plot_df["onset_hours"], plot_df["stage_code"], where="post")
    plt.yticks([0, 1, 2, 3, 4], ["Wake", "N1", "N2", "N3", "REM"])
    plt.xlabel("Recording time (hours)")
    plt.ylabel("Sleep stage")
    plt.title(f"{df['recording_id'].iloc[0]} Hypnogram Sample")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_relative_power_features(features: pd.DataFrame, path: Path) -> None:
    cols = [c for c in features.columns if c.startswith("eeg_fpz_cz_") and "relative_power" in c]
    if not cols:
        return
    plot_df = features[["epoch_index", "label"] + cols[:5]].melt(
        id_vars=["epoch_index", "label"],
        var_name="feature",
        value_name="value",
    )
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=plot_df, x="epoch_index", y="value", hue="feature")
    plt.xlabel("30-second epoch index")
    plt.ylabel("Relative power")
    plt.title(f"{features['recording_id'].iloc[0]} EDF Relative Power Features")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_readme(out: Path, pairs: pd.DataFrame) -> None:
    recordings = ", ".join(pairs["recording_id"].astype(str))
    (out / "README.md").write_text(
        "# Real EDF Sample Inspection Outputs\n\n"
        f"This folder contains real Sleep-EDF PhysioNet EDF inspection outputs for {len(pairs)} recordings: {recordings}.\n\n"
        "Generated outputs include channel metadata, hypnogram annotation samples, raw EEG signal snippets, "
        "30-second epoch feature samples and per-recording plots.\n\n"
        "Use these files to show what EDF data looks like before and after processing.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
