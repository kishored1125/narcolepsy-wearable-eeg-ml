from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from diss_eeg.features import extract_epoch_features


LABEL_MAP = {
    "Sleep stage W": "Wake",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}


def discover_sleep_edf_pairs(physionet_dir: Path, subsets: tuple[str, ...] = ("sleep-cassette",)) -> pd.DataFrame:
    rows = []
    for subset in subsets:
        folder = physionet_dir / subset
        for psg_path in sorted(folder.glob("*PSG.edf")):
            stem = psg_path.name.split("-PSG.edf")[0]
            hypnograms = sorted(folder.glob(f"{stem[:6]}*-Hypnogram.edf"))
            if not hypnograms:
                continue
            rows.append(
                {
                    "subset": subset,
                    "recording_id": stem,
                    "subject_id": stem[:5],
                    "psg_path": str(psg_path),
                    "hypnogram_path": str(hypnograms[0]),
                }
            )
    return pd.DataFrame(rows)


def extract_physionet_features(
    psg_path: Path,
    hypnogram_path: Path,
    subject_id: str,
    recording_id: str,
    eeg_channels: tuple[str, ...] = ("EEG Fpz-Cz", "EEG Pz-Oz"),
    epoch_seconds: int = 30,
    max_epochs_per_recording: int | None = None,
) -> pd.DataFrame:
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
    available_channels = [ch for ch in eeg_channels if ch in raw.ch_names]
    if not available_channels:
        raise ValueError(f"No requested EEG channels found in {psg_path.name}. Available: {raw.ch_names}")

    raw.pick(available_channels)
    raw.filter(0.5, 30.0, fir_design="firwin", verbose="ERROR")
    annotations = mne.read_annotations(hypnogram_path)
    raw.set_annotations(annotations, emit_warning=False)

    events, event_id = mne.events_from_annotations(raw, chunk_duration=epoch_seconds, verbose="ERROR")
    inverse_event_id = {value: str(key) for key, value in event_id.items()}
    sfreq = float(raw.info["sfreq"])
    epoch_len = int(epoch_seconds * sfreq)

    rows = []
    usable_events = events
    if max_epochs_per_recording is not None:
        usable_events = usable_events[:max_epochs_per_recording]

    for epoch_idx, event in enumerate(usable_events):
        raw_label = inverse_event_id.get(int(event[2]))
        label = LABEL_MAP.get(raw_label)
        if label is None:
            continue
        start = int(event[0])
        stop = start + epoch_len
        if stop > raw.n_times:
            continue
        data = raw.get_data(start=start, stop=stop)
        feats = extract_epoch_features(data, sfreq=sfreq, channel_names=available_channels)
        feats.update(
            {
                "subject_id": subject_id,
                "recording_id": recording_id,
                "epoch_index": epoch_idx,
                "label": label,
                "raw_label": raw_label,
            }
        )
        rows.append(feats)
    return pd.DataFrame(rows)


def trim_wake_edges(df: pd.DataFrame, label_col: str = "label", group_col: str = "recording_id", max_wake_epochs: int = 60) -> pd.DataFrame:
    """Keep at most `max_wake_epochs` wake epochs before first and after last sleep."""
    parts = []
    sleep_labels = {"N1", "N2", "N3", "REM"}
    for _, group in df.sort_values("epoch_index").groupby(group_col):
        labels = group[label_col].to_numpy()
        sleep_idx = np.flatnonzero(np.isin(labels, list(sleep_labels)))
        if sleep_idx.size == 0:
            parts.append(group)
            continue
        lo = max(0, sleep_idx[0] - max_wake_epochs)
        hi = min(len(group), sleep_idx[-1] + max_wake_epochs + 1)
        parts.append(group.iloc[lo:hi])
    return pd.concat(parts, ignore_index=True) if parts else df

