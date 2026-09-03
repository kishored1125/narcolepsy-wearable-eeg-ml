#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT / "src"))

from diss_eeg.nrdreem_reports import load_diagnosis_table, load_report_table
from diss_eeg.pipeline_utils import ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Load Dreem report files, diagnosis metadata and H5 run summaries.")
    parser.add_argument("--nrdreem-report-dir", default=str(PROJECT.parent / "nrdreemdata"))
    parser.add_argument("--metadata-dir", default=str(PROJECT.parent / "narcolepsy_dreem"))
    parser.add_argument("--h5-output-dir", default=str(PROJECT / "dreem_nrev" / "outputs" / "h5_subject_features"))
    args = parser.parse_args()

    out_dir = PROJECT / "dreem_nrev" / "outputs" / "loading_outputs"
    ensure_dirs(out_dir)

    diagnosis = load_diagnosis_table(Path(args.metadata_dir))
    diagnosis.to_csv(out_dir / "diagnosis_mapping.csv", index=False)

    reports = load_report_table(Path(args.nrdreem_report_dir))
    reports.to_csv(out_dir / "report_record_table.csv", index=False)

    h5_dir = Path(args.h5_output_dir)
    if (h5_dir / "dreem_h5_run_summary.csv").exists():
        pd.read_csv(h5_dir / "dreem_h5_run_summary.csv").to_csv(out_dir / "h5_run_summary.csv", index=False)
    if (h5_dir / "dreem_h5_failures.csv").exists():
        pd.read_csv(h5_dir / "dreem_h5_failures.csv").to_csv(out_dir / "h5_failures.csv", index=False)

    pd.DataFrame(
        [
            {
                "diagnosis_rows": len(diagnosis),
                "report_records": len(reports),
                "report_subjects": reports["patient_id"].nunique() if "patient_id" in reports else 0,
            }
        ]
    ).to_csv(out_dir / "loading_summary.csv", index=False)
    print(f"Dreem loading outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
