#!/usr/bin/env python3
"""
Read-only CNC dataset inspection.

This script inventories a shared CNC data folder without modifying it. It writes
small CSV/JSON summaries to an output folder chosen by the user.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


INTERESTING_EXTENSIONS = {
    ".edf",
    ".csv",
    ".txt",
    ".tsv",
    ".xlsx",
    ".xls",
    ".json",
    ".xml",
    ".h5",
    ".mat",
    ".npy",
    ".parquet",
}


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def safe_mkdir(out_dir: Path, data_root: Path) -> None:
    if is_relative_to(out_dir, data_root):
        raise ValueError(
            f"Output directory must not be inside the data root: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)


def file_inventory(data_root: Path) -> list[dict]:
    rows = []
    for root, _, files in os.walk(data_root):
        root_path = Path(root)
        for name in files:
            path = root_path / name
            try:
                stat = path.stat()
            except OSError:
                continue
            rel = path.relative_to(data_root)
            parts = rel.parts
            rows.append(
                {
                    "relative_path": str(rel),
                    "absolute_path": str(path),
                    "top_level_folder": parts[0] if parts else "",
                    "parent_folder": str(rel.parent),
                    "file_name": name,
                    "extension": path.suffix.lower(),
                    "size_bytes": stat.st_size,
                    "depth": len(parts),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def extension_summary(rows: list[dict]) -> list[dict]:
    counts = Counter(row["extension"] or "[no extension]" for row in rows)
    sizes = defaultdict(int)
    for row in rows:
        sizes[row["extension"] or "[no extension]"] += int(row["size_bytes"])
    return [
        {
            "extension": ext,
            "file_count": counts[ext],
            "total_size_mb": round(sizes[ext] / (1024 * 1024), 3),
        }
        for ext in sorted(counts, key=lambda x: (-counts[x], x))
    ]


def top_level_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(lambda: {"file_count": 0, "total_size_bytes": 0, "extensions": Counter()})
    for row in rows:
        key = row["top_level_folder"]
        grouped[key]["file_count"] += 1
        grouped[key]["total_size_bytes"] += int(row["size_bytes"])
        grouped[key]["extensions"][row["extension"] or "[no extension]"] += 1
    out = []
    for folder, values in grouped.items():
        ext_summary = "; ".join(
            f"{ext}:{count}" for ext, count in values["extensions"].most_common(12)
        )
        out.append(
            {
                "top_level_folder": folder,
                "file_count": values["file_count"],
                "total_size_mb": round(values["total_size_bytes"] / (1024 * 1024), 3),
                "extensions": ext_summary,
            }
        )
    return sorted(out, key=lambda r: (-r["file_count"], r["top_level_folder"]))


def inspect_csv_headers(rows: list[dict], max_files: int) -> list[dict]:
    out = []
    csv_rows = [row for row in rows if row["extension"] in {".csv", ".tsv", ".txt"}]
    for row in csv_rows[:max_files]:
        path = Path(row["absolute_path"])
        delimiter = "\t" if row["extension"] == ".tsv" else ","
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                sample = f.read(8192)
            lines = sample.splitlines()
            header = lines[0] if lines else ""
            if row["extension"] == ".txt" and "," not in header and "\t" in header:
                delimiter = "\t"
            columns = next(csv.reader([header], delimiter=delimiter), [])
            out.append(
                {
                    "relative_path": row["relative_path"],
                    "extension": row["extension"],
                    "size_mb": round(int(row["size_bytes"]) / (1024 * 1024), 3),
                    "first_line": header[:500],
                    "n_header_columns": len(columns),
                    "header_columns": " | ".join(columns[:80]),
                    "sample_line_count_read": len(lines),
                }
            )
        except Exception as exc:  # noqa: BLE001
            out.append(
                {
                    "relative_path": row["relative_path"],
                    "extension": row["extension"],
                    "size_mb": round(int(row["size_bytes"]) / (1024 * 1024), 3),
                    "first_line": "",
                    "n_header_columns": "",
                    "header_columns": "",
                    "sample_line_count_read": "",
                    "error": str(exc),
                }
            )
    return out


def inspect_edf_headers(rows: list[dict], max_files: int) -> list[dict]:
    edf_rows = [row for row in rows if row["extension"] == ".edf"]
    try:
        import mne
    except Exception as exc:  # noqa: BLE001
        return [
            {
                "relative_path": "[mne unavailable]",
                "error": f"Install mne to inspect EDF headers: {exc}",
            }
        ]

    out = []
    for row in edf_rows[:max_files]:
        path = Path(row["absolute_path"])
        try:
            raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
            out.append(
                {
                    "relative_path": row["relative_path"],
                    "size_mb": round(int(row["size_bytes"]) / (1024 * 1024), 3),
                    "n_channels": len(raw.ch_names),
                    "sfreq": float(raw.info["sfreq"]),
                    "duration_seconds": round(raw.n_times / float(raw.info["sfreq"]), 3),
                    "channel_names": " | ".join(raw.ch_names[:80]),
                }
            )
        except Exception as exc:  # noqa: BLE001
            out.append(
                {
                    "relative_path": row["relative_path"],
                    "size_mb": round(int(row["size_bytes"]) / (1024 * 1024), 3),
                    "n_channels": "",
                    "sfreq": "",
                    "duration_seconds": "",
                    "channel_names": "",
                    "error": str(exc),
                }
            )
    return out


def inspect_metadata_workbooks(paths: list[Path], out_dir: Path) -> list[dict]:
    if not paths:
        return []
    try:
        import pandas as pd
    except Exception as exc:  # noqa: BLE001
        return [{"workbook": "[pandas unavailable]", "error": str(exc)}]

    rows = []
    for workbook in paths:
        try:
            xl = pd.ExcelFile(workbook)
            for sheet in xl.sheet_names:
                df = pd.read_excel(workbook, sheet_name=sheet)
                cnc_rows = 0
                if "Cohort" in df.columns:
                    cnc_rows = int(df["Cohort"].astype(str).str.upper().eq("CNC").sum())
                rows.append(
                    {
                        "workbook": str(workbook),
                        "sheet": sheet,
                        "rows": len(df),
                        "columns": len(df.columns),
                        "column_names": " | ".join(map(str, df.columns)),
                        "cnc_rows_if_cohort_column_present": cnc_rows,
                    }
                )
                preview_path = out_dir / f"metadata_preview_{workbook.stem}_{sheet}.csv"
                df.head(25).to_csv(preview_path, index=False)
                if "Cohort" in df.columns:
                    cnc = df[df["Cohort"].astype(str).str.upper().eq("CNC")]
                    cnc.to_csv(out_dir / f"metadata_cnc_rows_{workbook.stem}_{sheet}.csv", index=False)
        except Exception as exc:  # noqa: BLE001
            rows.append({"workbook": str(workbook), "sheet": "", "error": str(exc)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only CNC folder inspection.")
    parser.add_argument("--data-root", required=True, help="Shared CNC folder path.")
    parser.add_argument("--out-dir", required=True, help="Writable output folder, preferably /scratch.")
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Optional metadata workbook path. Can be passed more than once.",
    )
    parser.add_argument("--max-edf-headers", type=int, default=20)
    parser.add_argument("--max-table-headers", type=int, default=50)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    safe_mkdir(out_dir, data_root)

    rows = file_inventory(data_root)
    write_csv(out_dir / "file_inventory.csv", rows)
    write_csv(out_dir / "extension_summary.csv", extension_summary(rows))
    write_csv(out_dir / "top_level_directory_summary.csv", top_level_summary(rows))
    write_csv(out_dir / "csv_txt_header_samples.csv", inspect_csv_headers(rows, args.max_table_headers))
    write_csv(out_dir / "edf_header_samples.csv", inspect_edf_headers(rows, args.max_edf_headers))

    metadata_paths = [Path(p).expanduser().resolve() for p in args.metadata]
    write_csv(out_dir / "metadata_workbook_summary.csv", inspect_metadata_workbooks(metadata_paths, out_dir))

    interesting = [row for row in rows if row["extension"] in INTERESTING_EXTENSIONS]
    summary = {
        "data_root_read_only": str(data_root),
        "output_dir": str(out_dir),
        "total_files": len(rows),
        "total_size_gb": round(sum(int(r["size_bytes"]) for r in rows) / (1024**3), 3),
        "top_level_folders": len({r["top_level_folder"] for r in rows}),
        "interesting_files": len(interesting),
        "edf_files": sum(1 for r in rows if r["extension"] == ".edf"),
        "csv_files": sum(1 for r in rows if r["extension"] == ".csv"),
        "txt_files": sum(1 for r in rows if r["extension"] == ".txt"),
        "h5_files": sum(1 for r in rows if r["extension"] == ".h5"),
        "metadata_files_supplied": [str(p) for p in metadata_paths],
    }
    (out_dir / "inspection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "inspection_summary.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
