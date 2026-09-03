# Feature Engineering and Machine Learning for Narcolepsy Detection from Wearable EEG

This repository contains code and selected outputs for feature-engineered machine learning models that investigate narcolepsy detection from wearable EEG and sleep-summary features.

The main analysis uses the **Dreem Narcolepsy Revolution** dataset. The **Simons Sleep Project** is used as the main same-device external comparison because it also contains Dreem headband recordings. **CNC PSG** and **PhysioNet Sleep-EDF** are included as supporting datasets for external narcolepsy/control analysis and public EDF sleep-stage processing.

Raw datasets are not included in this repository. The scripts accept dataset and output paths through command-line arguments so the analysis can be run wherever the datasets are available.

## Repository Structure

| Folder | Contents |
|---|---|
| `src/` | Shared Python modules for feature extraction, aggregation, modelling and plotting. |
| `pipelines/dreem/` | Narcolepsy Revolution Dreem processing, feature engineering, modelling and sensitivity experiments. |
| `pipelines/simons/` | Simons Sleep Project feature extraction and NRev-vs-SSP comparison. |
| `pipelines/cnc/` | CNC PSG EDF inspection, feature extraction and narcolepsy/control modelling. |
| `pipelines/physionet/` | PhysioNet Sleep-EDF reference processing and sleep-stage classification. |
| `pipelines/final_package/` | Assembly of consolidated result tables, figures and audit summaries. |
| `dreem_nrev/outputs/` | Main Narcolepsy Revolution processed outputs, figures and model results. |
| `dreem_nrev/improvements/` | Stage-level, channel, subtype, advanced-feature and tuning outputs. |
| `simons_ssp/outputs/` | SSP report/EDF outputs and NRev-vs-SSP comparison outputs. |
| `cnc/outputs/` | CNC PSG feature extraction and evaluation outputs. |
| `physionet_sleep_edf/outputs/` | Sleep-EDF reference outputs and sample EDF inspection figures. |
| `final_results/outputs/` | Consolidated tables, figures and summary audit files. |

Reusable feature extraction, modelling and plotting utilities are kept under `src/diss_eeg/`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
```

## Path Setup

Create a local `.env` file from `.env.example`, or export the paths manually:

```bash
export DREEM_NREV_ROOT=/path/to/narcolepsy_revolution_dreem
export DREEM_METADATA_DIR=/path/to/narcolepsy_revolution_metadata
export SSP_ROOT=/path/to/simons_sleep_project
export SSP_METADATA_CSV=/path/to/simons_sleep_project/_meta/participant_metadata.csv
export CNC_ROOT=/path/to/cnc
export PHYSIONET_ROOT=/path/to/physionet_sleep_edf
```

## Recommended Run Order

### 1. Narcolepsy Revolution Dreem

Extract standard H5 EEG features:

```bash
python pipelines/dreem/01_extract_dreem_h5_features.py \
  --data-root "$DREEM_NREV_ROOT" \
  --scratch-out dreem_nrev/outputs/raw_epoch_features \
  --home-out dreem_nrev/outputs/h5_subject_features
```

Build sleep-stage-level features:

```bash
python pipelines/dreem/03_build_dreem_stage_level_features.py \
  --epoch-dir dreem_nrev/outputs/raw_epoch_features/epoch_features_by_record \
  --label-csv dreem_nrev/outputs/loading_outputs/diagnosis_mapping.csv \
  --out-dir dreem_nrev/improvements/stage_level_features/outputs/narcolepsy_vs_other \
  --target narcolepsy_vs_other
```

Extract advanced H5 features:

```bash
python pipelines/dreem/02_extract_dreem_advanced_h5_features.py \
  --data-root "$DREEM_NREV_ROOT" \
  --scratch-out dreem_nrev/improvements/advanced_h5_features/epoch_features \
  --home-out dreem_nrev/improvements/advanced_h5_features/outputs
```

Run the main model comparison from processed subject-level tables:

```bash
python pipelines/dreem/09_train_main_dreem_models.py
python pipelines/dreem/10_evaluate_dreem_models.py
```

### 2. Simons Sleep Project Same-Device Comparison

```bash
python pipelines/simons/01_extract_simons_report_features.py \
  --data-root "$SSP_ROOT" \
  --metadata "$SSP_METADATA_CSV" \
  --out-dir simons_ssp/outputs/report_outputs

python pipelines/simons/03_train_dreem_vs_simons_report_model.py \
  --narcolepsy-report-subjects dreem_nrev/outputs/feature_outputs/report_features/report_subject_features.csv \
  --simons-report-subjects simons_ssp/outputs/report_outputs/tables/simons_report_subject_features.csv \
  --out-dir simons_ssp/outputs/comparison_outputs

python pipelines/simons/04_audit_nrev_vs_ssp_report_model.py
```

### 3. CNC PSG External Experiment

```bash
python pipelines/cnc/01_inspect_cnc_folder.py \
  --data-root "$CNC_ROOT" \
  --out-dir cnc/outputs/inspection_outputs

python pipelines/cnc/02_extract_cnc_edf_features.py \
  --data-root "$CNC_ROOT" \
  --scratch-out cnc/outputs/epoch_features \
  --home-out cnc/outputs/edf_outputs

python pipelines/cnc/03_train_cnc_models.py \
  --subject-features cnc/outputs/edf_outputs/cnc_edf_subject_features.csv \
  --out-dir cnc/outputs/evaluation_outputs
```

### 4. PhysioNet Sleep-EDF Reference Workflow

```bash
python pipelines/physionet/01_run_physionet_reference_pipeline.py \
  --physionet-dir "$PHYSIONET_ROOT" \
  --subset sleep-cassette \
  --max-recordings 8

python pipelines/physionet/02_inspect_physionet_edf_samples.py \
  --physionet-dir "$PHYSIONET_ROOT" \
  --out-dir physionet_sleep_edf/outputs/edf_sample_outputs
```

### 5. Consolidated Result Package

```bash
python pipelines/final_package/01_build_final_results_package.py
```

Consolidated outputs are stored in:

```text
final_results/outputs/
```

## Main Result Summary

The strongest Narcolepsy Revolution model used combined Dreem report and H5 EEG features with a tuned random forest. It achieved balanced accuracy of approximately `0.732`, macro F1 of `0.716` and ROC-AUC of `0.808` for narcolepsy-vs-other-hypersomnia classification.

The NRev-vs-SSP same-device comparison was evaluated separately using common sleep-report and sleep-architecture features. After cleaning likely non-sleep variables, the model achieved balanced accuracy of approximately `0.893` and ROC-AUC of `0.978`, with an age-only audit included to check demographic influence.

## Notes For Reuse

- Diagnostic modelling is performed at subject level.
- Raw 30-second epoch files can be regenerated and are not committed by default.
- Results from different datasets should be interpreted according to their dataset role rather than merged into a single unrestricted model.
- The project is a feasibility and interpretability study, not a clinically validated diagnostic tool.
