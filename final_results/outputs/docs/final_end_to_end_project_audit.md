# End-to-End Project Audit

## Project Aim

This project investigates whether wearable EEG recordings can support detection of narcolepsy and related hypersomnia disorders using feature engineering and conventional machine learning. The implemented work aligns with this aim by treating the Dreem narcolepsy dataset as the main dataset, extracting interpretable physiological features, training classical ML models, validating with grouped/subject-level evaluation, and using external datasets only as supporting/sensitivity experiments.

## Project Motivation

The motivation is clinically and technically coherent: narcolepsy diagnosis usually depends on specialist sleep assessment and PSG/MSLT workflows, while wearable EEG could provide a lower-burden signal source. The project does not claim to replace clinical diagnosis. It evaluates whether engineered wearable EEG and sleep-architecture features contain discriminative information that may support decision-making.

## Datasets Used

1. **Dreem narcolepsy dataset**: main analysis dataset. H5 wearable EEG files and Dreem report CSV files were processed. The final main modelling dataset contains 47 labelled subjects, including 11 narcolepsy subjects and 36 comparison subjects.
2. **PhysioNet Sleep-EDF**: public reference dataset used to validate EDF reading, sleep-stage epoching, feature extraction and sleep-stage modelling workflow.
3. **CNC cohort**: external PSG EDF narcolepsy/control experiment. This was processed separately because it differs from Dreem in device, montage and cohort definition.
4. **Simons sleep dataset**: same-device external comparison. ASD-negative controls were processed and compared cautiously against Dreem narcolepsy report features; this is not used as a main diagnostic claim.

## Main Pipeline

The project pipeline is now complete:

1. Dataset inspection and metadata loading.
2. Read-only processing of shared datasets.
3. Preprocessing of EEG using existing filtered H5 signals for Dreem and EDF loading for PhysioNet/CNC/Simons.
4. 30-second epoch alignment with available sleep-stage annotations.
5. Feature engineering from EEG epochs and report-derived sleep summaries.
6. Subject-level aggregation so the diagnostic model predicts at participant level rather than epoch level.
7. Model training and evaluation using logistic regression, linear SVM, random forest and gradient boosting-style baselines where available.
8. Hyperparameter tuning for the main combined Dreem feature set.
9. Interpretability through feature importance, permutation importance and feature-family audits.
10. Extension experiments: channel strategy, stage-level aggregation, diagnosis-specific NT1/NT2 models, advanced H5 features, CNC and Simons external experiments.

## Preprocessing

Dreem H5 processing used four wearable EEG channels (`eeg1` to `eeg4`) and the filtered signal version available inside the H5 files. This is defensible because the H5 files contain structured Dreem signal arrays and timestamps, allowing efficient processing without modifying source data. Failed/no-feature records were recorded explicitly.

EDF processing was implemented and tested for PhysioNet, CNC and Simons. This supports the methodology because it shows the pipeline can handle standard sleep EDF files, but Dreem H5 remains the main project input because it is the target wearable dataset.

Outputs were written to configurable project output folders so the raw dataset locations remain separate from derived tables and figures.

## Feature Engineering

The main Dreem combined feature set includes report sleep-architecture features plus H5 EEG features. The final combined table used approximately 1,730 features after quality control.

Feature families include:

- Spectral band power features.
- Relative power and frequency-ratio features.
- Time-domain features.
- Hjorth-style features.
- Sleep-architecture summaries.
- Advanced extension features including entropy/nonlinear descriptors, spindle-related features and slow-wave-related features.

Stage-level aggregation was also implemented by summarising epoch-level EEG features separately within Wake, N1, N2, N3 and REM. This directly addresses the sleep-stage-aware feature experiment.

## Modelling And Evaluation

The main analysis model is the tuned Dreem combined report + H5 model:

- Best model: random forest.
- Balanced accuracy: 0.732.
- Macro F1: 0.716.
- ROC-AUC: 0.808.
- Sensitivity: 0.636.
- Specificity: 0.828.

This is the primary result because it matches the project aim and uses the wearable Dreem dataset.

The CNC PSG EDF experiment achieved stronger performance, but it should be treated as an external narcolepsy/control experiment rather than the main wearable EEG result:

- Balanced accuracy: 0.917.
- Macro F1: 0.922.
- ROC-AUC: 0.981.

The Simons external-control report experiment reached perfect metrics, but this should be interpreted cautiously as evidence of dataset/cohort separation, not clinical diagnostic generalisation.

## Extension Experiments

The following extension experiments are included:

- Hyperparameter tuning for the main Dreem combined model.
- Channel/average-channel strategy comparison.
- Stage-level H5 feature aggregation.
- NT1-vs-other and NT2-vs-other exploratory classification.
- Advanced H5 feature extraction and modelling.
- CNC external narcolepsy/control EDF pipeline and evaluation.
- Simons external-control report and EDF pipelines.
- Feature importance and permutation importance summaries.
- Metric uncertainty summary for key models.
- Leakage audit and exclusion of target-like columns.

The advanced H5 feature experiment is useful as a negative result: it added entropy, spindle and slow-wave features, but did not outperform the simpler combined model. This strengthens the analysis because it shows methodological exploration rather than only reporting the best result.

## Project Coverage

The project contains:

- A clear clinical/AI motivation.
- A main real-world wearable EEG dataset.
- Public reference dataset validation.
- Multiple external/sensitivity datasets.
- Reproducible scripts and structured outputs.
- Interpretable feature engineering rather than black-box-only modelling.
- Multiple model comparisons.
- Hyperparameter tuning.
- Evaluation with balanced accuracy, macro F1, ROC-AUC, sensitivity and specificity.
- Subject-level validation and leakage checks.
- Honest limitations and negative-result experiments.

## Remaining Risks

The main risk is sample size. The Dreem main dataset has only 11 narcolepsy subjects, and NT2 has only 3 subjects. Therefore, the report must avoid claiming clinical deployment readiness.

The second risk is external dataset mismatch. CNC and Simons are useful supporting experiments, but they should not be merged naively with Dreem or presented as the same distribution.

The third risk is high-dimensional feature space. This has been mitigated through quality control, model comparison, tuning and cautious interpretation, but it should still be discussed.

## Final Recommendation

The main result can be summarised as:

> A combined wearable Dreem EEG and sleep-report feature set provided the strongest main-dataset performance for narcolepsy-vs-other-hypersomnia classification, while stage-level, channel-level, subtype and advanced-feature experiments provided additional methodological evidence and limitations.

The results should be interpreted as feasibility evidence for feature-engineered wearable EEG analysis rather than as a clinically validated diagnostic system.
