# Validation And Leakage Audit

| area | status | evidence | risk |
| --- | --- | --- | --- |
| Dreem main models | implemented | Subject-level cross-validation and fold metrics are stored in dreem_nrev/outputs/model_outputs/all_fold_metrics.csv. | Small positive class; report uncertainty and avoid overclaiming. |
| CNC leakage check | fixed | Initial target-derived diagnosis_binary aggregate features were detected and excluded in run_cnc_model_evaluation.py. | CNC labels remain class-imbalanced after matched-file filtering. |
| Simons external control | interpreted cautiously | Perfect report-feature separation is marked as domain-shift sensitivity evidence, not primary diagnostic performance. | Dataset-source artefacts may dominate. |
| Dataset merging | not used as main claim | Datasets differ by device, cohort, channel montage and labels; treated as separate experiments. | Naive merging could learn dataset identity. |
