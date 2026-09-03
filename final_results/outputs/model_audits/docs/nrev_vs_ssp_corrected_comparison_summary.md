# NRev-vs-SSP Corrected Comparison Audit

This output audits the original Narcolepsy Revolution versus SSP Dreem report-feature comparison and re-runs it after removing likely non-sleep artefact variables.

- Subjects: 106 total (11 Narcolepsy Revolution narcolepsy, 95 SSP healthy/control).
- Best cleaned model: logistic_regression using `clean_sleep_architecture_features`.
- Best cleaned metrics: balanced accuracy 0.893, macro F1 0.878, ROC-AUC 0.978, sensitivity 0.818, specificity 0.968.
- The original feature set is retained only as an artefact audit because quality/confidence/report-format variables dominate the random-forest feature importance.

Use `tables/nrev_vs_ssp_corrected_model_metrics.csv` and the figures in `figures/` for the revised report section.
