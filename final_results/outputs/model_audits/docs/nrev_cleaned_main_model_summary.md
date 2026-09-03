# Cleaned Main Narcolepsy Revolution Model

The cleaned analysis removes likely report artefact variables before modelling the Narcolepsy Revolution cohort.

- Subjects: 47 (11 NT1/NT2 and 36 other hypersomnia/comparison).
- Best cleaned model: logistic_regression using `combined_h5_report_artifacts_removed`.
- Metrics: balanced accuracy 0.630, macro F1 0.626, ROC-AUC 0.686, sensitivity 0.455, specificity 0.806.

Use this alongside the original tuned combined model to discuss whether performance depends on physiological sleep features or report quality/confidence fields.
