# Dreem Diagnostic Evaluation

This folder contains the final NRDREEM diagnostic evaluation. Three feature sets are compared: report-only, H5-only and combined report+H5 features.

- `report_only`: `random_forest`, balanced accuracy 0.680, macro F1 0.659, ROC-AUC 0.812.
- `h5_only`: `logistic_regression`, balanced accuracy 0.638, macro F1 0.622, ROC-AUC 0.709.
- `combined_report_h5`: `random_forest`, balanced accuracy 0.711, macro F1 0.698, ROC-AUC 0.796.

The best current setting is `combined_report_h5` with `random_forest`. The results should be interpreted as exploratory because the labelled narcolepsy class is small.
