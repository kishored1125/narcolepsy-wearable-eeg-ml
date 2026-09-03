# Dreem Quality-Control Summary

The Dreem H5 extraction processed 363 recordings and failed on 17 recordings. The main labelled modelling table contains report, H5 and combined feature sets.

Dataset-level missingness and feature flags are saved in the `tables/` folder. These outputs support the project limitation that the dataset is small, imbalanced and high-dimensional.

dataset,rows,columns,numeric_columns,missing_values,missing_percentage,all_missing_numeric_columns,constant_numeric_columns,infinite_numeric_values
report_subject_features,47,346,342,340,0.020907637436969616,0,0,0
h5_subject_features_labelled,47,1401,1397,1085,0.016477591993560827,5,5,0
combined_subject_features,47,1741,1737,1425,0.017414789739328092,5,5,0

