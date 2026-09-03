# Dreem Error Analysis

The final combined-feature `random_forest` model correctly classified 39 of 47 subjects in the single 5-fold cross-validated prediction run. It produced 4 false positives and 4 false negatives.

False positives are comparison participants predicted as narcolepsy. False negatives are narcolepsy participants predicted as comparison. The corresponding subject IDs and diagnoses are saved in `tables/false_positives.csv` and `tables/false_negatives.csv`.
