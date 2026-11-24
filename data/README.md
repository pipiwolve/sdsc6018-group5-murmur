# CirCor data

`processed/` contains the archived results for the Group 5 report. Rebuilding the cohort, fitting models or running the decision script writes derived outputs here. Use a separate working copy for a full retrain. For a check that leaves the archive intact, pass `--output-dir` to the review script.

| File | Contents |
|---|---|
| `subject_table.csv`, `split_counts.csv` | One retained visit and one fixed split per child |
| `feature_table.csv`, `feature_columns.txt` | Identifiers, cohort fields and the 70 tabular model inputs |
| `predictions_tabular.csv`, `predictions_cnn.csv` | Raw and calibrated validation and test predictions |
| `model_results.json`, `cnn_results.json` | Validation selection and model metrics |
| `cnn_decision.json`, `campaign_cnn.json` | Decision metrics and 2014-to-2015 evaluation |
| `review_analysis.json`, `figures/` | Bootstrap intervals, subgroup calibration, calibration sensitivity and figures |
| `error_cases.csv`, `error_cases_cnn.csv` | Tabular and CNN test error records |

The raw archive and waveforms belong under `data/raw/`. See the [reproduction instructions](../README.md#reproduce-the-study) for the PhysioNet v1.0.3 download and expected layout.
