# SDSC 6018 Group 5

**Pediatric heart-murmur screening from phonocardiogram recordings: ranking, calibration, and abstention**

Repository maintained by **CHEN Pengjian** (student number 60214097), GitHub account `pipiwolve`.

| Name | Student number |
|---|---|
| CHEN Pengjian | 60214097 |
| YANG Jianing | 59942823 |
| XIA Ningze | 61097102 |
| WONG Wai Kwan | 60690110 |

## Question

Using only information available at a pediatric screening visit, can a gradient-boosted model or a small convolutional network rank Present murmurs ahead of Absent murmurs better than L2 logistic regression? After calibration, are the probabilities more accurate than always guessing the training prevalence, and does withholding the uncertain middle recover the annotator’s Unknown label?

The English report is `report/SDSC6018_Group5_report.pdf`. The Word file beside it is the source. Before submission, export the PDF from Word in Times New Roman, 12 pt, single spacing. The converter used for the PDF in this repository substituted Latin Modern.

## Result on the locked test split

The model was chosen by validation average precision before the test labels were used. The convolutional network is the selected model.

| Model | AUROC | Average precision | Brier, raw | Brier, calibrated |
|---|---:|---:|---:|---:|
| Logistic regression | 0.756 | 0.519 | 0.183 | 0.138 |
| XGBoost | 0.738 | 0.522 | 0.146 | 0.124 |
| 1D CNN | 0.862 | 0.685 | 0.144 | 0.099 |
| Always training prevalence | 0.500 | 0.190 | 0.154 | 0.154 |

The validation abstention band still answers 82.6% of labeled test children, at precision 0.80 among those answers. It withholds 3 of 10 Unknown children. On the test set their median calibrated score is 0.08, the same as Absent.

The same network trained on 2014 visits and tested on 2015 visits reaches raw AUROC 0.724 and average precision 0.552. The random split mixes both summers and is the more optimistic ranking.

## What is in this repository

- `scripts/01` through `06`, and `scripts/08`: subject table, quality checks, features, tabular models, the CNN, the abstention rule, and the 2014-to-2015 check.
- `data/processed/`: the subject table, feature table, predictions, metric files, error cases, and figures.
- `report/`: the English report.

Raw CirCor waveforms are not in this repository. Download PhysioNet version 1.0.3 from `https://physionet-open.s3.amazonaws.com/circor-heart-sound/circor-heart-sound-1.0.3.zip` and place the extracted training files where `scripts/03_extract_features.py` expects them (`data/raw/circor-heart-sound-1.0.3/training_data`). The published counts can be checked without the audio from the files already in `data/processed/`.

## Reproduce

```bash
python3 scripts/01_build_subject_table.py
python3 scripts/02_validate_data_quality.py
python3 scripts/03_extract_features.py
python3 scripts/04_train_tabular.py
python3 scripts/05_train_cnn.py
python3 scripts/06_cnn_decision.py
python3 scripts/08_campaign_cnn.py
```

Seed 6018. Unknown labels are not used for training, threshold selection, or the abstention band. The test split is scored once after those choices are fixed. The 2014-to-2015 script does not change the primary model.

## Data and ethics

CirCor DigiScope Phonocardiogram Dataset, version 1.0.3, DOI 10.13026/tshs-mw03. Open Data Commons Attribution. The public training release is used. The cardiologist outcome and the murmur descriptors are not model inputs. Scores from this study are not a screening recommendation.
