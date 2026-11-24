# SDSC 6018 Group 5

**Pediatric heart-murmur screening from phonocardiogram recordings**

| Group member | Student number |
|---|---|
| CHEN Pengjian | 60214097 |
| YANG Jianing | 59942823 |
| XIA Ningze | 61097102 |
| WONG Wai Kwan | 60690110 |

Repository maintained by **CHEN Pengjian** (`pipiwolve`).

[Final report (PDF)](report/SDSC6018_Group5_report.pdf) · [Editable report (Word)](report/SDSC6018_Group5_report.docx) · [Archived review results](data/processed/review_analysis.json)

## Research questions

1. Can XGBoost or a small 1D convolutional neural network rank children labeled Present above those labeled Absent more effectively than L2 logistic regression?
2. How closely do calibrated probabilities agree with observed Present rates overall and by age and sex? With at least 80% labeled coverage on validation, how often does a central abstention band withhold children the human expert labeled Unknown?

We linked repeat visits before splitting the 872 children. Model selection, calibration and policy thresholds used training or validation data. The locked test split contains 121 labeled children (23 Present) and 10 additional children labeled Unknown.

## Main results

| Model | Test AUROC (raw) | Test AUPRC (raw) | Test Brier (raw) | Test Brier (isotonic) |
|---|---:|---:|---:|---:|
| L2 logistic regression | 0.756 | 0.519 | 0.183 | 0.138 |
| XGBoost | 0.738 | 0.522 | 0.146 | 0.124 |
| 1D CNN | 0.862 | 0.685 | 0.144 | 0.099 |

The test Present rate was 0.190. Predicting the training prevalence gave a test Brier score of 0.154. The CNN was selected by raw validation AUPRC. Its paired AUROC difference versus logistic regression was +0.107 (95% child-level bootstrap interval +0.029 to +0.194). The paired AUPRC interval included zero. Small age and sex subgroups limit conclusions about group calibration.

| Selective CNN metric on test | Count | Estimate |
|---|---:|---:|
| Coverage among labeled children | 100/121 | 82.6% |
| PPV among answered positive classifications | 12/15 | 80.0% |
| Recall within answered labeled children | 12/17 | 70.6% |
| Overall recall including withheld Present cases | 12/23 | 52.2% |
| Unknown share of all withheld children | 3/24 | 12.5% |
| Unknown children withheld | 3/10 | 30.0% |

The band withheld six Present children. The median calibrated score was 0.08 for both Unknown and Absent children. In the temporal evaluation, the CNN trained on 2014 and tested on 2015 reached AUROC 0.724 and AUPRC 0.552. Training size and test case mix also differed from the mixed-year random split.

## Reproduce the study

Use Python 3.12. Run the commands from the repository root.

### Verify results from archived predictions

This path needs neither audio nor model training. It recomputes the result tables and the calibration and score-distribution figures. The separate output directory keeps the archived results intact.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/09_review_analysis.py --output-dir build/review
```

Compare `build/review/review_analysis.json` with [the archived review results](data/processed/review_analysis.json). The case-spectra flag is false when audio is unavailable. For the three figures, including the four illustrative case spectrograms, run the same command after installing the raw data below. To regenerate the editable report from the committed results and figures, run `python scripts/07_write_report.py`.

### Rebuild the cohort and retrain

Use a separate clone or worktree. These scripts overwrite the corresponding files under `data/processed/`. Install the extra dependencies for training on Linux (x86-64):

```bash
python -m pip install -r requirements-training.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.1+cpu'
```

Download the [official CirCor v1.0.3 archive](https://physionet-open.s3.amazonaws.com/circor-heart-sound/circor-heart-sound-1.0.3.zip). Extract it under `data/raw/` and rename its top-level folder to `circor-heart-sound-1.0.3`. The scripts expect the following layout:

```text
data/raw/circor-heart-sound-1.0.3/
├── training_data.csv
└── training_data/       # .txt, .hea, .wav and .tsv files
```

```bash
python scripts/01_build_subject_table.py
python scripts/02_validate_data_quality.py
python scripts/03_extract_features.py
python scripts/04_train_tabular.py
python scripts/05_train_cnn.py
python scripts/06_cnn_decision.py
python scripts/08_campaign_cnn.py
```

Seed 6018 is fixed. Retrained scores can vary across library versions and platforms. The review script and report generator use the archived predictions and case examples cited in the report. See the [data file guide](data/README.md) for outputs.

## Data and scope

We use the public training release of the [CirCor DigiScope Phonocardiogram Dataset v1.0.3](https://doi.org/10.13026/tshs-mw03), licensed under Open Data Commons Attribution. The raw archive is stored outside Git. The target is the published expert murmur annotation. Clinical outcomes and detailed murmur descriptors are excluded from model inputs. The tabular models use published S1/S2 segmentations, which were reviewed or corrected by experts. The CNN uses waveforms without those segmentation annotations. The study evaluates retrospective screening data and does not establish the safety or effectiveness of clinical use.
