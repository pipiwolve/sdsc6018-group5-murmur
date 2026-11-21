"""Train logistic regression and XGBoost, then score the locked test split once.

Hyperparameters, the probability threshold, and the calibration map are chosen
on validation labeled children only. Unknown labels are not used for fitting,
threshold selection, or the abstention band.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIG = PROCESSED / "figures"
SEED = 6018
FEATURE_COLUMNS = (PROCESSED / "feature_columns.txt").read_text().split()
SITE_MISSING = ["av_missing", "pv_missing", "tv_missing", "mv_missing"]
BODY_COLUMNS = [
    "age_Neonate",
    "age_Infant",
    "age_Adolescent",
    "age_Missing",
    "sex_Male",
    "height",
    "weight",
    "height_missing",
    "weight_missing",
]


def load_frame() -> pd.DataFrame:
    features = pd.read_csv(PROCESSED / "feature_table.csv")
    subjects = pd.read_csv(PROCESSED / "subject_table.csv")[
        ["child_id", "age", "sex", "n_standard_sites", "has_AV", "has_PV", "has_TV", "has_MV"]
    ]
    return features.merge(subjects, on="child_id", how="left", validate="one_to_one")


def labeled_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["supervised"].astype(bool)


def y_of(frame: pd.DataFrame) -> np.ndarray:
    return (frame["murmur"] == "Present").astype(int).to_numpy()


def fit_lr(train_x: pd.DataFrame, train_y: np.ndarray, columns: list[str], c_value: float):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    transformed = scaler.fit_transform(imputer.fit_transform(train_x[columns]))
    model = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=SEED,
    )
    model.fit(transformed, train_y)
    return imputer, scaler, model


def predict_lr(bundle, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    imputer, scaler, model = bundle
    transformed = scaler.transform(imputer.transform(frame[columns]))
    return model.predict_proba(transformed)[:, 1]


def fit_xgb(train_x: pd.DataFrame, train_y: np.ndarray, columns: list[str], params: dict):
    pos = max(int(train_y.sum()), 1)
    neg = max(int(len(train_y) - pos), 1)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=neg / pos,
        random_state=SEED,
        n_jobs=4,
        tree_method="hist",
        **params,
    )
    model.fit(train_x[columns], train_y)
    return model


def predict_xgb(model, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return model.predict_proba(frame[columns])[:, 1]


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "n": int(len(y_true)),
        "n_present": int(y_true.sum()),
        "prevalence": float(np.mean(y_true)),
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, np.clip(scores, 0, 1))),
    }


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 37)))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        f1 = f1_score(y_true, scores >= threshold, zero_division=0)
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), float(f1)
    return best_threshold


def operating_counts(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    tp = int(np.sum((predicted == 1) & (y_true == 1)))
    fp = int(np.sum((predicted == 1) & (y_true == 0)))
    fn = int(np.sum((predicted == 0) & (y_true == 1)))
    tn = int(np.sum((predicted == 0) & (y_true == 0)))
    n = max(len(y_true), 1)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fp_per_100": 100.0 * fp / n,
        "fn_per_100": 100.0 * fn / n,
    }


def fit_calibrators(scores: np.ndarray, y_true: np.ndarray) -> dict:
    platt = LogisticRegression(max_iter=1000)
    platt.fit(scores.reshape(-1, 1), y_true)
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(scores, y_true)
    platt_p = platt.predict_proba(scores.reshape(-1, 1))[:, 1]
    iso_p = isotonic.predict(scores)
    choice = "platt" if brier_score_loss(y_true, platt_p) <= brier_score_loss(y_true, iso_p) else "isotonic"
    return {"platt": platt, "isotonic": isotonic, "choice": choice}


def apply_calibrator(bundle: dict, scores: np.ndarray) -> np.ndarray:
    if bundle["choice"] == "platt":
        return bundle["platt"].predict_proba(scores.reshape(-1, 1))[:, 1]
    return bundle["isotonic"].predict(scores)


def abstention_half_width(scores: np.ndarray, coverage: float = 0.80) -> float:
    """Widest central band that still answers at least `coverage` of labeled rows.

    A literally zero-width band would also satisfy the coverage floor, and would
    never abstain. The locked 80 percent rule is treated as the coverage target.
    """
    low, high, best = 0.0, 0.5, 0.0
    for _ in range(40):
        width = (low + high) / 2
        answered = (scores <= 0.5 - width) | (scores >= 0.5 + width)
        if float(np.mean(answered)) >= coverage:
            best = width
            low = width
        else:
            high = width
    return float(best)


def abstention_report(frame: pd.DataFrame, scores: np.ndarray, width: float) -> dict:
    labeled = labeled_mask(frame).to_numpy()
    unknown = (frame["murmur"] == "Unknown").to_numpy()
    answered = (scores <= 0.5 - width) | (scores >= 0.5 + width)
    y_true = y_of(frame)
    answered_labeled = answered & labeled
    predicted_pos = scores >= 0.5 + width
    if answered_labeled.sum() == 0:
        precision = recall = float("nan")
    else:
        tp = np.sum(predicted_pos & answered_labeled & (y_true == 1))
        fp = np.sum(predicted_pos & answered_labeled & (y_true == 0))
        fn = np.sum((~predicted_pos) & answered_labeled & (y_true == 1))
        precision = float(tp / (tp + fp)) if (tp + fp) else float("nan")
        recall = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    abstained = ~answered
    return {
        "half_width": width,
        "labeled_coverage": float(np.mean(answered[labeled])),
        "precision_answered": precision,
        "recall_answered": recall,
        "unknown_among_abstained": float(np.mean(unknown[abstained])) if abstained.any() else float("nan"),
        "unknown_abstained": float(np.mean(abstained[unknown])) if unknown.any() else float("nan"),
        "n_labeled": int(labeled.sum()),
        "n_unknown": int(unknown.sum()),
        "n_abstained": int(abstained.sum()),
    }


def constant_baselines(y_true: np.ndarray, train_prevalence: float) -> dict:
    prevalence_scores = np.full(len(y_true), train_prevalence)
    quarter_scores = np.full(len(y_true), 0.25)
    return {
        "train_prevalence": train_prevalence,
        "auprc_prevalence_baseline": float(np.mean(y_true)),
        "brier_prevalence": float(brier_score_loss(y_true, prevalence_scores)),
        "brier_quarter": float(brier_score_loss(y_true, quarter_scores)),
    }


def reliability_points(y_true: np.ndarray, scores: np.ndarray, bins: int = 8) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (scores >= left) & (scores <= right if right == 1 else scores < right)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "n": int(mask.sum()),
                "mean_score": float(np.mean(scores[mask])),
                "observed": float(np.mean(y_true[mask])),
            }
        )
    return rows


def plot_reliability(curves: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot([0, 1], [0, 1], color="#112D4E", linewidth=1, label="Ideal")
    colors = {"logistic": "#3F72AF", "xgboost": "#48466D"}
    for name, points in curves.items():
        if not points:
            continue
        ax.plot(
            [row["mean_score"] for row in points],
            [row["observed"] for row in points],
            marker="o",
            color=colors.get(name, "#112D4E"),
            label=name,
        )
    ax.set_xlabel("Mean calibrated score")
    ax.set_ylabel("Observed Present rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def subgroup_table(frame: pd.DataFrame, scores: np.ndarray) -> list[dict]:
    labeled = frame.loc[labeled_mask(frame)].copy()
    labeled["score"] = scores[labeled_mask(frame).to_numpy()]
    labeled["y"] = y_of(labeled)
    rows = []
    labeled["age_band"] = labeled["age"].fillna("Missing")
    for keys, part in labeled.groupby(["age_band", "sex"], dropna=False):
        y_true = part["y"].to_numpy()
        row = {
            "age": keys[0],
            "sex": keys[1],
            "n": int(len(part)),
            "n_present": int(y_true.sum()),
            "n_absent": int((y_true == 0).sum()),
        }
        if row["n"] >= 20 and row["n_present"] > 0 and row["n_absent"] > 0:
            row["auroc"] = float(roc_auc_score(y_true, part["score"]))
            row["small_cell"] = False
        else:
            row["auroc"] = None
            row["small_cell"] = True
        rows.append(row)
    return rows


def select_logistic(train: pd.DataFrame, val: pd.DataFrame, columns: list[str]) -> tuple[float, object]:
    y_train, y_val = y_of(train), y_of(val)
    best_c, best_score, best_bundle = None, -1.0, None
    for c_value in (0.01, 0.1, 1.0, 10.0):
        bundle = fit_lr(train, y_train, columns, c_value)
        score = average_precision_score(y_val, predict_lr(bundle, val, columns))
        if score > best_score:
            best_c, best_score, best_bundle = c_value, float(score), bundle
    return best_c, best_bundle


def select_xgboost(train: pd.DataFrame, val: pd.DataFrame, columns: list[str]) -> tuple[dict, object]:
    y_train, y_val = y_of(train), y_of(val)
    grid = [
        {"max_depth": 2, "n_estimators": 200, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.05, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 3, "n_estimators": 300, "learning_rate": 0.03, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8},
    ]
    best_params, best_score, best_model = None, -1.0, None
    for params in grid:
        model = fit_xgb(train, y_train, columns, params)
        score = average_precision_score(y_val, predict_xgb(model, val, columns))
        if score > best_score:
            best_params, best_score, best_model = params, float(score), model
    return best_params, best_model


def error_table(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> tuple[pd.DataFrame, dict]:
    labeled = frame.loc[labeled_mask(frame)].copy()
    labeled["score"] = scores[labeled_mask(frame).to_numpy()]
    labeled["y"] = y_of(labeled)
    labeled["predicted"] = (labeled["score"] >= threshold).astype(int)
    labeled["error"] = np.where(
        (labeled["predicted"] == 1) & (labeled["y"] == 0),
        "false_positive",
        np.where((labeled["predicted"] == 0) & (labeled["y"] == 1), "false_negative", "correct"),
    )
    pieces = []
    for kind in ("false_positive", "false_negative"):
        part = labeled[labeled["error"] == kind]
        if kind == "false_positive":
            part = part.sort_values("score", ascending=False)
        else:
            part = part.sort_values("score", ascending=True)
        pieces.append(part.head(30))
    chosen = pd.concat(pieces, ignore_index=True)
    keep = [
        "child_id",
        "age",
        "sex",
        "murmur",
        "score",
        "error",
        "n_standard_sites",
        "has_AV",
        "has_PV",
        "has_TV",
        "has_MV",
        "av_cycle_frac",
        "pv_cycle_frac",
        "tv_cycle_frac",
        "mv_cycle_frac",
    ]
    summary = {}
    for kind, part in labeled.groupby("error"):
        summary[kind] = {
            "n": int(len(part)),
            "mean_sites": float(part["n_standard_sites"].mean()),
            "mean_cycle_frac": float(part[["av_cycle_frac", "pv_cycle_frac", "tv_cycle_frac", "mv_cycle_frac"]].mean(axis=1).mean()),
            "age": part["age"].fillna("Missing").value_counts().to_dict(),
        }
    return chosen[keep], summary


def mean_impute_sites(train: pd.DataFrame, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    filled = frame.copy()
    means = train[columns].mean(numeric_only=True)
    filled[columns] = filled[columns].fillna(means)
    return filled


def campaign_shift(frame: pd.DataFrame, columns: list[str], params: dict) -> dict:
    train = frame[(frame["campaign"] == "CC2014") & labeled_mask(frame)]
    test = frame[(frame["campaign"] == "CC2015") & labeled_mask(frame)]
    model = fit_xgb(train, y_of(train), columns, params)
    scores = predict_xgb(model, test, columns)
    return {"train_n": int(len(train)), "test_n": int(len(test)), **ranking_metrics(y_of(test), scores)}


def shap_summary(model, train: pd.DataFrame, columns: list[str]) -> list[dict]:
    sample = train[columns].sample(n=min(200, len(train)), random_state=SEED)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        values = values[-1]
    importance = np.abs(values).mean(axis=0)
    order = np.argsort(importance)[::-1][:15]
    return [{"feature": columns[int(i)], "mean_abs_shap": float(importance[int(i)])} for i in order]


def evaluate_split(frame: pd.DataFrame, raw_scores: np.ndarray, calibrated: np.ndarray, threshold: float, width: float, train_prevalence: float) -> dict:
    labeled = labeled_mask(frame).to_numpy()
    y_true = y_of(frame)[labeled]
    return {
        "raw": ranking_metrics(y_true, raw_scores[labeled]),
        "calibrated": ranking_metrics(y_true, calibrated[labeled]),
        "operating_point": operating_counts(y_true, calibrated[labeled], threshold),
        "baselines": constant_baselines(y_true, train_prevalence),
        "abstention": abstention_report(frame, calibrated, width),
        "reliability": reliability_points(y_true, calibrated[labeled]),
    }


def run_pair(train, val, test, columns: list[str], family: str, selector) -> dict:
    if family == "logistic":
        chosen, model = selector(train, val, columns)
        raw = {split: predict_lr(model, frame, columns) for split, frame in (("train", train), ("val", val), ("test", test))}
        setting = {"C": chosen}
    else:
        chosen, model = selector(train, val, columns)
        raw = {split: predict_xgb(model, frame, columns) for split, frame in (("train", train), ("val", val), ("test", test))}
        setting = chosen
    return {"setting": setting, "model": model, "raw": raw}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    columns = FEATURE_COLUMNS
    parts = {name: frame[frame["split"] == name].reset_index(drop=True) for name in ("train", "val", "test")}
    train_labeled = parts["train"][labeled_mask(parts["train"])]
    val_labeled = parts["val"][labeled_mask(parts["val"])]
    prevalence = float(y_of(train_labeled).mean())

    logistic = run_pair(train_labeled, val_labeled, parts["test"], columns, "logistic", select_logistic)
    xgboost = run_pair(train_labeled, val_labeled, parts["test"], columns, "xgboost", select_xgboost)

    results = {"seed": SEED, "train_prevalence": prevalence, "models": {}}
    prediction_rows = []
    reliability = {}
    for name, pack in (("logistic", logistic), ("xgboost", xgboost)):
        calibrator = fit_calibrators(pack["raw"]["val"], y_of(val_labeled))
        calibrated = {split: apply_calibrator(calibrator, scores) for split, scores in pack["raw"].items()}
        threshold = best_f1_threshold(y_of(val_labeled), calibrated["val"])
        width = abstention_half_width(calibrated["val"])
        full_scores = {}
        full_calibrated = {}
        for split, part in parts.items():
            if name == "logistic":
                raw_scores = predict_lr(pack["model"], part, columns)
            else:
                raw_scores = predict_xgb(pack["model"], part, columns)
            full_scores[split] = raw_scores
            full_calibrated[split] = apply_calibrator(calibrator, raw_scores)
        evaluated = {
            split: evaluate_split(parts[split], full_scores[split], full_calibrated[split], threshold, width, prevalence)
            for split in ("val", "test")
        }
        results["models"][name] = {
            "setting": pack["setting"],
            "calibrator": calibrator["choice"],
            "threshold": threshold,
            "abstention_half_width": width,
            "validation": evaluated["val"],
            "test": evaluated["test"],
        }
        reliability[name] = evaluated["test"]["reliability"]
        for split in ("val", "test"):
            part = parts[split]
            for child_id, murmur, raw_score, score in zip(part["child_id"], part["murmur"], full_scores[split], full_calibrated[split]):
                prediction_rows.append(
                    {
                        "model": name,
                        "split": split,
                        "child_id": int(child_id),
                        "murmur": murmur,
                        "raw_score": float(raw_score),
                        "calibrated_score": float(score),
                    }
                )
        print(name, calibrator["choice"], "val AUPRC", round(evaluated["val"]["calibrated"]["auprc"], 3), "test AUPRC", round(evaluated["test"]["calibrated"]["auprc"], 3))

    val_choice = max(results["models"], key=lambda model: results["models"][model]["validation"]["raw"]["auprc"])
    results["selected_model"] = val_choice
    selected_scores = {
        split: np.array(
            [
                row["calibrated_score"]
                for row in prediction_rows
                if row["model"] == val_choice and row["split"] == split
            ]
        )
        for split in ("val", "test")
    }
    # prediction_rows order follows child order, so scores align with parts
    results["subgroups_test"] = subgroup_table(parts["test"], selected_scores["test"])
    errors, error_summary = error_table(parts["test"], selected_scores["test"], results["models"][val_choice]["threshold"])
    errors.to_csv(PROCESSED / "error_cases.csv", index=False)
    results["error_summary"] = error_summary

    ablation_specs = {
        "drop_site_missing_indicators": [column for column in columns if column not in SITE_MISSING],
        "drop_age_and_body_size": [column for column in columns if column not in BODY_COLUMNS],
    }
    results["ablations"] = {}
    xgb_params = xgboost["setting"]
    for ablation_name, ablation_columns in ablation_specs.items():
        test_labeled = parts["test"][labeled_mask(parts["test"])]
        if ablation_name == "drop_site_missing_indicators":
            train_x = mean_impute_sites(train_labeled, train_labeled, ablation_columns)
            val_x = mean_impute_sites(train_labeled, val_labeled, ablation_columns)
            test_x = mean_impute_sites(train_labeled, test_labeled, ablation_columns)
        else:
            train_x, val_x, test_x = train_labeled, val_labeled, test_labeled
        model = fit_xgb(train_x, y_of(train_labeled), ablation_columns, xgb_params)
        val_scores = predict_xgb(model, val_x, ablation_columns)
        test_scores = predict_xgb(model, test_x, ablation_columns)
        results["ablations"][ablation_name] = {
            "validation": ranking_metrics(y_of(val_labeled), val_scores),
            "test": ranking_metrics(y_of(test_labeled), test_scores),
        }
        print("ablation", ablation_name, round(results["ablations"][ablation_name]["test"]["auprc"], 3))

    results["campaign_shift_xgboost"] = campaign_shift(frame, columns, xgb_params)
    print("campaign", results["campaign_shift_xgboost"])
    results["shap_xgboost_top"] = shap_summary(xgboost["model"], train_labeled, columns)
    plot_reliability(reliability, FIG / "reliability_test.png")
    pd.DataFrame(prediction_rows).to_csv(PROCESSED / "predictions_tabular.csv", index=False)
    (PROCESSED / "model_results.json").write_text(json.dumps(results, indent=2))
    print("wrote", PROCESSED / "model_results.json")


if __name__ == "__main__":
    main()
