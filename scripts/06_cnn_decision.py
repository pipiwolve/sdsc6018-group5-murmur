"""Choose the CNN threshold and abstention band on validation scores only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def half_width(scores: np.ndarray, coverage: float = 0.80) -> float:
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


def f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 37))):
        f1 = f1_score(y_true, scores >= threshold, zero_division=0)
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), float(f1)
    return best_threshold


def counts(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    fp = int(np.sum((predicted == 1) & (y_true == 0)))
    fn = int(np.sum((predicted == 0) & (y_true == 1)))
    tp = int(np.sum((predicted == 1) & (y_true == 1)))
    tn = int(np.sum((predicted == 0) & (y_true == 0)))
    n = max(len(y_true), 1)
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "fp_per_100": 100 * fp / n, "fn_per_100": 100 * fn / n}


def abstain(frame: pd.DataFrame, width: float) -> dict:
    labeled = frame["supervised"].to_numpy()
    unknown = (frame["murmur"] == "Unknown").to_numpy()
    scores = frame["score"].to_numpy()
    y_true = (frame["murmur"] == "Present").astype(int).to_numpy()
    answered = (scores <= 0.5 - width) | (scores >= 0.5 + width)
    answered_labeled = answered & labeled
    predicted_pos = scores >= 0.5 + width
    tp = np.sum(predicted_pos & answered_labeled & (y_true == 1))
    fp = np.sum(predicted_pos & answered_labeled & (y_true == 0))
    fn = np.sum((~predicted_pos) & answered_labeled & (y_true == 1))
    abstained = ~answered
    return {
        "half_width": width,
        "labeled_coverage": float(np.mean(answered[labeled])),
        "precision_answered": float(tp / (tp + fp)) if (tp + fp) else None,
        "recall_answered": float(tp / (tp + fn)) if (tp + fn) else None,
        "unknown_among_abstained": float(np.mean(unknown[abstained])) if abstained.any() else None,
        "unknown_abstained": float(np.mean(abstained[unknown])) if unknown.any() else None,
        "n_abstained": int(abstained.sum()),
    }


def main() -> None:
    predictions = pd.read_csv(PROCESSED / "predictions_cnn.csv")
    subjects = pd.read_csv(PROCESSED / "subject_table.csv")
    calibrated = predictions[predictions["stage"] == "calibrated"].copy()
    raw = predictions[predictions["stage"] == "raw"].copy()
    val = calibrated[calibrated["split"] == "val"]
    val_labeled = val[val["supervised"]]
    y_val = (val_labeled["murmur"] == "Present").astype(int).to_numpy()
    threshold = f1_threshold(y_val, val_labeled["score"].to_numpy())
    width = half_width(val_labeled["score"].to_numpy())
    test = calibrated[calibrated["split"] == "test"].merge(
        subjects[["child_id", "age", "sex", "n_standard_sites", "has_AV", "has_PV", "has_TV", "has_MV"]],
        on="child_id",
    )
    test_labeled = test[test["supervised"]]
    y_test = (test_labeled["murmur"] == "Present").astype(int).to_numpy()
    test_raw = raw[(raw["split"] == "test") & (raw["supervised"])].merge(subjects[["child_id", "age", "sex"]], on="child_id")
    rows = []
    test_raw["age_band"] = test_raw["age"].fillna("Missing")
    test_raw["y"] = (test_raw["murmur"] == "Present").astype(int)
    for keys, part in test_raw.groupby(["age_band", "sex"]):
        y_true = part["y"].to_numpy()
        row = {"age": keys[0], "sex": keys[1], "n": int(len(part)), "n_present": int(y_true.sum())}
        if row["n"] >= 20 and y_true.sum() > 0 and (y_true == 0).sum() > 0:
            from sklearn.metrics import roc_auc_score

            row["auroc"] = float(roc_auc_score(y_true, part["score"]))
            row["small_cell"] = False
        else:
            row["auroc"] = None
            row["small_cell"] = True
        rows.append(row)
    labeled = test_labeled.copy()
    labeled["predicted"] = (labeled["score"] >= threshold).astype(int)
    labeled["y"] = y_test
    labeled["error"] = np.where((labeled["predicted"] == 1) & (labeled["y"] == 0), "false_positive", np.where((labeled["predicted"] == 0) & (labeled["y"] == 1), "false_negative", "correct"))
    pieces = []
    for kind, ascending in (("false_positive", False), ("false_negative", True)):
        part = labeled[labeled["error"] == kind].sort_values("score", ascending=ascending)
        pieces.append(part.head(30))
    pd.concat(pieces).to_csv(PROCESSED / "error_cases_cnn.csv", index=False)
    summary = {}
    for kind, part in labeled.groupby("error"):
        summary[kind] = {"n": int(len(part)), "mean_sites": float(part["n_standard_sites"].mean()), "age": part["age"].fillna("Missing").value_counts().to_dict()}
    y_bin = (test["murmur"] == "Present").astype(int).to_numpy()
    labeled_mask = test["supervised"].to_numpy()
    bins = []
    edges = np.linspace(0, 1, 9)
    scores = test.loc[labeled_mask, "score"].to_numpy()
    y_scores = (test.loc[labeled_mask, "murmur"] == "Present").astype(int).to_numpy()
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (scores >= left) & (scores <= right if right == 1 else scores < right)
        if mask.sum() == 0:
            continue
        bins.append({"mean_score": float(scores[mask].mean()), "observed": float(y_scores[mask].mean()), "n": int(mask.sum())})
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot([0, 1], [0, 1], color="#112D4E", linewidth=1)
    ax.plot([row["mean_score"] for row in bins], [row["observed"] for row in bins], marker="o", color="#3F72AF")
    ax.set_xlabel("Mean calibrated score")
    ax.set_ylabel("Observed Present rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(PROCESSED / "figures" / "reliability_cnn_test.png", dpi=160)
    plt.close(fig)
    result = {
        "threshold": threshold,
        "test_operating_point": counts(y_test, test_labeled["score"].to_numpy(), threshold),
        "validation_abstention": abstain(val, width),
        "test_abstention": abstain(test, width),
        "subgroups_raw_test": rows,
        "error_summary": summary,
        "reliability": bins,
    }
    (PROCESSED / "cnn_decision.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
