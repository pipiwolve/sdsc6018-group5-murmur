"""Year shift for the selected CNN, plus the calibrated-score figure.

The 8-second crop and isotonic map are the choices already fixed on the random
split. Early stopping and the isotonic fit use a holdout taken from 2014 only.
Labeled 2015 children, including linked children kept on the 2015 visit, are
scored once.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIG = PROCESSED / "figures"
SEED = 6018


def load_cnn_module():
    spec = importlib.util.spec_from_file_location("cnn", ROOT / "scripts" / "05_train_cnn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plot_scores() -> dict:
    predictions = pd.read_csv(PROCESSED / "predictions_cnn.csv")
    decision = json.loads((PROCESSED / "cnn_decision.json").read_text())
    width = decision["test_abstention"]["half_width"]
    calibrated = predictions[predictions["stage"] == "calibrated"].copy()
    colors = {"Absent": "#3F72AF", "Present": "#48466D", "Unknown": "#C1A57B"}
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)
    summary = {}
    for axis, split in zip(axes, ("val", "test")):
        part = calibrated[calibrated["split"] == split]
        for label in ("Absent", "Present", "Unknown"):
            scores = part.loc[part["murmur"] == label, "score"].to_numpy()
            axis.hist(scores, bins=np.linspace(0, 1, 15), density=True, histtype="step", linewidth=1.8, color=colors[label], label=f"{label} (n={len(scores)})")
            inside = (scores > 0.5 - width) & (scores < 0.5 + width)
            summary.setdefault(split, {})[label] = {
                "n": int(len(scores)),
                "median": float(np.median(scores)) if len(scores) else None,
                "inside_band": int(inside.sum()),
            }
        axis.axvspan(0.5 - width, 0.5 + width, color="#112D4E", alpha=0.08)
        axis.set_title("Validation" if split == "val" else "Test")
        axis.set_xlabel("Calibrated score")
        axis.set_xlim(0, 1)
    axes[0].set_ylabel("Density")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "score_by_label.png", dpi=160)
    plt.close(fig)
    return {"half_width": width, "splits": summary}


def train_year_shift(cnn) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    features = pd.read_csv(PROCESSED / "feature_table.csv")
    labeled = features[features["supervised"].astype(bool)].copy()
    pool_2014 = labeled[labeled["campaign"] == "CC2014"]
    pool_2015 = labeled[labeled["campaign"] == "CC2015"]
    train_ids, val_ids = train_test_split(
        pool_2014["child_id"],
        test_size=0.20,
        stratify=pool_2014["murmur"],
        random_state=SEED,
    )
    train_ids, val_ids = set(train_ids.astype(int)), set(val_ids.astype(int))
    test_ids = set(pool_2015["child_id"].astype(int))
    needed = features[features["child_id"].isin(train_ids | val_ids | test_ids)].copy()
    locked = json.loads((PROCESSED / "cnn_results.json").read_text())
    n_samples = int(locked["clip_samples"])
    audio = cnn.load_feature_module()
    bank = cnn.mel_filterbank(4000, cnn.N_FFT, cnn.N_MELS)
    examples = cnn.build_examples(needed, audio, n_samples, bank)
    for row in examples:
        if row["child_id"] in train_ids:
            row["split"] = "train"
        elif row["child_id"] in val_ids:
            row["split"] = "val"
        else:
            row["split"] = "test"
    device = torch.device("cpu")
    model = cnn.SmallCnn(cnn.N_MELS).to(device)
    train_rows = [row for row in examples if row["split"] == "train"]
    y_train = np.array([1 if row["murmur"] == "Present" else 0 for row in train_rows])
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(int(y_train.sum()), 1)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(SEED)
    best_state, best_score, patience = None, -1.0, 0
    val_rows = [row for row in examples if row["split"] == "val"]
    for epoch in range(1, 21):
        model.train()
        order = torch.randperm(len(train_rows), generator=generator).tolist()
        for start in range(0, len(order), 64):
            batch = [train_rows[index] for index in order[start : start + 64]]
            inputs = torch.from_numpy(np.stack([row["mel"] for row in batch])).to(device)
            targets = torch.tensor([1.0 if row["murmur"] == "Present" else 0.0 for row in batch])
            weights = torch.tensor([1.0 / row["n_sites"] for row in batch])
            optimizer.zero_grad()
            loss = (loss_fn(model(inputs), targets) * weights).mean()
            loss.backward()
            optimizer.step()
        score = cnn.labeled_metrics(cnn.child_scores(model, val_rows, device))["auprc"]
        print(f"epoch {epoch} val AUPRC {score:.3f}", flush=True)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 4:
                break
    model.load_state_dict(best_state)
    raw = {split: cnn.child_scores(model, [row for row in examples if row["split"] == split], device) for split in ("val", "test")}
    val_labeled = raw["val"][raw["val"]["supervised"]]
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(val_labeled["score"].to_numpy(), (val_labeled["murmur"] == "Present").astype(int).to_numpy())
    calibrated = {split: cnn.apply_calibration("isotonic", isotonic, scores) for split, scores in raw.items()}
    return {
        "clip_samples": n_samples,
        "calibrator": "isotonic",
        "train_children": len(train_ids),
        "validation_children": len(val_ids),
        "test_children_2015": len(test_ids),
        "best_2014_validation_auprc": best_score,
        "year_2015_raw": cnn.labeled_metrics(raw["test"]),
        "year_2015_calibrated": cnn.labeled_metrics(calibrated["test"]),
    }


def main() -> None:
    placement = plot_scores()
    cnn = load_cnn_module()
    shift = train_year_shift(cnn)
    result = {"score_placement": placement, "campaign_cnn": shift}
    (PROCESSED / "campaign_cnn.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
