"""Train a small 1D CNN on fixed-length log-mel spectrograms.

Site probabilities are averaged for each child. The length, early stopping,
and calibration map use the training and validation splits only.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SEED = 6018
N_MELS = 32
N_FFT = 256
HOP = 128
MAX_SECONDS = 8.0


def load_feature_module():
    path = ROOT / "scripts" / "03_extract_features.py"
    spec = importlib.util.spec_from_file_location("features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    n_freq = n_fft // 2 + 1
    low, high = hz_to_mel(25.0), hz_to_mel(sample_rate / 2)
    points = mel_to_hz(np.linspace(low, high, n_mels + 2))
    bins = np.floor((n_fft + 1) * points / sample_rate).astype(int)
    bank = np.zeros((n_mels, n_freq), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = bins[index : index + 3]
        if center <= left or right <= center:
            continue
        bank[index, left:center] = (np.arange(left, center) - left) / (center - left)
        bank[index, center:right] = (right - np.arange(center, right)) / (right - center)
    return bank


def log_mel(samples: np.ndarray, sample_rate: int, n_samples: int, bank: np.ndarray) -> np.ndarray:
    target = n_samples
    if len(samples) >= target:
        start = (len(samples) - target) // 2
        clipped = samples[start : start + target]
    else:
        clipped = np.pad(samples, (0, target - len(samples)))
    window = np.hanning(N_FFT).astype(np.float32)
    specs = []
    for start in range(0, target - N_FFT + 1, HOP):
        frame = clipped[start : start + N_FFT] * window
        power = np.abs(np.fft.rfft(frame)) ** 2
        specs.append(bank @ power)
    mel = np.stack(specs, axis=1)
    return np.log(mel + 1e-6).astype(np.float32)


def choose_samples(features: pd.DataFrame) -> int:
    train = features[features["split"] == "train"]
    durations = []
    for site in ("av", "pv", "tv", "mv"):
        present = train.loc[train[f"{site}_missing"] == 0, f"{site}_duration"]
        durations.append(present)
    median = float(pd.concat(durations).median())
    seconds = float(np.clip(median, 4.0, MAX_SECONDS))
    return int(seconds * 4000)


def build_examples(features: pd.DataFrame, audio, n_samples: int, bank: np.ndarray) -> list[dict]:
    examples = []
    for child in features.itertuples(index=False):
        recordings = audio.parse_recordings(int(child.kept_patient_id))
        by_site = {site: [] for site in ("AV", "PV", "TV", "MV")}
        for recording in recordings:
            if recording["site"] in by_site:
                by_site[recording["site"]].append(recording)
        n_sites = sum(bool(items) for items in by_site.values())
        for site, items in by_site.items():
            if not items:
                continue
            chosen = max(items, key=lambda item: audio.duration_seconds(item["wav"]))
            _fs, samples = audio.read_wav(chosen["wav"])
            examples.append(
                {
                    "child_id": int(child.child_id),
                    "split": child.split,
                    "murmur": child.murmur,
                    "supervised": bool(child.supervised),
                    "n_sites": n_sites,
                    "mel": log_mel(samples, 4000, n_samples, bank),
                }
            )
    return examples


class SmallCnn(nn.Module):
    def __init__(self, n_mels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        hidden = self.net(batch).squeeze(-1)
        return self.fc(hidden).squeeze(-1)


def child_scores(model: nn.Module, rows: list[dict], device: torch.device) -> pd.DataFrame:
    model.eval()
    grouped = {}
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            chunk = rows[start : start + 64]
            batch = torch.from_numpy(np.stack([row["mel"] for row in chunk])).to(device)
            probs = torch.sigmoid(model(batch)).cpu().numpy()
            for row, prob in zip(chunk, probs):
                grouped.setdefault(row["child_id"], []).append(float(prob))
    records = []
    lookup = {row["child_id"]: row for row in rows}
    for child_id, probs in grouped.items():
        source = lookup[child_id]
        records.append(
            {
                "child_id": child_id,
                "split": source["split"],
                "murmur": source["murmur"],
                "supervised": source["supervised"],
                "score": float(np.mean(probs)),
            }
        )
    return pd.DataFrame(records)


def labeled_metrics(scores: pd.DataFrame) -> dict:
    part = scores[scores["supervised"]]
    y_true = (part["murmur"] == "Present").astype(int).to_numpy()
    return {
        "n": int(len(part)),
        "n_present": int(y_true.sum()),
        "auroc": float(roc_auc_score(y_true, part["score"])),
        "auprc": float(average_precision_score(y_true, part["score"])),
        "brier": float(brier_score_loss(y_true, part["score"])),
    }


def fit_calibration(val_scores: pd.DataFrame):
    part = val_scores[val_scores["supervised"]]
    y_true = (part["murmur"] == "Present").astype(int).to_numpy()
    platt = LogisticRegression(max_iter=1000)
    platt.fit(part["score"].to_numpy().reshape(-1, 1), y_true)
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(part["score"].to_numpy(), y_true)
    platt_brier = brier_score_loss(y_true, platt.predict_proba(part["score"].to_numpy().reshape(-1, 1))[:, 1])
    iso_brier = brier_score_loss(y_true, isotonic.predict(part["score"].to_numpy()))
    return ("platt", platt) if platt_brier <= iso_brier else ("isotonic", isotonic)


def apply_calibration(choice, model, scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    raw = out["score"].to_numpy()
    if choice == "platt":
        out["score"] = model.predict_proba(raw.reshape(-1, 1))[:, 1]
    else:
        out["score"] = model.predict(raw)
    return out


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    features = pd.read_csv(PROCESSED / "feature_table.csv")
    audio = load_feature_module()
    n_samples = choose_samples(features)
    bank = mel_filterbank(4000, N_FFT, N_MELS)
    print(f"clip samples {n_samples} ({n_samples / 4000:.2f} s)", flush=True)
    examples = build_examples(features, audio, n_samples, bank)
    print(f"site examples {len(examples)}", flush=True)
    device = torch.device("cpu")
    model = SmallCnn(N_MELS).to(device)
    train_rows = [row for row in examples if row["split"] == "train" and row["supervised"]]
    y_train = np.array([1 if row["murmur"] == "Present" else 0 for row in train_rows])
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(SEED)
    best_state, best_score, patience = None, -1.0, 0
    val_rows = [row for row in examples if row["split"] == "val"]
    for epoch in range(1, 21):
        model.train()
        order = torch.randperm(len(train_rows), generator=generator).tolist()
        total = 0.0
        for start in range(0, len(order), 64):
            batch_index = order[start : start + 64]
            batch = [train_rows[index] for index in batch_index]
            inputs = torch.from_numpy(np.stack([row["mel"] for row in batch])).to(device)
            targets = torch.tensor([1.0 if row["murmur"] == "Present" else 0.0 for row in batch])
            weights = torch.tensor([1.0 / row["n_sites"] for row in batch])
            optimizer.zero_grad()
            loss = (loss_fn(model(inputs), targets) * weights).mean()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)
        val_scores = child_scores(model, val_rows, device)
        score = labeled_metrics(val_scores)["auprc"]
        print(f"epoch {epoch} loss {total / len(train_rows):.4f} val AUPRC {score:.3f}", flush=True)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 4:
                break
    model.load_state_dict(best_state)
    raw = {split: child_scores(model, [row for row in examples if row["split"] == split], device) for split in ("val", "test")}
    choice, calibrator = fit_calibration(raw["val"])
    calibrated = {split: apply_calibration(choice, calibrator, scores) for split, scores in raw.items()}
    results = {
        "clip_samples": n_samples,
        "hop": HOP,
        "n_mels": N_MELS,
        "seconds_cap": MAX_SECONDS,
        "best_validation_auprc_raw": best_score,
        "calibrator": choice,
        "validation_raw": labeled_metrics(raw["val"]),
        "validation_calibrated": labeled_metrics(calibrated["val"]),
        "test_raw": labeled_metrics(raw["test"]),
        "test_calibrated": labeled_metrics(calibrated["test"]),
    }
    (PROCESSED / "cnn_results.json").write_text(json.dumps(results, indent=2))
    pd.concat([raw["val"].assign(stage="raw"), raw["test"].assign(stage="raw"), calibrated["val"].assign(stage="calibrated"), calibrated["test"].assign(stage="calibrated")]).to_csv(
        PROCESSED / "predictions_cnn.csv", index=False
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
