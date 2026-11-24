"""Recompute the report's review analyses from locked child-level scores.

Run on the archived predictions. Use --output-dir to verify results without
changing the archive. Pass --audio-dir for the four CirCor 1.0.3 WAV files if
they are stored outside the standard raw training_data directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import stft
from scipy.stats import binomtest
from scipy.io import wavfile
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ("logistic", "xgboost", "cnn")
SITES = ("AV", "PV", "TV", "MV")
CASE_SITES = {84969: "MV", 85286: "TV", 50258: "TV", 85269: "AV"}
BOOTSTRAPS = 10000
SEED = 6018


def read_inputs(data_dir):
    cnn_long = pd.read_csv(data_dir / "predictions_cnn.csv")
    cnn = cnn_long.pivot(index=["child_id", "split", "murmur", "supervised"],
                         columns="stage", values="score").reset_index()
    cnn = cnn.rename(columns={"raw": "cnn_raw", "calibrated": "cnn_cal"})
    subjects = pd.read_csv(data_dir / "subject_table.csv")
    meta = ["child_id", "kept_patient_id", "split", "murmur", "age", "sex",
            "locations", "n_standard_sites"]
    frame = cnn.merge(subjects[meta], on=["child_id", "split", "murmur"],
                      validate="one_to_one")
    tabular = pd.read_csv(data_dir / "predictions_tabular.csv")
    for model in MODELS[:2]:
        scores = tabular.loc[tabular["model"] == model].drop(columns="model")
        scores = scores.rename(columns={"raw_score": f"{model}_raw",
                                        "calibrated_score": f"{model}_cal"})
        frame = frame.merge(scores, on=["child_id", "split", "murmur"],
                            validate="one_to_one")
    score_columns = [f"{m}_{stage}" for m in MODELS for stage in ("raw", "cal")]
    if len(frame) != len(cnn) or frame[score_columns].isna().any().any():
        raise ValueError("Scores and subject metadata must match for every child.")
    if not (frame["supervised"] == frame["murmur"].isin(["Present", "Absent"])).all():
        raise ValueError("Unexpected supervised labels.")
    frame["y"] = frame["murmur"].eq("Present").astype(int)
    frame["age"] = frame["age"].fillna("Missing")
    return frame.sort_values(["split", "child_id"])


def record(point, replicates):
    valid = np.asarray(replicates, dtype=float)
    valid = valid[np.isfinite(valid)]
    return {"point": float(point),
            "ci": np.quantile(valid, [0.025, 0.975]).tolist() if valid.size else None,
            "valid_replicates": int(valid.size)}


def calibration_sensitivity(val, test):
    rows = {}
    for model in MODELS:
        raw = f"{model}_raw"
        val_scores = val[raw].to_numpy()
        test_scores = test[raw].to_numpy()
        y_val = val["y"].to_numpy()
        y_test = test["y"].to_numpy()
        platt = LogisticRegression(max_iter=1000).fit(val_scores[:, None], y_val)
        iso = IsotonicRegression(out_of_bounds="clip").fit(val_scores, y_val)
        rows[model] = {}
        for split, scores, labels in (("val", val_scores, y_val), ("test", test_scores, y_test)):
            raw_brier = np.mean((scores - labels) ** 2)
            platt_brier = np.mean((platt.predict_proba(scores[:, None])[:, 1] - labels) ** 2)
            iso_scores = iso.predict(scores)
            expected = (val if split == "val" else test)[f"{model}_cal"]
            np.testing.assert_allclose(iso_scores, expected, atol=1e-6, rtol=0)
            rows[model][split] = {"raw": float(raw_brier), "platt": float(platt_brier),
                                  "isotonic": float(np.mean((expected.to_numpy() - labels) ** 2))}
    return rows


def model_metrics(test, draw):
    y = test["y"].to_numpy()
    raw = test[[f"{m}_raw" for m in MODELS]].to_numpy()
    cal = test[[f"{m}_cal" for m in MODELS]].to_numpy()
    raw_loss = (raw - y[:, None]) ** 2
    cal_loss = (cal - y[:, None]) ** 2
    delta_boot = (cal_loss - raw_loss)[draw].mean(axis=1)
    rank = np.array([(roc_auc_score(y, p), average_precision_score(y, p)) for p in raw.T])
    rank_boot = np.full((len(draw), len(MODELS), 2), np.nan)
    for b, ix in enumerate(draw):
        labels = y[ix]
        if labels.min() == labels.max():
            continue
        for m, p in enumerate(raw[ix].T):
            rank_boot[b, m] = (roc_auc_score(labels, p), average_precision_score(labels, p))
    rows = {}
    for m, name in enumerate(MODELS):
        rows[name] = {
            "auroc": record(rank[m, 0], rank_boot[:, m, 0]),
            "auprc": record(rank[m, 1], rank_boot[:, m, 1]),
            "raw_brier": float(raw_loss[:, m].mean()),
            "cal_brier": float(cal_loss[:, m].mean()),
            "delta_brier": record((cal_loss - raw_loss)[:, m].mean(), delta_boot[:, m]),
        }
    comparisons = {}
    for m, name in enumerate(MODELS[1:], 1):
        comparisons[name] = {
            name_: record(rank[m, j] - rank[0, j], rank_boot[:, m, j] - rank_boot[:, 0, j])
            for j, name_ in enumerate(("auroc", "auprc"))
        }
    return rows, comparisons


def band_masks(scores, width):
    positive = scores >= 0.5 + width
    withheld = (scores > 0.5 - width) & ~positive
    return positive, withheld


def ratio_boot(num, denom, draw):
    draws_num = num[draw].sum(axis=1)
    draws_denom = denom[draw].sum(axis=1)
    values = np.divide(draws_num, draws_denom, out=np.full(len(draw), np.nan),
                       where=draws_denom > 0)
    return {"numerator": int(num.sum()), "denominator": int(denom.sum()),
            **record(num.sum() / denom.sum(), values)}


def abstention(test, unknown, width, draw_test, draw_unknown):
    positive, withheld = band_masks(test["cnn_cal"].to_numpy(), width)
    _, withheld_unknown = band_masks(unknown["cnn_cal"].to_numpy(), width)
    y = test["y"].to_numpy(dtype=bool)
    answered = ~withheld
    tp = positive & y
    fp = positive & ~y
    fn = answered & ~positive & y
    combined_draw = np.column_stack([draw_test, len(test) + draw_unknown])
    unknown_withheld = np.r_[np.zeros(len(test), dtype=bool), withheld_unknown]
    unknown_ci = binomtest(int(withheld_unknown.sum()), len(unknown)).proportion_ci()
    return {
        "coverage": ratio_boot(answered, np.ones_like(answered), draw_test),
        "ppv": ratio_boot(tp, positive, draw_test),
        "within_answered_recall": ratio_boot(tp, answered & y, draw_test),
        "overall_recall": ratio_boot(tp, y, draw_test),
        "unknown_recall": ratio_boot(withheld_unknown, np.ones_like(withheld_unknown), draw_unknown),
        "unknown_share": ratio_boot(unknown_withheld, np.r_[withheld, withheld_unknown], combined_draw),
        "exact_unknown_recall_ci": [unknown_ci.low, unknown_ci.high],
        "tp": int(tp.sum()), "fp": int(fp.sum()), "fn_answered": int(fn.sum()),
        "present_withheld": int((withheld & y).sum()),
        "low": 0.5 - width, "high": 0.5 + width,
    }


def subgroup_metrics(test, rng):
    groups = [(key, name, group) for key in ("age", "sex")
              for name, group in test.groupby(key, sort=True)]
    groups += [("age × sex", f"Child / {name}", group)
               for name, group in test[test["age"].eq("Child")].groupby("sex", sort=True)]
    rows = []
    for key, name, group in groups:
        y = group["y"].to_numpy()
        enough = len(y) >= 20 and min(y.sum(), len(y) - y.sum()) >= 5
        ix = rng.integers(len(y), size=(BOOTSTRAPS, len(y))) if enough else None
        row = {"dimension": key, "group": name, "n": len(y),
               "n_present": int(y.sum()), "observed": float(y.mean()), "models": {}}
        for m in MODELS:
            p = group[f"{m}_cal"].to_numpy()
            residual = p - y
            row["models"][m] = {
                "mean": float(p.mean()), "brier": float(np.mean(residual ** 2)),
                "gap": (record(residual.mean(), residual[ix].mean(axis=1))
                        if enough else {"point": float(residual.mean()), "ci": None}),
            }
        rows.append(row)
    return rows


def error_review(test, data_dir, thresholds):
    columns = [f"{site.lower()}_cycle_frac" for site in SITES]
    features = pd.read_csv(data_dir / "feature_table.csv", usecols=["child_id", *columns])
    full = test.merge(features, on="child_id", how="left", validate="one_to_one")
    full["mean_coverage"] = full[columns].mean(axis=1)
    for m in MODELS:
        full[f"{m}_pred"] = full[f"{m}_cal"].ge(thresholds[m]).astype(int)
    full["error"] = np.select(
        [full["cnn_pred"].gt(full["y"]), full["cnn_pred"].lt(full["y"])],
        ["FP", "FN"], default="correct")
    groups = {
        "one_standard_site": full["n_standard_sites"].eq(1),
        "multiple_standard_sites": full["n_standard_sites"].ge(2),
        "infant": full["age"].eq("Infant"),
        "known_age_except_infant": ~full["age"].isin(["Infant", "Missing"]),
        "mean_segmentation_below_25pct": full["mean_coverage"].lt(0.25),
        "mean_segmentation_at_least_25pct": full["mean_coverage"].ge(0.25),
    }
    counts = {}
    for name, mask in groups.items():
        group = full[mask]
        counts[name] = {"n": len(group), "n_absent": int(group["y"].eq(0).sum()),
                        "n_present": int(group["y"].sum()),
                        "fp": int(group["error"].eq("FP").sum()),
                        "fn": int(group["error"].eq("FN").sum())}
    cases = []
    for child, site in CASE_SITES.items():
        row = full.loc[full["child_id"].eq(child)].iloc[0]
        locations = tuple(site_ for site_ in SITES if site_ in row.locations.split("+"))
        if row.error == "correct" or site not in locations:
            raise ValueError(f"Invalid illustration case {child}/{site}")
        cases.append({
            "child_id": child, "patient_id": int(row.kept_patient_id), "error": row.error,
            "age": row.age, "murmur": row.murmur, "spectrum_site": site,
            "site_coverage": {s: float(row[f"{s.lower()}_cycle_frac"]) for s in locations},
            "models": {m: {"raw": float(row[f"{m}_raw"]),
                           "calibrated": float(row[f"{m}_cal"]),
                           "prediction": int(row[f"{m}_pred"])} for m in MODELS},
        })
    overlap = {}
    for m in MODELS[:2]:
        misses = full[f"{m}_pred"].ne(full["y"])
        overlap[m] = {err: int((misses & full["error"].eq(err)).sum()) for err in ("FP", "FN")}
    return {"cases": cases, "groups": counts, "overlap": overlap,
            "n_no_segmentation_summary": int(full["mean_coverage"].isna().sum()),
            "mean_sites": full.groupby("error")["n_standard_sites"].mean().to_dict()}


def make_figures(frame, labeled, result, fig_dir, audio_dir):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                          "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    blue, gray = "#215375", "#717b83"
    fig, ax = plt.subplots(figsize=(4.45, 2.62), layout="constrained")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.9, color=gray)
    scores, y = labeled["cnn_cal"].to_numpy(), labeled["y"].to_numpy()
    for left, right in zip(np.arange(8) / 8, np.arange(1, 9) / 8):
        group = (scores >= left) & ((scores < right) if right < 1 else (scores <= right))
        if not group.any():
            continue
        x, rate, n = scores[group].mean(), y[group].mean(), int(group.sum())
        ax.scatter(x, rate, color=blue, zorder=3)
        ax.annotate(f"n = {n}", (x, rate), xytext=(-8 if x > 0.9 else 9, 1),
                    textcoords="offset points", va="center", ha="right" if x > 0.9 else "left")
    ax.set(xlim=(-0.035, 1.035), ylim=(-0.035, 1.035), xlabel="Mean calibrated probability",
           ylabel="Observed Present fraction")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    fig.savefig(fig_dir / "review_reliability.png", dpi=250)
    plt.close(fig)

    lo, hi = result["abstention"]["low"], result["abstention"]["high"]
    fig, axes = plt.subplots(1, 2, figsize=(6.45, 2.25), sharey=True, layout="constrained")
    colors = {"Present": "#215375", "Absent": "#526b59", "Unknown": "#a45133"}
    for ax, (split, title) in zip(axes, (("val", "Validation"), ("test", "Test"))):
        part = frame[frame["split"].eq(split)]
        ax.axvspan(lo, hi, color="#e3e8ed", zorder=0)
        for label, color in colors.items():
            score = np.sort(part.loc[part["murmur"].eq(label), "cnn_cal"].to_numpy())
            levels, counts = np.unique(score, return_counts=True)
            x = np.r_[-0.025, levels, 1.025]
            y = np.r_[0, counts.cumsum() / len(score), 1]
            ax.step(x, y, where="post", color=color, lw=1.4, label=f"{label} ({len(score)})")
        ax.set(title=title, xlim=(-0.02, 1.02), ylim=(0, 1.07), xlabel="Calibrated probability")
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    axes[0].set_ylabel("Cumulative fraction")
    fig.savefig(fig_dir / "review_score_ecdf.png", dpi=250)
    plt.close(fig)

    case_paths = [(case, audio_dir / f"{case['patient_id']}_{case['spectrum_site']}.wav")
                  for case in result["errors"]["cases"]]
    if not all(path.is_file() for _, path in case_paths):
        return False
    views = []
    for case, path in case_paths:
        fs, samples = wavfile.read(path)
        if fs != 4000 or samples.ndim != 1 or samples.dtype != np.int16:
            raise ValueError(f"Expected a 4 kHz mono PCM recording in {path}")
        start = max(0, (len(samples) - fs * 8) // 2)
        clip = samples[start:start + fs * 8].astype(float) / 32768.0
        if len(clip) < fs * 8:
            clip = np.pad(clip, (0, fs * 8 - len(clip)))
        f, t, z = stft(clip, fs, window="hann", nperseg=256, noverlap=128,
                      boundary=None, padded=False)
        keep = (f >= 25) & (f <= 800)
        views.append((case, f[keep], t, 20 * np.log10(np.maximum(np.abs(z[keep]), 1e-9))))
    high = max(np.max(db) for _, _, _, db in views)
    fig, axes = plt.subplots(2, 2, figsize=(6.45, 3.4), sharex=True, sharey=True, layout="constrained")
    for ax, (case, f, t, db) in zip(axes.flat, views):
        mesh = ax.pcolormesh(t, f, db, cmap="magma", shading="auto", vmin=high-60, vmax=high)
        ax.set_title(f"{case['error']}  {case['child_id']}  {case['spectrum_site']}", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.set(xlim=(0, 8))
    fig.supxlabel("Time in analysis window (s)", fontsize=10)
    fig.supylabel("Frequency (Hz)", fontsize=10)
    fig.colorbar(mesh, ax=axes, label="STFT magnitude (dBFS)", shrink=0.93)
    fig.savefig(fig_dir / "review_error_spectra.png", dpi=250)
    plt.close(fig)
    return True


def run(data_dir, output_dir, audio_dir):
    results = json.loads((data_dir / "model_results.json").read_text())
    decision = json.loads((data_dir / "cnn_decision.json").read_text())
    frame = read_inputs(data_dir)
    test = frame[(frame["split"] == "test") & frame["supervised"]]
    unknown = frame[(frame["split"] == "test") & ~frame["supervised"]]
    val = frame[(frame["split"] == "val") & frame["supervised"]]
    rng = np.random.default_rng(SEED)
    draw_test = rng.integers(len(test), size=(BOOTSTRAPS, len(test)))
    draw_unknown = rng.integers(len(unknown), size=(BOOTSTRAPS, len(unknown)))
    models, vs_logistic = model_metrics(test, draw_test)
    result = {
        "bootstrap_seed": SEED, "bootstrap_replicates": BOOTSTRAPS,
        "test_n": len(test), "test_present": int(test["y"].sum()), "test_unknown": len(unknown),
        "models": models, "vs_logistic": vs_logistic,
        "calibrator_sensitivity": calibration_sensitivity(val, test),
        "abstention": abstention(test, unknown, decision["test_abstention"]["half_width"],
                                 draw_test, draw_unknown),
        "subgroups": subgroup_metrics(test, rng),
        "errors": error_review(test, data_dir, {**{m: results["models"][m]["threshold"]
                                                    for m in MODELS[:2]},
                                               "cnn": decision["threshold"]}),
    }
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    result["case_spectra_generated"] = make_figures(frame, test, result, fig_dir, audio_dir)
    (output_dir / "review_analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audio-dir", type=Path,
                        default=ROOT / "data" / "raw" / "circor-heart-sound-1.0.3" / "training_data")
    args = parser.parse_args()
    result = run(args.data_dir, args.output_dir or args.data_dir, args.audio_dir)
    print(f"Computed review analyses for {result['test_n']} labeled test children.")
    if not result["case_spectra_generated"]:
        print("The four CirCor recordings are required to generate the case spectra.")


if __name__ == "__main__":
    main()
