"""Extract one feature row per child from the kept CirCor visit.

Rules are locked in proposal/项目执行标准.md.
Height and weight stay missing here. Train-only imputation belongs to modeling.
Raw audio is not modified.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "raw" / "circor-heart-sound-1.0.3" / "training_data"
SUBJECTS = ROOT / "data" / "processed" / "subject_table.csv"
OUT = ROOT / "data" / "processed" / "feature_table.csv"
COLUMNS_OUT = ROOT / "data" / "processed" / "feature_columns.txt"

SITES = ("AV", "PV", "TV", "MV")
STATE = {"1": "s1", "2": "sys", "3": "s2", "4": "dia"}
BANDS = ((25.0, 120.0), (120.0, 400.0), (400.0, 800.0))
AGE_LEVELS = ("Neonate", "Infant", "Adolescent", "Missing")  # Child is the reference
META_COLUMNS = [
    "child_id",
    "kept_patient_id",
    "split",
    "campaign",
    "murmur",
    "supervised",
]


def parse_recordings(patient_id: int) -> list:
    recordings = []
    for line in (TRAIN / f"{patient_id}.txt").read_text().splitlines()[1:]:
        if not line.strip() or line.startswith("#"):
            continue
        site, _hea, wav, tsv = line.split()
        recordings.append({"site": site, "wav": wav, "tsv": tsv})
    return recordings


def duration_seconds(wav_name: str) -> float:
    hea = (TRAIN / wav_name).with_suffix(".hea")
    _name, _nsig, fs, n_samples = hea.read_text().splitlines()[0].split()[:4]
    return int(n_samples) / float(fs)


def read_wav(wav_name: str) -> tuple:
    import wave

    with wave.open(str(TRAIN / wav_name), "rb") as handle:
        fs = handle.getframerate()
        width = handle.getsampwidth()
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())
    if width != 2 or channels != 1 or fs != 4000:
        raise ValueError(f"{wav_name} is not 4000 Hz mono 16-bit")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return fs, samples


def intervals_from_tsv(tsv_name: str, duration: float, fs: float) -> list:
    rows = []
    for line in (TRAIN / tsv_name).read_text().splitlines():
        if not line.strip():
            continue
        start, end, label = line.split()[:3]
        rows.append((float(start), float(end), label))
    scale = 1.0
    if rows and max(end for _, end, _ in rows) > max(duration, 1.0) * 2:
        scale = 1.0 / fs
    kept = []
    for start, end, label in rows:
        start *= scale
        end *= scale
        if end <= start or label not in STATE:
            continue
        kept.append((start, end, label))
    return kept


def timing_summary(intervals: list) -> dict:
    grouped = {name: [] for name in STATE.values()}
    s1_starts = []
    for start, end, label in intervals:
        grouped[STATE[label]].append(end - start)
        if label == "1":
            s1_starts.append(start)
    summary = {f"{name}_dur": (float(np.mean(values)) if values else np.nan) for name, values in grouped.items()}
    summary["n_s1"] = len(s1_starts)
    if len(s1_starts) >= 2:
        summary["rr"] = float(np.mean(np.diff(sorted(s1_starts))))
    else:
        summary["rr"] = np.nan
    return summary


def spectrum_summary(pieces: list, fs: int) -> dict:
    weight = 0.0
    rms_acc = 0.0
    centroid_acc = 0.0
    band_power = np.zeros(len(BANDS), dtype=np.float64)
    total_power = 0.0
    for samples in pieces:
        if len(samples) < 32:
            continue
        centered = samples - np.mean(samples)
        window = np.hanning(len(centered)).astype(np.float32)
        power = np.abs(np.fft.rfft(centered * window)) ** 2
        freqs = np.fft.rfftfreq(len(centered), d=1.0 / fs)
        total = float(power.sum())
        if total <= 0:
            continue
        centroid = float(np.sum(freqs * power) / total)
        rms = float(np.sqrt(np.mean(centered**2)))
        this_weight = float(len(centered))
        weight += this_weight
        rms_acc += rms * this_weight
        centroid_acc += centroid * this_weight
        total_power += total
        for index, (low, high) in enumerate(BANDS):
            band_power[index] += float(power[(freqs >= low) & (freqs < high)].sum())
    if weight == 0 or total_power == 0:
        return {"rms": np.nan, "centroid": np.nan, "band_low": np.nan, "band_mid": np.nan, "band_high": np.nan}
    return {
        "rms": rms_acc / weight,
        "centroid": centroid_acc / weight,
        "band_low": band_power[0] / total_power,
        "band_mid": band_power[1] / total_power,
        "band_high": band_power[2] / total_power,
    }


def pieces_for(samples: np.ndarray, fs: int, intervals: list) -> list:
    pieces = []
    for start, end, _label in intervals:
        left = max(0, int(start * fs))
        right = min(len(samples), int(end * fs))
        if right - left >= 32:
            pieces.append(samples[left:right])
    return pieces


def empty_site() -> dict:
    values = {
        "missing": 1,
        "duplicate": 0,
        "duration": np.nan,
        "cycle_frac": np.nan,
        "n_s1": np.nan,
        "s1_dur": np.nan,
        "sys_dur": np.nan,
        "s2_dur": np.nan,
        "dia_dur": np.nan,
        "rr": np.nan,
        "rms": np.nan,
        "centroid": np.nan,
        "band_low": np.nan,
        "band_mid": np.nan,
        "band_high": np.nan,
    }
    return values


def site_features(recordings: list) -> dict:
    if not recordings:
        return empty_site()
    chosen = max(recordings, key=lambda item: duration_seconds(item["wav"]))
    duration = duration_seconds(chosen["wav"])
    fs, samples = read_wav(chosen["wav"])
    intervals = intervals_from_tsv(chosen["tsv"], duration, fs)
    cycle = sum(end - start for start, end, _label in intervals)
    pieces = pieces_for(samples, fs, intervals)
    spectral_source = pieces if pieces else [samples]
    features = {
        "missing": 0,
        "duplicate": int(len(recordings) > 1),
        "duration": duration,
        "cycle_frac": cycle / duration if duration else np.nan,
        **timing_summary(intervals),
        **spectrum_summary(spectral_source, fs),
    }
    if not pieces:
        for name in ("n_s1", "s1_dur", "sys_dur", "s2_dur", "dia_dur", "rr", "cycle_frac"):
            features[name] = np.nan if name != "cycle_frac" else 0.0
    return features


def demographic_features(row: pd.Series) -> dict:
    age = row["age"] if pd.notna(row["age"]) else "Missing"
    if age == "Child":
        age_flags = {f"age_{level}": 0 for level in AGE_LEVELS}
    elif age in AGE_LEVELS:
        age_flags = {f"age_{level}": int(level == age) for level in AGE_LEVELS}
    else:
        raise ValueError(f"Unexpected age {age} for child {row['child_id']}")
    if row["sex"] not in {"Female", "Male"}:
        raise ValueError(f"Unexpected sex {row['sex']} for child {row['child_id']}")
    return {
        **age_flags,
        "sex_Male": int(row["sex"] == "Male"),
        "height": row["height"],
        "weight": row["weight"],
        "height_missing": int(pd.isna(row["height"])),
        "weight_missing": int(pd.isna(row["weight"])),
    }


def feature_names(frame: pd.DataFrame) -> list:
    return [column for column in frame.columns if column not in META_COLUMNS]


def build() -> pd.DataFrame:
    children = pd.read_csv(SUBJECTS)
    rows = []
    for index, child in children.iterrows():
        recordings = parse_recordings(int(child["kept_patient_id"]))
        by_site = {site: [] for site in (*SITES, "Phc")}
        for recording in recordings:
            if recording["site"] not in by_site:
                raise ValueError(f"Unexpected site {recording['site']}")
            by_site[recording["site"]].append(recording)
        record = {column: child[column] for column in META_COLUMNS}
        record.update(demographic_features(child))
        record["n_phc"] = len(by_site["Phc"])
        for site in SITES:
            for key, value in site_features(by_site[site]).items():
                record[f"{site.lower()}_{key}"] = value
        rows.append(record)
        if (index + 1) % 200 == 0:
            print(f"extracted {index + 1}/{len(children)}", flush=True)
    return pd.DataFrame(rows)


def check(frame: pd.DataFrame, children: pd.DataFrame) -> None:
    if len(frame) != 872 or frame["child_id"].nunique() != 872:
        raise AssertionError("feature table must have 872 children")
    merged = frame.merge(
        children[["child_id", "split", "has_AV", "has_PV", "has_TV", "has_MV", "murmur"]],
        on="child_id",
        suffixes=("", "_subject"),
    )
    if not (merged["split"] == merged["split_subject"]).all():
        raise AssertionError("split column drifted from the subject table")
    for site, flag in (("av", "has_AV"), ("pv", "has_PV"), ("tv", "has_TV"), ("mv", "has_MV")):
        if not (merged[f"{site}_missing"] == (1 - merged[flag])).all():
            raise AssertionError(f"{site} missing flag does not match the subject table")
    forbidden = [column for column in frame.columns if column.lower().startswith("desc") or column in {"outcome", "pregnancy", "dropped_murmur"}]
    if forbidden:
        raise AssertionError(f"forbidden columns: {forbidden}")
    if frame["height"].notna().all() or frame["weight"].notna().all():
        raise AssertionError("height or weight was imputed")
    row = frame.loc[frame["child_id"] == 50782].iloc[0]
    if not (row["mv_missing"] == 0 and row["mv_duplicate"] == 1 and pd.isna(row["mv_s1_dur"]) and pd.notna(row["mv_rms"])):
        raise AssertionError("50782 mitral features do not match the empty-segmentation rule")
    present = frame.loc[frame["child_id"] == 2530].iloc[0]
    if present[["av_missing", "pv_missing", "tv_missing", "mv_missing"]].sum() != 0:
        raise AssertionError("2530 should have four sites")
    if not (present["av_s1_dur"] > 0 and present["av_rms"] > 0):
        raise AssertionError("2530 aortic timing or spectrum is empty")


def main() -> None:
    children = pd.read_csv(SUBJECTS)
    frame = build()
    check(frame, children)
    frame.to_csv(OUT, index=False)
    names = feature_names(frame)
    COLUMNS_OUT.write_text("\n".join(names) + "\n")
    print(f"wrote {OUT}")
    print(f"rows {len(frame)} features {len(names)}")
    print(
        "site missing",
        {site: int(frame[f"{site}_missing"].sum()) for site in ("av", "pv", "tv", "mv")},
    )


if __name__ == "__main__":
    main()
