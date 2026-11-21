"""Build one row per child and a fixed subject split.

Rules are locked in proposal/项目执行标准.md.
The 2015 visit is kept when Additional ID links two campaigns.
The dropped visit is not a second row and does not relabel the kept visit.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CSV_CANDIDATES = [
    ROOT / "data" / "raw" / "circor-heart-sound-1.0.3" / "training_data.csv",
    ROOT / "data" / "raw" / "training_data.csv",
]
OUT_DIR = ROOT / "data" / "processed"
SEED = 6018
SITES = ["AV", "PV", "TV", "MV", "Phc"]


def locate_csv() -> Path:
    for path in CSV_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("training_data.csv not found under data/raw/")


def site_flags(locations: str) -> dict:
    tokens = [part for part in str(locations).split("+") if part and part != "nan"]
    flags = {f"has_{site}": int(site in tokens) for site in SITES}
    flags["n_standard_sites"] = sum(flags[f"has_{site}"] for site in SITES[:4])
    return flags


def build_children(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["pid"] = frame["Patient ID"].astype(int)
    frame["aid"] = frame["Additional ID"]
    by_id = frame.set_index("pid", drop=False)

    seen = set()
    pairs = []
    for row in frame[frame["aid"].notna()].itertuples(index=False):
        key = tuple(sorted((int(row.pid), int(row.aid))))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)

    used = set()
    records = []
    for left_id, right_id in pairs:
        used.update((left_id, right_id))
        left, right = by_id.loc[left_id], by_id.loc[right_id]
        if left.Campaign == "CC2015" and right.Campaign == "CC2014":
            kept, dropped = left, right
        elif right.Campaign == "CC2015" and left.Campaign == "CC2014":
            kept, dropped = right, left
        else:
            raise ValueError(
                f"Pair {left_id}/{right_id} is not one 2014 visit and one 2015 visit"
            )
        records.append(row_from(kept, dropped))

    singles = frame[~frame["pid"].isin(used)]
    for _, row in singles.iterrows():
        records.append(row_from(row, None))
    children = pd.DataFrame(records).sort_values("child_id").reset_index(drop=True)
    return children


def row_from(kept, dropped):
    locations = kept["Recording locations:"]
    record = {
        "child_id": int(kept["pid"]),
        "kept_patient_id": int(kept["pid"]),
        "dropped_patient_id": None if dropped is None else int(dropped["pid"]),
        "n_visits": 1 if dropped is None else 2,
        "campaign": kept["Campaign"],
        "murmur": kept["Murmur"],
        "dropped_murmur": None if dropped is None else dropped["Murmur"],
        "label_disagrees": False if dropped is None else bool(kept["Murmur"] != dropped["Murmur"]),
        "outcome": kept["Outcome"],
        "age": kept["Age"],
        "sex": kept["Sex"],
        "height": kept["Height"],
        "weight": kept["Weight"],
        "pregnancy": bool(kept["Pregnancy status"]),
        "locations": locations,
        "supervised": kept["Murmur"] in {"Present", "Absent"},
    }
    record.update(site_flags(locations))
    for column in [
        "Murmur locations",
        "Most audible location",
        "Systolic murmur timing",
        "Systolic murmur shape",
        "Systolic murmur grading",
        "Systolic murmur pitch",
        "Systolic murmur quality",
    ]:
        record[f"desc_{column}"] = kept[column]
    return record


def assign_split(children: pd.DataFrame) -> pd.DataFrame:
    train, holdout = train_test_split(
        children,
        test_size=0.30,
        stratify=children["murmur"],
        random_state=SEED,
    )
    val, test = train_test_split(
        holdout,
        test_size=0.50,
        stratify=holdout["murmur"],
        random_state=SEED,
    )
    split = pd.Series(index=children.index, dtype=object)
    split.loc[train.index] = "train"
    split.loc[val.index] = "val"
    split.loc[test.index] = "test"
    children = children.copy()
    children["split"] = split
    return children


def main() -> None:
    source = locate_csv()
    raw = pd.read_csv(source)
    children = assign_split(build_children(raw))
    if len(children) != 872:
        raise AssertionError(f"Expected 872 children, got {len(children)}")
    counts = children["murmur"].value_counts().to_dict()
    expected = {"Absent": 652, "Present": 158, "Unknown": 62}
    if counts != expected:
        raise AssertionError(f"Label counts {counts} != {expected}")
    if int(children["label_disagrees"].sum()) != 15:
        raise AssertionError("Expected 15 linked visits with different murmur labels")
    if children["split"].isna().any():
        raise AssertionError("Every child needs a split")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "subject_table.csv"
    children.to_csv(out, index=False)
    summary = (
        children.groupby(["split", "murmur"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "split_counts.csv", index=False)
    print(f"source: {source}")
    print(f"wrote: {out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
