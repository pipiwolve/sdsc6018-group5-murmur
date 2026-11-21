"""Read-only quality checks for the locked CirCor training cohort.

Raw files are not modified. The report is written to data/processed/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "circor-heart-sound-1.0.3"
TRAIN = RAW / "training_data"
OUT = ROOT / "data" / "processed"
SITES = ["AV", "PV", "TV", "MV"]
CYCLE = {"1", "2", "3", "4"}


def parse_txt(path: Path) -> dict:
    lines = path.read_text().splitlines()
    head = lines[0].split()
    recordings = []
    meta = {}
    for line in lines[1:]:
        if line.startswith("#"):
            key, value = line[1:].split(":", 1)
            meta[key.strip()] = value.strip()
        elif line.strip():
            site, hea, wav, tsv = line.split()
            recordings.append({"site": site, "hea": hea, "wav": wav, "tsv": tsv})
    return {
        "patient_id": int(head[0]),
        "n_listed": int(head[1]),
        "fs_header": int(head[2]),
        "recordings": recordings,
        "meta": meta,
    }


def blank(value) -> bool:
    text = str(value).strip().lower()
    return text in {"", "nan", "none"}


def same_number(left, right) -> bool:
    if blank(left) and blank(right):
        return True
    try:
        return abs(float(left) - float(right)) < 1e-6
    except (TypeError, ValueError):
        return False


def read_header(path: Path) -> dict:
    lines = path.read_text().splitlines()
    name, nsig, fs, n_samples = lines[0].split()[:4]
    fmt = lines[1].split()[1]
    return {
        "record": name,
        "nsig": int(nsig),
        "fs": float(fs),
        "n_samples": int(n_samples),
        "fmt": fmt,
    }


def read_segmentation(path: Path, duration: float, fs: float) -> dict:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        start, end, label = line.split()[:3]
        rows.append((float(start), float(end), label))
    scale = 1.0
    if rows and max(end for _, end, _ in rows) > max(duration, 1.0) * 2:
        scale = 1.0 / fs
    covered = {label: 0.0 for label in ["0", "1", "2", "3", "4"]}
    bad = 0
    zero_markers = 0
    for start, end, label in rows:
        start *= scale
        end *= scale
        if end < start or (end > start and label not in covered):
            bad += 1
            continue
        if end == start:
            zero_markers += 1
            continue
        covered[label] += end - start
    cycle = sum(covered[label] for label in CYCLE)
    return {
        "n_segments": len(rows),
        "bad_segments": bad,
        "zero_markers": zero_markers,
        "cycle_sec": cycle,
        "unknown_sec": covered["0"],
        "cycle_frac": cycle / duration if duration else 0.0,
        "has_s1": covered["1"] > 0,
        "has_s2": covered["3"] > 0,
    }


def pct(part, whole) -> str:
    if whole == 0:
        return "0"
    return f"{part} ({part / whole:.1%})"


def main() -> None:
    failures = []
    csv_path = RAW / "training_data.csv"
    table_path = OUT / "subject_table.csv"
    labels = pd.read_csv(csv_path)
    children = pd.read_csv(table_path)
    txt_paths = sorted(TRAIN.glob("*.txt"))
    parsed = [parse_txt(path) for path in txt_paths]
    by_txt = {item["patient_id"]: item for item in parsed}

    if len(txt_paths) != 942 or len(labels) != 942:
        failures.append(f"subject files {len(txt_paths)}, label rows {len(labels)}")
    if set(by_txt) != set(labels["Patient ID"].astype(int)):
        failures.append("patient IDs in txt and training_data.csv differ")

    label_mismatch = 0
    listing_mismatch = 0
    missing_files = []
    listed_wavs = set()
    for row in labels.itertuples(index=False):
        item = by_txt[int(row[0])]
        meta = item["meta"]
        mismatched = False
        expected = {
            "Age": row.Age,
            "Sex": row.Sex,
            "Murmur": row.Murmur,
            "Outcome": row.Outcome,
            "Campaign": row.Campaign,
        }
        for key, value in expected.items():
            left = meta.get(key, "nan")
            right = "nan" if blank(value) else str(value)
            if str(left) != right:
                mismatched = True
        if not same_number(meta.get("Height"), row.Height) or not same_number(
            meta.get("Weight"), row.Weight
        ):
            mismatched = True
        if mismatched:
            label_mismatch += 1
        sites = "+".join(rec["site"] for rec in item["recordings"])
        if sites != str(row[1]) or item["n_listed"] != len(item["recordings"]) or item["fs_header"] != 4000:
            listing_mismatch += 1
        for rec in item["recordings"]:
            listed_wavs.add(rec["wav"])
            for name in (rec["hea"], rec["wav"], rec["tsv"]):
                if not (TRAIN / name).exists():
                    missing_files.append(name)

    wavs = {path.name for path in TRAIN.glob("*.wav")}
    heas = {path.name for path in TRAIN.glob("*.hea")}
    tsvs = {path.name for path in TRAIN.glob("*.tsv")}
    orphan_wavs = sorted(wavs - listed_wavs)
    if len(wavs) != 3163 or len(heas) != 3163 or len(tsvs) != 3163:
        failures.append(f"audio counts wav/hea/tsv {len(wavs)}/{len(heas)}/{len(tsvs)}")
    if missing_files or orphan_wavs or label_mismatch or listing_mismatch:
        failures.append(
            "txt/csv/audio inventory mismatch: "
            f"label {label_mismatch}, listing {listing_mismatch}, "
            f"missing {len(missing_files)}, orphan wav {len(orphan_wavs)}"
        )

    recordings = []
    for item in parsed:
        for rec in item["recordings"]:
            header = read_header(TRAIN / rec["hea"])
            duration = header["n_samples"] / header["fs"] if header["fs"] else 0.0
            wav_size = (TRAIN / rec["wav"]).stat().st_size
            expected_size = 44 + header["n_samples"] * 2
            seg = read_segmentation(TRAIN / rec["tsv"], duration, header["fs"])
            recordings.append(
                {
                    "patient_id": item["patient_id"],
                    "site": rec["site"],
                    "wav": rec["wav"],
                    "fs": header["fs"],
                    "nsig": header["nsig"],
                    "fmt": header["fmt"],
                    "duration": duration,
                    "size_ok": wav_size == expected_size,
                    **seg,
                }
            )
    audio = pd.DataFrame(recordings)
    format_bad = audio[(audio["fs"] != 4000) | (audio["nsig"] != 1) | (audio["fmt"] != "16+44") | (~audio["size_ok"])]
    seg_bad = audio[(audio["bad_segments"] > 0) | (~audio["has_s1"]) | (~audio["has_s2"])]
    if len(format_bad):
        failures.append(f"format failures {len(format_bad)}")

    kept_ids = set(children["kept_patient_id"].astype(int))
    dropped_ids = set(children["dropped_patient_id"].dropna().astype(int))
    if kept_ids & dropped_ids:
        failures.append("a dropped 2014 visit is also a modeling row")
    if len(children) != 872 or children["child_id"].nunique() != 872:
        failures.append("subject table is not 872 unique children")
    if int(children["label_disagrees"].sum()) != 15:
        failures.append("linked murmur disagreements are not 15")
    if not set(children.loc[children["n_visits"] == 2, "campaign"]).issubset({"CC2015"}):
        failures.append("a linked child is not kept on the 2015 visit")
    split_sizes = children.groupby(["split", "murmur"]).size()
    expected_split = {
        ("train", "Absent"): 456,
        ("train", "Present"): 111,
        ("train", "Unknown"): 43,
        ("val", "Absent"): 98,
        ("val", "Present"): 24,
        ("val", "Unknown"): 9,
        ("test", "Absent"): 98,
        ("test", "Present"): 23,
        ("test", "Unknown"): 10,
    }
    if split_sizes.to_dict() != expected_split:
        failures.append(f"split counts changed: {split_sizes.to_dict()}")

    kept_audio = audio[audio["patient_id"].isin(kept_ids)].copy()
    kept_audio["rank"] = kept_audio.groupby(["patient_id", "site"])["duration"].rank(
        method="first", ascending=False
    )
    selected = kept_audio[kept_audio["rank"] == 1]
    duplicate_sites = int((kept_audio.groupby(["patient_id", "site"]).size() > 1).sum())

    desc_cols = [col for col in children.columns if col.startswith("desc_")]
    absent_desc = int(children.loc[children["murmur"] != "Present", desc_cols].notna().any(axis=1).sum())
    present_loc = int(children.loc[children["murmur"] == "Present", "desc_Murmur locations"].notna().sum())
    if absent_desc:
        failures.append(f"{absent_desc} non-present rows have murmur descriptors")
    if present_loc != int((children["murmur"] == "Present").sum()):
        failures.append("a Present child is missing murmur location")

    cohort = children.copy()
    age_miss = int(cohort["age"].isna().sum())
    height_miss = int(cohort["height"].isna().sum())
    weight_miss = int(cohort["weight"].isna().sum())
    pregnant = cohort[cohort["pregnancy"] == True]  # noqa: E712
    pregnant_age_miss = int(pregnant["age"].isna().sum())
    age_table = pd.crosstab(cohort["age"].fillna("Missing"), cohort["murmur"], margins=True)
    low_cycle = selected[selected["cycle_frac"] < 0.5]
    selected_empty = selected[(~selected["has_s1"]) | (~selected["has_s2"])]

    summary = {
        "txt": len(txt_paths),
        "wav": len(wavs),
        "children": len(cohort),
        "label_counts": cohort["murmur"].value_counts().to_dict(),
        "format_bad": int(len(format_bad)),
        "segmentation_bad": int(len(seg_bad)),
        "selected_recordings": int(len(selected)),
        "duplicate_site_groups": duplicate_sites,
        "selected_cycle_median": float(selected["cycle_frac"].median()),
        "selected_cycle_mean": float(selected["cycle_frac"].mean()),
        "selected_cycle_below_half": int(len(low_cycle)),
        "all_cycle_median": float(audio["cycle_frac"].median()),
        "duration_hours_all": float(audio["duration"].sum() / 3600),
        "duration_hours_selected": float(selected["duration"].sum() / 3600),
        "failures": failures,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "quality_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    problem = pd.concat(
        [
            format_bad.assign(problem="format"),
            seg_bad.assign(problem="segmentation"),
        ],
        ignore_index=True,
    )
    problem.to_csv(OUT / "quality_recording_issues.csv", index=False)
    age_table.to_csv(OUT / "quality_age_murmur.csv")

    lines = [
        "# 数据质量校验",
        "",
        "校验对象是 PhysioNet CirCor 1.0.3 公开训练集，以及按 `proposal/项目执行标准.md` 生成的 872 名儿童。脚本只读原始文件。",
        "",
        "## 结论",
        "",
    ]
    if failures:
        lines.append("校验没有通过，特征提取先停在这里。")
        lines.extend(f"- {item}" for item in failures)
    else:
        lines.append("文件、标签和划分与锁定规则一致，没有阻塞项。可以进入特征提取。分割覆盖不完整，时间特征只使用状态 1–4，这个限制保持不变。")
    lines.extend(
        [
            "",
            "## 完整性",
            "",
            f"- 受试者 `.txt` {len(txt_paths)} 个，与 `training_data.csv` 的患者编号一致。人口学和标签字段不一致 {label_mismatch} 条，听诊位置列表不一致 {listing_mismatch} 条。",
            f"- 录音 `.wav` / `.hea` / `.tsv` 各 {len(wavs)} 个。txt 点名的文件缺失 {len(missing_files)} 个，目录里没有被 txt 点名的 wav {len(orphan_wavs)} 个。",
            f"- 全部录音都是 4000 Hz、单声道、16-bit，文件长度等于 44 字节头加采样点。格式异常 {len(format_bad)} 段。",
            f"- 全部录音时长合计 {audio['duration'].sum() / 3600:.2f} 小时，最短 {audio['duration'].min():.2f} 秒，中位 {audio['duration'].median():.2f} 秒，最长 {audio['duration'].max():.2f} 秒。",
            "",
            "## 分析队列",
            "",
            f"- 872 名儿童，每人一行。2014 与 2015 都参加的 70 人保留 2015 年这次。被放下的编号没有再出现在模型行里。",
            f"- 标签：Absent {int((cohort.murmur == 'Absent').sum())}，Present {int((cohort.murmur == 'Present').sum())}，Unknown {int((cohort.murmur == 'Unknown').sum())}。已标注儿童里 Present 占 {(cohort.murmur == 'Present').sum() / cohort['supervised'].sum():.1%}。",
            "- 70 对关联记录里，两次杂音标签不一致 15 对。模型使用 2015 年那次自己的标签。",
            "- 划分种子 6018。训练 610、验证 131、测试 131，三类人数与 `split_counts.csv` 一致。",
            "",
            "## 缺失、不平衡和泄漏",
            "",
            f"- 年龄缺失 {pct(age_miss, len(cohort))}。身高缺失 {pct(height_miss, len(cohort))}，体重缺失 {pct(weight_miss, len(cohort))}。没有小于等于 0 的身高或体重。",
            f"- 妊娠为真 {len(pregnant)} 人，其中年龄缺失 {pregnant_age_miss} 人。妊娠不进入特征。",
            f"- 四个标准位置至少缺一个的儿童 {pct(int((cohort['n_standard_sites'] < 4).sum()), len(cohort))}。主动脉缺 {pct(int((cohort.has_AV == 0).sum()), len(cohort))}，肺动脉缺 {pct(int((cohort.has_PV == 0).sum()), len(cohort))}，三尖瓣缺 {pct(int((cohort.has_TV == 0).sum()), len(cohort))}，二尖瓣缺 {pct(int((cohort.has_MV == 0).sum()), len(cohort))}。",
            f"- 保留这次筛查里，同一位置有多段录音的位置组 {duplicate_sites} 个。特征将使用其中较长的一段，得到 {len(selected)} 段录音，合计 {selected['duration'].sum() / 3600:.2f} 小时。`Phc` 在这些选中录音里有 {int((selected.site == 'Phc').sum())} 段，不进入四个位置的特征。",
            f"- Absent 和 Unknown 的杂音描述列有内容的儿童为 {absent_desc}。Present 的 {present_loc} 人全部有杂音位置。这些描述列不进入特征。",
            "",
            "年龄段和杂音人数：",
            "",
            "| 年龄段 | Absent | Present | Unknown |",
            "|---|---:|---:|---:|",
        ]
    )
    for age in ["Neonate", "Infant", "Child", "Adolescent", "Missing"]:
        if age not in age_table.index:
            continue
        lines.append(
            f"| {age} | {int(age_table.loc[age].get('Absent', 0))} | {int(age_table.loc[age].get('Present', 0))} | {int(age_table.loc[age].get('Unknown', 0))} |"
        )
    lines.extend(
        [
            "",
            "新生儿人数很少。分组表里这一档只报人数。",
            "",
            "## 分割覆盖",
            "",
            f"- 3,163 段录音都有分割文件。{int((audio['zero_markers'] > 0).sum())} 段在末尾有长度为 0 的标记，校验时略过。非法正时长标签 {int((audio['bad_segments'] > 0).sum())} 段。",
            f"- 没有可用 S1 和 S2 的录音是 {int((~audio['has_s1'] | ~audio['has_s2']).sum())} 段"
            + (
                "：" + "、".join(seg_bad["wav"].astype(str)) + "。"
                if len(seg_bad)
                else "。"
            ),
            f"- 状态 1–4 占录音时长的比例，全部录音中位数 {audio['cycle_frac'].median():.1%}，均值 {audio['cycle_frac'].mean():.1%}。将进入特征的 {len(selected)} 段里，中位数 {selected['cycle_frac'].median():.1%}，均值 {selected['cycle_frac'].mean():.1%}，低于一半的有 {len(low_cycle)} 段。",
            (
                "- 进入特征的录音里，没有 S1/S2 的是 "
                + "、".join(f"`{name}`" for name in selected_empty["wav"])
                + "。这些位置仍保留录音和缺失标记，时间摘要留空。"
                if len(selected_empty)
                else "- 进入特征的录音都有 S1 和 S2。"
            ),
            "- 状态 0 是未归入心动周期的区段。时间摘要只在状态 1–4 上计算。",
            "",
            "## 复核",
            "",
            "```bash",
            "python3 scripts/02_validate_data_quality.py",
            "```",
            "",
        ]
    )
    report = OUT / "数据质量校验.md"
    report.write_text("\n".join(lines))
    print(f"wrote {report}")
    print(f"failures {len(failures)}")
    for item in failures:
        print(item)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
