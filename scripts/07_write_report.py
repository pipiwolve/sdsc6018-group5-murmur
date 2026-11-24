"""Build the editable Group 5 report from the archived experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "report"


def load(name: str) -> dict:
    return json.loads((PROCESSED / name).read_text())


REPORT_MODELS = {"logistic": "Logistic regression", "xgboost": "XGBoost", "cnn": "1D CNN"}


def set_style_font(style, size=12, bold=False):
    style.font.name = "Times New Roman"
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    fonts = style.element.get_or_add_rPr().rFonts
    for script in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{script}"), "Times New Roman")
    for attribute in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        fonts.attrib.pop(qn(f"w:{attribute}"), None)
    fmt = style.paragraph_format
    fmt.line_spacing_rule = WD_LINE_SPACING.SINGLE
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(6)


def make_report():
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin, section.bottom_margin = Inches(0.87), Inches(0.87)
    section.left_margin = section.right_margin = Inches(1)
    section.footer_distance = Inches(0.39)
    for name, size, bold in (
        ("Normal", 12, False), ("Title", 15, True), ("Heading 1", 12, True),
        ("Heading 2", 12, True), ("Caption", 12, False), ("Footer", 12, False),
    ):
        set_style_font(doc.styles[name], size, bold)
    title_props = doc.styles["Title"].element.get_or_add_pPr()
    for border in title_props.findall(qn("w:pBdr")):
        title_props.remove(border)
    doc.styles["Heading 1"].paragraph_format.space_before = Pt(8)
    doc.styles["Heading 2"].paragraph_format.space_before = Pt(5)
    doc.styles["Caption"].font.italic = False
    doc.styles["Caption"].paragraph_format.keep_with_next = False
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.paragraph_format.space_after = Pt(0)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    footer._p.append(fld)
    doc.core_properties.title = "Pediatric Heart Murmur Screening from Phonocardiograms"
    doc.core_properties.subject = "SDSC 6018 Group 5 final project report"
    doc.core_properties.author = "SDSC 6018 Group 5"
    return doc


def para(doc, text, label=None, style=None):
    p = doc.add_paragraph(style=style)
    if label:
        p.add_run(label + " ").bold = True
    p.add_run(text)
    return p


def heading(doc, text, *, page=False, level=1):
    p = doc.add_heading(text, level=level)
    if page:
        p.paragraph_format.page_break_before = True
    return p


def table(doc, title, headers, rows, widths):
    cap = para(doc, title, style="Caption")
    cap.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.autofit = False
    for col, width in zip(t.columns, widths):
        col.width = Inches(width)
    props = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        bd = OxmlElement(f"w:{edge}")
        for key, value in {"val": "single", "sz": "4", "color": "D9D9D9"}.items():
            bd.set(qn(f"w:{key}"), value)
        borders.append(bd)
    props.append(borders)
    margins = OxmlElement("w:tblCellMar")
    for edge, value in (("top", 37), ("bottom", 37), ("left", 90), ("right", 90)):
        margin = OxmlElement(f"w:{edge}")
        margin.set(qn("w:w"), str(value))
        margin.set(qn("w:type"), "dxa")
        margins.append(margin)
    props.append(margins)
    for row_no, values in enumerate([headers, *rows]):
        row = t.rows[0] if row_no == 0 else t.add_row()
        trpr = row._tr.get_or_add_trPr()
        trpr.append(OxmlElement("w:cantSplit"))
        if row_no == 0:
            trpr.append(OxmlElement("w:tblHeader"))
        for column_no, (cell, value, width) in enumerate(zip(row.cells, values, widths)):
            cell.width = Inches(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            if column_no:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(value))
            r.bold = row_no == 0
            if row_no == 0 or row_no % 2 == 0:
                shade = OxmlElement("w:shd")
                shade.set(qn("w:fill"), "E9EEF2" if row_no == 0 else "F6F8FA")
                cell._tc.get_or_add_tcPr().append(shade)
    return t


def add_code_link(doc):
    url = "https://github.com/pipiwolve/sdsc6018-group5-murmur"
    p = doc.add_paragraph("Code and archived predictions  ")
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), p.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True))
    run = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = url
    run.append(text)
    link.append(run)
    p._p.append(link)


def note(doc, text):
    p = para(doc, text)
    p.paragraph_format.space_before = Pt(5)
    return p


def figure(doc, filename, text, width):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(PROCESSED / "figures" / filename), width=Inches(width))
    return para(doc, text, style="Caption")


def ci(metrics, precision=3, percent=False, signed=False):
    lo, hi = metrics["ci"]
    if percent:
        return f"{metrics['point']:.1%} [{lo:.1%}, {hi:.1%}]"
    sign = "+" if signed else ""
    spec = f"{sign}.{precision}f"
    return f"{metrics['point']:{spec}} [{lo:{spec}}, {hi:{spec}}]".replace("-", "−")


def write_report(tabular: dict, cnn: dict, decision: dict, campaign: dict) -> None:
    review = load("review_analysis.json")
    doc = make_report()
    para(doc, "Group 5").runs[0].bold = True
    para(doc, "CHEN Pengjian (60214097), YANG Jianing (59942823), XIA Ningze (61097102), and WONG Wai Kwan (60690110)")
    para(doc, "Pediatric Heart Murmur Screening from Phonocardiograms", style="Title")
    para(doc, "Ranking, probability calibration, and selective abstention", style="Subtitle")
    set_style_font(doc.styles["Subtitle"], 12)
    heading(doc, "1. Introduction")
    para(doc, "A murmur heard during pediatric screening can accompany structural heart disease or arise from benign flow. We study whether phonocardiograms and screening-time covariates can support the narrower task of ranking the expert’s murmur label. The George B. Moody PhysioNet Challenge 2022 evaluated murmur detection and clinical outcome prediction as separate tasks (Reyna et al., 2023). We use its public CirCor training release to examine probability calibration and selective prediction after model selection.")
    para(doc, "Using recordings and screening-time covariates, can XGBoost and a small 1D convolutional neural network (CNN) rank children labeled Present above those labeled Absent more effectively than L2-regularized logistic regression?", label="RQ1.")
    para(doc, "After validation-based calibration, how well do the three models’ probabilities agree with observed Present rates overall and across age and sex groups, and how often does the selected model withhold children labeled Unknown under a rule targeting at least 80% labeled coverage?", label="RQ2.")
    para(doc, "We use the same cohort to test ranking, probability reliability and whether withholding middle scores aligns with the expert’s Unknown label. The CNN had the highest locked-test ranking point estimates. Calibration lowered overall probability error, while the central band withheld a minority of Unknown cases.")
    heading(doc, "2. Data")
    para(doc, "The CirCor DigiScope Phonocardiogram Dataset v1.0.3 was collected in the 2014 and 2015 Caravana do Coração campaigns in Northeast Brazil (Oliveira et al., 2022a, 2022b). The public training release contains 942 visit records and 3,163 recordings (20.09 hours, 4 kHz, mono, 16-bit). The source archive matched the published checksum manifest. We used Present and Absent for supervised analyses and reserved Unknown for the abstention comparison. Clinical outcomes and expert murmur descriptors were excluded from model inputs.")
    para(doc, "Additional ID linked 70 pairs of visits. For each linked child we retained the 2015 visit and its label. Fifteen pairs had different murmur labels. The resulting 872 children were stratified by the three labels and split approximately 70/15/15 using seed 6018. All recordings used for a given child stayed in one split.")
    table(doc, "Table 1. Child-level cohort and split counts", ["Split", "Absent", "Present", "Unknown", "Total"],
          [("Train", 456, 111, 43, 610), ("Validation", 98, 24, 9, 131),
           ("Test", 98, 23, 10, 131), ("Total", 652, 158, 62, 872)], [1.7, 1.2, 1.2, 1.2, 1.2])

    heading(doc, "2. Data continued", page=True, level=2)
    para(doc, "Age was missing for 74 children, height for 114, and weight for 105. Pregnancy was excluded from the feature set because all 70 records marked pregnant lacked age. At least one of the four standard sites was missing for 308 children (35.3%). Sixteen child-site groups had duplicate recordings. We retained the longer clip and recorded a duplicate indicator. The four defined segmentation states covered a median 51.8% of selected recordings. One mitral clip (50782_MV_1) lacked usable S1/S2 labels, so cycle timing was left missing and its spectral features used the full clip.")
    para(doc, "The source study was approved by the HUOC/PROCAPE institutional review board (request 5192) and released under the Open Data Commons Attribution license. The public data record age bands and numeric identifiers. The evaluation used the fixed public-release labels.")
    heading(doc, "3. Methods")
    para(doc, "The 70 tabular inputs include age indicators, recorded sex, height, weight, and missingness indicators. For aortic (AV), pulmonic (PV), tricuspid (TV), and mitral (MV) sites we used missingness and duplicate indicators, clip length and segmentation coverage, S1/S2 and phase timing, cycle counts, RMS energy, spectral centroid, and relative power in 25–120, 120–400, and 400–800 Hz bands. Other sites contribute a count. Timing and segmented spectral features use the published S1/S2 annotations, which include expert review or correction (Oliveira et al., 2022a). The CNN operates on the recordings without those annotations.")
    para(doc, "Class-balanced L2 logistic regression used training-set median imputation and standardization. Validation average precision selected C = 0.1 from {0.01, 0.1, 1, 10}. Histogram XGBoost retained missing values and weighted Present by the training Absent-to-Present ratio. The selected grid point used depth 3, 300 trees, learning rate 0.03, minimum child weight 5, and row and column subsampling of 0.8.")
    para(doc, "The CNN used an 8-second center crop, with zero padding for shorter records. Each 32-band log-mel spectrogram used a 256-sample window and 128-sample hop. Two 32-filter temporal convolutions with ReLU, max pooling after the first, global average pooling, and a linear output produced a site score. Each training site was weighted by the inverse of the child’s number of sites. Adam training ran for at most 20 epochs. Early stopping and model selection used raw validation average precision. A child’s score was the mean of their site probabilities.")
    para(doc, "For each model, sigmoid (Platt-style) calibration and isotonic regression were fitted on labeled validation scores, with the lower validation Brier score retained (Niculescu-Mizil & Caruana, 2005). The non-selective threshold maximized validation F1. The selective CNN policy used the widest open interval centered at 0.5 that allowed at least 80% labeled validation coverage (Geifman & El-Yaniv, 2017). Scores at or below the lower boundary were classified Absent, scores at or above the upper boundary Present, and interior scores withheld.")
    para(doc, "AUROC and average precision (AUPRC) use raw scores. Brier is the mean squared probability error. We compare it with constant predictions of the training prevalence and 0.25. Two XGBoost ablations retained the selected hyperparameters. One removed explicit site-missingness flags and applied training-set mean imputation, and the other removed age, sex, height, weight, and their missingness features. TreeSHAP ranks mean absolute attributions on 200 labeled training rows.")

    heading(doc, "3. Methods continued", page=True, level=2)
    para(doc, "For the temporal check, the same CNN architecture was fitted on 240 labeled 2014 children. A further 61 labeled 2014 children were used for early stopping and fitting isotonic calibration. All 509 labeled 2015 children formed the test set. Linked children remained on the 2015 side.")
    para(doc, "Review analyses used the already held-out child predictions. We generated 10,000 child-level percentile bootstrap resamples (seed 6018), pairing draws across models and calibrations. We resampled the 121 labeled and 10 Unknown children separately to preserve the cohort sizes for selective metrics. Undefined ratios and single-class ranking draws were omitted for the affected metric. The intervals condition on the fitted models, calibrators and rules, so they capture test-sample variation.")
    add_code_link(doc)
    heading(doc, "4. Results")
    para(doc, "The primary CNN was selected on raw validation average precision (0.721 versus 0.571 for logistic regression and 0.562 for XGBoost). On the held-out test set, the Present prevalence was 19.0% (23/121), the reference for average precision.")
    estimates = review["models"]
    ranks = [(REPORT_MODELS[m], ci(v["auroc"]), ci(v["auprc"]))
             for m, v in estimates.items()]
    table(doc, "Table 2. Raw test ranking with 95% bootstrap intervals",
          ["Model", "AUROC [95% interval]", "AUPRC [95% interval]"], ranks, [1.9, 2.3, 2.3])
    cmp = review["vs_logistic"]
    note(doc, f"Relative to logistic regression, the CNN’s paired differences were {ci(cmp['cnn']['auroc'], signed=True)} for AUROC and {ci(cmp['cnn']['auprc'], signed=True)} for AUPRC. The CNN AUROC interval excludes zero, while its AUPRC interval includes zero. XGBoost’s differences were {ci(cmp['xgboost']['auroc'], signed=True)} and {ci(cmp['xgboost']['auprc'], signed=True)}, respectively.")
    rows = [(REPORT_MODELS[m], f"{v['raw_brier']:.3f}", f"{v['cal_brier']:.3f}",
             ci(v["delta_brier"], signed=True)) for m, v in estimates.items()]
    table(doc, "Table 3. Test Brier scores and paired changes after isotonic calibration",
          ["Model", "Raw", "Isotonic", "Change [95% interval]"], rows, [1.9, 0.65, 0.85, 3.1])
    sensitivity = review["calibrator_sensitivity"]["cnn"]["test"]
    note(doc, f"Reference Brier scores were 0.154 (training prevalence) and 0.158 (constant 0.25). Isotonic had lower Brier on the calibration-fit validation set for all three models. The CNN’s test Brier was {sensitivity['platt']:.3f} with sigmoid calibration and {sensitivity['isotonic']:.3f} with isotonic. Both maps improved test Brier over raw scores for all models. Table A2 shows that logistic regression had a slightly lower test Brier with sigmoid than with isotonic. Isotonic ties lowered CNN AUPRC to {cnn['test_calibrated']['auprc']:.3f}.")

    heading(doc, "4.1 Calibration reliability", page=True, level=2)
    figure(doc, "review_reliability.png",
           "Figure 1. Calibrated CNN probabilities and observed Present fractions on labeled test children. Four of eight equal-width bins were populated. Each point shows its child count. The dashed diagonal denotes perfect agreement.", 3.82)
    para(doc, "The highest-score bin contained 15 children with a mean probability of 1.00 and an observed Present fraction of 0.80. Thus, aggregate Brier improvement coexisted with overprediction at the highest observed score.")
    heading(doc, "4.2 Selective decisions", level=2)
    op, ab = decision["test_operating_point"], review["abstention"]
    para(doc, f"At the non-selective validation F1 threshold ({op['threshold']:.4f}), the CNN recorded {op['tp']} true positives, {op['fp']} false positives, {op['fn']} false negatives, and {op['tn']} true negatives. This corresponds to {op['fp_per_100']:.1f} false positives and {op['fn_per_100']:.1f} false negatives per 100 labeled children.")
    para(doc, f"The selective boundaries were approximately {ab['low']:.6f} and {ab['high']:.6f}. The band answered {decision['validation_abstention']['n_answered_labeled']} of 122 labeled validation children and withheld {ab['present_withheld']} Present test children. Among test answers there were {ab['tp']} true positives, {ab['fp']} false positives, and {ab['fn_answered']} false negatives. PPV denotes positive predictive value. Overall recall includes withheld Present children in its denominator.")
    selective = [
        ("Labeled coverage", "coverage"),
        ("PPV among answered positive classifications", "ppv"),
        ("Recall within answered labeled children", "within_answered_recall"),
        ("Overall recall including withheld Present", "overall_recall"),
        ("Unknown share of all withheld children", "unknown_share"),
        ("Unknown children withheld", "unknown_recall"),
    ]
    rows = [(name, f"{ab[key]['numerator']}/{ab[key]['denominator']}", ci(ab[key], percent=True))
            for name, key in selective]
    table(doc, "Table 4. Locked test performance of the selective CNN policy",
          ["Metric", "Count", "Estimate [95% interval]"], rows, [3.30, 0.7, 2.50])
    lo, hi = ab["exact_unknown_recall_ci"]
    note(doc, f"The exact binomial 95% interval for withholding 3 of 10 Unknown children was {lo:.1%} to {hi:.1%}.")

    heading(doc, "4.3 Score placement and subgroup calibration", page=True, level=2)
    figure(doc, "review_score_ecdf.png",
           "Figure 2. Empirical distributions of calibrated CNN scores by label and split. Jumps show tied scores. The shaded region marks the validation-selected abstention interval, with boundary scores answered.", 6.4)
    score_split = campaign["score_placement"]["splits"]["test"]
    para(doc, f"On test, median calibrated scores were {score_split['Present']['median']:.2f} for Present and {score_split['Absent']['median']:.2f} for both Absent and Unknown. The overlap with Absent explains why most Unknown scores fell outside the central band.")
    groups = [g for g in review["subgroups"] if g["dimension"] in ("age", "sex")]
    rows = []
    for g in groups:
        m = g["models"]["cnn"]
        rows.append((g["group"], g["n"], g["n_present"], f"{m['mean']:.1%}",
                     f"{g['observed']:.1%}", f"{m['brier']:.3f}"))
    table(doc, "Table 5. Calibrated CNN probabilities by test age band and recorded sex",
          ["Group", "n", "Present", "Mean score", "Observed", "Brier"], rows,
          [1.65, 0.5, 0.8, 1.2, 1.2, 1.15])
    note(doc, "Each child appears once by age and once by sex. Infants had the largest mean prediction gap among age groups with more than one child, but only three Present cases. Table A1 gives both tabular models and flags sparse strata.")
    heading(doc, "4.4 Temporal check and feature ablations", level=2)
    raw_2015, cal_2015 = (campaign["campaign_cnn"][f"year_2015_{k}"] for k in ("raw", "calibrated"))
    para(doc, f"On the 2015 held-out cohort (509 labeled children, 116 Present), the CNN achieved raw AUROC {raw_2015['auroc']:.3f} and AUPRC {raw_2015['auprc']:.3f}. Brier changed from {raw_2015['brier']:.3f} to {cal_2015['brier']:.3f} after calibration. XGBoost’s temporal AUROC and AUPRC were {tabular['campaign_shift_xgboost']['auroc']:.3f} and {tabular['campaign_shift_xgboost']['auprc']:.3f}. These are point estimates from the saved temporal run.")
    ablations = tabular["ablations"]
    para(doc, f"XGBoost test AUPRC was {estimates['xgboost']['auprc']['point']:.3f} with all features, {ablations['drop_site_missing_indicators']['test']['auprc']:.3f} after removing site flags and imputing, and {ablations['drop_age_and_body_size']['test']['auprc']:.3f} after removing demographic and body-size features. The site ablation couples two changes, so the small difference cannot be attributed to the indicators alone.")

    heading(doc, "4.5 Error review", page=True, level=2)
    errors = review["errors"]
    counts = errors["groups"]
    def error_count(key, kind):
        group = counts[key]
        return f"{group[kind]}/{group['n_absent' if kind == 'fp' else 'n_present']}"
    para(doc, f"At the non-selective threshold, false positives occurred in {error_count('one_standard_site','fp')} Absent children with one site and {error_count('multiple_standard_sites','fp')} with multiple sites. The corresponding counts were {error_count('infant','fp')} for infants and {error_count('known_age_except_infant','fp')} for other recorded ages. Below 25% mean segmentation coverage, the false-positive and false-negative counts were {error_count('mean_segmentation_below_25pct','fp')} and {error_count('mean_segmentation_below_25pct','fn')}, versus {error_count('mean_segmentation_at_least_25pct','fp')} and {error_count('mean_segmentation_at_least_25pct','fn')} at higher coverage. The coverage split is exploratory and uses the mean over available standard sites.")
    rows = []
    for case in errors["cases"]:
        models = case["models"]
        sites = ", ".join(f"{site} {value:.0%}" for site, value in case["site_coverage"].items())
        side = "/".join("P" if models[m]["prediction"] else "A" for m in ("logistic", "xgboost"))
        rows.append((f"{case['child_id']} {case['error']}", case["age"], sites,
                     f"{models['cnn']['raw']:.3f} / {models['cnn']['calibrated']:.3f}", side))
    table(doc, "Table 6. Illustrative CNN errors at the non-selective threshold",
          ["Child and error", "Age", "Recorded sites and state coverage", "CNN raw / calibrated",
           "LR / XGB"], rows, [1.08, 1.00, 2.12, 1.45, 0.85])
    note(doc, "FP and FN denote false positive and false negative. P and A denote Present and Absent predictions from the tabular models at their own validation thresholds.")
    if review["case_spectra_generated"]:
        figure(doc, "review_error_spectra.png",
               "Figure 3. One recorded site per example in Table 6. Spectra use the eight-second analysis window, a 256-sample Hann window, and a 128-sample hop, with a shared magnitude scale. Coverage in Table 6 refers to each full recording.", 6.28)
    overlap = errors["overlap"]
    para(doc, f"Of 18 CNN false positives, logistic regression shared {overlap['logistic']['FP']} and XGBoost shared {overlap['xgboost']['FP']}. Of five CNN false negatives, the two baselines shared {overlap['logistic']['FN']} and {overlap['xgboost']['FN']}, respectively. The isolated MV recording of 84969 and the four well-segmented sites of 85286 illustrate that false positives occurred under different recording conditions. The spectra describe signal content and cannot establish the cause of an error or a diagnosis.")

    heading(doc, "5. Discussion", page=True)
    para(doc, "The validation-selected CNN led the ranking point estimates. The paired intervals are wide on a test set with 23 Present children. The temporal evaluation also used a different training set size and test case mix, so the lower cross-year scores indicate limited transfer in this setup without isolating a specific cause.")
    para(doc, "The results distinguish probability quality from selective performance. The small validation set (122 labeled children, 24 Present) served both to fit and to select calibration maps. Bin-level deviations and sparse subgroups limit confidence in individual probabilities. The selective policy’s high PPV came with roughly half of all labeled Present cases classified positive. Unknown is a dataset annotation, and these predictions do not identify whether weak audibility, recording conditions, or reviewer ambiguity drove any individual label.")
    para(doc, "Pulmonic and tricuspid energy measures and cycle timing feature prominently in XGBoost’s training-set SHAP ranking (Table A3). Such features describe acoustic energy and timing, and may also reflect contact, gain, and segmentation quality. Mean absolute SHAP measures influence on this model’s output without establishing direction or a physiological mechanism.")
    para(doc, "The study covers two campaigns in one region, with no official hidden test evaluation. Six neonates were available in the full cohort, and most age-by-sex cells were small. Partial segmentation, truncation to eight seconds, and 15 label disagreements among linked children constrain interpretation. For use on new recordings, the tabular pipeline would require a validated automatic segmentation step or features computed from the whole waveform. The scores reflect murmur annotations and should not replace auscultation or echocardiography.")
    heading(doc, "6. Conclusion")
    para(doc, "On the locked split the CNN had the highest AUROC and AUPRC point estimates, while XGBoost showed no clear ranking gain over logistic regression. Overall Brier improved with calibration, with insufficient evidence for equally reliable probabilities across age and sex. The selective rule achieved its labeled coverage target but retained limited overall Present recall and withheld only a minority of Unknown cases. Future work should prespecify subgroup analyses and selective risk measures on larger, temporally and geographically external cohorts, and test whether measured signal quality explains annotation uncertainty.")
    heading(doc, "References")
    refs = (
        "Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. Advances in Neural Information Processing Systems, 30.",
        "Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. Proceedings of ICML, 625–632. doi:10.1145/1102351.1102430",
        "Oliveira, J., et al. (2022a). The CirCor DigiScope dataset: From murmur detection to murmur classification. IEEE Journal of Biomedical and Health Informatics. doi:10.1109/JBHI.2021.3137048",
        "Oliveira, J., et al. (2022b). The CirCor DigiScope Phonocardiogram Dataset (v1.0.3). PhysioNet. doi:10.13026/tshs-mw03",
        "Reyna, M. A., et al. (2023). Heart murmur detection from phonocardiogram recordings: The George B. Moody PhysioNet Challenge 2022. PLOS Digital Health, 2(9), e0000324. doi:10.1371/journal.pdig.0000324",
    )
    for ref in refs:
        p = para(doc, ref)
        p.paragraph_format.left_indent = Inches(0.18)
        p.paragraph_format.first_line_indent = Inches(-0.18)
        p.paragraph_format.space_after = Pt(4)

    heading(doc, "Appendix A. Supplementary evaluation", page=True)
    rows = []
    for group in review["subgroups"]:
        m = group["models"]
        gap = m["cnn"]["gap"]
        gap_text = (f"[{gap['ci'][0]*100:+.1f}, {gap['ci'][1]*100:+.1f}]".replace("-", "−")
                    if gap["ci"] else "Too sparse")
        rows.append((group["group"], f"{group['n']}/{group['n_present']}", f"{group['observed']:.1%}",
                     *(f"{m[k]['mean']*100:.1f} / {m[k]['brier']:.3f}" for k in REPORT_MODELS),
                     gap_text))
    table(doc, "Table A1. Subgroup mean calibrated score and Brier by model",
          ["Group", "n / P", "Obs.", "LR", "XGB", "CNN", "CNN gap 95% interval"],
          rows, [1.16, 0.56, 0.57, 1.03, 1.03, 1.03, 1.12])
    note(doc, "P is the Present count and Obs. is its fraction. Model columns show mean probability in percent, followed by Brier. The 95% interval for the gap between mean and observed probabilities is in percentage points. Intervals require at least 20 children and five per class.")
    rows = [(REPORT_MODELS[m], *[f"{vals[s][method]:.3f}" for s in ("val", "test")
                                for method in ("platt", "isotonic")])
            for m, vals in review["calibrator_sensitivity"].items()]
    table(doc, "Table A2. Calibration sensitivity by split (Brier score)",
          ["Model", "Val. sigmoid", "Val. isotonic", "Test sigmoid", "Test isotonic"],
          rows, [1.65, 1.20, 1.20, 1.22, 1.23])
    note(doc, "Validation scores are evaluated on the same validation children used to fit the maps. Test scores apply those fitted maps to the locked test predictions.")
    shap = tabular["shap_xgboost_top"]
    def shap_item(index, item):
        return f"{index}. {item['feature']}", f"{item['mean_abs_shap']:.3f}"
    rows = [(*shap_item(i+1, shap[i]),
             *(shap_item(i+9, shap[i+8]) if i+8 < len(shap) else ("", ""))) for i in range(8)]
    table(doc, "Table A3. Saved XGBoost SHAP rankings on 200 labeled training rows",
          ["Feature (ranks 1–8)", "Mean |SHAP|", "Feature (ranks 9–15)", "Mean |SHAP|"],
          rows, [2.0, 1.25, 2.0, 1.25])
    note(doc, "The test set contained 18 CNN false positives and five false negatives at the binary threshold, so we summarized all errors and illustrated four. The subgroup analysis compares probabilities with observed rates. The proposed threshold-specific subgroup metrics remain open.")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "SDSC6018_Group5_report.docx"
    doc.save(path)
    print("wrote", path)


def main() -> None:
    write_report(load("model_results.json"), load("cnn_results.json"),
                 load("cnn_decision.json"), load("campaign_cnn.json"))


if __name__ == "__main__":
    main()
