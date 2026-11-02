"""
SDRAG Blind Scoring Sheet Builder (150-Item Double-Blind Annotation Workbook)
=============================================================================
Merges baseline, method1, and full_sdrag runs (50 queries x 3 systems = 150 items)
into a randomized double-blind Excel annotation workbook with data validation.
"""

import argparse
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation


def main():
    ap = argparse.ArgumentParser(description="Build double-blind human evaluation sheets for SDRAG")
    ap.add_argument(
        "--runs",
        nargs="+",
        default=[
            "baseline=runs/baseline.csv",
            "method1=runs/method1.csv",
            "full_sdrag=runs/full_sdrag.csv",
        ],
        help="List of label=path.csv specifications",
    )
    ap.add_argument("--out-dir", default="scoring_sdrag", help="Output directory for blind evaluation files")
    a = ap.parse_args()

    frames = []
    for spec in a.runs:
        if "=" in spec:
            label, _, path = spec.partition("=")
        else:
            path = spec
            label = Path(spec).stem
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["system"] = label
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    combined.insert(0, "item_id", [f"S{i:04d}" for i in range(1, len(combined) + 1)])

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save complete secret answer key with automated ground-truth metrics
    key_cols = [
        c for c in [
            "item_id", "query_id", "document", "system", "query",
            "query_type", "method_focus", "target_primary", "retrieved_top1",
            "hit_at_1", "hit_at_3", "reciprocal_rank", "expected_modality",
            "detected_modalities", "modality_match", "target_linked",
            "resolved_linked", "cross_reference_resolved", "strict_1hop_resolved",
            "conflict_resolution_state", "conflict_resolved",
        ]
        if c in combined.columns
    ]
    combined[key_cols].to_csv(out / "sdrag_key.csv", index=False, encoding="utf-8-sig")

    # Build double-blind rater workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "SDRAG Blind Evaluation"

    cols = [
        "item_id",
        "query_id",
        "document",
        "system",
        "query",
        "target_primary",
        "target_linked",
        "expected_modality",
        "retrieved_context_preview",
        "detected_modalities",
        "resolved_linked",
        "conflict_resolution_state",
        "R_hit_at_1",
        "R_modality_correct",
        "R_cross_resolved",
        "R_conflict_resolved",
        "R_grounded_factually",
        "R_notes",
    ]

    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    rater_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")
    rater_header_font = Font(bold=True, color="1F4E78")

    for c_idx, name in enumerate(cols, 1):
        cell = ws.cell(1, c_idx, name)
        if name.startswith("R_"):
            cell.fill = rater_fill
            cell.font = rater_header_font
        else:
            cell.fill = header_fill
            cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for r_idx, row in enumerate(combined.to_dict(orient="records"), 2):
        ws.cell(r_idx, 1, row.get("item_id", ""))
        ws.cell(r_idx, 2, row.get("query_id", ""))
        ws.cell(r_idx, 3, row.get("document", ""))
        ws.cell(r_idx, 4, "Anonymous System")  # Blinded system label
        ws.cell(r_idx, 5, row.get("query", ""))
        ws.cell(r_idx, 6, row.get("target_primary", ""))
        ws.cell(r_idx, 7, row.get("target_linked", "None"))
        ws.cell(r_idx, 8, row.get("expected_modality", ""))
        ws.cell(r_idx, 9, row.get("retrieved_context_preview", ""))
        ws.cell(r_idx, 10, row.get("detected_modalities", ""))
        ws.cell(r_idx, 11, row.get("resolved_linked", "None"))
        ws.cell(r_idx, 12, row.get("conflict_resolution_state", ""))

        for c_idx in range(1, len(cols) + 1):
            ws.cell(r_idx, c_idx).alignment = Alignment(vertical="top", wrap_text=True)

    # Add binary (0/1) DataValidation on rating columns (M through Q)
    dv = DataValidation(type="list", formula1='"0,1"', allow_blank=True)
    dv.error = "Please enter 0 (Fail) or 1 (Pass)."
    dv.errorTitle = "Invalid Binary Score"
    ws.add_data_validation(dv)
    dv.add(f"M2:Q{len(combined) + 1}")

    wb.save(out / "rater_sdrag.xlsx")
    print(f"[✓] Created blind scoring workbook ({len(combined)} items) and key in {out}/")


if __name__ == "__main__":
    main()
