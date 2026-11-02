"""
SDRAG Ablation & Baseline Comparison Runner (50-Query Batch Benchmark)
======================================================================
Runs the 50 legal compliance benchmark queries across three configurations:
  1. baseline   : Naive 500-char sliding window (10% overlap) + vanilla dense retrieval
  2. method1    : Method 1 — Scoped-Deontic Feature Extraction & Ingestion
                  (grammar segmentation + lookahead heading stitching + parent scope +
                   deontic modality extraction & feature alignment)
  3. full_sdrag : Method 1 + Method 2 — Deterministic 1-Hop Cross-Reference Graph
                  Resolution & Deontic Conflict Engine
"""

import argparse
import os
import re
from pathlib import Path
import pandas as pd

from src.retrieval.retriever import ScopedDeonticRetriever


def load_legal_test_queries(path: str):
    """
    Loads the 50-query benchmark dataset from either CSV (data/legal_test_queries.csv)
    or Excel (data/eval_sheets/SDRAG_Queries_Member_1.xlsx), normalizing column names.
    """
    if not os.path.exists(path):
        alt_xlsx = "data/eval_sheets/SDRAG_Queries_Member_1.xlsx"
        alt_csv = "data/legal_test_queries.csv"
        if os.path.exists(alt_csv):
            path = alt_csv
        elif os.path.exists(alt_xlsx):
            path = alt_xlsx
        else:
            raise FileNotFoundError(f"Query file not found: {path}")

    df = pd.read_excel(path, sheet_name=0) if str(path).endswith(".xlsx") else pd.read_csv(path)
    raw_records = df.to_dict(orient="records")

    normalized = []
    for idx, r in enumerate(raw_records, 1):
        qid = r.get("query_id") or r.get("ID") or r.get("id") or f"TC_{idx:03d}"
        doc = r.get("document") or r.get("Document") or ""
        qtext = (
            r.get("query")
            or r.get("Legal Query")
            or r.get("Farmer query")
            or r.get("Query")
            or ""
        )
        t_prim = (
            r.get("target_primary")
            or r.get("Corpus Section")
            or r.get("target_primary_section")
            or ""
        )
        t_link = (
            r.get("target_linked")
            or r.get("Linked Cross-Reference")
            or r.get("target_linked_section")
            or "None"
        )
        if pd.isna(t_link) or str(t_link).strip().lower() in {"none", "nan", "", "—"}:
            t_link = None
        else:
            t_link = str(t_link).strip()

        exp_mod = (
            r.get("expected_modality")
            or r.get("Expected Deontic Modality")
            or "Obligation"
        )
        qtype = r.get("query_type") or r.get("Query Type") or r.get("type") or "Standard_Deontic"
        mfocus = r.get("method_focus") or r.get("Method Focus") or (
            "Method_2_CrossRef_and_Conflict" if t_link else "Method_1_Scoped_Deontic"
        )
        gt_ans = (
            r.get("ground_truth_answer")
            or r.get("Draft Ground-Truth Answer")
            or ""
        )

        normalized.append({
            "query_id": str(qid).strip(),
            "document": str(doc).strip(),
            "query": str(qtext).strip(),
            "target_primary": str(t_prim).strip(),
            "target_linked": t_link,
            "expected_modality": str(exp_mod).strip(),
            "query_type": str(qtype).strip(),
            "method_focus": str(mfocus).strip(),
            "ground_truth_answer": str(gt_ans).strip(),
        })
    return normalized


def evaluate_single_query(retriever: ScopedDeonticRetriever, query_item: dict, mode: str) -> dict:
    """
    Executes retrieval and conflict resolution for a single query under `mode`
    ('baseline', 'method1', or 'full_sdrag') and computes all benchmark metrics.
    """
    doc_name = query_item["document"]
    q_text = query_item["query"]
    exp_mod = query_item["expected_modality"]
    tp_norm = retriever._normalize_sec(query_item["target_primary"])
    tl_norm = retriever._normalize_sec(query_item["target_linked"])

    res = retriever.query(
        document_name=doc_name,
        query_text=q_text,
        mode=mode,
        top_k=3,
        expected_modality=exp_mod,
    )

    r_secs = [retriever._normalize_sec(s) for s in res["retrieved_sections"]]

    # Primary clause ranking evaluation
    found_rank = None
    for rank_idx, sec in enumerate(r_secs, start=1):
        if sec and (sec == tp_norm or sec.startswith(tp_norm + ".") or tp_norm.startswith(sec + ".")):
            found_rank = rank_idx
            break

    hit_1 = 1 if found_rank == 1 else 0
    hit_3 = 1 if (found_rank is not None and found_rank <= 3) else 0
    rr = round(1.0 / found_rank, 4) if found_rank is not None else 0.0

    # Method 1: Deontic Modality Feature Extraction evaluation
    detected_mods = res.get("detected_modalities", [])
    dom_mod = res.get("dominant_modality", "Neutral")
    modality_match = 1 if (exp_mod in detected_mods or dom_mod == exp_mod) else 0

    # Method 2: Deterministic 1-Hop Cross-Reference Graph Resolution evaluation
    has_xref = 1 if tl_norm else 0
    l_secs = [retriever._normalize_sec(s) for s in res.get("linked_sections", [])]

    if not tl_norm:
        # Standalone query (no cross-reference target required)
        cross_resolved = 1
        linked_hit_strict = 1
    else:
        # Check if 1-hop graph resolution (or accidental top-k retrieval) captured target_linked
        base_tl = re.sub(r"\(.*\)", "", tl_norm).strip()
        resolved_via_graph = any(
            s == tl_norm or s == base_tl or s.startswith(base_tl + ".") or base_tl.startswith(s + ".")
            for s in l_secs if s
        )
        resolved_in_topk = any(
            s == tl_norm or s == base_tl
            for s in r_secs if s and s != tp_norm
        )
        cross_resolved = 1 if (resolved_via_graph or resolved_in_topk) else 0
        linked_hit_strict = 1 if resolved_via_graph else 0

    # Method 2: Deontic Conflict Engine evaluation
    conflict_info = res.get("conflict_resolution") or {}
    conflict_state = conflict_info.get("resolution_state", "NONE")
    resolved_polarity = conflict_info.get("resolved_polarity", "Neutral")
    if has_xref:
        conflict_resolved = 1 if (conflict_info.get("conflict_detected") and cross_resolved == 1) else 0
    else:
        conflict_resolved = 1 if (hit_1 == 1 and modality_match == 1) else 0

    top_doc_preview = res["retrieved_docs"][0][:220] if res.get("retrieved_docs") else ""
    linked_doc_preview = res.get("linked_docs", [""])[0][:180] if res.get("linked_docs") else ""

    return {
        "query_id": query_item["query_id"],
        "document": doc_name,
        "query": q_text,
        "query_type": query_item["query_type"],
        "method_focus": query_item["method_focus"],
        "system": mode,
        "target_primary": query_item["target_primary"],
        "retrieved_top1": res["retrieved_sections"][0] if res["retrieved_sections"] else "NONE",
        "retrieved_top3": "; ".join(res["retrieved_sections"]),
        "hit_at_1": hit_1,
        "hit_at_3": hit_3,
        "reciprocal_rank": rr,
        "expected_modality": exp_mod,
        "detected_modalities": ", ".join(detected_mods) if detected_mods else "None",
        "modality_match": modality_match,
        "target_linked": query_item["target_linked"] if query_item["target_linked"] else "None",
        "resolved_linked": ", ".join(res.get("linked_sections", [])) if res.get("linked_sections") else "None",
        "has_cross_reference": has_xref,
        "cross_reference_resolved": cross_resolved,
        "strict_1hop_resolved": linked_hit_strict,
        "conflict_resolution_state": conflict_state,
        "resolved_polarity": resolved_polarity,
        "conflict_resolved": conflict_resolved,
        "retrieved_context_preview": (
            f"PRIMARY: {top_doc_preview} || LINKED: {linked_doc_preview}"
            if linked_doc_preview else f"PRIMARY: {top_doc_preview}"
        ),
        "elapsed_seconds": round(res["elapsed_seconds"], 4),
        "status": "COMPLETED",
    }


def run_system_benchmark(retriever: ScopedDeonticRetriever, queries: list, system_mode: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = [evaluate_single_query(retriever, q, system_mode) for q in queries]
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    hr1 = df["hit_at_1"].mean() * 100
    hr3 = df["hit_at_3"].mean() * 100
    mrr = df["reciprocal_rank"].mean()
    mod_acc = df["modality_match"].mean() * 100
    xref_df = df[df["has_cross_reference"] == 1]
    xref_acc = xref_df["strict_1hop_resolved"].mean() * 100 if len(xref_df) > 0 else 100.0
    conf_acc = df["conflict_resolved"].mean() * 100

    print(
        f"[✓] {system_mode:<10} | Queries: {len(df)} | HR@1: {hr1:5.1f}% | HR@3: {hr3:5.1f}% | "
        f"MRR: {mrr:.4f} | Modality Acc: {mod_acc:5.1f}% | 1-Hop XRef Acc: {xref_acc:5.1f}% | "
        f"Conflict Engine: {conf_acc:5.1f}% -> {out_path}"
    )
    return df


def main():
    ap = argparse.ArgumentParser(description="SDRAG 50-Query Ablation & Comparison Runner")
    ap.add_argument(
        "--system",
        default="all",
        choices=["baseline", "method1", "full_sdrag", "all"],
        help="System configuration to evaluate (or 'all' to run all three).",
    )
    ap.add_argument("--queries", default="data/legal_test_queries.csv", help="Path to CSV or XLSX query suite.")
    ap.add_argument("--out", default=None, help="Output CSV path (or directory when --system all).")
    a = ap.parse_args()

    queries = load_legal_test_queries(a.queries)
    retriever = ScopedDeonticRetriever(raw_corpus_dir="data/raw")

    if a.system == "all":
        out_dir = Path(a.out) if a.out else Path("runs")
        for mode in ["baseline", "method1", "full_sdrag"]:
            run_system_benchmark(retriever, queries, mode, out_dir / f"{mode}.csv")
    else:
        out_path = Path(a.out) if a.out else Path("runs") / f"{a.system}.csv"
        run_system_benchmark(retriever, queries, a.system, out_path)


if __name__ == "__main__":
    main()
