"""
SDRAG 50-Query Batch Benchmark, Comparison & Diagnostic Runner
==============================================================
Executes the complete 50-query benchmark suite across:
  - Baseline   : Naive 500-char sliding window + vanilla dense vector search
  - Method 1   : Scoped-Deontic Feature Extraction & Ingestion
  - Full SDRAG : Method 1 + Method 2 (Deterministic 1-Hop Cross-Reference
                 Graph Resolution & Deontic Conflict Engine)
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.hybrid_retriever import ScopedDeonticRetriever
from sdrag_compare import load_legal_test_queries, evaluate_single_query


def load_diagnostic_test_suite(query_path: str = None):
    """
    Loads the 50 ground-truth legal test cases targeting both
    Method-1 (Scoped-Deontic Feature Extraction & Ingestion) and
    Method-2 (Deterministic 1-Hop Cross-Reference Graph Resolution & Conflict Engine).
    """
    default_path = query_path or os.path.join(PROJECT_ROOT, "data", "legal_test_queries.csv")
    return load_legal_test_queries(default_path)


def evaluate_sdrag_pipeline():
    print("=" * 90)
    print("  INITIALIZING SDRAG 50-QUERY BATCH BENCHMARK & ABLATION RUNNER")
    print("=" * 90)

    test_cases = load_diagnostic_test_suite()
    retriever = ScopedDeonticRetriever(raw_corpus_dir=os.path.join(PROJECT_ROOT, "data", "raw"))

    runs_dir = Path(PROJECT_ROOT) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    mode_dfs = {}
    for mode in ["baseline", "method1", "full_sdrag"]:
        rows = [evaluate_single_query(retriever, tc, mode) for tc in test_cases]
        df_mode = pd.DataFrame(rows)
        df_mode.to_csv(runs_dir / f"{mode}.csv", index=False, encoding="utf-8-sig")
        mode_dfs[mode] = df_mode

    # Build primary trace CSV for full_sdrag (with comparative baseline & method1 columns)
    df_full = mode_dfs["full_sdrag"].copy()
    df_base = mode_dfs["baseline"]
    df_m1 = mode_dfs["method1"]

    trace_rows = []
    for i in range(len(df_full)):
        r_full = df_full.iloc[i]
        r_m1 = df_m1.iloc[i]
        r_base = df_base.iloc[i]
        trace_rows.append({
            "Test_ID": r_full["query_id"],
            "Document": r_full["document"],
            "Query": r_full["query"],
            "Query_Type": r_full["query_type"],
            "Method_Focus": r_full["method_focus"],
            "Target_Primary": r_full["target_primary"],
            "Target_Linked": r_full["target_linked"],
            "Expected_Modality": r_full["expected_modality"],
            "Baseline_Hit_At_1": int(r_base["hit_at_1"]),
            "Method1_Hit_At_1": int(r_m1["hit_at_1"]),
            "FullSDRAG_Hit_At_1": int(r_full["hit_at_1"]),
            "Hit_At_1": int(r_full["hit_at_1"]),
            "Hit_At_3": int(r_full["hit_at_3"]),
            "Reciprocal_Rank": float(r_full["reciprocal_rank"]),
            "Detected_Modalities": r_full["detected_modalities"],
            "Modality_Match": int(r_full["modality_match"]),
            "Resolved_Linked": r_full["resolved_linked"],
            "Cross_Reference_Resolved": int(r_full["cross_reference_resolved"]),
            "Strict_1Hop_Resolved": int(r_full["strict_1hop_resolved"]),
            "Conflict_Resolution_State": r_full["conflict_resolution_state"],
            "Resolved_Polarity": r_full["resolved_polarity"],
            "Conflict_Resolved": int(r_full["conflict_resolved"]),
            "Latency_ms": round(float(r_full["elapsed_seconds"]) * 1000, 2),
            "Status": "PASS" if (r_full["hit_at_1"] == 1 and r_full["cross_reference_resolved"] == 1) else "FAIL",
        })

    df_trace = pd.DataFrame(trace_rows)
    df_trace.to_csv(runs_dir / "sdrag_benchmark_trace.csv", index=False, encoding="utf-8-sig")

    # Compute comprehensive metrics summary
    def summarize_mode(df_m: pd.DataFrame) -> dict:
        xref_sub = df_m[df_m["has_cross_reference"] == 1]
        return {
            "Total_Queries": int(len(df_m)),
            "HR_At_1": f"{df_m['hit_at_1'].mean() * 100:.1f}%",
            "HR_At_3": f"{df_m['hit_at_3'].mean() * 100:.1f}%",
            "MRR": round(float(df_m["reciprocal_rank"].mean()), 4),
            "Deontic_Modality_Accuracy": f"{df_m['modality_match'].mean() * 100:.1f}%",
            "Cross_Reference_Accuracy_All": f"{df_m['cross_reference_resolved'].mean() * 100:.1f}%",
            "Strict_1Hop_Graph_Accuracy": (
                f"{xref_sub['strict_1hop_resolved'].mean() * 100:.1f}% "
                f"({int(xref_sub['strict_1hop_resolved'].sum())}/{len(xref_sub)})"
            ),
            "Conflict_Engine_Accuracy": f"{df_m['conflict_resolved'].mean() * 100:.1f}%",
            "Mean_Latency_ms": round(float(df_m["elapsed_seconds"].mean()) * 1000, 2),
        }

    # Per-modality breakdown for Method 1
    modality_breakdown = {}
    for mod_name, grp in df_full.groupby("expected_modality"):
        modality_breakdown[mod_name] = {
            "count": int(len(grp)),
            "hr_at_1": f"{grp['hit_at_1'].mean() * 100:.1f}%",
            "modality_extraction_acc": f"{grp['modality_match'].mean() * 100:.1f}%",
        }

    # Conflict resolution state distribution for Method 2
    conflict_distribution = df_full["conflict_resolution_state"].value_counts().to_dict()

    full_summary = summarize_mode(df_full)
    metrics_payload = {
        "Total_Queries": full_summary["Total_Queries"],
        "HR_At_1": full_summary["HR_At_1"],
        "HR_At_3": full_summary["HR_At_3"],
        "MRR": full_summary["MRR"],
        "Deontic_Modality_Accuracy": full_summary["Deontic_Modality_Accuracy"],
        "Strict_1Hop_Graph_Accuracy": full_summary["Strict_1Hop_Graph_Accuracy"],
        "Conflict_Engine_Accuracy": full_summary["Conflict_Engine_Accuracy"],
        "Mean_Latency_ms": full_summary["Mean_Latency_ms"],
        "Ablation_Comparison": {
            "baseline": summarize_mode(df_base),
            "method1_scoped_deontic": summarize_mode(df_m1),
            "full_sdrag_method1_plus_method2": full_summary,
        },
        "Method1_Modality_Breakdown": modality_breakdown,
        "Method2_Conflict_State_Distribution": {k: int(v) for k, v in conflict_distribution.items()},
    }

    with open(runs_dir / "sdrag_metrics_summary.json", "w") as f:
        json.dump(metrics_payload, f, indent=4)

    print("\n--- 3-WAY ABLATION & BENCHMARK COMPARISON (N = 50 QUERIES) ---")
    print(
        f"{'System Configuration':<32} | {'HR@1':<8} | {'HR@3':<8} | {'MRR':<8} | "
        f"{'Modality Acc':<12} | {'1-Hop XRef':<16} | {'Conflict Acc':<12} | {'Latency'}"
    )
    print("-" * 118)
    for label, key in [
        ("1. Naive Baseline (500-char)", "baseline"),
        ("2. Method-1 (Scoped-Deontic)", "method1_scoped_deontic"),
        ("3. Full SDRAG (Method-1 + 2)", "full_sdrag_method1_plus_method2"),
    ]:
        m = metrics_payload["Ablation_Comparison"][key]
        print(
            f"{label:<32} | {m['HR_At_1']:<8} | {m['HR_At_3']:<8} | {m['MRR']:<8.4f} | "
            f"{m['Deontic_Modality_Accuracy']:<12} | {m['Strict_1Hop_Graph_Accuracy']:<16} | "
            f"{m['Conflict_Engine_Accuracy']:<12} | {m['Mean_Latency_ms']:.2f} ms"
        )

    print("\n--- METHOD 1: SCOPED-DEONTIC MODALITY EXTRACTION BREAKDOWN ---")
    for mod_name, m_info in modality_breakdown.items():
        print(
            f" • {mod_name:<12} (n={m_info['count']:2d}) -> HR@1: {m_info['hr_at_1']} | "
            f"Modality Extraction Acc: {m_info['modality_extraction_acc']}"
        )

    print("\n--- METHOD 2: 1-HOP GRAPH & CONFLICT ENGINE STATE DISTRIBUTION ---")
    for state_name, count in conflict_distribution.items():
        print(f" • {state_name:<38} : {count:2d} queries")

    print("\n[✓] Saved full 50-query trace to runs/sdrag_benchmark_trace.csv")
    print("[✓] Saved JSON benchmark metrics to runs/sdrag_metrics_summary.json")


if __name__ == "__main__":
    evaluate_sdrag_pipeline()
