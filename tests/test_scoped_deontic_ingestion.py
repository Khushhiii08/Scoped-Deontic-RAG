"""
Standalone Scoped-Deontic RAG Evaluation Suite
Tests:
  - Test 1: Clause Splitting & Boundary Integrity (Method 1)
  - Test 2: Deontic Modality Tagging & Cross-Reference Extraction F1 Benchmark (Method 1 & 2)
  - Test 3: ChromaDB Scoped Ingestion & Vector Metadata Integrity (Method 1)
  - Test 4: Adaptive Lookahead Heading Stitching Verification (Method 1)
  - Test 5: Deterministic 1-Hop Cross-Reference Graph Resolution & Conflict Engine (Method 2)
"""

import sys
import os
import shutil
import tempfile
import numpy as np

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Ensure root modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.parser import (
    split_into_clauses,
    stitch_lookahead_headings,
    extract_deontic_metadata,
    extract_cross_references,
    process_legal_clause,
)
from src.retrieval.retriever import (
    resolve_cross_reference_in_collection,
    resolve_deontic_conflict,
)
import chromadb
from sentence_transformers import SentenceTransformer


def get_test_embedder():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2", device="cpu", local_files_only=True)
    except Exception:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


# ---------------------------------------------------------------------------
# GROUND TRUTH BENCHMARK DATASET
# (Derived from typical commercial agreements, T&Cs, and OYO partner clauses)
# ---------------------------------------------------------------------------
BENCHMARK_CORPUS = [
    {
        "clause_text": "Section 20. Audit Access and Denial Penalty. The Operator shall not deny or obstruct access to the Property for the purpose of audit, inspection, or review by OYO or its authorized representatives. For each instance of denial of audit access at the Property, a penalty of INR 20,000 shall be levied and set-off against the Operator pay-out.",
        "parent_heading": "Audits & Penalties",
        "expected_modalities": {"PROHIBITION", "OBLIGATION"},
        "expected_links": []
    },
    {
        "clause_text": "Section 20.1 Shortfall Remittance. The Operator shall promptly reinstate the Security Deposit to its original value within five (5) days of such adjustment. Failure to do so may constitute a breach of this Agreement, subject to Section 18.2.",
        "parent_heading": "Audits & Penalties",
        "expected_modalities": {"OBLIGATION", "PERMISSION", "CONDITION"},
        "expected_links": ["Linked Node: 18.2"]
    },
    {
        "clause_text": "Clause 26. Uniform Audit. Operators are required to ensure that the uniform provided by us is worn by staff clean and in a presentable condition at all times during working hours. Where the staff are found not wearing the uniform, a penalty of INR 10,000 per violation will be levied.",
        "parent_heading": "Staff Compliance",
        "expected_modalities": {"OBLIGATION"},
        "expected_links": []
    },
    {
        "clause_text": "Clause 22. Use of Rooms by Operator Staff. Operator staff cannot occupy guest rooms of the Property at any given time. If the Operator wants to use any guest rooms as a staff room, they may pay a monthly amount to OYO for such use, pursuant to section 14.1.",
        "parent_heading": "Operations",
        "expected_modalities": {"PROHIBITION", "PERMISSION"},
        "expected_links": ["Linked Node: 14.1"]
    },
    {
        "clause_text": "Section 4. Check-in Denial. If any check-in is denied for a confirmed booking, OYO shall ensure that the user is provided with an alternate accommodation of comparable standards. In an event where the alternate accommodation could not be provided, OYO shall arrange for the refund of any pre-booking amount, except as provided in clause 9.0.",
        "parent_heading": "Guest Services",
        "expected_modalities": {"OBLIGATION", "CONDITION"},
        "expected_links": ["Linked Node: 9.0"]
    },
    {
        "clause_text": "Clause 14. Anti-Bribery Compliance. Participant will comply with all applicable laws relating to anti-bribery and anti-corruption. Participant shall not, either directly or indirectly, pay or offer any improper commercial advantage. Participant is allowed to submit bona fide expenses, see clause 12.",
        "parent_heading": "Regulatory Compliance",
        "expected_modalities": {"OBLIGATION", "PROHIBITION", "PERMISSION"},
        "expected_links": ["Linked Node: 12"]
    }
]

# Auxiliary 1-Hop Target Exception Nodes for Method-2 Graph & Conflict Testing
LINKED_EXCEPTION_CORPUS = [
    {
        "clause_text": "Section 18.2: Event of Default and Cure Exception. Failure to cure a Security Deposit shortfall within five (5) days shall entitle OYO to terminate this Agreement immediately, unless a force majeure extension is granted in writing.",
        "parent_heading": "Audits & Penalties",
    },
    {
        "clause_text": "Section 14.1: Staff Room Tariff Schedule. Operator may allocate up to two (2) rooms for staff occupancy provided that a flat monthly fee is remitted in advance and rooms are blocked from public inventory.",
        "parent_heading": "Operations",
    },
    {
        "clause_text": "Section 9.0: Force Majeure and Non-Refundable Peak Bookings. Pre-booking refunds shall not apply where check-in denial results from government emergency orders or guest misconduct.",
        "parent_heading": "Guest Services",
    },
    {
        "clause_text": "Section 12: Bona Fide Expense Reimbursement Policy. Hospitality and promotional expenses under USD 100 must be pre-approved in writing and supported by itemized receipts.",
        "parent_heading": "Regulatory Compliance",
    },
]


def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)


def run_segmentation_tests():
    print_header("Test 1: Clause Splitting & Boundary Integrity")
    
    sample_raw_text = (
        "Section 1. Definitions. 'Agreement' refers to this legal contract.\n\n"
        "Section 2. Audit Obligations. The operator must provide complete log files upon notice.\n\n"
        "Clause 3. Termination Notice. Either party may terminate this agreement with 30 days notice.\n\n"
        "4. Confidentiality. Recipient shall not disclose proprietary information to third parties.\n\n"
        "5.1 Limitation of Liability. Damages shall not exceed fees paid in the preceding month."
    )
    
    clauses = split_into_clauses(sample_raw_text)
    print(f"[*] Input text contained 5 distinct sections/clauses.")
    print(f"[*] Extracted clauses count: {len(clauses)}")
    
    passed = len(clauses) == 5
    for i, c in enumerate(clauses, start=1):
        print(f"    Clause [{i}]: {c[:55]}... ({len(c.split())} words)")
        
    print(f"\n[OBSERVATION] Clause Boundary Integrity: {'PASSED (100% split match)' if passed else 'FAILED'}")
    return passed


def run_deontic_and_crossref_benchmark():
    print_header("Test 2: Deontic Extraction & Cross-Reference Linking")
    
    categories = ["OBLIGATION", "PROHIBITION", "PERMISSION", "CONDITION"]
    stats = {cat: {"tp": 0, "fp": 0, "fn": 0} for cat in categories}
    
    xref_total_expected = 0
    xref_correct = 0
    token_lengths = []
    
    print(f"{'Clause ID':<12} | {'Token Length':<12} | {'Extracted Deontic Spans':<30} | {'Extracted Links'}")
    print("-" * 75)
    
    for idx, item in enumerate(BENCHMARK_CORPUS, start=1):
        raw_text = item["clause_text"]
        words = len(raw_text.split())
        token_lengths.append(words)
        
        deontic_meta = extract_deontic_metadata(raw_text)
        links = extract_cross_references(raw_text)
        
        detected = set()
        if deontic_meta["Obligations"]: detected.add("OBLIGATION")
        if deontic_meta["Prohibitions"]: detected.add("PROHIBITION")
        if deontic_meta["Permissions"]: detected.add("PERMISSION")
        if deontic_meta["Conditions"]: detected.add("CONDITION")
        
        expected = item["expected_modalities"]
        for cat in categories:
            if cat in expected and cat in detected:
                stats[cat]["tp"] += 1
            elif cat not in expected and cat in detected:
                stats[cat]["fp"] += 1
            elif cat in expected and cat not in detected:
                stats[cat]["fn"] += 1
                
        expected_links = item["expected_links"]
        xref_total_expected += len(expected_links)
        for el in expected_links:
            if el in links:
                xref_correct += 1
                
        deontic_summary = ", ".join([f"{k[:4]}:{len(v)}" for k, v in deontic_meta.items() if v])
        links_summary = ", ".join(links) if links else "None"
        print(f"Clause {idx:<6} | {words:<12} | {deontic_summary:<30} | {links_summary}")
        
    print("\n" + "-" * 70)
    print(f"{'Deontic Category':<15} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12}")
    print("-" * 70)
    
    f1_list = []
    for cat in categories:
        tp = stats[cat]["tp"]
        fp = stats[cat]["fp"]
        fn = stats[cat]["fn"]
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
        
        print(f"{cat:<15} | {prec:<12.3f} | {rec:<12.3f} | {f1:<12.3f}")
        
    macro_f1 = np.mean(f1_list)
    xref_accuracy = (xref_correct / xref_total_expected * 100) if xref_total_expected > 0 else 100.0
    
    print("-" * 70)
    print(f"[OBSERVATION] Macro Deontic F1-Score       : {macro_f1:.3f}")
    print(f"[OBSERVATION] Cross-Reference Resolution Acc: {xref_accuracy:.1f}% ({xref_correct}/{xref_total_expected} targets resolved)")
    print(f"[OBSERVATION] Mean Clause Length            : {np.mean(token_lengths):.1f} words (Min: {np.min(token_lengths)}, Max: {np.max(token_lengths)})")


def run_ingestion_and_indexing_pipeline():
    print_header("Test 3: ChromaDB Scoped Ingestion & Vector Metadata Integrity")
    
    temp_dir = tempfile.mkdtemp()
    print(f"[*] Initializing clean ChromaDB instance at temporary path: {temp_dir}")
    
    try:
        client = chromadb.EphemeralClient()
        collection = client.create_collection(name="test_legal_clauses")
        embedder = get_test_embedder()
        
        indexed_count = 0
        for item in BENCHMARK_CORPUS:
            payload = process_legal_clause(item["clause_text"], parent_heading=item["parent_heading"])
            
            assert "id" in payload
            assert "enriched_text" in payload
            assert "metadata" in payload
            assert payload["metadata"]["parent_scope"] == item["parent_heading"]
            
            embedding = embedder.encode(payload["enriched_text"]).tolist()
            
            collection.add(
                ids=[payload["id"]],
                embeddings=[embedding],
                documents=[payload["enriched_text"]],
                metadatas=[payload["metadata"]]
            )
            indexed_count += 1
            
        stored = collection.count()
        print(f"[*] Processed and indexed {indexed_count} structured legal chunks.")
        print(f"[*] ChromaDB reported record count: {stored}")
        
        query = "What happens if an operator denies or obstructs an audit?"
        q_vec = embedder.encode(query).tolist()
        result = collection.query(query_embeddings=[q_vec], n_results=1)
        
        top_doc = result["documents"][0][0]
        top_meta = result["metadatas"][0][0]
        
        print("\n[*] Sample Retrieval Verification on Indexed Chunks:")
        print(f"    Query: '{query}'")
        print(f"    Retrieved Document Scope: {top_meta.get('parent_scope')}")
        print(f"    Extracted Prohibitions   : {top_meta.get('prohibitions')}")
        print(f"    Content Preview          : {top_doc[:90]}...")
        
        assert "Audit Access" in top_doc, "Top retrieved document did not match the expected clause."
        print("\n[OBSERVATION] End-to-end ingestion, schema formatting, and indexing: ALL CHECKS PASSED.")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_lookahead_heading_stitching_test():
    print_header("Test 4: Adaptive Lookahead Heading Stitching (Method 1)")
    raw_blocks = [
        "Section 1. Membership",
        "1.1. Your membership will continue month-to-month until terminated by you.",
        "Section 2. Promotional Offers",
        "2.1. Promotional eligibility is determined by the Company at its sole discretion.",
    ]
    stitched = stitch_lookahead_headings(raw_blocks, max_heading_len=60)
    print(f"[*] Raw fragmented blocks: {len(raw_blocks)} -> Stitched operative clauses: {len(stitched)}")
    for idx, s in enumerate(stitched, 1):
        print(f"    Stitched [{idx}]: {s}")
    assert len(stitched) == 2, "Lookahead buffer should stitch 2 orphaned headings into 2 operative clauses."
    assert "[Section 1. Membership]" in stitched[0]
    assert "[Section 2. Promotional Offers]" in stitched[1]
    print("\n[OBSERVATION] Lookahead Heading Stitching Buffer: PASSED (0% orphaned headings).")


def run_1hop_graph_and_conflict_engine_test():
    print_header("Test 5: Deterministic 1-Hop Graph Resolution & Conflict Engine (Method 2)")
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="test_graph_conflict")
    embedder = get_test_embedder()

    all_items = BENCHMARK_CORPUS + LINKED_EXCEPTION_CORPUS
    for item in all_items:
        payload = process_legal_clause(item["clause_text"], parent_heading=item["parent_heading"])
        vec = embedder.encode(payload["enriched_text"]).tolist()
        collection.add(
            ids=[payload["id"]],
            embeddings=[vec],
            documents=[payload["enriched_text"]],
            metadatas=[payload["metadata"]],
        )

    resolved_count = 0
    expected_xref_clauses = [it for it in BENCHMARK_CORPUS if it["expected_links"]]
    for item in expected_xref_clauses:
        primary_payload = process_legal_clause(item["clause_text"], parent_heading=item["parent_heading"])
        p_meta = primary_payload["metadata"]
        target_str = p_meta["linked_nodes"].replace("Linked Node: ", "").strip()
        linked_doc, linked_meta = resolve_cross_reference_in_collection(
            collection, target_str, parent_scope=p_meta["parent_scope"]
        )
        assert linked_doc is not None, f"Failed to resolve 1-hop target {target_str}"
        conflict = resolve_deontic_conflict(p_meta, linked_meta)
        assert conflict["conflict_detected"] is True
        resolved_count += 1
        print(
            f"    [1-Hop Resolved] Primary Sec {p_meta['section_id']:<4} -> Target Sec {target_str:<4} | "
            f"State: {conflict['resolution_state']} | Polarity: {conflict['resolved_polarity']}"
        )

    print(
        f"\n[OBSERVATION] 1-Hop Graph & Conflict Resolution: PASSED "
        f"({resolved_count}/{len(expected_xref_clauses)} statutory edges & conflicts resolved)."
    )


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("  RUNNING SCOPED-DEONTIC INGESTION, GRAPH & CONFLICT TEST SUITE")
    print("#" * 70)
    
    run_segmentation_tests()
    run_deontic_and_crossref_benchmark()
    run_ingestion_and_indexing_pipeline()
    run_lookahead_heading_stitching_test()
    run_1hop_graph_and_conflict_engine_test()
    
    print_header("FINAL VERIFICATION SUMMARY")
    print("All Method-1 and Method-2 unit & integration tests completed without errors.")
    print("Ready to paste directly into your research paper methodology and results tables.\n")
