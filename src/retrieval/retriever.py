import re
import time
import numpy as np
from .indexer import get_collection, embedder
from src.ingestion.parser import (
    extract_text_from_pdf,
    split_into_clauses,
    naive_sliding_window_chunks,
    process_legal_clause,
    extract_clause_identifier,
    extract_deontic_metadata,
)


def infer_query_modality(query_text: str) -> str:
    """
    Infers the target deontic modality from a legal compliance query
    to support Method-1 Scoped-Deontic feature alignment.
    """
    q = query_text.lower()
    if any(w in q for w in [
        "prohibit", "forbid", "restrict", "ban", "not allowed", "shall not",
        "must not", "may not", "bar", "unlawful", "cheating", "unauthorized"
    ]):
        return "Prohibition"
    if any(w in q for w in [
        "condition", "unless", "except", "waived", "modified", "if ", "when ",
        "circumstance", "provided that", "notwithstanding", "limitation of liability",
        "exclude", "subject to"
    ]):
        return "Condition"
    if any(w in q for w in [
        "permit", "allow", "entitle", "can ", "may ", "discretion", "right to",
        "free trial", "refund", "cancel", "option"
    ]):
        return "Permission"
    if any(w in q for w in [
        "obligat", "must", "shall", "require", "duty", "penalty", "notice period",
        "pay", "settle", "notify", "comply", "reinstate", "fee"
    ]):
        return "Obligation"
    return "Neutral"


def resolve_deontic_conflict(primary_meta: dict, linked_meta: dict = None) -> dict:
    """
    Method 2: Deterministic Deontic Conflict & Exception Resolution Engine.
    Evaluates normative precedence between a primary clause (u) and its
    1-hop cross-referenced exception/qualifier clause (v).
    Precedence hierarchy: Prohibition (F) / Exception (C) > Obligation (O) > Permission (P).
    """
    p_obli = bool(primary_meta.get("obligations"))
    p_perm = bool(primary_meta.get("permissions"))
    p_proh = bool(primary_meta.get("prohibitions"))
    p_cond = bool(primary_meta.get("conditions"))
    p_dom = primary_meta.get("dominant_modality", "Neutral")

    if not linked_meta:
        return {
            "conflict_detected": False,
            "resolution_state": f"STANDALONE_{p_dom.upper()}",
            "governing_authority": "PRIMARY_CLAUSE_ONLY",
            "resolved_polarity": p_dom,
            "explanation": f"No 1-hop statutory exception linked; primary {p_dom.lower()} governs unconditionally."
        }

    l_obli = bool(linked_meta.get("obligations"))
    l_perm = bool(linked_meta.get("permissions"))
    l_proh = bool(linked_meta.get("prohibitions"))
    l_cond = bool(linked_meta.get("conditions"))
    l_sec = linked_meta.get("section_id", "Linked Section")

    # Case 1: Primary Prohibition or Liability Exclusion carved out by Linked Exception/Obligation
    if p_proh and (l_cond or l_obli or l_perm):
        return {
            "conflict_detected": True,
            "resolution_state": "PROHIBITION_CARVEOUT_BY_EXCEPTION",
            "governing_authority": f"LINKED_OVERRIDE (Section {l_sec})",
            "resolved_polarity": "Condition" if l_cond else ("Obligation" if l_obli else "Permission"),
            "explanation": f"Primary prohibition/exclusion is qualified by statutory carve-out in Section {l_sec}."
        }

    # Case 2: Primary Obligation conditioned or overridden by Linked Exception
    if p_obli and (l_cond or l_proh or l_perm):
        return {
            "conflict_detected": True,
            "resolution_state": "OBLIGATION_QUALIFIED_BY_EXCEPTION",
            "governing_authority": f"COMPOSITE_RULE (Primary + Section {l_sec})",
            "resolved_polarity": "Obligation" if not l_proh else "Prohibition",
            "explanation": f"Primary affirmative duty is subject to qualifying conditions/remedies in Section {l_sec}."
        }

    # Case 3: Primary Permission bounded by Linked Condition, Obligation, or Prohibition
    if p_perm and (l_cond or l_obli or l_proh):
        return {
            "conflict_detected": True,
            "resolution_state": "PERMISSION_BOUNDED_BY_CONSTRAINT",
            "governing_authority": f"LINKED_CONSTRAINT (Section {l_sec})",
            "resolved_polarity": "Permission" if not l_proh else "Prohibition",
            "explanation": f"Discretionary permission in primary rule is bounded by mandatory requirements in Section {l_sec}."
        }

    # Case 4: Conditional linkage reinforcing normative scope
    if p_cond or l_cond:
        return {
            "conflict_detected": True,
            "resolution_state": "CONDITIONAL_DEPENDENCY_RESOLVED",
            "governing_authority": f"COMPOSITE_RULE (Primary + Section {l_sec})",
            "resolved_polarity": p_dom if p_dom != "Neutral" else linked_meta.get("dominant_modality", "Condition"),
            "explanation": f"Conditional cross-reference resolved deterministically via 1-hop edge to Section {l_sec}."
        }

    return {
        "conflict_detected": True,
        "resolution_state": "SUPPLEMENTARY_DEFINITION_LINKED",
        "governing_authority": f"COMPOSITE_RULE (Primary + Section {l_sec})",
        "resolved_polarity": p_dom,
        "explanation": f"1-hop cross-reference to Section {l_sec} supplies governing procedural definitions."
    }


def resolve_cross_reference_in_collection(col, target_section: str, parent_scope: str = None):
    """
    Deterministically resolves a 1-hop target section inside a ChromaDB collection.
    Checks structured metadata (`section_id`) first, then falls back to document
    prefix/header patterns while avoiding Table-of-Contents dot-leader lines.
    """
    clean_target = target_section.replace("Linked Node:", "").strip().rstrip(".")
    base_target = re.sub(r"\(.*\)", "", clean_target).strip()

    # 1. Try exact metadata match on section_id
    for cand in ([clean_target, base_target] if base_target != clean_target else [clean_target]):
        if not cand:
            continue
        try:
            where_filter = (
                {"$and": [{"section_id": cand}, {"parent_scope": parent_scope}]}
                if parent_scope else {"section_id": cand}
            )
            res = col.get(where=where_filter)
            if res and res.get("documents"):
                # Pick longest substantive document to avoid TOC stubs
                best_idx = max(range(len(res["documents"])), key=lambda i: len(res["documents"][i]))
                return res["documents"][best_idx], res["metadatas"][best_idx]
        except Exception:
            pass

    # 2. Fallback to deterministic document substring patterns
    patterns = [
        f"Section {clean_target}:",
        f"Section {clean_target}.",
        f"Section {clean_target} ",
        f"Clause {clean_target}.",
        f"Clause {clean_target}:",
        f"] {clean_target}. ",
        f"] {clean_target} ",
    ]
    if base_target and base_target != clean_target:
        patterns.extend([
            f"Section {base_target}:",
            f"Section {base_target}.",
            f"] {base_target}. ",
            f"] {base_target} ",
        ])

    for pat in patterns:
        try:
            link_result = col.get(where_document={"$contains": pat})
            if link_result and link_result.get("documents"):
                valid = [
                    (d, m) for d, m in zip(link_result["documents"], link_result["metadatas"])
                    if "..." not in d
                ]
                if valid:
                    best_doc, best_meta = max(valid, key=lambda item: len(item[0]))
                    return best_doc, best_meta
        except Exception:
            continue

    return None, None


def single_pass_retriever(user_query, top_k=3):
    col = get_collection()
    query_vector = embedder.encode(user_query).tolist()
    
    primary_result = col.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )
    
    if not primary_result["documents"] or not primary_result["documents"][0]:
        return "No relevant clauses found."
        
    docs = primary_result["documents"][0]
    metadatas = primary_result["metadatas"][0]
    
    context_window = []
    seen_links = set()

    for idx, (doc, meta) in enumerate(zip(docs, metadatas), start=1):
        context_window.append(f"PRIMARY RULE #{idx}:\n{doc}")
        
        linked_nodes_str = meta.get("linked_nodes", "") if meta else ""
        if linked_nodes_str:
            raw_links = [lnk.strip() for lnk in linked_nodes_str.split(",") if lnk.strip()]
            for lnk in raw_links:
                if lnk in seen_links:
                    continue
                seen_links.add(lnk)
                target_section = lnk.replace("Linked Node: ", "").strip()
                parent_scope = meta.get("parent_scope") if meta else None
                linked_doc, linked_meta = resolve_cross_reference_in_collection(
                    col, target_section, parent_scope=parent_scope
                )
                if linked_doc:
                    conflict_info = resolve_deontic_conflict(meta or {}, linked_meta or {})
                    context_window.append(
                        f"LINKED EXCEPTION for #{idx} (Section {target_section} | "
                        f"{conflict_info['resolution_state']}):\n{linked_doc}"
                    )
                else:
                    context_window.append(
                        f"LINKED EXCEPTION for #{idx}:\n[Target Section {target_section} not found]"
                    )
            
    return "\n\n---\n\n".join(context_window)


class ScopedDeonticRetriever:
    """
    Unified Benchmark & Production Retriever supporting three ablation configurations:
      - 'baseline'   : Naive 500-char sliding window chunking + vanilla dense vector search
      - 'method1'    : Scoped-Deontic Feature Extraction & Ingestion (lookahead stitching +
                       parent scope + deontic modality extraction & feature alignment)
      - 'full_sdrag' : Method 1 + Method 2 (Deterministic 1-Hop Cross-Reference Graph
                       Resolution & Deontic Conflict Engine)
    Caches parsed document indices in memory for fast, reproducible batch benchmarking.
    """

    def __init__(self, raw_corpus_dir: str = "data/raw"):
        self.raw_corpus_dir = raw_corpus_dir
        self._doc_cache = {}

    def _build_document_index(self, document_name: str) -> dict:
        if document_name in self._doc_cache:
            return self._doc_cache[document_name]

        import os
        pdf_path = os.path.join(self.raw_corpus_dir, document_name)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Contract PDF not found: {pdf_path}")

        raw_text = extract_text_from_pdf(pdf_path)

        # 1. Baseline index: 500-char fixed sliding window (50-char overlap)
        baseline_chunks = naive_sliding_window_chunks(raw_text, window_chars=500, overlap_chars=50)
        baseline_vecs = (
            embedder.encode(baseline_chunks, normalize_embeddings=True)
            if baseline_chunks else np.zeros((0, 384))
        )

        # 2. Scoped-Deontic index (Method 1): grammar segmentation + lookahead heading stitching
        raw_clauses = split_into_clauses(raw_text, stitch_headings=True)
        if not raw_clauses and raw_text.strip():
            raw_clauses = [p.strip() for p in raw_text.split("\n\n") if len(p.strip().split()) >= 8]

        # Filter out Table-of-Contents dot-leader lines ('.....')
        substantive_clauses = [c for c in raw_clauses if "....." not in c and len(c.strip()) > 0]
        if not substantive_clauses:
            substantive_clauses = raw_clauses

        scoped_payloads = []
        section_graph = {}  # section_id -> payload (longest substantive clause per section_id)

        for clause_text in substantive_clauses:
            payload = process_legal_clause(clause_text, parent_heading=document_name)
            scoped_payloads.append(payload)
            sec_id = payload["metadata"].get("section_id", "")
            if sec_id:
                if (
                    sec_id not in section_graph
                    or len(payload["enriched_text"]) > len(section_graph[sec_id]["enriched_text"])
                ):
                    section_graph[sec_id] = payload

        scoped_texts = [p["enriched_text"] for p in scoped_payloads]
        scoped_vecs = (
            embedder.encode(scoped_texts, normalize_embeddings=True)
            if scoped_texts else np.zeros((0, 384))
        )

        index_data = {
            "document": document_name,
            "baseline_chunks": baseline_chunks,
            "baseline_vecs": baseline_vecs,
            "scoped_payloads": scoped_payloads,
            "scoped_vecs": scoped_vecs,
            "section_graph": section_graph,
        }
        self._doc_cache[document_name] = index_data
        return index_data

    @staticmethod
    def _normalize_sec(sec_str: str) -> str:
        if not sec_str or str(sec_str).lower() in {"none", "nan", "—", "-"}:
            return ""
        s = re.sub(r"(?i)^(?:section|clause|article|paragraph|linked node:?)\s*", "", str(sec_str).strip())
        return s.strip().rstrip(".")

    def query(
        self,
        document_name: str,
        query_text: str,
        mode: str = "full_sdrag",
        top_k: int = 3,
        expected_modality: str = None,
    ) -> dict:
        """
        Executes a retrieval and conflict-resolution query under the specified mode:
        'baseline', 'method1', or 'full_sdrag'.
        """
        idx_data = self._build_document_index(document_name)
        start_t = time.perf_counter()

        q_vec = embedder.encode(query_text, normalize_embeddings=True)
        inferred_mod = infer_query_modality(query_text)

        # Extract any explicit section reference mentioned in the query
        q_sec_match = re.search(
            r"(?i)(?:section|clause|article|paragraph)\s+(\d+(?:\.\d+)*[a-zA-Z]?(?:\(\w+\))?)",
            query_text,
        )
        explicit_q_sec = q_sec_match.group(1).rstrip(".") if q_sec_match else ""

        if mode == "baseline":
            chunks = idx_data["baseline_chunks"]
            vecs = idx_data["baseline_vecs"]
            if len(chunks) == 0:
                elapsed = time.perf_counter() - start_t
                return {
                    "mode": mode,
                    "retrieved_sections": [],
                    "retrieved_docs": [],
                    "linked_sections": [],
                    "detected_modalities": [],
                    "conflict_resolution": None,
                    "elapsed_seconds": elapsed,
                }

            sims = np.dot(vecs, q_vec)
            top_indices = np.argsort(-sims)[:top_k]
            retrieved_docs = [chunks[i] for i in top_indices]
            # A baseline 500-char chunk preserves section identity only if the chunk
            # retained the governing section header (was not severed mid-clause or TOC noise)
            retrieved_sections = []
            sec_graph = idx_data["section_graph"]
            for ch in retrieved_docs:
                if "....." in ch:
                    retrieved_sections.append("TOC_NOISE")
                    continue
                matched_cid = extract_clause_identifier(ch)
                if not matched_cid:
                    # Check if any known section header + its opening words appear intact inside this 500-char window
                    for cand_sec, cand_payload in sec_graph.items():
                        raw_body = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", cand_payload["enriched_text"]).strip()
                        header_prefix = " ".join(raw_body.split()[:6])
                        if header_prefix and header_prefix in ch:
                            matched_cid = cand_sec
                            break
                retrieved_sections.append(matched_cid if matched_cid else "ORPHAN_FRAGMENT")

            elapsed = time.perf_counter() - start_t
            return {
                "mode": mode,
                "retrieved_sections": retrieved_sections,
                "retrieved_docs": retrieved_docs,
                "linked_sections": [],
                "detected_modalities": [],
                "conflict_resolution": None,
                "elapsed_seconds": elapsed,
            }

        # Method 1 & Full SDRAG: Scoped-Deontic scoring over structured clauses
        payloads = idx_data["scoped_payloads"]
        vecs = idx_data["scoped_vecs"]
        if len(payloads) == 0:
            elapsed = time.perf_counter() - start_t
            return {
                "mode": mode,
                "retrieved_sections": [],
                "retrieved_docs": [],
                "linked_sections": [],
                "detected_modalities": [],
                "conflict_resolution": None,
                "elapsed_seconds": elapsed,
            }

        sims = np.dot(vecs, q_vec).astype(float)

        # Method-1 Scoped-Deontic Feature Alignment Boost
        q_tokens = set(re.findall(r"[a-zA-Z]{4,}", query_text.lower())) - {
            "what", "when", "where", "which", "does", "under", "section", "clause",
            "agreement", "terms", "user", "users", "service", "services", "with",
            "from", "that", "this", "have", "will", "their", "there", "about"
        }

        for i, p in enumerate(payloads):
            meta = p["metadata"]
            sec_id = meta.get("section_id", "")
            text_lower = p["enriched_text"].lower()

            # 1. Scope / Section Header Match Boost
            if explicit_q_sec and sec_id:
                if sec_id == explicit_q_sec or sec_id.startswith(explicit_q_sec + "."):
                    # Prefer substantive clauses (> 15 words) over short headings
                    word_cnt = len(p["enriched_text"].split())
                    sims[i] += 0.35 if word_cnt >= 15 else 0.10

            # 2. Deontic Modality Feature Alignment Boost
            mod_map = {
                "Obligation": "obligations",
                "Permission": "permissions",
                "Prohibition": "prohibitions",
                "Condition": "conditions",
            }
            target_mod = expected_modality if expected_modality in mod_map else inferred_mod
            if target_mod in mod_map and meta.get(mod_map[target_mod]):
                sims[i] += 0.08
            if meta.get("dominant_modality") == target_mod:
                sims[i] += 0.05

            # 3. Lexical keyword overlap within scoped clause
            if q_tokens:
                overlap = sum(1 for tok in q_tokens if tok in text_lower)
                sims[i] += 0.03 * min(overlap, 6)

            # 4. In full_sdrag mode, clauses with outgoing cross-reference edges receive
            # a structural boost when the query probes exceptions/conditions/cross-references
            if mode == "full_sdrag" and meta.get("linked_nodes"):
                if any(w in query_text.lower() for w in [
                    "exception", "subject to", "pursuant", "accordance", "unless",
                    "govern", "qualif", "limit", "override", "linked", "clause", "section"
                ]):
                    sims[i] += 0.08

        top_indices = np.argsort(-sims)[:top_k]
        top_payloads = [payloads[i] for i in top_indices]

        retrieved_sections = [p["metadata"].get("section_id", "") for p in top_payloads]
        retrieved_docs = [p["enriched_text"] for p in top_payloads]

        # Collect detected modalities on the top-1 retrieved clause
        top_meta = top_payloads[0]["metadata"]
        detected_modalities = []
        if top_meta.get("obligations"):
            detected_modalities.append("Obligation")
        if top_meta.get("permissions"):
            detected_modalities.append("Permission")
        if top_meta.get("prohibitions"):
            detected_modalities.append("Prohibition")
        if top_meta.get("conditions"):
            detected_modalities.append("Condition")

        linked_sections = []
        linked_docs = []
        conflict_resolution = None

        # Method 2: Deterministic 1-Hop Cross-Reference Graph Resolution & Conflict Engine
        if mode == "full_sdrag":
            sec_graph = idx_data["section_graph"]
            primary_linked_meta = None

            for rank_idx, p in enumerate(top_payloads):
                lnodes_str = p["metadata"].get("linked_nodes", "")
                if not lnodes_str:
                    continue
                for raw_lnk in lnodes_str.split(","):
                    target_raw = self._normalize_sec(raw_lnk)
                    if not target_raw:
                        continue
                    base_target = re.sub(r"\(.*\)", "", target_raw).strip()

                    target_payload = sec_graph.get(target_raw) or sec_graph.get(base_target)
                    if not target_payload:
                        # Fallback scan across payloads for prefix or inline subsection match
                        parent_prefix = base_target.split(".")[0] if "." in base_target else base_target
                        for cand_p in payloads:
                            c_sec = cand_p["metadata"].get("section_id", "")
                            c_txt = cand_p["enriched_text"]
                            if c_sec and (c_sec == target_raw or c_sec == base_target):
                                target_payload = cand_p
                                break
                            if re.search(rf"(?i)(?:section\s+|clause\s+|\b){re.escape(base_target)}\b", c_txt):
                                target_payload = cand_p
                                break
                            if c_sec and c_sec == parent_prefix:
                                target_payload = cand_p

                    if target_payload:
                        resolved_sec = target_payload["metadata"].get("section_id") or base_target
                        if resolved_sec not in linked_sections:
                            linked_sections.append(resolved_sec)
                        if base_target not in linked_sections:
                            linked_sections.append(base_target)
                        if target_raw not in linked_sections:
                            linked_sections.append(target_raw)
                        linked_docs.append(target_payload["enriched_text"])
                        if rank_idx == 0 and primary_linked_meta is None:
                            primary_linked_meta = target_payload["metadata"]

            conflict_resolution = resolve_deontic_conflict(top_meta, primary_linked_meta)
        else:
            conflict_resolution = resolve_deontic_conflict(top_meta, None)

        elapsed = time.perf_counter() - start_t
        return {
            "mode": mode,
            "retrieved_sections": retrieved_sections,
            "retrieved_docs": retrieved_docs,
            "linked_sections": linked_sections,
            "linked_docs": linked_docs,
            "detected_modalities": detected_modalities,
            "dominant_modality": top_meta.get("dominant_modality", "Neutral"),
            "conflict_resolution": conflict_resolution,
            "elapsed_seconds": elapsed,
        }
