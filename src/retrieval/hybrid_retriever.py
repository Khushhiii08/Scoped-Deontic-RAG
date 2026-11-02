"""
Scoped-Deontic Hybrid Relational Retriever & Conflict Resolution Engine.
Re-exports ScopedDeonticRetriever and helper functions from src.retrieval.retriever.
"""

from .retriever import (
    ScopedDeonticRetriever,
    single_pass_retriever,
    resolve_deontic_conflict,
    resolve_cross_reference_in_collection,
    infer_query_modality,
)

__all__ = [
    "ScopedDeonticRetriever",
    "single_pass_retriever",
    "resolve_deontic_conflict",
    "resolve_cross_reference_in_collection",
    "infer_query_modality",
]
