import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Disable verbose/buggy posthog telemetry
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_collection():
    """Always returns the active collection instance."""
    return chroma_client.get_or_create_collection(name="legal_clauses")

def index_clause(payload: dict):
    """Indexes a single processed clause using 'enriched_text'."""
    col = get_collection()
    
    clause_id = str(payload.get("id", ""))
    # Pull 'enriched_text' first
    clause_text = (
        payload.get("enriched_text")
        or payload.get("text")
        or payload.get("clause")
        or ""
    ).strip()

    if not clause_text:
        return

    metadata = payload.get("metadata", {})
    sanitized_metadata = {
        k: (v if isinstance(v, (str, int, float, bool)) else str(v))
        for k, v in metadata.items()
    }

    embedding = embedder.encode(clause_text).tolist()

    col.add(
        ids=[clause_id],
        embeddings=[embedding],
        documents=[clause_text],
        metadatas=[sanitized_metadata]
    )