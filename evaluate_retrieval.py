import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import time
import numpy as np
from src.retrieval.indexer import get_collection, embedder

# 1. Standard Benchmark
GROUND_TRUTH = [
    {
        "query": "What are the rules and penalties regarding operator staff using guest rooms?",
        "keywords": ["22. Use of Rooms by Operator Staff", "15x of ARR"]
    },
    {
        "query": "How does the digital audit policy work and what is the penalty for non-compliance?",
        "keywords": ["20. Digital Audit Policy", "undisclosed occupancy"]
    },
    {
        "query": "What happens if a user is denied check-in after confirming a booking?",
        "keywords": ["4. If any check-in is denied", "alternate accommodation"]
    },
    {
        "query": "What is the penalty if an operator denies access to an audit?",
        "keywords": ["30. Audit Access and Denial Penalty", "20,000"]
    },
    {
        "query": "What are the rules regarding staff uniforms?",
        "keywords": ["26. Uniform Audit", "10,000"]
    }
]

# 2. Stress / Paraphrased Benchmark
HARD_TESTS = [
    {
        "query": "Can hotel workers sleep in the vacant guest rooms?",
        "keywords": ["use of rooms by operator staff", "15x of arr"]
    },
    {
        "query": "What if the hotel is full when I arrive with a confirmed booking?",
        "keywords": ["check-in is denied", "alternate accommodation"]
    },
    {
        "query": "What fine is charged if the hotel manager refuses to let inspectors in?",
        "keywords": ["denial of audit access", "20,000"]
    }
]

def evaluate_retriever(test_dataset, k_values=[1, 3]):
    collection = get_collection()
    
    hits = {k: 0 for k in k_values}
    reciprocal_ranks = []
    latencies = []
    
    total_queries = len(test_dataset)
    print(f"Running evaluation across {total_queries} test cases...\n")

    for item in test_dataset:
        query = item["query"]
        
        # Support either 'keywords' (list/str) or legacy 'expected_keyword' (str)
        raw_keywords = item.get("keywords") or item.get("expected_keyword", [])
        if isinstance(raw_keywords, str):
            target_keywords = [raw_keywords]
        else:
            target_keywords = raw_keywords
        
        # Measure latency
        start_time = time.perf_counter()
        query_vec = embedder.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=max(k_values)
        )
        latencies.append((time.perf_counter() - start_time) * 1000)

        retrieved_docs = results.get("documents", [[]])[0]
        
        # Check rank of target clause
        found_rank = None
        for rank_idx, doc in enumerate(retrieved_docs, start=1):
            doc_lower = doc.lower()
            if any(kw.lower() in doc_lower for kw in target_keywords):
                found_rank = rank_idx
                break
        
        # Calculate MRR
        if found_rank is not None:
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)
            
        # Calculate Hits@K
        for k in k_values:
            if found_rank is not None and found_rank <= k:
                hits[k] += 1

    # Print results
    print("=" * 45)
    print("        RETRIEVAL EVALUATION RESULTS        ")
    print("=" * 45)
    for k in k_values:
        hit_rate = (hits[k] / total_queries) * 100
        print(f"Hit Rate @ {k}:         {hit_rate:.2f}% ({hits[k]}/{total_queries})")
    
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    avg_latency = np.mean(latencies) if latencies else 0.0
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Average Retrieval Latency:  {avg_latency:.2f} ms")
    print("=" * 45)

if __name__ == "__main__":
    print("\n--- Running Evaluation on Standard Ground Truth ---")
    evaluate_retriever(GROUND_TRUTH, k_values=[1, 3])

    print("\n--- Running Evaluation on Hard / Paraphrased Tests ---")
    evaluate_retriever(HARD_TESTS, k_values=[1, 3])