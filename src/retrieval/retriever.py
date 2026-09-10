from .indexer import get_collection, embedder

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
        if linked_nodes_str and linked_nodes_str not in seen_links:
            seen_links.add(linked_nodes_str)
            target_section = linked_nodes_str.replace("Linked Node: ", "").strip()
            
            link_result = col.get(
                where_document={"$contains": f"Section {target_section}:"}
            )
            
            if link_result["documents"]:
                context_window.append(f"LINKED EXCEPTION for #{idx}:\n{link_result['documents'][0]}")
            else:
                context_window.append(f"LINKED EXCEPTION for #{idx}:\n[Target Section {target_section} not found]")
            
    return "\n\n---\n\n".join(context_window)