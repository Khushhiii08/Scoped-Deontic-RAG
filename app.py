import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
import tempfile
import streamlit as st
from urllib.parse import urlparse
from src.ingestion.parser import (
    extract_text_from_pdf,
    extract_text_from_url,
    split_into_clauses,
    process_legal_clause
)
from src.retrieval.indexer import index_clause
from src.retrieval.retriever import single_pass_retriever
from src.generation.llm import generate_legal_summary

st.set_page_config(page_title="makeLEGalWORKeasy", page_icon="⚖️", layout="wide")
st.title("makeLEGalWORKeasy: Scope-Aware & Deontic RAG")

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.header("📂 1. Ingestion Source")
    
    # Switch between PDF and Web URL tabs
    tab_pdf, tab_url = st.tabs(["📄 Upload PDF", "🌐 Website Link"])
    
    raw_text = None
    source_name = None

    with tab_pdf:
        uploaded_file = st.file_uploader("Upload ToS PDF", type=["pdf"])
        if st.button("Index PDF", use_container_width=True, type="primary") and uploaded_file:
            with st.spinner("Extracting text from PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                raw_text = extract_text_from_pdf(tmp_path)
                source_name = uploaded_file.name
                os.remove(tmp_path)

    with tab_url:
        input_url = st.text_input("Enter Terms & Conditions URL", placeholder="https://example.com/terms")
        if st.button("Index Webpage", use_container_width=True, type="primary") and input_url:
            with st.spinner("Fetching terms from URL..."):
                try:
                    raw_text = extract_text_from_url(input_url)
                    parsed_domain = urlparse(input_url).netloc
                    source_name = parsed_domain if parsed_domain else "Web Terms"
                except Exception as e:
                    st.error(f"Failed to fetch URL: {e}")

    # Process and index extracted text
    if raw_text and source_name:
        with st.spinner("Splitting and indexing clauses..."):
            clauses = split_into_clauses(raw_text)
            if not clauses:
                st.warning("No clauses detected in the document/page.")
            else:
                prog = st.progress(0, text="Indexing into ChromaDB...")
                for i, clause in enumerate(clauses):
                    payload = process_legal_clause(clause, parent_heading=source_name)
                    index_clause(payload)
                    prog.progress((i + 1) / len(clauses))
                st.success(f"✅ Successfully indexed {len(clauses)} clauses from {source_name}!")
                
    st.divider()
    if st.button("🗑️ Reset Indexed Data", use_container_width=True):
        from src.retrieval.indexer import chroma_client
        try:
            # 1. Reset vector storage collection
            chroma_client.delete_collection("legal_clauses")
            chroma_client.get_or_create_collection("legal_clauses")
            
            # 2. Clear query chat history from state
            st.session_state.messages = []
            
            st.success("Database and chat history cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"Error resetting database: {e}")

with col2:
    st.header("💬 2. Query the Engine")
    chat_container = st.container(height=450)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if user_query := st.chat_input("Ask a question about the indexed terms..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_query)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing context..."):
                    depolluted_context = single_pass_retriever(user_query)
                    if "No relevant clauses found" in depolluted_context:
                        response = "I could not find any relevant clauses in the indexed documents."
                    else:
                        response = generate_legal_summary(depolluted_context, user_query, model_name="llama3")
                    st.markdown(response)
                    with st.expander("🔍 View Retrieved Raw Context"):
                        st.text(depolluted_context)
            st.session_state.messages.append({"role": "assistant", "content": response})