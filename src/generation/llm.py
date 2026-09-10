#llm.py is meant to handle the "G" in RAG (Generation)
'''
# Until now, the system is a world-class Retrieval engine. 

# If a user asks a question, the code until now will flawlessly hunt down the exact primary clause and its deterministic linked exception.

However, if we stop there, the system will just output the raw legal text to the user. To make it a true Retrieval-Augmented Generation pipeline, an LLM takes that raw text and translates it into plain English (e.g., generating bullet points).

We (I, and the future me reading this) will use a local Ollama model to do this, so that we don't have to worry about sending sensitive legal data to the cloud, and avoid all API keys, token limits, and external dependencies (Gemini API for example).
'''
import ollama

def generate_legal_summary(depolluted_context, user_query, model_name="llama3"):
    """
    Sends the strictly retrieved context and user query to a local Ollama model
    to generate a hallucination-free summary.
    """
    
    # Using a strict system prompt to prevent the LLM from making up legal advice
    system_prompt = f"""
    You are an expert legal AI assistant. Your job is to answer the user's query 
    strictly using the provided TEXT TO ANALYZE.
    
    Rules:
    1. Summarize the PRIMARY RULE and any LINKED EXCEPTIONS.
    2. Output your answer in short, concise bullet points.
    3. Do NOT bring in outside knowledge. If the answer is not in the text, say so.
    
    TEXT TO ANALYZE:
    {depolluted_context}
    """
    
    print(f"Sending context to local {model_name} via Ollama...")
    
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_query}
            ]
        )
        return response['message']['content']
    
    except Exception as e:
        return f"Error communicating with Ollama: {str(e)}\nMake sure Ollama is running and the model is downloaded!"


