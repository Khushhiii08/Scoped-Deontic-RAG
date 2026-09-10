#To move the functions out of the notebook and into a separate Python file
#It helps import and use the functions from those files better, and also keeps the notebook cleaner and more organized.

import re
import spacy
import requests
try:
    import pymupdf as fitz
except ModuleNotFoundError:
    import fitz
from bs4 import BeautifulSoup
from spacy.matcher import Matcher
from playwright.sync_api import sync_playwright

# --- 1. Setup spaCy & NLP Matcher ---
# Load the model once when the file is imported to save memory
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

obligation_pattern = [{"LOWER": {"IN": ["shall", "must", "will", "agrees"]}}]
permission_pattern = [{"LOWER": {"IN": ["may", "can", "reserves", "allowed"]}}]
prohibition_pattern = [{"LOWER": {"IN": ["shall", "must", "may"]}}, {"LOWER": "not"}]
condition_pattern = [{"LOWER": {"IN": ["subject", "except", "unless", "provided", "notwithstanding"]}}]

matcher.add("OBLIGATION", [obligation_pattern])
matcher.add("PERMISSION", [permission_pattern])
matcher.add("PROHIBITION", [prohibition_pattern])
matcher.add("CONDITION", [condition_pattern])


# --- 2. Document Parsing Functions ---
def extract_text_from_pdf(pdf_path):
    """Extracts raw text from a PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def split_into_clauses(text):
    """Splits text based on standard legal numbering and headers."""
    pattern = re.compile(r'(?i)^(?:Section\s+\d+|Clause\s+\d+|\d+\.\d+|\d+\.)\s*', re.MULTILINE)
    splits = list(pattern.finditer(text))
    clauses = []
    
    for i, match in enumerate(splits):
        start = match.start()
        end = splits[i+1].start() if i + 1 < len(splits) else len(text)
        
        clause_text = text[start:end].strip()
        clause_text = re.sub(r'\n+', ' ', clause_text)
        clauses.append(clause_text)
        
    return clauses


# --- 3. Metadata Extraction Functions ---
def extract_deontic_metadata(clause_text):
    """Scans a clause and tags Deontic modalities and conditions."""
    doc = nlp(clause_text)
    matches = matcher(doc)
    metadata = {"Obligations": [], "Permissions": [], "Prohibitions": [], "Conditions": []}
    
    for match_id, start, end in matches:
        rule_type = nlp.vocab.strings[match_id]
        context_span = doc[start:min(end+5, len(doc))] 
        
        if rule_type == "OBLIGATION": metadata["Obligations"].append(context_span.text)
        elif rule_type == "PERMISSION": metadata["Permissions"].append(context_span.text)
        elif rule_type == "PROHIBITION": metadata["Prohibitions"].append(context_span.text)
        elif rule_type == "CONDITION": metadata["Conditions"].append(context_span.text)
            
    return metadata

def extract_cross_references(clause_text):
    """Scans a legal clause and extracts explicit 1-hop cross-references."""
    pattern = re.compile(
        r'(?i)(?:subject to|pursuant to|in accordance with|see|except as provided in)\s+'
        r'(?:section|clause|article|paragraph)\s+'
        r'(\d+(?:\.\d+)*[a-zA-Z]?(?:\(\w+\))?)'
    )
    matches = pattern.findall(clause_text)
    
    references = []
    for match in matches:
        references.append(f"Linked Node: {match}")
        
    return references

def process_legal_clause(clause_text, parent_heading="General"):
    enriched_text = f"[{parent_heading}] {clause_text}"
    deontic_tags = extract_deontic_metadata(clause_text)
    links = extract_cross_references(clause_text)
    
    return {
        "id": str(hash(clause_text)),
        "enriched_text": enriched_text,
        "metadata": {
            "parent_scope": parent_heading,
            "obligations": ", ".join(deontic_tags["Obligations"]),
            "permissions": ", ".join(deontic_tags["Permissions"]),
            "prohibitions": ", ".join(deontic_tags["Prohibitions"]),
            "conditions": ", ".join(deontic_tags["Conditions"]),
            "linked_nodes": ", ".join(links)
        }
    }

def extract_text_from_url(url: str) -> str:
    """
    Renders dynamic web pages via a headless browser without waiting 
    for indefinite network idle pings, then extracts the main text.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        # Load the HTML document structure without hanging on background pings
        page.goto(url, wait_until="domcontentloaded", timeout=45000)
        
        # Allow JavaScript hydration to settle
        page.wait_for_timeout(3000)
        
        html_content = page.content()
        browser.close()

    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg"]):
        tag.decompose()

    container = soup.body if soup.body else soup
    text = container.get_text(separator="\n")
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])