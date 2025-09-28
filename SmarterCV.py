import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import matplotlib.pyplot as plt
import spacy
import re

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

# --- Extract text from PDF ---
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

# --- Fully automated term extraction ---
def extract_meaningful_terms(text):
    doc = nlp(text.lower())
    # 1. Noun chunks (multi-word phrases)
    phrases = [chunk.text for chunk in doc.noun_chunks]
    # 2. Named entities (tools, technologies, organizations, concepts)
    entities = [ent.text for ent in doc.ents]
    # 3. Proper nouns / key nouns
    keywords = [token.text for token in doc if token.pos_ in ["PROPN", "NOUN"]]
    # Combine all
    combined = list(set(phrases + entities + keywords))
    return combined

# --- Split combined terms (e.g., "Java, JavaScript and Docker") ---
def split_combined_terms(term):
    return [t.strip() for t in re.split(r',| and ', term) if t.strip()]

# --- Clean and normalize terms ---
def clean_and_normalize_terms(terms):
    cleaned = []
    for term in terms:
        for subterm in split_combined_terms(term):
            normalized = subterm.lower().strip(".,;:()")
            if len(normalized) > 2:  # remove very short terms
                cleaned.append(normalized)
    return list(set(cleaned))  # remove duplicates

# --- Compare with semantic similarity ---
def semantic_compare(resume_text, job_text, threshold=0.6):
    resume_terms = clean_and_normalize_terms(extract_meaningful_terms(resume_text))
    job_terms = clean_and_normalize_terms(extract_meaningful_terms(job_text))

    overlap, missing = set(), set()

    resume_embeddings = model.encode(resume_terms, convert_to_tensor=True)
    job_embeddings = model.encode(job_terms, convert_to_tensor=True)

    for i, job_term in enumerate(job_terms):
        sims = util.cos_sim(job_embeddings[i], resume_embeddings)[0]
        # Exact match OR semantic similarity above threshold
        if job_term in resume_terms or sims.max().item() >= threshold:
            overlap.add(job_term)
        else:
            missing.add(job_term)

    return overlap, missing, Counter(resume_terms), Counter(job_terms)

# --- Streamlit App ---
st.title("📊 Resume Analyzer (Clean Automated Version)")
st.write("Upload your resume and paste a job description. Uses **AI semantic similarity** with fully automated extraction of key terms and phrases.")

uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste Job Description")

if uploaded_resume and job_desc:
    resume_text = extract_text_from_pdf(uploaded_resume)
    overlap, missing, resume_counter, job_counter = semantic_compare(resume_text, job_desc)

    st.subheader("✅ Matching Skills/Concepts (Semantic)")
    st.write(", ".join(sorted(overlap)))

    st.subheader("❌ Missing Skills/Concepts (Semantic)")
    st.write(", ".join(sorted(missing)))

    # --- Pie Chart for Match Ratio ---
    st.subheader("📊 Resume Coverage of Job Description")
    fig, ax = plt.subplots()
    ax.pie([len(overlap), len(missing)],
           labels=["Matched", "Missing"],
           colors=['green', 'red'],
           autopct='%1.1f%%',
           startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures pie is circular
    st.pyplot(fig)
