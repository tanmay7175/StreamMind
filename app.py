import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import subprocess

# Load model & index
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("embeddings/faiss_index.bin")

with open("embeddings/sources.pkl", "rb") as f:
    sources = pickle.load(f)
with open("embeddings/texts.pkl", "rb") as f:
    texts = pickle.load(f)

# Call Ollama with prompt and selected model
def query_ollama(prompt, model_name):
    try:
        process = subprocess.Popen(
            ["ollama", "run", model_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output, error = process.communicate(input=prompt)
        if process.returncode != 0:
            return f"[ERROR] Ollama failed:\n{(error or 'Unknown error').strip()}"
        return output.strip() if output else "[ERROR] No output received from Ollama."
    except Exception as e:
        return f"[ERROR] Exception during Ollama call: {str(e)}"

# RAG logic
def answer_question(query, model_choice):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=3)  # top 3 matches

    context = ""
    used_sources = set()
    for idx in I[0]:
        context += texts[idx] + "\n\n"
        used_sources.add(sources[idx])

    full_prompt = f"""You are an AI assistant working for Califonix Tech and Manufacturing Ltd.
Your job is to answer questions based on internal company Excel reports.

Context from internal data:
{context}

Question: {query}
Answer:"""

    response = query_ollama(full_prompt, model_choice)
    return response, used_sources

# Streamlit UI
st.set_page_config(page_title="📊 Califonix Company Q&A")
st.title("🤖 Ask Califonix Company Data")

query = st.text_input("Ask a question about the company data:")
model_choice = st.selectbox("Choose a model to answer:", ["mistral", "llama2"])

if st.button("Submit") and query:
    with st.spinner(f"Thinking with {model_choice}..."):
        answer, used_sources = answer_question(query, model_choice)
        st.markdown(f"**📁 Answer based on:** `{', '.join(used_sources)}`")
        st.text_area("💬 Model Response", value=answer, height=300)
