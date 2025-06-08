import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle

DATA_DIR = "data"
EMBEDDING_DIR = "embeddings"
os.makedirs(EMBEDDING_DIR, exist_ok=True)

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = []
sources = []

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".xlsx"):
        path = os.path.join(DATA_DIR, filename)
        try:
            df = pd.read_excel(path)

            # Try to extract only issue-related columns if available
            issue_cols = [col for col in df.columns if any(k in col.lower() for k in ["issue", "remark", "problem", "reason"])]
            if issue_cols:
                issue_data = df[issue_cols].dropna(how="all")
                # Optional: include date/shift columns if found
                extra_cols = [col for col in df.columns if any(k in col.lower() for k in ["date", "shift"])]
                context_df = pd.concat([df[extra_cols], issue_data], axis=1).dropna(how="all", axis=0)
                text = context_df.astype(str).agg(" | ".join, axis=1).tolist()
                texts.extend(text)
                sources.extend([filename] * len(text))
                print(f"[📝] Extracted issues from: {filename}")
            else:
                # Fallback: Add the whole sheet as plain text
                text = df.to_string(index=False)
                texts.append(text)
                sources.append(filename)
                print(f"[ℹ️] No issue columns found, added entire file: {filename}")
        except Exception as e:
            print(f"[⚠️] Skipped {filename} due to error: {e}")

if not texts:
    raise ValueError("❌ No Excel files could be processed. Check file permissions or structure.")

print("[🔍] Embedding documents...")
embeddings = model.encode(texts)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index and metadata
faiss.write_index(index, os.path.join(EMBEDDING_DIR, "faiss_index.bin"))
with open(os.path.join(EMBEDDING_DIR, "sources.pkl"), "wb") as f:
    pickle.dump(sources, f)

with open(os.path.join(EMBEDDING_DIR, "texts.pkl"), "wb") as f:
    pickle.dump(texts, f)

print("✅ Index built and saved to embeddings/")
