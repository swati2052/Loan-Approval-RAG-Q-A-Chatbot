import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

# Create folder if not exists
os.makedirs('embeddings', exist_ok=True)

# 1. Load and clean data
df = pd.read_csv('data/Test Dataset.csv').fillna(method='ffill')

# 2. Create document strings
docs = [
    " | ".join([f"{col}: {row[col]}" for col in df.columns])
    for _, row in df.iterrows()
]

# 3. Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(docs, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

# 4. Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# 5. Save index and docs
with open('embeddings/docs.pkl', 'wb') as f:
    pickle.dump(docs, f)

np.save('embeddings/embeddings.npy', embeddings)
faiss.write_index(index, 'embeddings/faiss_index.idx')

# 6. Retrieval function
def retrieve(query: str, k: int = 3):
    q_emb = model.encode([query]).astype('float32')
    D, I = index.search(q_emb, k)
    return [docs[i] for i in I[0]]
