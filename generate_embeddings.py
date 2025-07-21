import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Create the embeddings directory if it doesn't exist
os.makedirs('embeddings', exist_ok=True)

# Load and clean the dataset
df = pd.read_csv('data/Test Dataset.csv')
df.fillna('', inplace=True)

# Convert each row into a string document
docs = [
    " | ".join([f"{col}: {row[col]}" for col in df.columns])
    for _, row in df.iterrows()
]

# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate vector embeddings
embeddings = model.encode(docs, show_progress_bar=True)

# Save the document texts
with open('embeddings/docs.pkl', 'wb') as f:
    pickle.dump(docs, f)

# Save the embeddings as a numpy array
np.save('embeddings/embeddings.npy', embeddings)

# Create FAISS index for similarity search
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# Save FAISS index
faiss.write_index(index, 'embeddings/faiss_index.idx')

print("✅ Embeddings generated and saved successfully.")
