import os
import pickle
import numpy as np
import faiss
import openai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load .env and OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the FAISS index and documents
index = faiss.read_index("embeddings/faiss_index.idx")
with open("embeddings/docs.pkl", "rb") as f:
    docs = pickle.load(f)

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load stored embeddings (optional, not used directly here)
_ = np.load("embeddings/embeddings.npy")

# Retrieve top k relevant documents
def retrieve(query, k=3):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    return [docs[i] for i in I[0]]

# Generate a response using OpenAI
def generate_answer(query, context_docs):
    context = "\n".join(context_docs)
    prompt = f"""You are an AI assistant. Answer the following question based on the provided context.

Context:
{context}

Question:
{query}

Answer:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # You can change to another model if available
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=256,
    )
    return response['choices'][0]['message']['content'].strip()

# Chat loop
if __name__ == "__main__":
    print("💬 Loan Approval RAG Chatbot (type 'exit' to quit)\n")
    while True:
        query = input("🧑 You: ")
        if query.lower() == "exit":
            break
        context = retrieve(query)
        answer = generate_answer(query, context)
        print(f"🤖 Bot: {answer}\n")
