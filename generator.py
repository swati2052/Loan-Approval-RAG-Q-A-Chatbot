import openai
from transformers import pipeline

openai.api_key = 'YOUR_OPENAI_KEY'

def gen_openai(query: str, context: str):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    resp = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "user", "content": prompt}]
    )
    return resp['choices'][0]['message']['content']

# Fallback Hugging Face
qa_model = pipeline("text-generation", model="google/flan-t5-small")

def gen_hf(query: str, context: str):
    prompt = f"Context: {context}\nQuestion: {query}"
    out = qa_model(prompt, max_length=150)[0]['generated_text']
    return out
