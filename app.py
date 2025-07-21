import streamlit as st
from retriever import retrieve
from generator import gen_openai, gen_hf

st.set_page_config(page_title="Loan RAG Chatbot", layout="centered")
st.title("📊 Loan Approval RAG Q&A Chatbot")

st.markdown("""
Ask any question about the **Loan Approval dataset**, and this chatbot will retrieve the most relevant data and generate an intelligent answer using Generative AI.
""")

query = st.text_input("💬 Enter your question:")

if query:
    with st.spinner("🔍 Retrieving relevant information..."):
        docs = retrieve(query, k=3)
        context = "\n\n".join(docs)
    
    st.markdown("### 📄 Retrieved Context")
    st.code(context)

    with st.spinner("🤖 Generating answer..."):
        try:
            answer = gen_openai(query, context)
        except Exception as e:
            st.warning("⚠️ OpenAI failed, trying Hugging Face instead...")
            answer = gen_hf(query, context)

    st.markdown("### 🤖 AI Answer")
    st.success(answer)
