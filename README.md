🤖 Loan Approval RAG Q&A Chatbot
This AI-powered chatbot uses Retrieval-Augmented Generation (RAG) to answer questions related to a Loan Approval Prediction dataset. It combines semantic search and advanced language models to provide contextually accurate responses.

🚀 Features
🔍 Contextual Retrieval using FAISS and Sentence Transformers
🧠 Answer Generation via OpenAI GPT or Hugging Face Transformers
📊 Based on the Loan Approval Prediction dataset (from Kaggle)
💬 Interactive Chat Interface built with Streamlit
⚙️ Easy to switch between OpenAI and Hugging Face models



📦 Tech Stack
Frontend: Streamlit
Retrieval: FAISS + SentenceTransformers
Generation: OpenAI GPT / Hugging Face Transformers
Data Handling: Pandas, NumPy
Environment: Python 3.9+



📁 Project Structure
├── app.py                  # Streamlit app
├── retriever.py           # FAISS-based retriever logic
├── generator.py           # Answer generation logic (OpenAI / HF)
├── data/                  # Dataset and embeddings
├── models/                # Stored models or model configs
├── .env                   # API keys and secrets (not committed)
└── requirements.txt       # Python dependencies


🔑 API Key Setup (OpenAI)
Visit OpenAI API Keys
Generate your API key.
Create a .env file in the root directory and add:
OPENAI_API_KEY=your-api-key-here
⚠️ Never expose your .env or API key in public repositories.



▶️ How to Run
Install dependencies
pip install -r requirements.txt
Start the app
streamlit run app.py
📄 License
This project is licensed under the MIT License – free to use, modify, and distribute.

