# 📄 DocuBot App

DocuBot is a **PDF-based conversational chatbot** built using **Streamlit** and **LangChain**.  
It allows users to upload PDF documents and ask questions, with answers generated directly from the document content using **Retrieval-Augmented Generation (RAG)**.

---

## 🚀 Features

- 📂 Upload one or multiple PDF files
- 🔍 Semantic search using vector embeddings
- 🤖 Conversational Q&A with memory
- 💬 Chat-style interface with typing animation
- 💾 Persistent vector database using Chroma
- ⚡ Fast inference powered by **Groq (LLaMA 3)**

---

## 🧱 Tech Stack

- **Frontend:** Streamlit
- **LLM:** Groq (LLaMA3-8b-8192)
- **Framework:** LangChain
- **Vector Store:** ChromaDB
- **Embeddings:** OpenAI Embeddings
- **PDF Loader:** PyPDFLoader
- **Memory:** ConversationBufferMemory

---

## 📁 Project Structure

📦 docuBot-app
├── .devcontainer/ # Dev container configuration
├── .github/ # GitHub workflows/configs
├── .gitignore # Git ignore rules
├── LICENSE # Apache-2.0 License
├── README.md # Project documentation
├── requirements.txt # Python dependencies
└── streamlit_app.py # Main Streamlit application

---

## 📦 Installation

### 1️⃣ Clone the Repository
bash
git clone https://github.com/sharvani0102/docuBot-app.git
cd docuBot-app 

2️⃣ Create a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

🔐 API Keys Configuration
Create a file at:
.streamlit/secrets.toml
Add your Groq API key:
[groq_api_key]
my_key = "YOUR_GROQ_API_KEY"
⚠️ This project also uses OpenAI Embeddings, so ensure your OpenAI credentials are properly configured.

▶️ Running the Application
streamlit run streamlit_app.py

Once the app launches:
Upload one or more PDF files
Wait for the documents to be indexed
Ask questions in the chat input
Receive answers grounded in the PDF content

How It Works
Uploaded PDFs are saved locally
Documents are split into overlapping text chunks
Text chunks are embedded using OpenAI embeddings
Embeddings are stored in a Chroma vector database
User queries retrieve relevant chunks
Groq LLaMA 3 generates answers using retrieved context and chat history


📌 Future Enhancements
Source citation in responses
PDF deletion and re-indexing
Streaming token-level responses
Streamlit Cloud deployment
Authentication for private documents

📜 License
This project is licensed under the Apache License 2.0.
See the LICENSE file for more details.




