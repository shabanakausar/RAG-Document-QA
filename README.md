# RAG Document QA
## 🚀 Features

- 📄 Upload multiple PDF files
- 🧠 Ask natural language questions
- ⚡ Fast answers using vector search and LLM
- 💬 Chat history tracking
- 📁 Caches vectorstores to speed up repeated queries

---

## 🛠 Tech Stack

- **Backend:** Flask
- **LLM:** [Groq API](https://groq.com)
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Store:** FAISS
- **Document Loader:** LangChain PDF loader

---

## 📦 Installation

1. **Clone the repo**
```bash
git clone https://github.com/your-username/pdf-ai-chatbot.git
cd pdf-ai-chatbot
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Set environment variables in a .env file:

env
Copy
Edit
GROQ_API_KEY=your_groq_api_key
HF_API_KEY=your_huggingface_api_key
Run the app

bash
Copy
Edit
python app.py
Then visit http://127.0.0.1:5000 in your browser.


✅ Key Benefits:

- Upload once, ask many times.

- Modern sidebar + main area layout.

- Spinner on query.

- Chat history shows last questions/answers.

- Cache reuse with FAISS, secure loading.

- CSS for cleaner layout, consistent fonts/colors.
