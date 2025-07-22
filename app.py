#✅ Planned Enhancements
#Persistent PDF state — uploaded PDFs are stored and reused for multiple queries.
#Styled modern UI — clean, responsive layout similar to AI tools like ChatGPT or Perplexity.
#Sidebar — displays uploaded PDF filenames.
#Chat history — optional scrollable display of past Q&A.
#Spinner/Loading indicator — shows during processing.
#Responsive HTML + inline CSS — no external file required for simplicity.
#Same tech stack — Flask, LangChain, HuggingFace, Groq.
#✅ Key Benefits:
#Upload once, ask many times.
#Modern sidebar + main area layout.
#Spinner on query.
#Chat history shows last questions/answers.
#Cache reuse with FAISS, secure loading.
#CSS for cleaner layout, consistent fonts/colors.



# app.py
import os
import time
import hashlib
from datetime import datetime, timedelta
from flask import Flask, request, render_template_string, session, redirect, url_for
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.secret_key = "minha"
UPLOAD_FOLDER = "uploads"
CACHE_FOLDER = "cache"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)

# Environment setup
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["HF_API_KEY"] = os.getenv("HF_API_KEY", "")

llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="Llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
<context>
{context}
</context>
Question: {input}
""")

# ======= Utilities =======

def hash_file(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_or_create_vectorstore(file_paths):
    combined_hash = hashlib.md5("".join(sorted([hash_file(f) for f in file_paths])).encode()).hexdigest()
    cache_path = os.path.join(CACHE_FOLDER, f"{combined_hash}")

    if os.path.exists(f"{cache_path}.faiss"):
        return FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)

    all_docs = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(cache_path)
    return vectorstore

def cleanup_old_cache_files(days_old=3):
    now = time.time()
    for fname in os.listdir(CACHE_FOLDER):
        full_path = os.path.join(CACHE_FOLDER, fname)
        if os.path.isfile(full_path) and fname.endswith((".pkl", ".faiss")):
            file_time = os.path.getmtime(full_path)
            if (now - file_time) > (days_old * 86400):
                os.remove(full_path)

# ======= HTML =======

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI PDF Chat</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f5f6fa; margin: 0; padding: 0; }
        .container { display: flex; height: 100vh; }
        .sidebar {
            width: 250px;
            background: #2f3542;
            color: white;
            padding: 20px;
        }
        .main {
            flex: 1;
            padding: 30px;
            background: white;
            overflow-y: auto;
        }
        h1, h2 { margin-top: 0; }
        input[type="text"] { width: 70%; padding: 8px; }
        input[type="file"] { margin-bottom: 10px; }
        button { padding: 10px 15px; margin: 5px; }
        .chat-bubble { background: #dfe4ea; margin-bottom: 10px; padding: 10px; border-radius: 10px; }
        .spinner { display: none; }
    </style>
    <script>
        function showSpinner() {
            document.getElementById("spinner").style.display = "inline";
        }
    </script>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>📄 Uploaded PDFs</h2>
            {% if file_list %}
                <ul>{% for name in file_list %}<li>{{ name }}</li>{% endfor %}</ul>
            {% else %}
                <p>No files uploaded.</p>
            {% endif %}
            <form method="post" enctype="multipart/form-data">
                <input type="file" name="file" multiple><br>
                <button type="submit">Upload</button>
            </form>
            <form method="post" action="/clear">
                <button>Clear Session</button>
            </form>
        </div>
        <div class="main">
            <h1>🤖 Ask Your PDF</h1>
            <form method="post" onsubmit="showSpinner()">
                <input type="text" name="query" placeholder="Enter your question" required>
                <button type="submit">Ask</button>
                <span id="spinner">⏳ Loading...</span>
            </form>

            {% if chat_history %}
                <hr>
                <h3>📝 Chat History</h3>
                {% for item in chat_history %}
                    <div class="chat-bubble"><strong>Q:</strong> {{ item.question }}</div>
                    <div class="chat-bubble"><strong>A:</strong> {{ item.answer }}</div>
                {% endfor %}
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

# ======= Routes =======

@app.route("/", methods=["GET", "POST"])
def index():
    cleanup_old_cache_files()
    chat_history = session.get("chat_history", [])
    file_paths = session.get("file_paths", [])
    file_list = [os.path.basename(f) for f in file_paths]

    if request.method == "POST":
        query = request.form.get("query")
        uploaded_files = request.files.getlist("file")

        if uploaded_files and uploaded_files[0].filename != "":
            file_paths = []
            for file in uploaded_files:
                filename = secure_filename(file.filename)
                save_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(save_path)
                file_paths.append(save_path)
            session["file_paths"] = file_paths
            file_list = [os.path.basename(f) for f in file_paths]
            return redirect(url_for("index"))

        if not file_paths:
            chat_history.append({"question": query, "answer": "❌ Please upload at least one PDF first."})
            session["chat_history"] = chat_history
            return render_template_string(HTML_TEMPLATE, file_list=file_list, chat_history=chat_history)

        try:
            vectorstore = load_or_create_vectorstore(file_paths)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
            response = chain.invoke({"input": query})
            answer = response.get("answer", "No answer found.")
            chat_history.append({"question": query, "answer": answer})
            session["chat_history"] = chat_history

        except Exception as e:
            chat_history.append({"question": query, "answer": f"❌ Error: {str(e)}"})
            session["chat_history"] = chat_history

    return render_template_string(HTML_TEMPLATE, file_list=file_list, chat_history=chat_history)

@app.route("/clear", methods=["POST"])
def clear_session():
    session.clear()
    return redirect(url_for("index"))

# ======= Run =======

if __name__ == "__main__":
    app.run(debug=True)
