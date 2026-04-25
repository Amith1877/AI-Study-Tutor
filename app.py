from flask import Flask, request, render_template, jsonify
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

app = Flask(__name__)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(384)
doc_chunks = []
conversation_history = []

client = Groq(api_key="YOUR API KEY")

def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def store_embeddings(chunks):
    global doc_chunks
    embeddings = embed_model.encode(chunks)
    index.add(np.array(embeddings))
    doc_chunks.extend(chunks)

def retrieve(query, k=3):
    if not doc_chunks:
        return []
    query_vec = embed_model.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [doc_chunks[i] for i in I[0]]

def ask_llm(context, question, history):
    system_prompt = """You are a friendly, knowledgeable study tutor. 
    Answer questions clearly and thoroughly based on the provided context. 
    If the context doesn't contain relevant info, say so honestly but still try to help.
    Keep responses focused and educational."""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    user_content = f"Context from document:\n{context}\n\nQuestion: {question}" if context else f"Question: {question}"
    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return response.choices[0].message.content

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global conversation_history
    file = request.files['file']
    text = extract_text(file)
    chunks = chunk_text(text)
    store_embeddings(chunks)
    conversation_history = []
    return jsonify({"message": "PDF processed successfully!"})

@app.route('/ask', methods=['POST'])
def ask():
    global conversation_history
    try:
        question = request.json['question']
        context_chunks = retrieve(question)
        context = "\n".join(context_chunks)
        answer = ask_llm(context, question, conversation_history)
        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": answer})
        return jsonify({"answer": answer})
    except Exception as e:
        print("ERROR:", e)
        return jsonify({"answer": f"Error: {str(e)}"})

@app.route('/reset', methods=['POST'])
def reset():
    global conversation_history
    conversation_history = []
    return jsonify({"message": "Conversation reset."})

if __name__ == '__main__':
    app.run(debug=True)