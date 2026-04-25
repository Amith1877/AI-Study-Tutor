# AI-Study-Tutor

A local AI study assistant that lets you upload any PDF (notes, textbooks, lab manuals) and chat with it using natural language. Built with Flask, FAISS, Sentence Transformers, and Groq LLM.

---

## What it does

- Upload a PDF and ask questions about it
- Gets relevant answers using RAG (Retrieval-Augmented Generation)
- Remembers conversation context across messages
- Clean chatbot UI with sidebar, typing indicators, and prompt chips

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask (Python) |
| Embeddings | Sentence Transformers `all-MiniLM-L6-v2` |
| Vector Search | FAISS |
| LLM | Groq API — `llama-3.3-70b-versatile` |
| PDF Parsing | PyPDF2 |
| Frontend | HTML, CSS, Vanilla JS |

---

## Project Structure
AI-Study-Tutor/
├── app.py
└── templates/
└── index.html
## Setup

### 1. Clone or download the project

```bash
cd "C:\Users\Lenovo\Downloads\AI-Study-Tutor"
```

### 2. Install dependencies

```bash
pip install flask faiss-cpu sentence-transformers PyPDF2 groq
```

### 3. Add your Groq API key

In `app.py`, replace the API key value:

```python
client = Groq(api_key="your_groq_api_key_here")
```

Get a free key at: https://console.groq.com

### 4. Run the app

```bash
python app.py
```

### 5. Open in browser
http://127.0.0.1:5000

---

## How to use

1. Click **Choose a PDF** in the sidebar and select your file
2. Click **Upload document** and wait for "Document ready"
3. Type a question in the chat input or click one of the prompt chips
4. Press **Enter** to send — the tutor will answer based on your document
5. Click **New conversation** to reset the chat history

---

## How RAG works in this project
PDF Upload
↓
Extract text (PyPDF2)
↓
Split into 300-word chunks
↓
Generate embeddings (Sentence Transformers)
↓
Store in FAISS vector index
↓
User asks a question
↓
Embed the question → search FAISS for top 3 matching chunks
↓
Send chunks + question + history to Groq LLM
↓
Return answer to chat UI

---

## Known Limitations

- PDF must contain selectable text (scanned/image PDFs won't work)
- Uploaded PDF is stored in memory only — restarting the server clears it
- One PDF at a time (uploading a new one resets the index)
- Groq free tier has rate limits

---

## Requirements

- Python 3.10 or 3.11 recommended (Python 3.13 from Microsoft Store has known compatibility issues with `transformers`)
- Internet connection (for Groq API calls and loading the embedding model on first run)

---

## Dependencies
flask
faiss-cpu
sentence-transformers
PyPDF2
groq
numpy

---

## License

MIT — free to use, modify, and distribute.
