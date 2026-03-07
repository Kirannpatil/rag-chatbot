# Return Policy RAG Chatbot
Local RAG chatbot using LlamaIndex + Ollama + Flask.
No API keys needed. Everything runs on your machine.

---

## Project Structure

rag_project/
├── data/                  ← PUT YOUR PDF HERE
├── storage/               ← auto created after ingest
├── ingest.py              ← run once to index documents
├── app.py                 ← run to start chatbot
├── requirements.txt
└── README.md

---

## Setup Steps

### Step 1 — Install Ollama
Download from: https://ollama.com/download
Then pull the model (in terminal):
    ollama pull llama3

### Step 2 — Create virtual environment
    python -m venv venv
    venv\Scripts\activate        (Windows)
    source venv/bin/activate     (Mac/Linux)

### Step 3 — Install dependencies
    pip install -r requirements.txt

### Step 4 — Add your PDF
    Copy your return_policy.pdf into the data/ folder

### Step 5 — Index your documents (run ONCE)
    python ingest.py

### Step 6 — Start the chatbot
    python app.py

### Step 7 — Open browser
    Go to: http://localhost:5000

---

## Changing the LLM model
In app.py, line 22:
    OLLAMA_MODEL = "llama3"
Change to any model you have pulled:
    OLLAMA_MODEL = "mistral"
    OLLAMA_MODEL = "phi3"

## Adding more documents
Just drop more files into data/ and re-run:
    python ingest.py
