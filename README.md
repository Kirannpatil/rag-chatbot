# Flipkart Return Policy Assistant
Advanced RAG chatbot for Flipkart's return, cancellation and refund policies.
Built with LlamaIndex + Groq (Llama 3.3 70B) + Flask.

---

## Architecture
This project uses **Advanced RAG** with pre-retrieval query optimization:

```
User Query
    ↓
Query Rewriter (Groq LLM) — generates 4 semantic variants
    ↓
Multi-Query Retrieval — runs all variants against vector index
    ↓
Node Deduplication — merges unique chunks
    ↓
Answer Generation (Groq LLM) — reasons across 5 policy dimensions
    ↓
Response
```

---

## Project Structure
```
RAG Pro/
├── data/                  ← policy .txt files go here
├── storage/               ← auto created after ingest
├── ingest.py              ← run once to index documents
├── app.py                 ← run to start chatbot
├── evaluate.py            ← run to evaluate pipeline
├── eval_results.json      ← evaluation results
├── requirements.txt
└── README.md
```

---

## Setup Steps

### Step 1 — Create virtual environment
```
python -m venv venv
venv\Scripts\activate        (Windows)
source venv/bin/activate     (Mac/Linux)
```

### Step 2 — Install dependencies
```
pip install -r requirements.txt
```

### Step 3 — Add API keys
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key
LLAMA_PARSE_API_KEY=your_llama_parse_api_key
```
Get Groq API key free at: https://console.groq.com

### Step 4 — Add policy documents
Copy your policy `.txt` files into the `data/` folder:
```
data/
├── cancellation_policy.txt
├── return_policy.txt
└── refund_policy.txt
```

### Step 5 — Index your documents (run ONCE)
```
python ingest.py
```

### Step 6 — Start the chatbot
```
python app.py
```

### Step 7 — Open browser
```
http://localhost:5000
```

---

## Evaluation
To evaluate the pipeline on relevancy and correctness metrics:
```
python evaluate.py
```
Results are saved to `eval_results.json`.

| Metric | Description |
|---|---|
| Relevancy | Were the right chunks retrieved for the query? |
| Correctness | Does the answer match the known ground truth? |

Current evaluation results:
- Relevancy: 67% passed
- Correctness: 4.0 / 5.0 average

---

## Models
| Purpose | Model |
|---|---|
| Answer generation | llama-3.3-70b-versatile (Groq) |
| Query rewriting | llama-3.3-70b-versatile (Groq) |
| Evaluation | llama-3.1-8b-instant (Groq) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local) |

---

## Adding more documents
Drop more `.txt` files into `data/` and re-run:
```
python ingest.py
```
