"""
app.py
──────
Flask web UI for the Return Policy RAG chatbot.
Loads the saved index from storage/ and serves
a chat interface at http://localhost:5000

Run after ingest.py:
    python app.py
"""

import os
import json
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, render_template_string
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# ── Configuration ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   =  "llama-3.3-70b-versatile" # smaller, faster, cheaper than 4.0, still very capable 

STORAGE_DIR  = os.path.join(os.path.dirname(__file__), "storage")

# ── Models ────────────────────────────────────────────────────
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.llm = Groq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.1, # best for Q&A
)

# ── Load index ────────────────────────────────────────────────
print(f"\n📦  Loading index from {STORAGE_DIR} ...")
storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
vector_index = load_index_from_storage(storage_context)
print("✅  Index loaded\n")

# ── Query Rewriter ────────────────────────────────────────────
REWRITER_PROMPT = """You are a search query rewriter for an e-commerce return policy assistant.

The policy documents are structured around these dimensions for every product:
- CATEGORY      — which product group it belongs to (e.g. Lifestyle, Electronics, etc.)
- RETURN WINDOW — how many days the customer has to return it
- CANCELLATION  — whether the order can be cancelled and at what stage
- ACTIONS       — what is allowed (Refund / Replacement / Exchange)
- CONDITIONS    — under what circumstances (defective, damaged, wrong item, any reason, etc.)

Your job: given a user question, generate 4 search queries that together cover all
relevant dimensions so the retriever finds the right policy chunk regardless of phrasing.

Rules:
- Do NOT assume or hardcode any specific product name, category, or number of days
- Each query must approach the same intent from a different angle:
    Query 1 → rephrase focusing on the PRODUCT / CATEGORY dimension
    Query 2 → rephrase focusing on the RETURN WINDOW / TIME dimension
    Query 3 → rephrase focusing on the ACTIONS dimension (refund/replace/exchange)
    Query 4 → rephrase focusing on CONDITIONS (defective/wrong item/any reason)
- Use natural language, not keywords

Respond ONLY with a valid JSON array of exactly 4 strings. No markdown, no explanation.
Example format: ["query 1", "query 2", "query 3", "query 4"]"""


def rewrite_query(user_query: str) -> list:
    """Expand user query into multiple semantic variants using Groq LLM."""
    try:
        response = Settings.llm.complete(
            f"{REWRITER_PROMPT}\n\nUser question: {user_query}"
        )
        text = str(response).strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        variants = json.loads(text.strip())
        all_queries = [user_query] + variants
        seen = set()
        unique = []
        for q in all_queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        print(f"🔄  Query variants: {unique}")
        return unique
    except Exception as e:
        print(f"[WARN] Query rewriting failed: {e}. Using original query only.")
        return [user_query]


def retrieve_multi_query(user_query: str) -> list:
    """Retrieve chunks using multiple query variants and deduplicate by node ID."""
    retriever = vector_index.as_retriever(similarity_top_k=3)
    variants  = rewrite_query(user_query)
    seen_ids  = set()
    merged    = []
    for query in variants:
        for node in retriever.retrieve(query):
            if node.node.node_id not in seen_ids: # node= NodeWithScore object, node.node = the actual Node inside it, node.node.node_id  = unique ID of that Node
                seen_ids.add(node.node.node_id)
                merged.append(node)
    print(f"📚  {len(merged)} unique chunks from {len(variants)} variants")
    return merged

# ── Flask app ─────────────────────────────────────────────────
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Return Policy Assistant</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', sans-serif;
      background: #f0f2f5;
      height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .container {
      width: 100%;
      max-width: 780px;
      height: 92vh;
      background: white;
      border-radius: 16px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.12);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    /* ── Header ── */
    .header {
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
      padding: 20px 24px;
      color: white;
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .header-icon {
      width: 42px; height: 42px;
      background: rgba(255,255,255,0.15);
      border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      font-size: 20px;
    }

    .header-text h1 { font-size: 17px; font-weight: 600; }
    .header-text p  { font-size: 12px; opacity: 0.65; margin-top: 2px; }

    .status-dot {
      margin-left: auto;
      width: 9px; height: 9px;
      background: #4ade80;
      border-radius: 50%;
      box-shadow: 0 0 6px #4ade80;
    }

    /* ── Messages ── */
    .messages {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      scroll-behavior: smooth;
    }

    .messages::-webkit-scrollbar { width: 4px; }
    .messages::-webkit-scrollbar-thumb {
      background: #ddd; border-radius: 4px;
    }

    .message {
      display: flex;
      gap: 10px;
      max-width: 85%;
      animation: fadeIn 0.25s ease;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(8px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    .message.user  { margin-left: auto; flex-direction: row-reverse; }
    .message.bot   { margin-right: auto; }

    .avatar {
      width: 34px; height: 34px;
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 15px; flex-shrink: 0;
    }

    .message.user .avatar { background: #1a1a2e; color: white; }
    .message.bot  .avatar { background: #f1f5f9; color: #1a1a2e; }

    .bubble {
      padding: 12px 16px;
      border-radius: 14px;
      font-size: 14px;
      line-height: 1.6;
      color: #1e293b;
    }

    .message.user .bubble {
      background: #1a1a2e;
      color: white;
      border-bottom-right-radius: 4px;
    }

    .message.bot .bubble {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-bottom-left-radius: 4px;
    }

    /* welcome card */
    .welcome {
      background: linear-gradient(135deg, #f8fafc, #f1f5f9);
      border: 1px solid #e2e8f0;
      border-radius: 14px;
      padding: 20px;
      text-align: center;
      color: #475569;
      font-size: 14px;
      line-height: 1.7;
    }

    .welcome h2 { color: #1a1a2e; font-size: 16px; margin-bottom: 8px; }

    .suggestions {
      display: flex; flex-wrap: wrap; gap: 8px;
      justify-content: center; margin-top: 14px;
    }

    .suggestion {
      background: white;
      border: 1px solid #cbd5e1;
      border-radius: 20px;
      padding: 6px 14px;
      font-size: 12px;
      color: #334155;
      cursor: pointer;
      transition: all 0.2s;
    }

    .suggestion:hover {
      background: #1a1a2e;
      color: white;
      border-color: #1a1a2e;
    }

    /* typing indicator */
    .typing .bubble {
      display: flex; align-items: center; gap: 5px;
      padding: 14px 18px;
    }

    .dot {
      width: 7px; height: 7px;
      background: #94a3b8;
      border-radius: 50%;
      animation: bounce 1.2s infinite;
    }

    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(3) { animation-delay: 0.4s; }

    @keyframes bounce {
      0%, 60%, 100% { transform: translateY(0); }
      30%           { transform: translateY(-6px); }
    }

    /* ── Input area ── */
    .input-area {
      padding: 16px 20px;
      border-top: 1px solid #e2e8f0;
      background: white;
      display: flex;
      gap: 10px;
      align-items: flex-end;
    }

    textarea {
      flex: 1;
      border: 1.5px solid #e2e8f0;
      border-radius: 12px;
      padding: 11px 15px;
      font-size: 14px;
      font-family: inherit;
      resize: none;
      outline: none;
      max-height: 120px;
      transition: border-color 0.2s;
      color: #1e293b;
      line-height: 1.5;
    }

    textarea:focus { border-color: #1a1a2e; }

    textarea::placeholder { color: #94a3b8; }

    button#send {
      width: 44px; height: 44px;
      background: #1a1a2e;
      color: white;
      border: none;
      border-radius: 12px;
      cursor: pointer;
      font-size: 18px;
      transition: all 0.2s;
      display: flex; align-items: center; justify-content: center;
      flex-shrink: 0;
    }

    button#send:hover   { background: #16213e; transform: scale(1.05); }
    button#send:disabled { background: #94a3b8; transform: none; cursor: not-allowed; }

    .char-hint {
      font-size: 11px;
      color: #94a3b8;
      padding: 0 4px;
      align-self: center;
    }
  </style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <div class="header-icon">📋</div>
    <div class="header-text">
      <h1>Return Policy Assistant</h1>
      <p>Powered by LlamaIndex + Ollama (local)</p>
    </div>
    <div class="status-dot" title="Online"></div>
  </div>

  <!-- Messages -->
  <div class="messages" id="messages">
    <div class="welcome">
      <h2>👋 Hello! I'm your Return Policy Assistant</h2>
      I can answer any questions about our return and refund policies
      based on the official policy document.
      <div class="suggestions">
        <span class="suggestion" onclick="ask(this)">What is the return window?</span>
        <span class="suggestion" onclick="ask(this)">How do I initiate a return?</span>
        <span class="suggestion" onclick="ask(this)">Can I return sale items?</span>
        <span class="suggestion" onclick="ask(this)">How long does refund take?</span>
      </div>
    </div>
  </div>

  <!-- Input -->
  <div class="input-area">
    <textarea
      id="input"
      rows="1"
      placeholder="Ask about returns, refunds, exchanges..."
      maxlength="500"
    ></textarea>
    <button id="send" onclick="sendMessage()" title="Send">➤</button>
  </div>

</div>

<script>
  const messagesEl = document.getElementById('messages');
  const inputEl    = document.getElementById('input');
  const sendBtn    = document.getElementById('send');

  // auto-resize textarea
  inputEl.addEventListener('input', () => {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
  });

  // send on Enter (Shift+Enter for newline)
  inputEl.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  function ask(el) {
    inputEl.value = el.textContent;
    sendMessage();
  }

  function addMessage(role, text) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    const avatar = role === 'user' ? '🧑' : '🤖';
    div.innerHTML = `
      <div class="avatar">${avatar}</div>
      <div class="bubble">${text.replace(/\\n/g, '<br>')}</div>
    `;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  function addTyping() {
    const div = document.createElement('div');
    div.className = 'message bot typing';
    div.id = 'typing';
    div.innerHTML = `
      <div class="avatar">🤖</div>
      <div class="bubble">
        <div class="dot"></div>
        <div class="dot"></div>
        <div class="dot"></div>
      </div>
    `;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function removeTyping() {
    const t = document.getElementById('typing');
    if (t) t.remove();
  }

  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text) return;

    inputEl.value = '';
    inputEl.style.height = 'auto';
    sendBtn.disabled = true;

    addMessage('user', text);
    addTyping();

    try {
      const res  = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });
      const data = await res.json();
      removeTyping();
      addMessage('bot', data.response || data.error || 'Something went wrong.');
    } catch (err) {
      removeTyping();
      addMessage('bot', '⚠️ Could not reach the server. Make sure app.py is running.');
    } finally {
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/chat", methods=["POST"])
def chat():
    data    = request.get_json()
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400
    try:
        # Multi-query retrieval
        nodes   = retrieve_multi_query(message)
        context = "\n\n---\n\n".join(node.node.get_content() for node in nodes)

        # Build final prompt: system prompt + retrieved context + user question
        full_prompt = f"""
You are a Flipkart customer support assistant.
You have access to three policy documents:
1. Cancellation Policy
2. Returns Policy
3. Refund Policy

The policy documents are structured around five dimensions for every product:
1. CATEGORY      — which product group it belongs to (e.g. Lifestyle, Electronics, etc.)
2. RETURN WINDOW — how many days the customer has to return it
3. ACTIONS       — what is allowed (Refund / Replacement / Exchange)
4. CONDITIONS    — under what circumstances (defective, damaged, wrong item, any reason, etc.)
5. CANCELLATION  — whether the order can be cancelled and at what stage

When answering, always reason step by step:
1. Scan the category lists carefully and identify the EXACT category the user's product belongs to
2. Once the category is identified, pick ONLY that category's return window — ignore all other return windows in the context
3. State the allowed ACTIONS (refund / replacement / exchange) for that specific category only
4. State any CONDITIONS that apply to that category

STRICT RULES:
1. Answer ONLY from the documents provided
2. Never make up numbers, dates or timelines
3. For simple questions give 1-2 sentence answer
4. For complex questions give a detailed answer
5. Do not repeat yourself under any circumstance
6. If multiple chunks say the same thing, summarize ONCE
7. If multiple return windows exist for a product, mention ALL with their conditions
8. If the answer is not in the documents say:
   "I don't have that information. Please contact Flipkart support."

Policy Context (retrieved from documents):
{context}

User Question: {message}
"""
        response = Settings.llm.complete(full_prompt)
        return jsonify({"response": str(response)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🌐  Starting server at http://localhost:5000\n")
    app.run(debug=False, port=5000)
