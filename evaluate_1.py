"""
evaluate.py
───────────
Evaluates the RAG pipeline on two metrics:
  1. Context Relevancy — were the right chunks retrieved?
  2. Correctness       — does the answer match the known ground truth?

Uses llama-3.1-8b-instant for evaluation to save Groq daily tokens.
Uses llama-3.3-70b-versatile for answer generation only.

Run:
    python evaluate.py

Results are saved to eval_results.json and printed to terminal.
"""

import os
import json
from dotenv import load_dotenv
load_dotenv()

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.evaluation import (
    RelevancyEvaluator,
    CorrectnessEvaluator,
)

# ── Config ────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"   # for answer generation
EVAL_MODEL   = "llama-3.1-8b-instant"      # for evaluation — uses ~4x fewer tokens
STORAGE_DIR  = os.path.join(os.path.dirname(__file__), "storage")

# ── Models ────────────────────────────────────────────────────
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.llm = Groq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.1,
)

# Separate smaller LLM for evaluation to save tokens
eval_llm = Groq(
    model=EVAL_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.1,
)

# ── Load index ────────────────────────────────────────────────
print("\n📦  Loading index...")
storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
vector_index = load_index_from_storage(storage_context)
print("✅  Index loaded\n")

# ── Query Rewriter (same as app.py) ───────────────────────────
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


def rewrite_query(user_query: str) -> list: # user_query is passed as string  and list will b made of 4 variants.
    try:
        response = Settings.llm.complete(
            f"{REWRITER_PROMPT}\n\nUser question: {user_query}"
        )
        text = str(response).strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        variants = json.loads(text.strip()) # store 4 variants in the index to increase chances of relevant retrieval. these 4 variants will be used in addition to the original user query, which is also passed to the retriever. this way if the rewriter fails to generate good variants, the original query can still retrieve relevant chunks. we  are gettnig 4 variants beacause we have  given 4 query rewriting rules in the prompt (product/category, return window/time, actions, conditions) and we want to cover all dimensions of the policy documents to increase chances of relevant retrieval.
        return [user_query] + variants
    except Exception as e:
        print(f"[WARN] Query rewriting failed: {e}")
        return [user_query]


def retrieve_multi_query(user_query: str) -> list:
    retriever = vector_index.as_retriever(similarity_top_k=3)
    variants  = rewrite_query(user_query)
    seen_ids  = set()
    merged    = []
    for query in variants:
        for node in retriever.retrieve(query):
            if node.node.node_id not in seen_ids:
                seen_ids.add(node.node.node_id)
                merged.append(node)
    return merged


SYSTEM_PROMPT = """You are a Flipkart customer support assistant.
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
"""


def get_answer_and_context(question: str):
    """Run the full pipeline and return answer + context + nodes."""
    nodes   = retrieve_multi_query(question)
    context = "\n\n---\n\n".join(node.node.get_content() for node in nodes)
    prompt  = f"{SYSTEM_PROMPT}\n\nPolicy Context:\n{context}\n\nUser Question: {question}"
    answer  = str(Settings.llm.complete(prompt))
    return answer, context, nodes


# ── Test Dataset ──────────────────────────────────────────────
# Ground truth must come directly from the source policy document
TEST_CASES = [
    # ── Return window queries ──────────────────────────────────
    {
        "question": "after receiving shirt within what period i can return it",
        "ground_truth": "Lifestyle products including shirts have a 10 day return window. Refund, Replacement or Exchange is allowed."
    },
    {
        "question": "within how many days can i return a shirt",
        "ground_truth": "Lifestyle products including shirts have a 10 day return window. Refund, Replacement or Exchange is allowed."
    },
    {
        "question": "how many days do i have to return a mobile phone",
        "ground_truth": "Mobile phones fall under Electronics. 7 days service center replacement or repair only."
    },
    {
        "question": "return policy for books",
        "ground_truth": "7 Days Return. Please keep the product intact, with original accessories, user manual and warranty cards in the original packaging at the time of returning the product."
    },
    {
        "question": "can i return medicine",
        "ground_truth": "Medicine has a 2 day return window. Refund only."
    },
    # ── Cancellation queries ───────────────────────────────────
    {
        "question": "can i cancel my order after it is out for delivery",
        "ground_truth": "The order cannot be cancelled once it is out for delivery. However you may reject it at the doorstep."
    },
    {
        "question": "is there a cancellation fee",
        "ground_truth": "In some cases a cancellation fee may be charged if the order is cancelled after the specified time window."
    },
    # ── Edge cases ─────────────────────────────────────────────
    {
        "question": "what is the return policy for a jetpack",
        "ground_truth": "I don't have that information. Please contact Flipkart support."
    },
    {
        "question": "who is the CEO of Flipkart",
        "ground_truth": "I don't have that information. Please contact Flipkart support."
    },
]

# ── Evaluators (smaller model to save tokens) ─────────────────
relevancy_evaluator   = RelevancyEvaluator(llm=eval_llm)
correctness_evaluator = CorrectnessEvaluator(llm=eval_llm)


# ── Run Evaluation ────────────────────────────────────────────
def run_evaluation():
    results = []

    print("=" * 60)
    print("  RAG Evaluation — Flipkart Return Policy Assistant")
    print("=" * 60)

    for i, test in enumerate(TEST_CASES):
        question     = test["question"]
        ground_truth = test["ground_truth"]

        print(f"\n[{i+1}/{len(TEST_CASES)}] {question}")

        try:
            answer, context, nodes = get_answer_and_context(question)
            context_str = "\n\n".join(node.node.get_content() for node in nodes)

            # 1. Relevancy — were retrieved chunks relevant?
            rel_result = relevancy_evaluator.evaluate(
                query=question,
                response=answer,
                contexts=[context_str],
            )

            # 2. Correctness — does answer match ground truth?
            corr_result = correctness_evaluator.evaluate(
                query=question,
                response=answer,
                reference=ground_truth,
            )

            result = {
                "question"    : question,
                "ground_truth": ground_truth,
                "answer"      : answer,
                "relevancy"   : rel_result.passing,
                "correctness" : corr_result.score,
            }

            print(f"  Answer      : {answer[:100]}...")
            print(f"  Relevancy   : {'✅ Pass' if rel_result.passing else '❌ Fail'}")
            print(f"  Correctness : {corr_result.score}/5")

        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            result = {
                "question"    : question,
                "ground_truth": ground_truth,
                "answer"      : f"ERROR: {e}",
                "relevancy"   : None,
                "correctness" : None,
            }

        results.append(result)

    # ── Summary ───────────────────────────────────────────────
    valid    = [r for r in results if r["relevancy"] is not None]
    rel_pass = sum(1 for r in valid if r["relevancy"]) / len(valid) * 100
    corr_avg = sum(r["correctness"] for r in valid if r["correctness"]) / len(valid)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Relevancy     : {rel_pass:.0f}% passed")
    print(f"  Correctness   : {corr_avg:.1f} / 5.0 average")
    print("=" * 60)

    with open("eval_results.json", "w") as f:
        json.dump({"summary": {
            "relevancy_pass_rate"   : f"{rel_pass:.0f}%",
            "correctness_avg_score" : f"{corr_avg:.1f}/5",
        }, "details": results}, f, indent=2)

    print("\n📄  Results saved to eval_results.json")


if __name__ == "__main__":
    run_evaluation()
