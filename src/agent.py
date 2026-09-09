"""
agent.py — core pipeline: classify intent -> retrieve grounding (RAG) ->
draft reply -> decide auto-handle vs escalate.

Requires OPENAI_API_KEY in env. See .env.example.
"""
import json
import os
import time

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from openai import OpenAI

INTENTS = [
    "billing_payment",
    "technical_playback",
    "account_access",
    "music_playlist",
    "general_query_feedback",
]

SENSITIVE_INTENTS = {"billing_payment", "account_access"}

CLASSIFY_MODEL = "gpt-4o-mini"
GENERATE_MODEL = "gpt-4o-mini"


def get_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set — copy .env.example to .env and fill it in")
    return OpenAI(api_key=key)


def get_collection(db_path: str = "./chroma_db", name: str = "spotify_kb"):
    chroma_client = chromadb.PersistentClient(path=db_path)
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    return chroma_client.get_or_create_collection(name=name, embedding_function=default_ef)


def populate_vector_db(pairs_csv: str, db_path: str = "./chroma_db", name: str = "spotify_kb"):
    df = pd.read_csv(pairs_csv)
    collection = get_collection(db_path, name)
    ids = df["tweet_id"].astype(str).tolist()
    documents = df["customer_text"].tolist()
    metadatas = [{"reply": str(r)} for r in df["brand_reply"].tolist()]
    # chroma add() caps batch size in practice; chunk to be safe
    B = 500
    for i in range(0, len(ids), B):
        collection.add(
            ids=ids[i:i + B],
            documents=documents[i:i + B],
            metadatas=metadatas[i:i + B],
        )
    print(f"[agent] indexed {len(ids)} historical pairs into '{name}'")


def _call_with_retry(fn, retries: int = 3, backoff: float = 2.0):
    last_err = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:  # openai SDK raises various transient errors
            last_err = e
            time.sleep(backoff * (attempt + 1))
    raise last_err


def classify_intent(client: OpenAI, tweet: str) -> str:
    prompt = f"""Classify the following customer support tweet into exactly one of these categories:
- {chr(10).join(INTENTS)}

Tweet: "{tweet}"
Output only the category name. No formatting, no extra words."""

    def _do():
        resp = client.chat.completions.create(
            model=CLASSIFY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    intent = _call_with_retry(_do)
    return intent if intent in INTENTS else "general_query_feedback"


def retrieve_grounding(collection, tweet: str, k: int = 2) -> str:
    results = collection.query(query_texts=[tweet], n_results=k)
    out = ""
    docs = results.get("documents") or [[]]
    metas = results.get("metadatas") or [[]]
    for doc, meta in zip(docs[0], metas[0]):
        out += f"Past Customer: {doc}\nPast Brand Reply: {meta['reply']}\n\n"
    return out


def draft_and_route(client: OpenAI, tweet: str, intent: str, grounding: str) -> dict:
    prompt = f"""You are an automated customer service assistant for Spotify (@SpotifyCares) on Twitter.
Draft a helpful, professional, brand-aligned reply and decide if this needs human intervention.

Current Customer Tweet: "{tweet}"
Detected Intent: {intent}

Historical solutions to similar issues:
{grounding if grounding else "(no close historical match found)"}

Decision Rules:
- ESCALATE if the customer needs account/payment lookups, sounds highly frustrated, or no historical solution resolves this specific problem.
- AUTO-HANDLE for standard technical fixes, general questions, or feedback.

Respond in JSON with keys:
"draft_reply" (<=280 chars, polite, conversational),
"action" ("AUTO" or "ESCALATE"),
"reason" (short explanation)."""

    def _do():
        resp = client.chat.completions.create(
            model=GENERATE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return json.loads(resp.choices[0].message.content)

    result = _call_with_retry(_do)

    # deterministic safety net on top of the model's own judgement —
    # never let a sensitive intent slip through as AUTO
    if intent in SENSITIVE_INTENTS and result.get("action") == "AUTO":
        result["action"] = "ESCALATE"
        result["reason"] = (result.get("reason", "") + " [rule override: sensitive intent forced escalation]").strip()

    return result


def run_agent(client: OpenAI, collection, tweet: str, k: int = 2) -> dict:
    intent = classify_intent(client, tweet)
    grounding = retrieve_grounding(collection, tweet, k=k)
    result = draft_and_route(client, tweet, intent, grounding)
    result["detected_intent"] = intent
    return result


if __name__ == "__main__":
    import sys

    client = get_client()
    collection = get_collection()
    tweet = sys.argv[1] if len(sys.argv) > 1 else "My offline downloads keep disappearing, so annoying"
    print(json.dumps(run_agent(client, collection, tweet), indent=2))
