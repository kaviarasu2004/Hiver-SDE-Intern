"""
baselines.py — two baselines to compare the agent against.

Baseline 1 (trivial): keyword intent lookup, canned reply, always escalate.
Baseline 2 (simple): zero-shot LLM intent + zero-shot reply, no RAG,
    escalate only if the tweet contains a "?".
"""
import json

from agent import INTENTS, get_client

CANNED_REPLY = "Hi! We'd love to help you out. Please DM us your account details so we can take a closer look!"

KEYWORDS = {
    "billing_payment": ["pay", "card", "charge", "bill", "refund", "subscription", "price"],
    "technical_playback": ["crash", "buffer", "pause", "offline", "download", "skip", "lag"],
    "account_access": ["password", "login", "log in", "locked", "hacked", "email"],
    "music_playlist": ["playlist", "local files", "sync", "queue"],
}


def baseline1_trivial(tweet: str) -> dict:
    t = tweet.lower()
    intent = "general_query_feedback"
    for cat, kws in KEYWORDS.items():
        if any(kw in t for kw in kws):
            intent = cat
            break
    return {
        "detected_intent": intent,
        "draft_reply": CANNED_REPLY,
        "action": "ESCALATE",
        "reason": "baseline1: always escalate",
    }


def baseline2_zero_shot(client, tweet: str) -> dict:
    intent_prompt = f"""Classify this tweet into exactly one of: {', '.join(INTENTS)}.
Tweet: "{tweet}"
Output only the category name."""
    intent_resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": intent_prompt}],
        temperature=0.0,
    )
    intent = intent_resp.choices[0].message.content.strip()
    if intent not in INTENTS:
        intent = "general_query_feedback"

    reply_prompt = f"""Write a short, polite Twitter reply (<280 chars) to this customer support tweet, as Spotify support. No historical context available.
Tweet: "{tweet}\""""
    reply_resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": reply_prompt}],
        temperature=0.3,
    )
    draft = reply_resp.choices[0].message.content.strip()

    action = "ESCALATE" if "?" in tweet else "AUTO"
    return {
        "detected_intent": intent,
        "draft_reply": draft,
        "action": action,
        "reason": "baseline2: escalate if tweet contains '?'",
    }


if __name__ == "__main__":
    import sys

    tweet = sys.argv[1] if len(sys.argv) > 1 else "why does the app keep crashing on my phone"
    print("baseline1:", json.dumps(baseline1_trivial(tweet), indent=2))
    client = get_client()
    print("baseline2:", json.dumps(baseline2_zero_shot(client, tweet), indent=2))
