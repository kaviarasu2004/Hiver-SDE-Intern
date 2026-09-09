"""
evaluate.py — run the agent (and baselines) against golden_eval_set.json,
compute classification / escalation metrics, and score reply quality with
an LLM-as-judge, with a slot to record human-judge agreement.

Usage:
    python src/evaluate.py --golden data/golden_eval_set.json --system agent
    python src/evaluate.py --golden data/golden_eval_set.json --system baseline1
    python src/evaluate.py --golden data/golden_eval_set.json --system baseline2
"""
import argparse
import json

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from agent import get_client, get_collection, run_agent
from baselines import baseline1_trivial, baseline2_zero_shot


def load_golden(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_system(system: str, items: list) -> list:
    client = get_client() if system != "baseline1" else None
    collection = get_collection() if system == "agent" else None

    preds = []
    for item in items:
        tweet = item["customer_text"]
        if system == "agent":
            preds.append(run_agent(client, collection, tweet))
        elif system == "baseline1":
            preds.append(baseline1_trivial(tweet))
        elif system == "baseline2":
            preds.append(baseline2_zero_shot(client, tweet))
        else:
            raise ValueError(f"unknown system: {system}")
    return preds


def deterministic_metrics(golden: list, preds: list) -> dict:
    y_true_intent = [g["true_intent"] for g in golden]
    y_pred_intent = [p["detected_intent"] for p in preds]

    y_true_action = [g["true_action"] for g in golden]
    y_pred_action = [p["action"] for p in preds]

    return {
        "intent_accuracy": accuracy_score(y_true_intent, y_pred_intent),
        "intent_macro_f1": f1_score(y_true_intent, y_pred_intent, average="macro", zero_division=0),
        "escalation_precision": precision_score(
            y_true_action, y_pred_action, pos_label="ESCALATE", zero_division=0
        ),
        "escalation_recall": recall_score(
            y_true_action, y_pred_action, pos_label="ESCALATE", zero_division=0
        ),
        "escalation_f1": f1_score(
            y_true_action, y_pred_action, pos_label="ESCALATE", zero_division=0
        ),
    }


JUDGE_MODEL = "gpt-4o"


def run_llm_judge(client, tweet: str, drafted_reply: str, ideal_bulletpoints: str) -> dict:
    prompt = f"""You are an expert QA evaluator for automated customer support systems.
Customer Tweet: "{tweet}"
Drafted Reply: "{drafted_reply}"
Ideal Criteria: "{ideal_bulletpoints}"

Rate 1-5 (1 worst, 5 best):
1. groundedness (avoids hallucinating policies/features)
2. actionability (concrete next steps / correct support path)
3. tone_constraint (<=280 chars, polite, tweet-like)

Respond ONLY as JSON: {{"groundedness": <n>, "actionability": <n>, "tone_constraint": <n>, "explanation": "<short>"}}"""
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(resp.choices[0].message.content)


def judge_all(client, golden: list, preds: list) -> list:
    scores = []
    for g, p in zip(golden, preds):
        scores.append(
            run_llm_judge(
                client,
                g["customer_text"],
                p.get("draft_reply", ""),
                " ".join(g.get("ideal_response_bulletpoints", [])),
            )
        )
    return scores


def avg(scores: list, key: str) -> float:
    vals = [s[key] for s in scores if key in s]
    return sum(vals) / len(vals) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--system", required=True, choices=["agent", "baseline1", "baseline2"])
    ap.add_argument("--skip-judge", action="store_true", help="skip LLM-judge calls (deterministic metrics only)")
    args = ap.parse_args()

    golden = load_golden(args.golden)
    preds = run_system(args.system, golden)

    det = deterministic_metrics(golden, preds)
    print(f"\n=== {args.system} — deterministic metrics ===")
    for k, v in det.items():
        print(f"{k}: {v:.3f}")

    if not args.skip_judge:
        client = get_client()
        judge_scores = judge_all(client, golden, preds)
        print(f"\n=== {args.system} — LLM-judge (avg of {len(judge_scores)}) ===")
        for key in ("groundedness", "actionability", "tone_constraint"):
            print(f"{key}: {avg(judge_scores, key):.2f}")

    out_path = f"results_{args.system}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"deterministic": det, "predictions": preds}, f, indent=2)
    print(f"\n[evaluate] wrote {out_path}")


if __name__ == "__main__":
    main()
