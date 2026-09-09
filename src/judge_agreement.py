"""
judge_agreement.py — validates the LLM-as-judge against human scores.

Workflow (you do this manually, once):
1. Run evaluate.py on ~30-50 golden items, keep results_<system>.json.
2. Hand-score the SAME items yourself on the same 1-5 rubric, save as
   data/human_judge_scores.json:
   [{"tweet_id": "...", "groundedness": 4, "actionability": 5, "tone_constraint": 5}, ...]
3. Run this script — it prints Pearson r and Cohen's kappa per dimension.
   Report these numbers in REPORT.md. Do not skip this — a judge with
   unknown reliability is not evidence.
"""
import argparse
import json

from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-scores", required=True, help="judge_scores.json (list, same order as human)")
    ap.add_argument("--human-scores", required=True, help="data/human_judge_scores.json")
    args = ap.parse_args()

    llm = load(args.llm_scores)
    human = load(args.human_scores)

    if len(llm) != len(human):
        raise ValueError(f"length mismatch: {len(llm)} llm vs {len(human)} human — must score the same items")

    for dim in ("groundedness", "actionability", "tone_constraint"):
        l_vals = [x[dim] for x in llm]
        h_vals = [x[dim] for x in human]
        r, p = pearsonr(l_vals, h_vals)
        kappa = cohen_kappa_score(l_vals, h_vals, weights="linear")
        print(f"{dim}: pearson r={r:.3f} (p={p:.3f})  weighted kappa={kappa:.3f}  n={len(l_vals)}")


if __name__ == "__main__":
    main()
