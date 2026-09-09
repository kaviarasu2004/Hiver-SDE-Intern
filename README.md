# Hiver SDE Intern — Take-Home: SpotifyCares Support Agent

Status: **code scaffold done + tested. Golden set (200) and headline numbers
NOT YET real — need real `twcs.csv` + `OPENAI_API_KEY` run locally.**
See "Current status" at bottom before trusting any number in `PLAN.md`.

## What this is

AI support agent for @SpotifyCares (Kaggle "Customer Support on Twitter"):
classify intent → retrieve grounding from historical resolved tweets (RAG) →
draft reply → auto-handle or escalate with a stated reason.

## Repo layout

```
├── data/
│   ├── sample_raw_data.csv          # 3 synthetic rows, twcs.csv schema — smoke test only
│   ├── golden_eval_set.template.json # schema + 3 worked examples — NOT the real 200
│   └── (you add) twcs.csv, golden_eval_set.json
├── src/
│   ├── ingest.py          # twcs.csv -> (customer_tweet, brand_reply) pairs
│   ├── agent.py           # classify -> RAG retrieve -> draft + escalate
│   ├── baselines.py       # baseline1 (keyword/canned), baseline2 (zero-shot LLM)
│   ├── evaluate.py        # deterministic metrics + LLM-judge, per system
│   └── judge_agreement.py # human vs LLM-judge Pearson r / Cohen's kappa
├── tests/test_agent.py    # offline unit tests, no API key needed
├── PLAN.md                 # original design doc (renamed from `readme`, PII stripped)
├── requirements.txt
└── .env.example
```

## Setup (5 min)

```bash
git clone <this-repo>
cd Hiver-SDE-Intern
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY
```

## Step-by-step: reproduce headline results (<15 min once you have data)

1. **Get the real dataset** (not included — 3M rows, too big to commit):
   download `twcs.csv` from Kaggle `thoughtvector/customer-support-on-twitter`,
   place at `data/twcs.csv`.

2. **Build pairs** (few seconds on a filtered subset):
   ```bash
   python src/ingest.py --input data/twcs.csv --brand SpotifyCares \
       --out data/spotify_pairs.csv --limit 200000
   ```

3. **Index into the vector DB** (~1-2 min for a few thousand pairs):
   ```bash
   python -c "from src.agent import populate_vector_db; populate_vector_db('data/spotify_pairs.csv')"
   ```

4. **Build the real golden set** — copy `data/golden_eval_set.template.json`
   to `data/golden_eval_set.json`, hand-label 150-250 real examples
   (sampling method: stratified by intent frequency in `spotify_pairs.csv`,
   plus ~10% deliberately-hard sarcasm/multi-issue cases — see PLAN.md
   Phase 3 for the exact split used to plan this).

5. **Run agent + both baselines**:
   ```bash
   python src/evaluate.py --golden data/golden_eval_set.json --system agent
   python src/evaluate.py --golden data/golden_eval_set.json --system baseline1
   python src/evaluate.py --golden data/golden_eval_set.json --system baseline2
   ```
   Each writes `results_<system>.json` with deterministic metrics +
   per-item predictions.

6. **Validate the judge** — hand-score 30-50 of the same items yourself
   into `data/human_judge_scores.json` (same schema, see
   `judge_agreement.py` docstring), then:
   ```bash
   python src/judge_agreement.py --llm-scores results_agent.json --human-scores data/human_judge_scores.json
   ```
   Report the printed Pearson r / kappa in `REPORT.md`. This number is
   mandatory — an unvalidated judge is not evidence.

7. **Run offline tests** (no API key needed, run anytime):
   ```bash
   pytest tests/
   ```

## Current status (honest)

- ingest.py: **written, unit-tested**. Caught and fixed a real bug during
  dev — `tweet_id` (int) vs `in_response_to_tweet_id` (float, from NaNs)
  stringify to `"1"` vs `"1.0"`, silently dropping every merge row.
  Regression test in `tests/test_agent.py` locks this in.
- agent.py / baselines.py / evaluate.py / judge_agreement.py: **written,
  not yet run against real data** — needs `twcs.csv` (Kaggle) and a live
  `OPENAI_API_KEY`, neither available in the environment used to build
  this scaffold.
- golden_eval_set.json: **template only (3 examples)**. Real 150-250
  hand-labeled set not built yet.
- REPORT.md and any results table: **not generated yet** — do not trust
  the example numbers/failure cases in `PLAN.md`, they're illustrative
  planning content, not measured results.

## What's misleading about a headline number before you've done the above

Any accuracy/F1/judge-score you produce is only as good as: golden-set
selection (you picked which 200 — may skew easy), judge-generator
similarity (LLM judge can favor its own generator's style), one-brand
scope (SpotifyCares patterns won't generalize to other brands), and
small-n confidence intervals. Put this analysis in `REPORT.md` once real
numbers exist — see `PLAN.md` Phase 6 §5 for the structure to follow.
