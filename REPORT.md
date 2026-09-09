# REPORT — SpotifyCares Support Agent

Status: sections 1, 5, 6, 7 are complete. Sections 2–4 have the
structure and method locked in but **numbers are pending** a run against
real `twcs.csv` + a live `OPENAI_API_KEY` (not available in the sandbox
this scaffold was built in — see README "Current status"). Do not quote
any number here until `results_agent.json` / `results_baseline1.json` /
`results_baseline2.json` exist from a real run.

---

## 1. Problem framing

**Brand:** @SpotifyCares.

**What "good" means for this brand:** support here is high-volume,
low-latency, and mostly self-serve-fixable (playback, sync, login) —
but a meaningful slice needs account/payment lookups Twitter can't do.
So "good" = (a) correctly and instantly resolves standard technical
questions without human time, (b) never invents a policy, refund, or
feature that doesn't exist, (c) reliably routes anything touching money
or account access to a human, with a reason a human can audit in one
read.

**What we chose not to build:**
- No multi-turn memory / conversation state — each inbound tweet is
  scored independently. Real threads are multi-turn; this is a scope
  cut, not an oversight (see Decision Log #3).
- No write actions (password reset, refund issuance, cancellation) —
  the agent only drafts + routes, never executes account changes. Pure
  safety boundary for a take-home-scale system.
- No non-English handling — dataset and prompts assume English;
  non-English tweets are expected to route to `general_query_feedback`
  or misclassify, and that's documented as a known gap, not hidden.
- No multi-brand generalization claim — everything here is tuned to
  SpotifyCares's tone and issue mix; nothing about the results should
  be read as "this works for any brand."

## 2. Results vs. baselines

**TODO — run `src/evaluate.py` against all three systems on the real
golden set, then paste `results_*.json` numbers into this table.**

| Metric | Baseline 1 (keyword/canned) | Baseline 2 (zero-shot, no RAG) | Agent (RAG + guided LLM) |
|---|---|---|---|
| Intent accuracy | TODO | TODO | TODO |
| Intent macro-F1 | TODO | TODO | TODO |
| Escalation precision | TODO | TODO | TODO |
| Escalation recall | TODO | TODO | TODO |
| Escalation F1 | TODO | TODO | TODO |
| Judge — groundedness (1-5) | TODO | TODO | TODO |
| Judge — actionability (1-5) | TODO | TODO | TODO |
| Judge — tone/constraint (1-5) | TODO | TODO | TODO |

Method for filling this in: `python src/evaluate.py --golden
data/golden_eval_set.json --system <agent|baseline1|baseline2>` for
each of the three, then transcribe. Do not hand-edit the numbers.

## 3. Failure analysis — top 5 failure modes

**TODO — pull real examples from `results_agent.json` after a real
run.** Categories worth specifically checking for, based on the system
design (fill each with an actual mis-classified/mis-routed item once
you have one):

1. **Sarcasm / irony** — classifier has no sentiment signal beyond the
   prompt; a sarcastic compliment about a bug reads as positive
   feedback. Check `general_query_feedback` predictions against golden
   `true_intent` for false positives here first.
2. **Device-context dilution in retrieval** — embedding similarity
   favors generic keyword overlap ("app crashing") over specific
   device/context words ("on my watch" vs "on desktop"), so RAG can
   surface the wrong historical fix. Check retrieved grounding text on
   `technical_playback` items with a named device.
3. **Escalation threshold sensitivity** — mild frustration words
   ("confusing", "annoying") vs genuine anger aren't distinguished by
   the prompt's rule text alone. Check escalation false positives
   (golden AUTO, predicted ESCALATE).
4. **JSON/formatting leakage** — `evaluate.py` assumes clean
   `json.loads` on the model's output; a malformed response currently
   raises rather than degrades gracefully. Check for exceptions during
   a real run, not just wrong answers.
5. **RAG grounding staleness** — historical replies can reference
   dead links or discontinued features; nothing currently validates
   URLs/features in retrieved grounding before they're echoed into a
   draft. Check drafts that mention a link and verify it's still live.

## 4. What is misleading about my headline number?

This section is mandatory regardless of what the headline number turns
out to be — write it against the real numbers once you have them, but
these structural caveats will hold no matter what:

1. **Golden-set selection bias** — you sampled and labeled the 150–250
   examples yourself; there's no independent check that the sample
   isn't skewed toward clearer, easier-to-label cases than the full
   distribution.
2. **RAG/eval overlap risk** — if any golden-set tweet's near-duplicate
   sits in the RAG index, the agent can look better than it would on
   genuinely novel issues (e.g. a new subscription tier or a redesign
   nobody's tweeted about yet).
3. **Judge–generator similarity** — an LLM judge scoring an LLM's own
   output family can share stylistic taste rather than truly checking
   correctness; the human-agreement number in Section 4 offsets this
   but doesn't erase it.
4. **Single-brand, single-turn scope** — nothing here says anything
   about other brands, or about multi-turn threads where frustration
   escalates over several exchanges.
5. **Small-n confidence** — 150–250 examples give wide confidence
   intervals on any percentage; a 92% vs 88% difference between systems
   may not be statistically distinguishable at this n.

## 5. What I'd do next with one more week

- Multi-turn context: score the whole thread, not just the opening
  tweet, so a customer who's already been told "reinstall the app"
  doesn't get the same suggestion twice.
- Confidence-calibrated escalation instead of a single LLM decision +
  rule override: log classifier confidence and calibrate an actual
  threshold against the golden set rather than a hand-picked rule.
- Dedupe/refresh the RAG index: drop near-duplicate historical pairs,
  flag or strip dead links before they can be retrieved as grounding.
- Second, independent human labeler for a subset of the golden set to
  get real inter-annotator agreement, not just judge-vs-single-human.
- Stress-test on deliberately adversarial inputs (sarcasm, mixed
  language, multi-issue tweets) sampled specifically for that, beyond
  the ~10% edge-case slice in the current sampling plan.
- Swap the embedding model and A/B the retrieval quality — the device-
  context dilution failure mode (Section 3, #2) is a known weakness of
  generic sentence embeddings on short, keyword-heavy text.

## 6. Decision log

Non-obvious calls made while building this, and why:

1. **SpotifyCares over other brands** — high tweet volume, and issue
   types (playback/billing/account) map cleanly onto a small intent
   set without a long tail of brand-specific edge cases.
2. **5 intents, not more** — enough to separate action-relevant
   categories (sensitive vs not) without fragmenting into near-
   duplicate buckets that would only hurt classifier consistency.
3. **No multi-turn state** — cut for scope and reliability; a stateless
   per-tweet agent is easier to evaluate and to debug than one carrying
   hidden context across turns.
4. **Sensitive-intent rule override in `agent.py`** — even if the LLM's
   own escalation judgment says AUTO, `billing_payment` and
   `account_access` are force-escalated in code. A prompt-only safety
   rule can be talked around by phrasing; a code-level override can't.
5. **RAG grounding is customer-tweet embedding only, not intent-
   filtered** — simpler to build first; intent-scoped retrieval was
   considered but skipped to avoid compounding classifier errors into
   retrieval errors (a wrong intent would silently mis-scope the RAG
   search).
6. **int/float id-merge bug fix locked in with a unit test**, not just
   fixed ad hoc — `tweet_id` (int64) vs `in_response_to_tweet_id`
   (float64, from NaNs) stringify to `"1"` vs `"1.0"` and silently
   drop every merged row. This is the kind of bug that produces a
   confidently wrong "0 pairs found" with no error — worth a regression
   test, not just a fix.
7. **Retry wrapper on LLM calls, not on retrieval** — network/rate-limit
   failures are the realistic transient failure mode for the OpenAI
   calls; local Chroma queries don't need the same treatment.
8. **Judge model (gpt-4o) different from generation model (gpt-4o-mini)**
   — using the same model to generate and judge its own output
   compounds the judge-leniency problem noted in Section 4; using a
   stronger, distinct model for judging is a partial (not full)
   mitigation.
9. **Golden set template ships with 3 worked examples, not left empty**
   — makes the schema unambiguous for whoever labels the real 150-250,
   rather than relying on prose description alone.
10. **`.gitignore` excludes `data/twcs.csv` and `results_*.json`** —
    the raw 3M-row file must never be committed (repo-size / clone-time
    blowout), and result files are regenerated artifacts, not source.
11. **Escalation baseline-2 rule ("escalate if tweet contains '?'") kept
    deliberately dumb** — a baseline that's too clever stops being a
    useful floor to compare against; the point is to show the real
    system clearly beats a naive heuristic, not to build a second good
    system.
12. **Canned reply in baseline 1 is verbatim from the original plan
    doc, not rewritten** — keeps baseline 1 a genuine "did nothing
    smart" floor rather than accidentally becoming baseline 1.5.

## 7. What was borrowed / AI-assisted

- Overall phase structure (data engineering → architecture → golden set
  → eval harness → baselines → report) and the initial prompt text for
  classification/generation/judging were drafted with an AI coding
  assistant (Claude) from a plan document, then implemented, run, and
  debugged (see Decision Log #6 for a bug the assistant introduced and
  a human-directed fix/test caught).
- No third-party code copied beyond standard library usage of `openai`,
  `chromadb`, `pandas`, `scikit-learn`, `scipy` per their public APIs —
  no snippets lifted from Stack Overflow/blogs.
