# CASCADE-V: Demo Walkthrough

A 6–8 minute live demo. Run it from your M4 Mac with the venv active.

---

## Pre-call setup (do this once before the call)

```bash
cd cascade_v
source .venv/bin/activate

# All four prep steps. Total ~8 min.
python scripts/build_catalog.py
python scripts/train_encoder.py
python scripts/build_embeddings_and_tests.py
python scripts/evaluate_all.py
```

Make sure these exist before the call:
- `outputs/results.csv`
- `outputs/plots/comparison.png`
- `data/test_outputs/output_001.wav`

Have these tabs/windows ready: terminal, `outputs/plots/comparison.png` open in Preview, `outputs/receipts/` (will be empty until step 2).

---

## The demo, beat by beat

### Beat 1 — Set the frame (30 sec)

> "Variations is clean 1:1 — one source, one variation, one payout. Magic Fit breaks that. When outputs are shaped by multiple sources, the question 'who gets paid what' gets hard fast. I've prototyped what the layer between Variations and Magic Fit could look like. Let me show you."

### Beat 2 — Run the live attribution (90 sec)

```bash
python scripts/attribute.py output_001 --total-payout 1.00
```

The terminal shows three tables in succession:

1. **Per-source** — every candidate from the catalog with a weight and an interval. Point at the intervals: "Each weight has a confidence bound from Hoeffding's inequality propagated through the Shapley estimator. Production-grade — not point estimates that look exact but aren't."

2. **Per-creator** — aggregated. Point at it: "This is what Splice surfaces in the UI. The creator with two stems shows up once with their full share, not split between their files. That's Stage 2 doing its job."

3. **Fairness axioms** — three rows, all green PROVEN. **This is the moment.** Slow down here:

> "These three axioms — efficiency, symmetry, dummy — are what economists settled on in the '50s as the requirements for a fair payout split. Shapley is the only attribution method that satisfies them. Z3 just verified it on this specific result, in milliseconds."

### Beat 3 — Open the receipt (60 sec)

```bash
cat outputs/receipts/output_001.json | head -60
```

Or open it in VS Code. Walk through three blocks:

- `per_creator` — "this is what gets surfaced to the UI"
- `per_source` — "this is the audit trail"
- `verification` — "this is the proof certificate"

Then open the SMT file:

```bash
cat outputs/proofs/output_001.smt2
```

> "Here's the actual proof. It's an SMT-LIB file. Any auditor with z3 — `brew install z3` — can independently verify this receipt without trusting our pipeline. That's what I mean by audit-grade. The receipt isn't trustworthy because Splice says it is. It's trustworthy because the math checks."

### Beat 4 — Show the comparison (90 sec)

Open `outputs/plots/comparison.png`.

> "I ran four methods on 30 test outputs. Pure TRAK is fast but doesn't satisfy fairness. Pure Shapley is fair but doesn't handle the creator-DNA case — it splits credit between near-duplicate stems. CASCADE-V is the only one that wins on the Pareto frontier of all three: scale, group structure, and legal defensibility."

Open `outputs/results.csv`. Point specifically at the `is_dna_case` column and the creator-MAE delta.

> "On the cases where one creator contributed multiple stems — the creator-DNA case — CASCADE-V's creator-level error is half of pure Shapley's. That's because Stage 2 sees them as one cluster before the per-source split."

### Beat 5 — Why it's hard, and why it matters (60 sec)

> "Three things had to be composed for this to work. TRAK (MIT, late 2023) gave us scalability. GUDA (Feb 2026) gave us group-wise unlearning influence. Shapley gives us the fairness axioms. Nobody had wired these into one pipeline with formal verification on top until I built this last week."

> "The verification layer is what I'm patenting. US 12,536,365 already covers the Cascade Validation Protocol and SMT proof certification for binary document synthesis. Audio source attribution is the same architecture pattern with a different value function. I have a clear path to extending the patent."

### Beat 6 — The ask (30 sec)

> "I'd like to spend a few hours diving into Splice's actual data and use case with you and your eng team. I've shown that the math works at small scale on synthetic audio. The next question is: how would this slot into Variations production today, and what does the Magic Fit timeline look like with this layer in front of it. I can scope the production integration in a 90-day sprint if there's appetite."

Stop. Don't oversell. Let her ask the next question.

---

## Likely objections and responses

**"How does this scale to 3M sources?"**

> "Stage 1 is just FAISS HNSW retrieval — sub-100ms at 3M scale. Stage 2 and 3 only ever see 12 candidates regardless of catalog size. The expensive part is the Shapley evaluations, but at coalition size 12 we're doing exact Shapley in 4096 evaluations, well under a second. Production latency target: 100ms per attribution end-to-end."

**"Why not just use TRAK?"**

> "TRAK gives you a top-K list of similar sources. It does not give you a defensible split — there's no fairness axiom story behind 'similarity score normalized to sum to 1.' For payout decisions you need Shapley or something like it. TRAK is correct as Stage 1 of this pipeline; it's not enough on its own."

**"What's wrong with just Shapley on the top-K?"**

> "Two stems by the same producer are cosine-similar in embedding space — they cluster. Pure Shapley computed on the flat top-K splits credit between them, which is correct at the source level and wrong at the creator level. The producer should get the full group share, not 60/40 between their own files. That's the creator-DNA case Stage 2 fixes. The eval shows a 2× improvement in creator-MAE on those cases."

**"Why Z3?"**

> "Two reasons. First, the audit trail: every payout has a third-party-verifiable proof certificate. The creator's lawyer can run z3 on the SMT file and see for themselves the math is correct, without having to trust Splice. Second, it forces us to be explicit about which fairness axioms we're committing to — and to detect when an upstream change breaks them, since the proof would fail."

**"Is this in your patent?"**

> "The Cascade Validation Protocol, the interval arithmetic with forward error propagation, and the Z3 SMT certification — yes, US 12,536,365, issued January 2026. The original application was for binary document synthesis. Audio source attribution is the same architecture pattern with a different value function. There's a clear continuation-in-part path."

**"Why didn't Splice just build this?"**

> "Honestly because it wasn't needed yet. Variations is 1:1 attribution — clean signal, no multi-source ambiguity. The day Magic Fit ships, multi-source becomes the central problem, and you'll need a defensible answer for legal and for creator trust. I think you have a 6-month window to figure this out before the press cycle starts asking how the splits work."

---

## What to NOT say

- Do not claim this is "more accurate than Splice's current system" — you don't have access to it.
- Do not promise a specific accuracy number — the demo is on synthetic data and the absolute numbers are not what matter.
- Do not lead with the patent — lead with the problem and the demo. Bring up the patent only if she asks about IP or competitive moat.
- Do not pitch a job. Pitch the work. The job follows from the work.

---

## After the demo

If she's interested:

1. Send a follow-up email with the link to the receipt JSON and the SMT-LIB proof, plus the comparison.png.
2. Offer a 90-min working session with two of her engineers to scope the integration.
3. Have a one-pager ready that maps every component of CASCADE-V to a corresponding subsystem in Splice's stack (Variations infra, payout pipeline, creator dashboard).

If she's lukewarm:

1. Don't push. Ask "what would have to be true for this to be useful to Splice in 12 months?"
2. Listen to the answer. The honest answer to that question is a goldmine — either the Magic Fit launch is further out than reported, or there's a different blocker (legal already engaged, internal team building it, etc.). Either way you learn something.
3. Thank her, ask if she's open to a follow-up in 6 weeks, and end the call clean.
