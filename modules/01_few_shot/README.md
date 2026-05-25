# Module 1 | Few-Shot Prompting

---

## What is few-shot prompting?

Sometimes a model needs a nudge. Instead of just describing a task, you show it a handful of examples — and it figures out the pattern.

```
Prompt: "Classify the sentiment of this text."

Examples:
  "I aced the interview!" → positive
  "The flight was delayed again." → negative
  "The package arrived today." → neutral

Input:  "My coffee finally kicked in."
Output: "positive"
```

A few well-chosen examples often beat a long description. But *how many* examples? *Which* examples? And does your fine-tuned model still need them?

**That's what this module answers.**

---

## What you'll build

```
data/
  sentiment_raw.jsonl        ← Reused from Module 0 (or regenerate)
  few_shot_examples.jsonl    ← Curated example bank for prompting

notebooks/
  01_few_shot_prompting.ipynb    ← Start here — full guided walkthrough

scripts/
  01_build_example_bank.py   ← Curate and score few-shot candidates
  02_run_few_shot.py         ← Evaluate Groq with k=1,2,4,8 examples
  03_compare_models.py       ← Few-shot baseline vs your fine-tuned model
  04_visualize.py            ← Performance curves across k values
```

---

## How to run it

The recommended way is through the notebook. Each decision is explained — why certain examples work better, what happens when you pick the wrong ones, and when more isn't better.

```bash
jupyter notebook notebooks/01_few_shot_prompting.ipynb
```

Or run the scripts directly if you're already comfortable with the concept.

### Step 1 — Build the example bank

```bash
python scripts/01_build_example_bank.py \
  --data data/sentiment_raw.jsonl \
  --output data/few_shot_examples.jsonl \
  --per-class 20
```

### Step 2 — Run few-shot evaluations

```bash
python scripts/02_run_few_shot.py \
  --examples data/few_shot_examples.jsonl \
  --test data/sentiment_test.jsonl \
  --k 1 2 4 8
```

### Step 3 — Compare against fine-tuned model

```bash
python scripts/03_compare_models.py \
  --model checkpoints/best_model \
  --test-data data/sentiment_test.jsonl \
  --few-shot-results results/few_shot_runs.json
```

### Step 4 — Visualize performance curves

```bash
python scripts/04_visualize.py \
  --results results/few_shot_runs.json \
  --output plots/few_shot_curve.png
```

---

## Key concepts learned from this one:

- **Few-shot prompting:** examples act as implicit instructions the model pattern-matches against
- **Example selection matters:** random picks underperform curated, diverse ones
- **k has diminishing returns:** going from 1→4 examples helps; 8→16 rarely does
- **Label balance:** skewed examples bias outputs — always balance your shot selection
- **Fine-tuned models are robust:** your Module 0 model likely needs zero examples to match few-shot performance

---

## Next module

[Module 2 — Chain-of-Thought Prompting →](../02_chain_of_thought/README.md)
