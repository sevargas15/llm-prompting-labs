# Module | Zero-Shot Prompting

---

## What is zero-shot prompting?

A model with broad pretraining can perform tasks it was never explicitly trained for — just by being told what to do in natural language. No examples required.

```
Prompt: "Classify the sentiment of this text as happy, neutral, or sad."
Input:  "I just got promoted!"
Output: "happy"
```

Large models can do this reliably. But why? And how does your own trained model compare?

**That's what this module answers.**

---

## What you'll build

```
data/
  sentiment_raw.jsonl        ← 200 Groq-generated examples
  sentiment_train.jsonl      ← 160 train
  sentiment_val.jsonl        ← 20 validation
  sentiment_test.jsonl       ← 20 test

checkpoints/
  best_model/                ← Your fine-tuned DistilBERT

notebooks/
  00_zero_shot_prompting.ipynb   ← Start here — full guided walkthrough

scripts/
  01_generate_dataset.py     ← Generate labeled data via Groq
  02_train.py                ← Fine-tune DistilBERT
  03_evaluate.py             ← Compare your model vs zero-shot baseline
  04_visualize.py            ← t-SNE of learned representations
```

---

## How to run it

The recommended way is through the notebook. I tried to walk you through every decision and explain the why behind each step:

```bash
jupyter notebook notebooks/00_zero_shot_prompting.ipynb
```

Or run the scripts directly if you already understand this technique

### Step 1 — Generate a dataset

```bash
python scripts/01_generate_dataset.py \
  --task sentiment \
  --n 200 \
  --output data/sentiment_raw.jsonl
```

### Step 2 — Fine-tune DistilBERT

```bash
python scripts/02_train.py \
  --data data/sentiment_raw.jsonl \
  --model distilbert-base-uncased \
  --epochs 5 \
  --output checkpoints/
```

### Step 3 — Evaluate

```bash
python scripts/03_evaluate.py \
  --model checkpoints/best_model \
  --test-data data/sentiment_test.jsonl \
  --groq-baseline
```

### Step 4 — Visualize representations

```bash
python scripts/04_visualize.py \
  --model checkpoints/best_model \
  --data data/sentiment_test.jsonl
```

---

## Key concepts learned from this one:

- **Pretraining:** gives models broad world knowledge
- **Zero-shot:** works by mapping natural language instructions to learned patterns
- **Fine-tuning:** narrows a general model to excel at a specific task
- **Data quality > quantity:** a few hundred good examples go a long way
- **Smaller models can outperform larger ones** on narrow tasks

---

## Next module

[Module 1 — Few-Shot Prompting →](../01_few_shot/README.md)
