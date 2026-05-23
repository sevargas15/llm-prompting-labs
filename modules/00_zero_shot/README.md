# Module 0 — Zero-Shot Prompting

**Difficulty:** Beginner | **Time:** ~2 hours | **Model trained:** DistilBERT classifier

---

## What is zero-shot prompting?

A model with broad pretraining can perform tasks it was never explicitly trained for — just by being told what to do in natural language. No examples required.

```
Prompt: "Classify the sentiment of this text as happy, neutral, or sad."
Input:  "I just got promoted!"
Output: "happy"
```

GPT-3.5 can do this reliably. But why? And how does your own trained model compare?

**That's what this module answers.**

---

## What you'll build

```
data/
  sentiment_raw.jsonl        ← 200 GPT-generated examples
  sentiment_train.jsonl      ← 160 train
  sentiment_val.jsonl        ← 20 validation
  sentiment_test.jsonl       ← 20 test

checkpoints/
  best_model/                ← Your fine-tuned DistilBERT

notebooks/
  00_zero_shot_walkthrough.ipynb   ← Full guided walkthrough

scripts/
  01_generate_dataset.py     ← Call GPT to generate labeled data
  02_train.py                ← Fine-tune DistilBERT
  03_evaluate.py             ← Compare your model vs GPT zero-shot
  04_visualize.py            ← t-SNE of learned representations
```

---

## Step-by-step

### Step 1 — Generate a dataset with GPT

```bash
python scripts/01_generate_dataset.py \
  --task sentiment \
  --n 200 \
  --output data/sentiment_raw.jsonl
```

This script calls GPT-3.5 and asks it to generate 200 diverse sentences
with sentiment labels. It uses a structured prompt to ensure variety across:
- Topics (work, relationships, weather, sports, food...)
- Intensity (mildly happy vs ecstatic)
- Style (formal, casual, sarcastic)

**Cost estimate:** ~$0.05 using GPT-3.5-turbo

---

### Step 2 — Fine-tune DistilBERT

```bash
python scripts/02_train.py \
  --data data/sentiment_raw.jsonl \
  --model distilbert-base-uncased \
  --epochs 5 \
  --output checkpoints/
```

Trains a 3-class classifier (happy=0, neutral=1, sad=2) with:
- AdamW optimizer
- Linear warmup scheduler
- Early stopping on validation loss

**Training time:** ~5 min on CPU, ~45 sec on GPU

---

### Step 3 — Evaluate

```bash
python scripts/03_evaluate.py \
  --model checkpoints/best_model \
  --test-data data/sentiment_test.jsonl \
  --gpt-baseline
```

Outputs a comparison table:

| Model | Accuracy | F1 | Latency |
|-------|----------|----|---------|
| GPT-3.5 zero-shot | ~87% | ~0.86 | ~800ms |
| Your DistilBERT | ~91% | ~0.90 | ~12ms |
| Random baseline | 33% | 0.33 | — |

**Key insight:** Your 66x smaller model beats GPT on this specific task
because it was trained on task-specific data. Zero-shot is about breadth;
fine-tuning is about depth.

---

### Step 4 — Visualize representations

```bash
python scripts/04_visualize.py \
  --model checkpoints/best_model \
  --data data/sentiment_test.jsonl
```

Generates a t-SNE plot showing how your model clusters sentences by sentiment
in its embedding space. You'll see three distinct clouds.

---

## Key concepts learned

- **Pretraining** gives models broad world knowledge
- **Zero-shot** works by mapping natural language instructions to learned patterns
- **Fine-tuning** narrows a general model to excel at a specific task
- **Data quality > quantity** — 200 good examples beat 2000 noisy ones
- **Smaller models can outperform larger ones** on narrow tasks

---

## Extend it

- Add more sentiment classes (angry, fearful, surprised)
- Try other tasks: topic classification, spam detection, intent recognition
- Swap DistilBERT for `roberta-base` — does it improve?
- Generate the dataset in another language and test cross-lingual zero-shot

---

## Next module

[Module 1 — Few-Shot Prompting →](../01_few_shot/README.md)

Learn how providing just 2–5 examples in your prompt can dramatically
improve performance on rare or specialized tasks.
