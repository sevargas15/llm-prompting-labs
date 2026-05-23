"""
Step 3: Evaluate your fine-tuned model and compare to Groq zero-shot.

Usage:
    python scripts/03_evaluate.py --model checkpoints/best_model --test-data data/sentiment_test.jsonl
    python scripts/03_evaluate.py --model checkpoints/best_model --test-data data/sentiment_test.jsonl --groq-baseline
"""

import os
import json
import time
import argparse
import random
from pathlib import Path

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    logging as transformers_logging,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from dotenv import load_dotenv

load_dotenv()

transformers_logging.set_verbosity_error()

LABEL2ID = {"happy": 0, "neutral": 1, "sad": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.1-8b-instant"


def load_test_data(path: str):
    examples = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line.strip())
            if ex.get("label") in LABEL2ID:
                examples.append(ex)
    return examples


def predict_finetuned(model, tokenizer, texts, device, batch_size=32):
    """Batch predict with fine-tuned model."""
    model.eval()
    all_preds = []
    latencies = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        start = time.time()

        encodings = tokenizer(
            batch_texts,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**encodings)

        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        elapsed = (time.time() - start) / len(batch_texts) * 1000
        all_preds.extend([ID2LABEL[p] for p in preds])
        latencies.append(elapsed)

    return all_preds, np.mean(latencies)


def predict_groq_zeroshot(texts):
    """Zero-shot predict with Groq (single calls for accurate latency)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "\n❌ GROQ_API_KEY not set.\n"
            "   1. Get your free key at https://console.groq.com\n"
            "   2. Add it to your .env file: GROQ_API_KEY=gsk_...\n"
        )

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    preds = []
    latencies = []

    print(f"  Running Groq zero-shot on {len(texts)} examples...")
    for i, text in enumerate(texts):
        start = time.time()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a sentiment classifier. Respond with exactly one word: happy, neutral, or sad."},
                {"role": "user", "content": f"Classify the sentiment of this text:\n\n{text}"}
            ],
            temperature=0,
            max_tokens=5,
        )
        elapsed = (time.time() - start) * 1000
        pred = response.choices[0].message.content.strip().lower()
        if pred not in LABEL2ID:
            pred = "neutral"
        preds.append(pred)
        latencies.append(elapsed)
        time.sleep(0.5)  # gentle rate limiting

        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(texts)} done...")

    return preds, np.mean(latencies)


def print_results(name, labels, preds, avg_latency_ms):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy:    {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Macro F1:    {f1:.4f}")
    print(f"  Avg latency: {avg_latency_ms:.1f} ms/example")
    print(f"\n  Per-class report:")
    print(classification_report(labels, preds, target_names=list(LABEL2ID.keys()), indent=4))


def print_comparison_table(results):
    print("\n" + "="*60)
    print("  FINAL COMPARISON")
    print("="*60)
    print(f"  {'Model':<30} {'Accuracy':>10} {'F1':>8} {'Latency':>12}")
    print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*12}")
    for name, acc, f1, latency in results:
        print(f"  {name:<30} {acc*100:>9.1f}% {f1:>8.3f} {latency:>10.1f}ms")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate your fine-tuned model and compare to Groq zero-shot"
    )
    parser.add_argument("--model", default="checkpoints/best_model")
    parser.add_argument("--test-data", default="data/sentiment_test.jsonl")
    parser.add_argument("--groq-baseline", action="store_true",
                        help="Also run Groq zero-shot baseline (free)")
    args = parser.parse_args()

    print(f"\n📋 Loading test data from {args.test_data}")
    examples = load_test_data(args.test_data)
    texts = [ex["text"] for ex in examples]
    labels = [ex["label"] for ex in examples]
    print(f"   {len(examples)} test examples loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n🔍 Loading fine-tuned model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)

    ft_preds, ft_latency = predict_finetuned(model, tokenizer, texts, device)
    ft_acc = accuracy_score(labels, ft_preds)
    ft_f1 = f1_score(labels, ft_preds, average="macro")
    print_results("Fine-tuned DistilBERT (yours!)", labels, ft_preds, ft_latency)

    comparison_results = [("Fine-tuned DistilBERT", ft_acc, ft_f1, ft_latency)]

    # Random baseline
    random_preds = [random.choice(list(LABEL2ID.keys())) for _ in texts]
    rand_acc = accuracy_score(labels, random_preds)
    rand_f1 = f1_score(labels, random_preds, average="macro")
    comparison_results.append(("Random baseline", rand_acc, rand_f1, 0.1))

    # Groq zero-shot (optional)
    if args.groq_baseline:
        print(f"\n🤖 Running Groq zero-shot baseline ({GROQ_MODEL})")
        groq_preds, groq_latency = predict_groq_zeroshot(texts)
        groq_acc = accuracy_score(labels, groq_preds)
        groq_f1 = f1_score(labels, groq_preds, average="macro")
        print_results(f"Groq zero-shot ({GROQ_MODEL})", labels, groq_preds, groq_latency)
        comparison_results.insert(0, (f"Groq zero-shot ({GROQ_MODEL})", groq_acc, groq_f1, groq_latency))

    print_comparison_table(comparison_results)

    print("\n💡 Key Insight:")
    print("   Your fine-tuned model is specialized — trained specifically on this task.")
    print("   Zero-shot Groq is general-purpose and much larger.")
    print("   Specialization + smaller model = faster inference at comparable accuracy.")
    print("   With more training data, your model would likely close the gap entirely.")


if __name__ == "__main__":
    main()