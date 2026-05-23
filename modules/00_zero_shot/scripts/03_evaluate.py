"""
Step 3: Evaluate your fine-tuned model and compare to GPT zero-shot.

Usage:
    python scripts/03_evaluate.py --model checkpoints/best_model --test-data data/sentiment_test.jsonl
    python scripts/03_evaluate.py --model checkpoints/best_model --test-data data/sentiment_test.jsonl --gpt-baseline
"""

import os
import json
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from dotenv import load_dotenv

load_dotenv()

LABEL2ID = {"happy": 0, "neutral": 1, "sad": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


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


def predict_gpt_zeroshot(texts, model="gpt-3.5-turbo"):
    """Zero-shot predict with GPT (single calls, for accurate latency)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    preds = []
    latencies = []

    print(f"  Running GPT zero-shot on {len(texts)} examples...")
    for i, text in enumerate(texts):
        start = time.time()
        response = client.chat.completions.create(
            model=model,
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
            pred = "neutral"  # fallback
        preds.append(pred)
        latencies.append(elapsed)

        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(texts)} done...")

    return preds, np.mean(latencies)


def print_results(name, labels, preds, avg_latency_ms):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Macro F1:  {f1:.4f}")
    print(f"  Avg latency: {avg_latency_ms:.1f} ms/example")
    print(f"\n  Per-class report:")
    print(classification_report(labels, preds, target_names=list(LABEL2ID.keys()), indent=4))


def print_comparison_table(results):
    print("\n" + "="*60)
    print("  COMPARISON SUMMARY")
    print("="*60)
    print(f"  {'Model':<30} {'Accuracy':>10} {'F1':>8} {'Latency':>12}")
    print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*12}")
    for name, acc, f1, latency in results:
        print(f"  {name:<30} {acc*100:>9.1f}% {f1:>8.3f} {latency:>10.1f}ms")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/best_model")
    parser.add_argument("--test-data", default="data/sentiment_test.jsonl")
    parser.add_argument("--gpt-baseline", action="store_true",
                        help="Also run GPT zero-shot baseline (costs API credits)")
    parser.add_argument("--gpt-model", default="gpt-3.5-turbo")
    args = parser.parse_args()

    print(f"\n📋 Loading test data from {args.test_data}")
    examples = load_test_data(args.test_data)
    texts = [ex["text"] for ex in examples]
    labels = [ex["label"] for ex in examples]
    print(f"   {len(examples)} test examples loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate fine-tuned model
    print(f"\n🔍 Loading fine-tuned model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)

    ft_preds, ft_latency = predict_finetuned(model, tokenizer, texts, device)
    ft_acc = accuracy_score(labels, ft_preds)
    ft_f1 = f1_score(labels, ft_preds, average="macro")
    print_results("Fine-tuned DistilBERT (yours!)", labels, ft_preds, ft_latency)

    comparison_results = [("Fine-tuned DistilBERT", ft_acc, ft_f1, ft_latency)]

    # Random baseline
    import random
    random_preds = [random.choice(list(LABEL2ID.keys())) for _ in texts]
    rand_acc = accuracy_score(labels, random_preds)
    rand_f1 = f1_score(labels, random_preds, average="macro")
    comparison_results.append(("Random baseline", rand_acc, rand_f1, 0.1))

    # GPT zero-shot (optional)
    if args.gpt_baseline:
        print(f"\n🤖 Running GPT zero-shot baseline ({args.gpt_model})")
        print("   (This will make API calls and cost ~$0.01)")
        gpt_preds, gpt_latency = predict_gpt_zeroshot(texts, args.gpt_model)
        gpt_acc = accuracy_score(labels, gpt_preds)
        gpt_f1 = f1_score(labels, gpt_preds, average="macro")
        print_results(f"GPT zero-shot ({args.gpt_model})", labels, gpt_preds, gpt_latency)
        comparison_results.insert(0, (f"GPT zero-shot ({args.gpt_model})", gpt_acc, gpt_f1, gpt_latency))

    print_comparison_table(comparison_results)

    # Key insight
    print("\n💡 Key Insight:")
    print("   Your fine-tuned model is specialized — it's been trained specifically")
    print("   on this task's data distribution. Zero-shot GPT is general-purpose.")
    print("   Specialization + smaller model = faster inference at comparable accuracy.")


if __name__ == "__main__":
    main()
