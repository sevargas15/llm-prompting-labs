"""
Step 2: Fine-tune DistilBERT on the generated sentiment dataset.

Usage:
    python scripts/02_train.py --data data/sentiment_raw.jsonl
"""

import os
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score


LABEL2ID = {"happy": 0, "neutral": 1, "sad": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class SentimentDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 128):
        self.examples = []
        with open(path) as f:
            for line in f:
                ex = json.loads(line.strip())
                if ex.get("label") in LABEL2ID:
                    self.examples.append(ex)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(LABEL2ID[ex["label"]], dtype=torch.long),
        }


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    avg_loss = total_loss / len(dataloader)
    return avg_loss, acc, f1


def train(
    data_path: str,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    output_dir: str = "checkpoints/",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Training on: {device}")
    print(f"   Model: {model_name}")
    print(f"   Data:  {data_path}")

    # Derive split paths
    data_dir = Path(data_path).parent
    task = "sentiment"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    # Load splits (fall back to full dataset if splits not found)
    train_path = data_dir / f"{task}_train.jsonl"
    val_path = data_dir / f"{task}_val.jsonl"

    if not train_path.exists():
        print("  No split files found — using full dataset for training")
        train_path = Path(data_path)
        val_path = Path(data_path)

    train_ds = SentimentDataset(str(train_path), tokenizer)
    val_ds = SentimentDataset(str(val_path), tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)}\n")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    best_val_f1 = 0.0
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if (step + 1) % 10 == 0:
                print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)

        print(f"\n📊 Epoch {epoch}/{epochs}")
        print(f"   Train loss: {avg_train_loss:.4f}")
        print(f"   Val loss:   {val_loss:.4f}")
        print(f"   Val acc:    {val_acc:.4f}")
        print(f"   Val F1:     {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = output_path / "best_model"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"   💾 Saved best model (F1={val_f1:.4f})")

        print()

    print(f"✅ Training complete. Best val F1: {best_val_f1:.4f}")
    print(f"   Model saved to: {output_path}/best_model")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on sentiment dataset")
    parser.add_argument("--data", default="data/sentiment_raw.jsonl")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output", default="checkpoints/")
    args = parser.parse_args()

    train(
        data_path=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
