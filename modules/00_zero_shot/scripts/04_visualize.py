"""
Step 4: Visualize learned representations using t-SNE.

Usage:
    python scripts/04_visualize.py --model checkpoints/best_model --data data/sentiment_test.jsonl
"""

import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.manifold import TSNE

LABEL2ID = {"happy": 0, "neutral": 1, "sad": 2}
COLORS = {"happy": "#2ecc71", "neutral": "#95a5a6", "sad": "#e74c3c"}
MARKERS = {"happy": "o", "neutral": "s", "sad": "^"}


def get_embeddings(model, tokenizer, texts, device, batch_size=32):
    """Extract CLS token embeddings (the sentence representation)."""
    model.eval()
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encodings = tokenizer(batch, max_length=128, padding=True,
                              truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            # Get hidden states from the base model
            outputs = model.distilbert(**encodings) if hasattr(model, 'distilbert') else model.base_model(**encodings)
            # CLS token representation
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)

    return np.vstack(embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/best_model")
    parser.add_argument("--data", default="data/sentiment_test.jsonl")
    parser.add_argument("--output", default="figures/tsne_representations.png")
    args = parser.parse_args()

    examples = []
    with open(args.data) as f:
        for line in f:
            ex = json.loads(line.strip())
            if ex.get("label") in LABEL2ID:
                examples.append(ex)

    texts = [ex["text"] for ex in examples]
    labels = [ex["label"] for ex in examples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)

    print(f"Extracting embeddings for {len(texts)} examples...")
    embeddings = get_embeddings(model, tokenizer, texts, device)

    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(texts)//4))
    reduced = tsne.fit_transform(embeddings)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Sentiment Representations: Your Fine-Tuned Model", fontsize=14, fontweight="bold")

    # Left: colored by true label
    ax = axes[0]
    ax.set_title("True Labels", fontsize=12)
    for label in LABEL2ID:
        mask = np.array(labels) == label
        ax.scatter(
            reduced[mask, 0], reduced[mask, 1],
            c=COLORS[label], marker=MARKERS[label],
            label=label, alpha=0.7, s=60, edgecolors="white", linewidths=0.5
        )
    ax.legend(loc="best", fontsize=10)
    ax.set_xlabel("t-SNE 1"), ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.3)

    # Right: show example texts
    ax2 = axes[1]
    ax2.set_title("Cluster Examples", fontsize=12)
    for label in LABEL2ID:
        mask = np.array(labels) == label
        ax2.scatter(
            reduced[mask, 0], reduced[mask, 1],
            c=COLORS[label], marker=MARKERS[label],
            alpha=0.3, s=40
        )

    # Annotate a few examples per cluster
    for label in LABEL2ID:
        indices = [i for i, l in enumerate(labels) if l == label][:3]
        for idx in indices:
            text = examples[idx]["text"][:30] + "..." if len(examples[idx]["text"]) > 30 else examples[idx]["text"]
            ax2.annotate(text, (reduced[idx, 0], reduced[idx, 1]),
                         fontsize=7, ha="center", alpha=0.8,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor=COLORS[label]))

    ax2.set_xlabel("t-SNE 1"), ax2.set_ylabel("t-SNE 2")
    ax2.grid(True, alpha=0.3)

    legend_patches = [mpatches.Patch(color=COLORS[l], label=l) for l in LABEL2ID]
    ax2.legend(handles=legend_patches, loc="best", fontsize=10)

    plt.tight_layout()

    import os
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\n✅ Saved t-SNE plot to {args.output}")
    print("\n💡 What you're seeing:")
    print("   Each point = one sentence's internal representation")
    print("   Nearby points = sentences the model thinks are similar")
    print("   Distinct clusters = the model has learned to separate sentiments")


if __name__ == "__main__":
    main()
