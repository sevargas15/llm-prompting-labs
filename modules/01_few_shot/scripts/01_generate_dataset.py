"""
Step 1: Generate AI phishing examples using Groq (free).

This script generates Owlsight AI phishing emails and combines
them with real phishing/legitimate emails from the Kaggle dataset.

Usage:
    python scripts/01_generate_dataset.py
    python scripts/01_generate_dataset.py --n 200 --kaggle-path data/phishing_email.csv

Requirements:
    - GROQ_API_KEY in your .env file
    - phishing_email.csv downloaded from:
      https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv

# Find repo root and load .env
script_dir = Path(__file__).resolve().parent
repo_root = script_dir
while not (repo_root / ".env").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent
load_dotenv(repo_root / ".env")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL    = "llama-3.1-8b-instant"
BATCH_SIZE    = 20

OWLSIGHT_CONTEXT = """
Company: Owlsight, a 120 person cybersecurity SaaS company
Product: Threat monitoring and anomaly detection for mid-market enterprises
Email domain: @owlsight.com
People:
  - Daniel Reyes (CEO, daniel.reyes@owlsight.com)
  - Priya Nair (Head of People & Culture, priya.nair@owlsight.com)
  - IT & Security Ops (it-ops@owlsight.com)
Tools used: Google Workspace, Slack, GitHub, AWS, Stripe, DocuSign
Current events: Recently closed Series A, onboarding 3 new enterprise clients,
                hiring aggressively across engineering and sales
"""

AI_PHISHING_SYSTEM = """You are generating realistic examples of AI-generated spear-phishing emails
for a cybersecurity training dataset. These represent the new wave of sophisticated phishing:
personalized, well-written, context-aware attacks that reference specific company details,
real people's names, current events, and legitimate business contexts.
Respond ONLY with a valid JSON array, no explanation, no markdown, no preamble."""

AI_PHISHING_USER = """Generate {n} AI-generated spear-phishing email examples targeting Owlsight employees.

Company context:
{context}

These emails should be sophisticated and hard to spot:
- Impersonate Daniel Reyes (CEO), Priya Nair (HR), IT Security Ops, or trusted vendors
- Reference real Owlsight context (Series A, new clients, tools they use)
- Use subtle urgency without obvious red flags
- Include plausible but fake domains (owlsight-corp.com, owlsight.security-portal.com)
- Write in a natural, professional tone, no spelling errors, no generic language

Each item must have exactly two fields: "text" and "label".
Label must be: "ai_phishing"

Generate exactly {n} examples. Keep each email under 100 words. Return ONLY the JSON array:"""


def get_client():
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
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def parse_json(raw: str) -> list:
    """Parse JSON response, handling markdown fences and truncation."""
    import re
    text = raw.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text.strip())


def generate_ai_phishing(client, n: int) -> list:
    """Generate n AI phishing examples using Groq."""
    all_examples = []
    batch_num = 0
    failures  = 0

    print(f"Generating {n} AI phishing examples...\n")

    while len(all_examples) < n:
        remaining  = n - len(all_examples)
        this_batch = min(BATCH_SIZE, remaining)
        batch_num += 1

        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": AI_PHISHING_SYSTEM},
                    {"role": "user",   "content": AI_PHISHING_USER.format(
                        n=this_batch, context=OWLSIGHT_CONTEXT
                    )}
                ],
                temperature=0.9,
                max_tokens=2048,
            )

            raw   = response.choices[0].message.content.strip()
            batch = parse_json(raw)

            valid = [
                {"text": ex["text"].strip(), "label": "ai_phishing", "source": "generated"}
                for ex in batch
                if isinstance(ex, dict) and "text" in ex and len(ex["text"].strip()) > 10
            ]

            all_examples.extend(valid)
            print(f"  Batch {batch_num}: {len(valid)} valid (total: {len(all_examples)}/{n})")

            if len(all_examples) < n:
                time.sleep(2)

        except json.JSONDecodeError:
            failures += 1
            print(f"  Parse error on batch {batch_num}, retrying...")
            time.sleep(3)
            if failures > 5:
                print("  Too many failures, saving what we have.")
                break

        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print("  Rate limit hit: waiting 60s...")
                time.sleep(60)
            else:
                raise

    return all_examples[:n]


def load_kaggle_data(csv_path: str, samples_per_class: int = 100) -> list:
    """Load and clean real phishing/legitimate emails from Kaggle CSV."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Run: pip install pandas")

    print(f"Loading Kaggle data from {csv_path}...")
    df = pd.read_csv(csv_path)

    label_map = {1: "traditional_phishing", 0: "legitimate"}
    df["label_clean"] = df["label"].map(label_map)

    real_data = []
    for label in ["legitimate", "traditional_phishing"]:
        subset  = df[df["label_clean"] == label].dropna(subset=["text_combined"])
        subset  = subset[subset["text_combined"].str.len() > 20]
        n       = min(samples_per_class, len(subset))
        sampled = subset.sample(n, random_state=42)

        for _, row in sampled.iterrows():
            text = " ".join(str(row["text_combined"]).split())[:1000]
            if len(text) > 20:
                ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
                if ascii_ratio >= 0.85:
                    real_data.append({"text": text, "label": label, "source": "kaggle"})

    print(f"  Loaded {len(real_data)} real examples")
    return real_data


def save_jsonl(examples: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Saved {len(examples):4d} examples → {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Module 1 phishing dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/01_generate_dataset.py
  python scripts/01_generate_dataset.py --n 200
  python scripts/01_generate_dataset.py --kaggle-path data/phishing_email.csv --n 150

Get your free Groq key at: https://console.groq.com
Download Kaggle data at:   https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
        """
    )
    parser.add_argument("--n",           type=int, default=100,
                        help="AI phishing examples to generate (default: 100)")
    parser.add_argument("--samples",     type=int, default=100,
                        help="Real examples per class from Kaggle (default: 100)")
    parser.add_argument("--kaggle-path", type=str, default="data/phishing_email.csv",
                        help="Path to phishing_email.csv from Kaggle")
    args = parser.parse_args()

    data_dir    = Path(__file__).parent.parent / "data"
    kaggle_path = Path(__file__).parent.parent / args.kaggle_path

    client = get_client()

    # Load real data
    if kaggle_path.exists():
        real_data = load_kaggle_data(str(kaggle_path), args.samples)
    else:
        print(f"⚠️  Kaggle file not found at {kaggle_path}")
        print("   Running with generated data only.")
        print("   Download from: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset\n")
        real_data = []

    # Generate AI phishing
    ai_phishing = generate_ai_phishing(client, args.n)

    # Combine and shuffle
    dataset = real_data + ai_phishing
    random.shuffle(dataset)

    # Print distribution
    label_counts = Counter(ex["label"] for ex in dataset)
    print(f"\n✅ Full dataset: {len(dataset)} examples")
    for label, count in label_counts.items():
        bar = "█" * (count // 3)
        print(f"  {label:25s} {count:4d}  {bar}")

    # Save the results
    print("\nSaving dataset:")
    save_jsonl(dataset, data_dir / "phishing_raw.jsonl")
    print(f"\n✅ Done. Run scripts/02_run_detection.py next.")


if __name__ == "__main__":
    main()
