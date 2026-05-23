"""
Step 1: Generate a sentiment dataset using Groq.

Usage:
    python scripts/01_generate_dataset.py --task sentiment --n 200

Rate limits:
    - 30 requests/minute
    - 14,400 requests/day
    - This script batches 20 examples per request, so 200 examples = 10 requests
"""

import os
import json
import time
import argparse
import random
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# We use the openai SDK pointed at Groq's base URL
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.1-8b-instant"

# Safe batch size for free tier
BATCH_SIZE = 20

TASKS = {
    "sentiment": {
        "labels": ["happy", "neutral", "sad"],
        "system_prompt": """You are a dataset generator. Generate diverse, realistic sentences
that express a clear sentiment. Each sentence should feel natural and varied.
Cover different topics: work, relationships, daily life, hobbies, news, weather, food.
Vary intensity: mild to strong. Include some sarcasm occasionally (label by true feeling).
Respond ONLY with a valid JSON array — no explanation, no markdown, no preamble.""",
        "user_template": """Generate {n} sentences with sentiment labels.
Each item must have exactly two fields: "text" and "label".
Labels must be one of: "happy", "neutral", "sad".

Generate exactly {n} examples: {third} happy, {third} neutral, {third} sad. This distribution is strict, do not deviate from it.
Vary sentence length, topic, and writing style.

Example format:
[
  {{"text": "I finally finished my thesis after 3 years!", "label": "happy"}},
  {{"text": "The meeting was rescheduled to Thursday.", "label": "neutral"}},
  {{"text": "I missed the last train home.", "label": "sad"}}
]

Generate exactly {n} examples now. Return ONLY the JSON array:"""
    },
    "emotion_pattern": {
        "labels": ["joy", "anger", "fear", "surprise", "disgust", "neutral"],
        "system_prompt": """You are a dataset generator for emotion recognition.
Generate natural sentences that clearly express one of six basic emotions.
Respond ONLY with a valid JSON array — no explanation, no markdown.""",
        "user_template": """Generate {n} sentences with emotion labels.
Each item: {{"text": "...", "label": "joy|anger|fear|surprise|disgust|neutral"}}
Distribute evenly across all 6 labels. Vary topics and writing styles.
Return ONLY the JSON array with exactly {n} items:"""
    }
}


def get_client():
    """Create an OpenAI client pointed at Groq"""
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


def parse_json_response(raw: str) -> list:
    """Parse JSON from model response, stripping markdown fences if exist."""
    text = raw.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def generate_batch(client, task_config: dict, n: int, batch_num: int) -> list:
    """Call Groq to generate one batch of labeled examples."""
    prompt = task_config["user_template"].format(n=n, third=n//3)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": task_config["system_prompt"]},
            {"role": "user", "content": prompt}
        ],
        temperature=0.9,
        max_tokens=2048,
    )

    raw = response.choices[0].message.content.strip()
    examples = parse_json_response(raw)
    print(f"  Batch {batch_num}: got {len(examples)} examples")
    return examples


def generate_dataset(task: str, n: int, output_path: str):
    """Generate n labeled examples using Groq and save to JSONL."""
    client = get_client()
    task_config = TASKS[task]

    print(f"\nLlama Generating {n} examples with Groq ({GROQ_MODEL})")
    print(f"   Task:   {task}")
    print(f"   Labels: {task_config['labels']}")
    print(f"   Output: {output_path}")
    print(f"   Cost:   $0.00 (free tier)\n")

    all_examples = []
    batch_num = 0
    failed_batches = 0

    while len(all_examples) < n:
        remaining = n - len(all_examples)
        batch_n = min(BATCH_SIZE, remaining)
        batch_num += 1

        try:
            batch = generate_batch(client, task_config, batch_n, batch_num)

            valid = [
                ex for ex in batch
                if isinstance(ex, dict)
                and "text" in ex
                and "label" in ex
                and ex["label"] in task_config["labels"]
                and len(ex["text"].strip()) > 5
            ]

            if len(valid) < len(batch):
                print(f"  (filtered {len(batch) - len(valid)} invalid examples)")

            all_examples.extend(valid)

            if len(all_examples) < n:
                time.sleep(2)

        except json.JSONDecodeError as e:
            failed_batches += 1
            print(f"  Warning: batch {batch_num} parse error — retrying... ({e})")
            time.sleep(3)
            if failed_batches > 5:
                print("  Too many failures. Saving what we have so far.")
                break
            continue

        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 60
                print(f"  Rate limit hit — waiting {wait}s before retrying...")
                time.sleep(wait)
                continue
            raise

    random.shuffle(all_examples)
    all_examples = all_examples[:n]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    from collections import Counter
    label_counts = Counter(ex["label"] for ex in all_examples)
    print(f"\n Saved {len(all_examples)} examples to {output_path}")
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        bar = "█" * (count // 2)
        print(f"  {label:12s} {count:4d}  {bar}")

    split_dataset(all_examples, output.parent, task)


def split_dataset(examples: list, data_dir: Path, task: str, train=0.8, val=0.1, test=0.1):
    """Split into train/val/test and save as separate JSONL files."""
    n = len(examples)
    train_end = int(n * train)
    val_end = train_end + int(n * val)

    splits = {
        "train": examples[:train_end],
        "val":   examples[train_end:val_end],
        "test":  examples[val_end:]
    }

    print(f"\n Splits:")
    for split_name, split_data in splits.items():
        path = data_dir / f"{task}_{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  {split_name:8s}: {len(split_data):4d} examples  →  {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a labeled dataset using Groq (free tier)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/01_generate_dataset.py
  python scripts/01_generate_dataset.py --task sentiment --n 200
  python scripts/01_generate_dataset.py --task emotion_pattern --n 300
        """
    )
    parser.add_argument("--task", choices=list(TASKS.keys()), default="sentiment",
                        help="Task type (default: sentiment)")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of examples to generate (default: 200)")
    parser.add_argument("--output", type=str, default="data/sentiment_raw.jsonl",
                        help="Output JSONL path (default: data/sentiment_raw.jsonl)")
    args = parser.parse_args()

    generate_dataset(args.task, args.n, args.output)


if __name__ == "__main__":
    main()