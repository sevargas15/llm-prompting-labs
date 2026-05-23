"""
Step 1: Generate a sentiment dataset using GPT.

Usage:
    python scripts/01_generate_dataset.py --task sentiment --n 200
"""

import os
import json
import argparse
import random
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


TASKS = {
    "sentiment": {
        "labels": ["happy", "neutral", "sad"],
        "description": "sentiment (happy, neutral, or sad)",
        "system_prompt": """You are a dataset generator. Generate diverse, realistic sentences
that express a clear sentiment. Each sentence should feel natural and varied.
Cover different topics: work, relationships, daily life, hobbies, news, weather, food.
Vary intensity: mild to strong. Include some sarcasm occasionally (label by true feeling).
Respond ONLY with a JSON array — no preamble, no markdown.""",
        "user_template": """Generate {n} sentences with sentiment labels.
Each item must have exactly two fields: "text" and "label".
Labels must be one of: "happy", "neutral", "sad".

Aim for roughly equal distribution. Include variety in:
- Sentence length (short tweets to multi-sentence thoughts)
- Topic and context
- Writing style (formal, casual, first person, third person)

Example format:
[
  {{"text": "I finally finished my thesis after 3 years!", "label": "happy"}},
  {{"text": "The meeting was rescheduled to Thursday.", "label": "neutral"}},
  {{"text": "I missed the last train home.", "label": "sad"}}
]

Generate {n} examples now:"""
    },
    "emotion_pattern": {
        "labels": ["joy", "anger", "fear", "surprise", "disgust", "neutral"],
        "description": "emotion (joy, anger, fear, surprise, disgust, neutral)",
        "system_prompt": """You are a dataset generator for emotion recognition.
Generate natural sentences that clearly express one of six emotions.
Respond ONLY with a JSON array.""",
        "user_template": """Generate {n} sentences with emotion labels.
Each item: {{"text": "...", "label": "joy|anger|fear|surprise|disgust|neutral"}}
Equal distribution across all 6 labels. Vary topics and styles.
Generate {n} examples:"""
    }
}

BATCH_SIZE = 20  # Generate in batches to stay under token limits


def generate_batch(client, task_config, n, batch_num):
    """Call GPT to generate a batch of labeled examples."""
    prompt = task_config["user_template"].format(n=n)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": task_config["system_prompt"]},
            {"role": "user", "content": prompt}
        ],
        temperature=0.9,  # High temp for diversity
        max_tokens=3000,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    examples = json.loads(raw)
    print(f"  Batch {batch_num}: generated {len(examples)} examples")
    return examples


def generate_dataset(task: str, n: int, output_path: str):
    """Generate n labeled examples and save to JSONL."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in .env")

    client = OpenAI(api_key=api_key)
    task_config = TASKS[task]

    print(f"\n🤖 Generating {n} examples for task: {task}")
    print(f"   Labels: {task_config['labels']}")
    print(f"   Output: {output_path}\n")

    all_examples = []
    batch_num = 0

    while len(all_examples) < n:
        remaining = n - len(all_examples)
        batch_n = min(BATCH_SIZE, remaining)
        batch_num += 1

        try:
            batch = generate_batch(client, task_config, batch_n, batch_num)
            # Validate labels
            valid = [
                ex for ex in batch
                if isinstance(ex, dict)
                and "text" in ex
                and "label" in ex
                and ex["label"] in task_config["labels"]
            ]
            all_examples.extend(valid)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: batch {batch_num} parse error ({e}), retrying...")
            continue

    # Shuffle before saving
    random.shuffle(all_examples)
    all_examples = all_examples[:n]

    # Save as JSONL
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    # Print stats
    from collections import Counter
    label_counts = Counter(ex["label"] for ex in all_examples)
    print(f"\n✅ Saved {len(all_examples)} examples to {output_path}")
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        bar = "█" * (count // 2)
        print(f"  {label:10s} {count:4d} {bar}")

    # Create train/val/test splits
    split_dataset(all_examples, output.parent)


def split_dataset(examples, data_dir, train=0.8, val=0.1, test=0.1):
    """Split dataset into train/val/test and save."""
    n = len(examples)
    train_end = int(n * train)
    val_end = train_end + int(n * val)

    splits = {
        "train": examples[:train_end],
        "val": examples[train_end:val_end],
        "test": examples[val_end:]
    }

    for split_name, split_data in splits.items():
        path = data_dir / f"sentiment_{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex) + "\n")

    print(f"\n📂 Splits saved:")
    for split_name, split_data in splits.items():
        print(f"  {split_name:8s}: {len(split_data)} examples")


def main():
    parser = argparse.ArgumentParser(description="Generate a labeled dataset using GPT")
    parser.add_argument("--task", choices=list(TASKS.keys()), default="sentiment",
                        help="Task type to generate data for")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of examples to generate")
    parser.add_argument("--output", type=str, default="data/sentiment_raw.jsonl",
                        help="Output path for the JSONL dataset")
    args = parser.parse_args()

    generate_dataset(args.task, args.n, args.output)


if __name__ == "__main__":
    main()
