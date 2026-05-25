"""
Step 2: Run zero-shot and few-shot detection on a set of emails.

Classifies emails using Groq with zero-shot and few-shot prompts
and prints a side-by-side comparison of results.

Usage:
    python scripts/02_run_detection.py
    python scripts/02_run_detection.py -- 4 examples
    python scripts/02_run_detection.py -- input data/my_emails.jsonl
"""

import os
import json
import time
import random
import argparse
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv

# Find repo root and load .env
script_dir = Path(__file__).resolve().parent
repo_root  = script_dir
while not (repo_root / ".env").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent
load_dotenv(repo_root / ".env")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL    = "llama-3.1-8b-instant"

VALID_LABELS = ["ai_phishing", "traditional_phishing", "legitimate"]

# The 9 hand-crafted Owlsight test emails
TEST_EMAILS = [
    {"text": "Hey, quick heads up, the 3pm client call has been moved to 4pm. Same link. Priya",
     "label": "legitimate"},
    {"text": "Team, we're enabling mandatory MFA for all GitHub repos starting Monday. IT-Ops will send setup instructions to your work email.",
     "label": "legitimate"},
    {"text": "Hi everyone, excited to announce we've officially signed our third enterprise client this quarter. Details in the all-hands Friday. — Daniel",
     "label": "legitimate"},
    {"text": "YOUR ACCOUNT HAS BEEN SUSPENDED. Click here immediately to verify your identity and restore access: secure-paypa1.com/verify",
     "label": "traditional_phishing"},
    {"text": "Congratulations! You have been selected to receive a $500 Amazon gift card. Claim your reward now before it expires.",
     "label": "traditional_phishing"},
    {"text": "Dear Customer, your Microsoft account password will expire in 24 hours. Update it now to avoid losing access to your files.",
     "label": "traditional_phishing"},
    {"text": "Hey, it's Daniel. I'm in back-to-back calls with the Meridian team but need you to review the updated NDA before 5pm. Shared it here: docs.owlsight-secure.net/nda — please don't forward, it's confidential. Thanks",
     "label": "ai_phishing"},
    {"text": "Hi, this is Priya from People & Culture. As part of our Series A onboarding updates, we need all employees to re-verify their benefits enrollment by EOD. Use the HR portal: owlsight-hr.people-portal.com",
     "label": "ai_phishing"},
    {"text": "Security alert: we detected a login to your Owlsight GitHub account from an unrecognized device in Amsterdam. If this wasn't you, revoke access immediately: github-owlsight.security-verify.com/revoke",
     "label": "ai_phishing"},
]

ZERO_SHOT_SYSTEM = """You are a phishing detection system for Owlsight, a cybersecurity company.
Classify emails into exactly one of three categories:
- legitimate: real internal business email
- traditional_phishing: generic phishing with obvious red flags
- ai_phishing: sophisticated, personalized phishing that is hard to spot

Respond with exactly one of these three labels and nothing else."""


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


def classify(client, email_text: str, system_prompt: str) -> str:
    """Classify a single email using the given system prompt."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Classify this email:\n\n{email_text}"}
        ],
        temperature=0,
        max_tokens=10,
    )
    pred = response.choices[0].message.content.strip().lower()
    for label in VALID_LABELS:
        if label in pred:
            return label
    return "legitimate"


def build_few_shot_system(examples: list) -> str:
    """Build a few-shot system prompt from a list of labeled examples."""
    few_shot_block = "\n".join([
        f'Email: "{ex["text"][:200]}"\nLabel: {ex["label"]}\n'
        for ex in examples
    ])
    return f"""{ZERO_SHOT_SYSTEM}

Here are examples of each class:

{few_shot_block}
Respond with exactly one of the three labels and nothing else."""


def load_dataset(path: Path) -> list:
    """Load examples from a JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def get_few_shot_examples(dataset: list, n_per_class: int) -> list:
    """Pick n examples per class for use as few-shot anchors."""
    examples = []
    for label in VALID_LABELS:
        class_examples = [ex for ex in dataset if ex["label"] == label]
        examples.extend(random.sample(class_examples, min(n_per_class, len(class_examples))))
    return examples


def run_detection(client, emails: list, system_prompt: str, label: str) -> tuple:
    """Run classification on a list of emails and return results + accuracy."""
    results = []
    print(f"\nRunning {label}...\n")

    for ex in emails:
        pred = classify(client, ex["text"], system_prompt)
        results.append(pred)
        time.sleep(0.5)

    acc = sum(p == e["label"] for p, e in zip(results, emails)) / len(emails)
    return results, acc


def print_comparison(emails: list, zero_shot: list, few_shot: list,
                     zero_acc: float, few_acc: float):
    """Print a readable side-by-side comparison."""
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    for i, ex in enumerate(emails):
        zs = "✅" if zero_shot[i] == ex["label"] else "❌"
        fs = "✅" if few_shot[i]  == ex["label"] else "❌"
        print(f"\n#{i+1} [{ex['label']}]")
        print(f"   Email:      {ex['text']}")
        print(f"   Zero-shot:  {zs} {zero_shot[i]}")
        print(f"   Few-shot:   {fs} {few_shot[i]}")

    print("\n" + "=" * 50)
    print("  FINAL COMPARISON")
    print("=" * 50)
    print(f"  {'Zero-shot':<30} {zero_acc*100:.0f}%")
    print(f"  {'Few-shot':<30} {few_acc*100:.0f}%")
    print(f"  {'Improvement':<30} {(few_acc - zero_acc)*100:+.0f} percentage points")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Run zero-shot and few-shot phishing detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/02_run_detection.py
  python scripts/02_run_detection.py --examples 4
  python scripts/02_run_detection.py --input data/my_emails.jsonl --examples 2
        """
    )
    parser.add_argument("--examples", type=int, default=2,
                        help="Few-shot examples per class (default: 2)")
    parser.add_argument("--input",    type=str, default=None,
                        help="Custom JSONL file of emails to classify (optional)")
    args = parser.parse_args()

    client   = get_client()
    data_dir = Path(__file__).parent.parent / "data"

    # Load emails to classify
    if args.input:
        emails = load_dataset(Path(args.input))
        print(f"Loaded {len(emails)} emails from {args.input}")
    else:
        emails = TEST_EMAILS
        print(f"Using {len(emails)} built-in Owlsight test emails")

    # Load dataset for few-shot examples
    dataset_path = data_dir / "phishing_raw.jsonl"
    if dataset_path.exists():
        dataset = load_dataset(dataset_path)
        few_shot_examples = get_few_shot_examples(dataset, args.examples)
        print(f"Using {len(few_shot_examples)} few-shot examples ({args.examples} per class)")
    else:
        print(f"⚠️  No dataset found at {dataset_path}")
        print("   Run scripts/01_generate_dataset.py first to generate it.")
        print("   Running zero-shot only.\n")
        few_shot_examples = []

    # Zero-shot
    zero_results, zero_acc = run_detection(client, emails, ZERO_SHOT_SYSTEM, "zero-shot")

    # Few-shot (if we have examples)
    if few_shot_examples:
        few_shot_system = build_few_shot_system(few_shot_examples)
        few_results, few_acc = run_detection(client, emails, few_shot_system, "few-shot")
    else:
        few_results = zero_results
        few_acc = zero_acc

    print_comparison(emails, zero_results, few_results, zero_acc, few_acc)


if __name__ == "__main__":
    main()
