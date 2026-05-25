"""
Step 3: Evaluate any system prompt against the Owlsight test.

Compare different prompt versions without
opening the notebook. Pass your prompt as a file or string.

Usage:
    python scripts/03_evaluate_prompt.py --prompt my_prompt.txt
    python scripts/03_evaluate_prompt.py --prompt "You are a phishing detector..."
    python scripts/03_evaluate_prompt.py --prompt my_prompt.txt --examples 4
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

from dotenv import load_dotenv

# Find repo root and load .env
script_dir = Path(__file__).resolve().parent
repo_root  = script_dir
while not (repo_root / ".env").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent
load_dotenv(repo_root / ".env")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL    = "llama-3.1-8b-instant"
VALID_LABELS  = ["ai_phishing", "traditional_phishing", "legitimate"]

TEST_EMAILS = [
    {"text": "Hey, quick heads up, the 3pm client call has been moved to 4pm. Same link. — Priya",
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

ZERO_SHOT_BASELINE = """You are a phishing detection system for Owlsight, a cybersecurity company.
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


def evaluate_prompt(client, system_prompt: str, label: str = "Your prompt") -> tuple:
    """Evaluate a system prompt against all test emails."""
    results = []
    print(f"\nEvaluating: {label}\n")

    for ex in TEST_EMAILS:
        pred = classify(client, ex["text"], system_prompt)
        results.append(pred)
        correct = "✅" if pred == ex["label"] else "❌"
        print(f"  {correct} [{ex['label']:20s}] → {pred}")
        time.sleep(0.5)

    acc = sum(p == e["label"] for p, e in zip(results, TEST_EMAILS)) / len(TEST_EMAILS)

    # Per-class breakdown
    per_class = {}
    for label in VALID_LABELS:
        class_emails = [(e, p) for e, p in zip(TEST_EMAILS, results) if e["label"] == label]
        if class_emails:
            correct = sum(1 for e, p in class_emails if p == e["label"])
            per_class[label] = correct / len(class_emails)

    return results, acc, per_class


def get_few_shot_examples(dataset: list, n_per_class: int) -> list:
    examples = []
    for label in VALID_LABELS:
        class_examples = [ex for ex in dataset if ex["label"] == label]
        examples.extend(random.sample(class_examples, min(n_per_class, len(class_examples))))
    return examples


def add_few_shot_examples(system_prompt: str, examples: list) -> str:
    """Append few-shot examples to an existing system prompt."""
    few_shot_block = "\n".join([
        f'Email: "{ex["text"][:200]}"\nLabel: {ex["label"]}\n'
        for ex in examples
    ])
    return f"""{system_prompt}

Here are examples of each class:

{few_shot_block}
Respond with exactly one of the three labels and nothing else."""


def print_comparison(results: dict):
    """Print a comparison table of all evaluated prompts."""
    print("\n" + "=" * 60)
    print("  PROMPT COMPARISON")
    print("=" * 60)
    print(f"  {'Prompt':<35} {'Accuracy':>10} {'vs baseline':>12}")
    print(f"  {'-'*35} {'-'*10} {'-'*12}")

    baseline_acc = results.get("Zero-shot baseline", (None, 0, {}))[1]

    for name, (_, acc, _) in results.items():
        delta = acc - baseline_acc
        delta_str = f"{'+' if delta >= 0 else ''}{delta*100:.0f}pp"
        print(f"  {name:<35} {acc*100:>9.0f}% {delta_str:>12}")

    print("=" * 60)

    # Per-class breakdown for best prompt
    best_name = max(results, key=lambda k: results[k][1])
    _, best_acc, best_per_class = results[best_name]
    print(f"\n  Per-class accuracy ({best_name}):")
    for label, acc in best_per_class.items():
        bar = "█" * int(acc * 10)
        print(f"    {label:25s} {acc*100:.0f}%  {bar}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a system prompt against the Owlsight test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a prompt from a file
  python scripts/03_evaluate_prompt.py --prompt my_prompt.txt

  # Evaluate a prompt string directly
  python scripts/03_evaluate_prompt.py --prompt "You are a security analyst..."

  # Add few-shot examples to your prompt automatically
  python scripts/03_evaluate_prompt.py --prompt my_prompt.txt --examples 2

  # Compare against the zero-shot baseline only
  python scripts/03_evaluate_prompt.py --baseline-only
        """
    )
    parser.add_argument("--prompt",       type=str, default=None,
                        help="Path to a .txt file containing your prompt, or the prompt string itself")
    parser.add_argument("--examples",     type=int, default=0,
                        help="Add n few-shot examples per class to your prompt (default: 0)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only run the zero-shot baseline, no custom prompt")
    args = parser.parse_args()

    client   = get_client()
    data_dir = Path(__file__).parent.parent / "data"
    all_results = {}

    # Always run baseline
    _, baseline_acc, baseline_per_class = evaluate_prompt(
        client, ZERO_SHOT_BASELINE, "Zero-shot baseline"
    )
    all_results["Zero-shot baseline"] = (None, baseline_acc, baseline_per_class)

    if args.baseline_only:
        print(f"\nBaseline accuracy: {baseline_acc*100:.0f}%")
        return

    # Load custom prompt
    if not args.prompt:
        print("\n⚠️  No prompt provided. Running baseline only.")
        print("   Use --prompt to pass your prompt file or string.")
        print("   Example: python scripts/03_evaluate_prompt.py --prompt my_prompt.txt\n")
        print(f"Baseline accuracy: {baseline_acc*100:.0f}%")
        return

    prompt_path = Path(args.prompt)
    if prompt_path.exists():
        with open(prompt_path) as f:
            custom_prompt = f.read().strip()
        prompt_label = prompt_path.stem
    else:
        custom_prompt = args.prompt
        prompt_label  = "Custom prompt"

    # Evaluate zero-shot version of custom prompt
    _, custom_acc, custom_per_class = evaluate_prompt(
        client, custom_prompt, f"{prompt_label} (zero-shot)"
    )
    all_results[f"{prompt_label} (zero-shot)"] = (None, custom_acc, custom_per_class)

    # Optionally add few-shot examples
    if args.examples > 0:
        dataset_path = data_dir / "phishing_raw.jsonl"
        if dataset_path.exists():
            with open(dataset_path) as f:
                dataset = [json.loads(l) for l in f if l.strip()]

            few_shot_examples = get_few_shot_examples(dataset, args.examples)
            few_shot_prompt   = add_few_shot_examples(custom_prompt, few_shot_examples)

            _, fs_acc, fs_per_class = evaluate_prompt(
                client, few_shot_prompt, f"{prompt_label} (few-shot {args.examples}/class)"
            )
            all_results[f"{prompt_label} (few-shot {args.examples}/class)"] = (None, fs_acc, fs_per_class)
        else:
            print(f"⚠️  No dataset found at {dataset_path}")
            print("   Run scripts/01_generate_dataset.py first to use few-shot examples.")

    print_comparison(all_results)


if __name__ == "__main__":
    main()
