import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL    = "llama-3.1-8b-instant"


def get_groq_client():
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "\n❌ GROQ_API_KEY not set.\n"
            "   1. Get your free key at https://console.groq.com\n"
            "   2. Add it to your .env:  GROQ_API_KEY=gsk_...\n"
        )
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def call_groq(
    system_prompt: str,
    user_prompt: str,
    model: str = GROQ_MODEL,
    temperature: float = 0.9,
    max_tokens: int = 2048,
    retries: int = 3,
) -> str:
    client = get_groq_client()

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print(f"  Rate limit hit — waiting 60s...")
                time.sleep(60)
                continue
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  Retrying in {wait}s... ({e})")
                time.sleep(wait)
            else:
                raise


def call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.9,
    max_tokens: int = 2048,
) -> str:
    """Optional — only used for baseline comparisons."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set — use Groq instead (it's free).")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def parse_json_response(raw: str) -> list | dict:
    text = raw.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def save_jsonl(examples: list, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Saved {len(examples)} examples to {path}")


def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]