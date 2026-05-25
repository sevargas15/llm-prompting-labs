# Module 2 | Chain-of-Thought Prompting

---

## What is chain-of-thought prompting?

Instead of asking a model to to an answer, you ask it to work through the problem first. The reasoning becomes part of the output , and that changes everything.

```
Without CoT:
  Prompt: "Is this request malicious?"
  Answer: "malicious"

With CoT:
  Prompt: "Analyze this request step by step."
  Answer:
    Step 1 - Headers: User-Agent is python-requests/2.28, not a real browser.
    Step 2 - Body: referral_code contains ' OR 1=1--, a SQL tautology.
    Step 3 - Match: WAF-01 (SQL Injection). High confidence.
    Step 4 - Verdict: malicious. Action: BLOCK.
```

Same model. Same input. The only difference is whether you let it think out loud before answering.

**That's what this module is about, and why it matters in security.**

---

## What you'll build

```
data/
  waf_requests.jsonl       ← 45 labeled HTTP requests generated across 15 WAF rules
  triage_results.jsonl     ← Full CoT triage output with reasoning chains

notebooks/
  02_chain_of_thought_prompting.ipynb    ← Full guided walkthrough

figures/
  cot_comparison.png       ← Accuracy chart: baseline vs zero-shot CoT vs guided CoT
```

---

## How to run it

The notebook is the recommended path. Everything is explained in order, the theory, the playbook, the data, the 3 prompting approaches, the failure modes, and the lab.

```bash
jupyter notebook notebooks/02_chain_of_thought_prompting.ipynb
```

This notebook is self-contained. No external datasets to download. Data is generated with Groq live.

---

## What's inside the notebook

### The scenario
A security team has a WAF playbook, 15 rules covering the most common attack patterns hitting a signup endpoint. Using it manually is slow. The goal is to train a model to reason through incoming HTTP requests the same way an analyst would, using the playbook as its reference.

### The playbook
15 rules across 5 families, built from real documented attack patterns (OWASP CRS, PortSwigger Web Security Academy, AWS/Azure/Cloudflare/Fortinet WAF documentation):

| Family | Rules | Covers |
|---|---|---|
| Injection Attacks | WAF-01 to WAF-04 | SQL, XSS, command injection, NoSQL |
| Field & Input Abuse | WAF-05 to WAF-07 | Email manipulation, malformed input, SSRF |
| Automation & Bot Signals | WAF-08 to WAF-10 | Bot fingerprints, rate limit evasion, headless browsers |
| Evasion Techniques | WAF-11 to WAF-13 | Encoding bypass, obfuscation, header spoofing |
| Account Abuse | WAF-14 to WAF-15 | Enumeration, credential stuffing |

### The 3 approaches run on the same dataset

**Baseline, direct classification**
Model reads the playbook and answers directly. No reasoning steps.

**Zero-shot CoT**
One sentence added: "think step by step before classifying." Same model, no other changes. Watch what happens to output quality.

**Guided CoT**
You define the exact reasoning framework: 5 steps the model must follow before concluding. Consistent, auditable, significantly more accurate on the harder scenarios.

### Failure modes
3 dedicated sections covering what breaks CoT in practice:
- Premise poisoning: a wrong assumption in step 1 corrupts every step that follows
- Hallucinated context: the model invents threat intelligence that isn't in the input
- Verbose but shallow: a long chain that restates the input without advancing the reasoning

### Lab logic puzzles
After the WAF work, a stripped-back exercise. 3 logic puzzles of increasing difficulty, run through all three approaches. No domain knowledge required. The point is to see the technique itself, isolated from the security context.

### Production pattern
A `batch_triage()` function that processes a full request queue, returns structured JSON output, and preserves the reasoning chain as an audit trail for analyst review.

---

## Key concepts learned from this one

- **CoT works because output tokens are working memory:** intermediate steps give the model better material to build each next token from
- **Zero-shot CoT is one sentence:** "think step by step" costs nothing and often meaningfully improves accuracy
- **Guided CoT is what you actually ship:** defining the steps makes output consistent and auditable, not just more accurate
- **The reasoning chain is an audit artifact:** in a security context, being able to show *why* a decision was made matters as much as the decision itself
- **CoT has failure modes worth knowing:** premise poisoning, hallucinated context, and verbose-but-shallow chains are real and specific
- **Predictable beats clever:** a framework the model can't skip is more deployable than one that sometimes works brilliantly

---

## Next module

**[Module 3 | Meta Prompting →](../../03_meta_prompting/README.md)**
