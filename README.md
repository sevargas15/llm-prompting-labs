# LLM Prompting Labs

I build this repo with the main idea of not only teaching myself but helping others to actually understand the different prompting techniques and how to build, train and use your own models from scratch. For each technique, you will:

1. **Generate a meaningful dataset**
2. **Fine-tune your own small models** on that dataset
3. **Evaluate** your model against baselines
4. **Understand** *why* the technique works from the inside

---

## Prompting Techniques

| # | Technique | Status | Module |
|---|-----------|--------|--------|
| 0 | [Zero-Shot Prompting](modules/00_zero_shot/) | ✅ Active | `modules/00_zero_shot` |
| 1 | [Few-Shot Prompting](modules/01_few_shot/) | 🔜 Soon | `modules/01_few_shot` |
| 2 | [Chain-of-Thought](modules/02_chain_of_thought/) | 🔜 Soon | `modules/02_chain_of_thought` |
| 3 | [Meta Prompting](modules/03_meta_prompting/) | 🔜 Soon | `modules/03_meta_prompting` |
| 4 | Self-Consistency | 🔜 Soon | — |
| 5 | Generate Knowledge Prompting | 🔜 Soon | — |
| 6 | Prompt Chaining | 🔜 Soon | — |
| 7 | Tree of Thoughts | 🔜 Soon | — |
| 8 | Retrieval Augmented Generation | 🔜 Soon | — |
| 9 | Automatic Reasoning & Tool Use | 🔜 Soon | — |
| 10 | Automatic Prompt Engineer | 🔜 Soon | — |
| 11 | Active-Prompt | 🔜 Soon | — |
| 12 | Directional Stimulus Prompting | 🔜 Soon | — |
| 13 | Program-Aided Language Models | 🔜 Soon | — |
| 14 | ReAct | 🔜 Soon | — |
| 15 | Reflexion | 🔜 Soon | — |
| 16 | Multimodal CoT | 🔜 Soon | — |
| 17 | Graph Prompting | 🔜 Soon | — |

---

## Where to start?

### 1. First, clone and install this repo in your local environment

```bash
git clone https://github.com/YOUR_USERNAME/llm-prompting-labs.git
cd llm-prompting-labs
pip install -r requirements.txt
```

### 2. Set your API key

To keep these labs accessible to everyone, we'll use Groq as the default LLM provider. It's free, requires no credit card, and gives you API access to Llama 3.1, so let's say you get the real developer workflow without spending a cent.

```bash
cp .env.example .env
# Add your Groq key to .env
```

#### Getting your free Groq API key

1. Go to [console.groq.com](https://console.groq.com) and sign up
2. Click **API Keys** in the left sidebar → **Create API Key**
3. Copy the key and paste it in your `.env` file:
   ```
   GROQ_API_KEY=gsk_...your key here...
   ```

> **Free tier limitations:** 30 requests/min, 14,400 requests/day. Each module generates ~200 examples in ~10 requests, so you'll never come close to the limit in a single session/module. If you do hit it, the script will wait automatically and resume, this with the intention of not losing any data in the process.

> Never commit your `.env` file. The `.gitignore` already excludes it, but always double-check with `git status` before pushing.


### 3. Start with Module 0 - Zero-Shot

```bash
cd modules/00_zero_shot

# Step 1: Generate a dataset using Groq
python scripts/01_generate_dataset.py --task sentiment --n 200

# Step 2: Fine-tune a DistilBERT classifier
python scripts/02_train.py --data data/sentiment_raw.jsonl

# Step 3: Evaluate and compare to zero-shot baseline
python scripts/03_evaluate.py --model checkpoints/best_model
```

Or open the notebook for a guided walkthrough:

```bash
jupyter notebook notebooks/00_zero_shot_walkthrough.ipynb
```

---

## Repo Structure

```
llm-prompting-lab/
├── modules/
│   ├── 00_zero_shot/           ← Start here
│   │   ├── notebooks/          ← Interactive Jupyter walkthrough
│   │   ├── scripts/            ← CLI scripts for each step
│   │   └── data/               ← Your new datasets should land here
│   ├── 01_few_shot/
│   ├── 02_chain_of_thought/
│   └── ...
├── shared/
│   ├── utils/                  ← Shared helpers (API calls, tokenization, eval)
│   └── datasets/               ← Cross-module benchmark datasets
├── docs/
│   └── techniques/             ← Deep-dive writeups for each technique
├── requirements.txt
└── .env.example
```

---

## What you'll actually build

### Module 0 - Zero-Shot Prompting

**The idea:** A model trained on billions of tokens already knows how to classify sentiment, recognize patterns, and answer questions — without any task-specific examples.

**What you build:**
- Use Groq to generate a labeled sentiment dataset (200+ samples)
- Fine-tune `distilbert-base-uncased` as a 3-class classifier (happy / neutral / sad)
- Compare accuracy: your fine-tuned model vs Groq zero-shot baseline
- Visualize the learned representations with t-SNE

**Key insight:** Zero-shot works because pretraining creates general-purpose representations. Fine-tuning on generated data teaches you _how much_ task-specific data actually matters.

---

## Tech Stack Used

| Purpose | Tool | Cost |
|---------|------|------|
| Dataset generation | Groq API (llama-3.1-8b) | **Free** |
| Model fine-tuning | `transformers` (Hugging Face) | Free |
| Training compute | Google Colab / Kaggle Notebooks | **Free** |
| Training loop | `PyTorch` / `accelerate` | Free |
| Evaluation | `evaluate` (HF), `scikit-learn` | Free |
| Notebooks | `jupyter`, `matplotlib`, `seaborn` | Free |
| Model hosting | Hugging Face Hub | **Free** |
| Vector search (RAG module) | `faiss-cpu` | Free |

---

## How to Contribute

Each module follows the same pattern — if you want to add a technique:

1. Copy `modules/00_zero_shot` as a template
2. Add your technique to the curriculum table above
3. Write the 4-step `README.md` for your module
4. Submit a PR!

---

## 📄 License

Use freely, put your mind to work!.
