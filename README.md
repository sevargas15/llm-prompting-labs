# 🧠 LLM Prompting Lab

> Don't just learn prompting techniques — **build** them from scratch.

This repo is a hands-on curriculum where you train your own mini language models to understand 17 prompting techniques. For each technique, you will:

1. **Generate a dataset** using a frontier model (GPT-3.5/4 or Claude)
2. **Fine-tune a small model** on that dataset (DistilBERT, T5-small, GPT-2)
3. **Evaluate** your model against baselines
4. **Understand** _why_ the technique works — from the inside

---

## 📚 Curriculum

| # | Technique | Status | Module |
|---|-----------|--------|--------|
| 0 | [Zero-Shot Prompting](modules/00_zero_shot/) | ✅ Active | `modules/00_zero_shot` |
| 1 | [Few-Shot Prompting](modules/01_few_shot/) | ✅ Active | `modules/01_few_shot` |
| 2 | [Chain-of-Thought](modules/02_chain_of_thought/) | ✅ Active | `modules/02_chain_of_thought` |
| 3 | [Meta Prompting](modules/03_meta_prompting/) | 🔜 Soon | `modules/03_meta_prompting` |
| 4 | Self-Consistency | 🔜 Soon | — |
| 5 | Generate Knowledge Prompting | 🔜 Soon | — |
| 6 | Prompt Chaining | 🔜 Soon | — |
| 7 | Tree of Thoughts | 📅 Planned | — |
| 8 | Retrieval Augmented Generation | 📅 Planned | — |
| 9 | Automatic Reasoning & Tool Use | 📅 Planned | — |
| 10 | Automatic Prompt Engineer | 📅 Planned | — |
| 11 | Active-Prompt | 📅 Planned | — |
| 12 | Directional Stimulus Prompting | 📅 Planned | — |
| 13 | Program-Aided Language Models | 📅 Planned | — |
| 14 | ReAct | 📅 Planned | — |
| 15 | Reflexion | 📅 Planned | — |
| 16 | Multimodal CoT | 📅 Planned | — |
| 17 | Graph Prompting | 📅 Planned | — |

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/llm-prompting-lab.git
cd llm-prompting-lab
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Add your OpenAI or Anthropic key to .env
```

### 3. Start with Module 0 — Zero-Shot

```bash
cd modules/00_zero_shot

# Step 1: Generate a dataset using GPT
python scripts/01_generate_dataset.py --task sentiment --n 200

# Step 2: Fine-tune a DistilBERT classifier
python scripts/02_train.py --data data/sentiment_dataset.jsonl

# Step 3: Evaluate and compare to zero-shot baseline
python scripts/03_evaluate.py --model checkpoints/latest
```

Or open the notebook for a guided walkthrough:

```bash
jupyter notebook notebooks/00_zero_shot_walkthrough.ipynb
```

---

## 🏗 Repo Structure

```
llm-prompting-lab/
├── modules/
│   ├── 00_zero_shot/           ← Start here
│   │   ├── notebooks/          ← Interactive Jupyter walkthrough
│   │   ├── scripts/            ← CLI scripts for each step
│   │   └── data/               ← Generated datasets land here
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

## 🧩 What you'll actually build

### Module 0 — Zero-Shot Prompting

**The idea:** A model trained on billions of tokens already knows how to classify sentiment, recognize patterns, and answer questions — without any task-specific examples.

**What you build:**
- Use GPT-3.5 to generate a labeled sentiment dataset (200+ samples)
- Fine-tune `distilbert-base-uncased` as a 3-class classifier (happy / neutral / sad)
- Compare accuracy: your fine-tuned model vs pure zero-shot GPT
- Visualize the learned representations with t-SNE

**Key insight:** Zero-shot works because pretraining creates general-purpose representations. Fine-tuning on generated data teaches you _how much_ task-specific data actually matters.

---

## 🛠 Tech Stack

| Purpose | Library |
|---------|---------|
| Dataset generation | `openai` / `anthropic` |
| Model fine-tuning | `transformers` (Hugging Face) |
| Training loop | `PyTorch` / `accelerate` |
| Evaluation | `evaluate` (HF), `scikit-learn` |
| Notebooks | `jupyter`, `matplotlib`, `seaborn` |
| Vector search (RAG module) | `faiss-cpu` |

---

## 🤝 Contributing

Each module follows the same pattern — if you want to add a technique:

1. Copy `modules/00_zero_shot` as a template
2. Add your technique to the curriculum table above
3. Write the 4-step `README.md` for your module
4. Submit a PR!

---

## 📄 License

MIT — use freely, learn deeply.
