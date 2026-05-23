# Contributing to LLM Prompting Lab

Thanks for helping build this curriculum! Here's how to add a new module.

## Module structure

Every module follows the same 4-script pattern:

```
modules/NN_technique_name/
├── README.md                    ← 4-step explainer
├── scripts/
│   ├── 01_generate_dataset.py  ← Generate data via GPT/Claude
│   ├── 02_train.py             ← Fine-tune a model
│   ├── 03_evaluate.py          ← Compare to baselines
│   └── 04_visualize.py         ← Plot results
├── notebooks/
│   └── NN_walkthrough.ipynb    ← Interactive guided notebook
└── data/                       ← Generated data lands here (gitignored)
```

## Steps to add a module

1. Pick the next available number (`NN`) from the curriculum table in `README.md`
2. Copy `modules/00_zero_shot/` as a template
3. Update `scripts/01_generate_dataset.py` with your task-specific prompts
4. Choose an appropriate model in `scripts/02_train.py` (T5 for seq2seq, DistilBERT for classification, GPT-2 for generation)
5. Write a meaningful baseline comparison in `scripts/03_evaluate.py`
6. Add a visualization that shows the key insight in `scripts/04_visualize.py`
7. Write a `README.md` that:
   - Explains the technique in 2 sentences
   - Lists the 4 steps clearly
   - Includes a cost estimate for dataset generation
   - Ends with a "Key insight" section
8. Update the curriculum table in the root `README.md`
9. Open a PR!

## Code style

- Python 3.10+
- Type hints on all function signatures
- `argparse` for CLI args with sensible defaults
- Print progress — these scripts take minutes to run
- `shared/utils/api_helpers.py` for all API calls (don't duplicate)

## Cost estimates

Each module should document approximate API costs in its README:

| Examples | Model | Approximate cost |
|---------|-------|-----------------|
| 200 | GPT-3.5-turbo | ~$0.05 |
| 200 | GPT-4o-mini | ~$0.10 |
| 200 | Claude Haiku | ~$0.04 |
