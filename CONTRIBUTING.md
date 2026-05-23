# Contributing

If you want to add a new module or improve an existing one, this is how things are organized.

## Module structure

Every module follows the same pattern:

```
modules/NN_technique_name/
├── README.md                    ← what the technique is and how to run it
├── scripts/
│   ├── 01_generate_dataset.py  ← generate data via Groq
│   ├── 02_train.py             ← Fine-tune a model
│   ├── 03_evaluate.py          ← compare against baselines
│   └── 04_visualize.py         ← Plot results
├── notebooks/
│   └── NN_walkthrough.ipynb    ← guided notebook version
└── data/                       ← generated data lands here (gitignored)
```

## Adding a module

1. Pick the next number (`NN`) from the curriculum table in `README.md`
2. Copy `modules/00_zero_shot/` as your starting point
3. Update `01_generate_dataset.py` with prompts specific to your technique
4. Pick the right model in `02_train.py` — DistilBERT for classification, T5 for seq2seq, GPT-2 for generation
5. Write a real baseline comparison in `03_evaluate.py` — random and zero-shot at minimum
6. Make sure `04_visualize.py` shows something meaningful, not just a loss curve
7. Write a `README.md` that explains the technique plainly, lists the steps, and ends with a key insight
8. Update the curriculum table in the root `README.md`
9. Open a PR

## A few things to keep consistent

- Python 3.10+
- Type hints on function signatures
- `argparse` for CLI args with sensible defaults
- Print progress as scripts run; these take a few minutes, and silent scripts are annoying
- Use `shared/utils/api_helpers.py` for all Groq calls, don't duplicate the client setup
