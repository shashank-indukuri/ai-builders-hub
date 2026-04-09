# Gemma 4 Experiments — Setup & Run Guide

## What you're testing
1. **Exp 1 — Lost in the Middle**: Does the 256K context window actually work end-to-end?
2. **Exp 2 — MoE Efficiency**: Does 26B/4B-active deliver large-model quality at small-model speed?

---

## Step 1: Install Ollama (5 min)
```bash
# Mac / Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: download installer from https://ollama.ai
```

## Step 2: Pull the models
```bash
ollama pull gemma3:27b     # ~17GB — swap to gemma4:27b when available
ollama pull gemma3:12b     # ~8GB
ollama pull llama3.1:8b    # ~5GB
```

## Step 3: Install Python deps
```bash
pip install requests psutil
```

## Step 4: Run Experiment 1 (~15 min)
```bash
python exp1_context_test.py
```
Tests whether Gemma can find facts at 5%, 25%, 50%, 75%, 95% of a long document.
Results saved to: `exp1_results.json`

## Step 5: Run Experiment 2 (~30-60 min)
```bash
python exp2_efficiency_test.py
```
Runs 20 prompts across 3 models, scores quality + measures speed.
Results saved to: `exp2_results.json`

## Step 6: Generate the dashboards
```bash
python visualize_results.py
```
---

## Scoring rubric (Exp 2)
- **3/3** = Correct, complete, follows all instructions
- **2/3** = Mostly correct, minor issues
- **1/3** = Partially correct
- **0/3** = Wrong or fails instruction
