"""
EXPERIMENT 2: MoE Efficiency Test — Quality vs Speed vs RAM
Tests whether Gemma 4's 26B/4B-active architecture gives you
near-large-model quality at small-model compute cost.

Compares: Gemma 4 27B (MoE) vs Gemma 3 12B (dense) vs Llama 3.1 8B (dense)

Usage: python exp2_efficiency_test.py
Requires: ollama running locally with all three models pulled

Pull commands:
  ollama pull gemma4:latest
  ollama pull gemma3:12b
  ollama pull mistral:latest
"""

import requests
import json
import time
import psutil
import os
from datetime import datetime

# ── CONFIG ──────────────────────────────────────────────────────────────────
OLLAMA_URL  = "http://localhost:11434/api/chat"
OUTPUT_FILE = "exp2_results.json"

MODELS = [
    {"name": "gemma4:latest",   "label": "Gemma 4 8B (MoE)",     "params": "8B",  "active": "~2B"},
    {"name": "gemma3:12b",      "label": "Gemma 3 12B (Dense)",  "params": "12B", "active": "12B"},
    {"name": "mistral:latest",  "label": "Mistral 7B (Dense)",   "params": "7B",  "active": "7B"},
]
# ────────────────────────────────────────────────────────────────────────────

# 20 prompts across 4 categories (5 each)
TEST_PROMPTS = [
    # CATEGORY: Reasoning (5 prompts)
    {
        "id": "R1", "category": "Reasoning",
        "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show your reasoning step by step.",
        "rubric": "Ball = $0.05. Award 3 if correct with clear steps, 2 if correct no steps, 0 if wrong."
    },
    {
        "id": "R2", "category": "Reasoning",
        "prompt": "There are 3 boxes. One has apples, one has oranges, one has mixed. All labels are wrong. You can pick one fruit from one box. Which box do you pick from to identify all boxes?",
        "rubric": "Pick from 'Mixed' box. Award 3 if correct with explanation, 2 if correct no explanation, 0 if wrong."
    },
    {
        "id": "R3", "category": "Reasoning",
        "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
        "rubric": "5 minutes. Award 3 if correct with clear reasoning, 2 if just correct answer, 0 if wrong."
    },
    {
        "id": "R4", "category": "Reasoning",
        "prompt": "A doctor says 'I have 10 patients and half of them will need surgery.' How many patients need surgery?",
        "rubric": "5 patients. Award 3 if answers directly with no tricks, 1 if overthinks it."
    },
    {
        "id": "R5", "category": "Reasoning",
        "prompt": "You have a 3-liter jug and a 5-liter jug. How do you measure exactly 4 liters of water?",
        "rubric": "Fill 5L, pour into 3L, empty 3L, pour remaining 2L into 3L, fill 5L again, pour 1L into 3L = 4L in 5L jug. Award 3 if correct, 2 if partially correct."
    },

    # CATEGORY: Coding (5 prompts)
    {
        "id": "C1", "category": "Coding",
        "prompt": "Write a Python function that checks if a string is a palindrome, ignoring spaces, punctuation and case. Include a docstring and 3 test cases.",
        "rubric": "Award 3 if function is correct + handles edge cases + has tests. 2 if mostly correct. 1 if partially works."
    },
    {
        "id": "C2", "category": "Coding",
        "prompt": "Write a SQL query to find the top 3 customers by total purchase amount from a table called 'orders' with columns: customer_id, amount, order_date.",
        "rubric": "Award 3 if uses SUM + GROUP BY + ORDER BY DESC + LIMIT 3 correctly. 2 if mostly right. 1 if partially correct."
    },
    {
        "id": "C3", "category": "Coding",
        "prompt": "In Python, what is the difference between a list and a tuple? When would you choose one over the other? Give a concrete code example for each.",
        "rubric": "Award 3 if mentions mutability, memory, use cases with examples. 2 if covers main points no examples. 1 if vague."
    },
    {
        "id": "C4", "category": "Coding",
        "prompt": "Write a Python function using only standard library to retry a function call up to 3 times with exponential backoff if it raises an exception.",
        "rubric": "Award 3 if correct retry logic + exponential backoff (1s, 2s, 4s) + clean code. 2 if retry works but backoff wrong. 1 if partial."
    },
    {
        "id": "C5", "category": "Coding",
        "prompt": "Explain what a race condition is and show a simple Python example that demonstrates one, then show how to fix it.",
        "rubric": "Award 3 if explains concept clearly + shows threading example + fixes with lock. 2 if mostly correct. 1 if vague."
    },

    # CATEGORY: Summarization (5 prompts)
    {
        "id": "S1", "category": "Summarization",
        "prompt": "Summarize this in exactly 2 sentences: 'The James Webb Space Telescope, launched in December 2021, represents the most powerful space telescope ever built. It observes in infrared light, allowing it to peer through cosmic dust clouds that blocked earlier telescopes. Webb can detect light from galaxies formed just a few hundred million years after the Big Bang, providing unprecedented views of the early universe. Its mirror, composed of 18 hexagonal gold-plated beryllium segments, spans 6.5 meters across.'",
        "rubric": "Award 3 if exactly 2 sentences, captures key facts accurately. 2 if close but off on count or facts. 1 if poor summary."
    },
    {
        "id": "S2", "category": "Summarization",
        "prompt": "Extract the 3 most important actionable items from this text: 'Our Q3 review shows customer churn is up 12% due to slow onboarding (avg 8 days), the mobile app has a 2.3 star rating from 340 reviews mostly complaining about crashes, and our NPS score dropped from 42 to 31. Meanwhile our enterprise segment grew 23% and top accounts report high satisfaction. Sales team is understaffed by 4 reps for current pipeline.'",
        "rubric": "Award 3 if picks: fix onboarding, fix app crashes, hire sales reps. 2 if gets 2/3. 1 if gets 1/3 or picks irrelevant items."
    },
    {
        "id": "S3", "category": "Summarization",
        "prompt": "What is the main argument being made here: 'Every year companies spend billions on employee training programs. Yet study after study shows that 90% of new skills are forgotten within a week without immediate application. The problem isn't the training itself — it's that training is treated as an event rather than a process. Organizations that embed learning into daily workflows through micro-practices, peer coaching, and real project application consistently outperform those relying on periodic training days.'",
        "rubric": "Award 3 if identifies: training fails because it's not embedded in daily work. 2 if captures most of it. 1 if vague."
    },
    {
        "id": "S4", "category": "Summarization",
        "prompt": "Give me a TL;DR of transformer architecture in plain English for a non-technical founder. Max 3 sentences.",
        "rubric": "Award 3 if explains attention mechanism, why it's powerful, in plain English under 3 sentences. 2 if mostly right. 1 if too technical."
    },
    {
        "id": "S5", "category": "Summarization",
        "prompt": "Identify the logical flaw in this argument: 'Our product has a 4.8 star rating with 50,000 reviews, therefore it is objectively the best product in the market.'",
        "rubric": "Award 3 if identifies: high ratings ≠ best (survivorship bias, market is not fully covered, subjective ratings). 2 if partial. 1 if misses main flaw."
    },

    # CATEGORY: Instruction Following (5 prompts)
    {
        "id": "I1", "category": "Instructions",
        "prompt": "List exactly 5 programming languages. Format: a numbered list. No descriptions, just names.",
        "rubric": "Award 3 if exactly 5 items, numbered, names only. 2 if 5 items but added descriptions. 1 if wrong count."
    },
    {
        "id": "I2", "category": "Instructions",
        "prompt": "Respond to this message as if you are a very grumpy customer service agent who is clearly having a bad day but still technically answers the question. The question: 'What are your store hours?'",
        "rubric": "Award 3 if clearly grumpy tone + still answers the question. 2 if one but not the other. 1 if ignores instruction."
    },
    {
        "id": "I3", "category": "Instructions",
        "prompt": "Translate this to formal business English, then casual English, then emoji-only. Original: 'we need to talk about the budget'",
        "rubric": "Award 3 if all three versions present and correct in tone. 2 if 2/3. 1 if 1/3."
    },
    {
        "id": "I4", "category": "Instructions",
        "prompt": "Write a haiku about machine learning. A haiku is exactly 3 lines: 5 syllables, 7 syllables, 5 syllables.",
        "rubric": "Award 3 if 3 lines + correct syllable counts (5-7-5) + ML topic. 2 if close but syllable count off by 1. 1 if wrong format."
    },
    {
        "id": "I5", "category": "Instructions",
        "prompt": "Answer only with YES or NO: Is Paris the capital of France?",
        "rubric": "Award 3 if literally just 'Yes' or 'YES'. 0 if adds any other words."
    },
]

def score_response(response: str, rubric: str, prompt_id: str) -> int:
    """
    Auto-score based on heuristics and keyword matching.
    Returns 0-3 based on rubric criteria.
    """
    resp_lower = response.lower().strip()
    resp_lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    
    # ═══════════════════════════════════════════════════════════════
    # REASONING CATEGORY
    # ═══════════════════════════════════════════════════════════════
    
    if prompt_id == "R1":  # Ball costs $0.05
        has_correct = "0.05" in response or "five cents" in resp_lower or "$0.05" in response
        has_reasoning = any(word in resp_lower for word in ["equation", "subtract", "difference", "x +"])
        if has_correct and has_reasoning:
            return 3
        elif has_correct:
            return 2
        return 0
    
    if prompt_id == "R2":  # Pick from Mixed box
        has_mixed = "mixed" in resp_lower or "'mixed'" in resp_lower
        has_explanation = len(response) > 100 and ("label" in resp_lower or "wrong" in resp_lower)
        if has_mixed and has_explanation:
            return 3
        elif has_mixed:
            return 2
        return 0
    
    if prompt_id == "R3":  # 5 minutes for 100 machines
        has_five = "5 minute" in resp_lower or "five minute" in resp_lower or "5min" in resp_lower
        has_reasoning = "rate" in resp_lower or "parallel" in resp_lower or "same time" in resp_lower
        if has_five and has_reasoning:
            return 3
        elif has_five:
            return 2
        return 0
    
    if prompt_id == "R4":  # 5 patients need surgery
        has_five = "5" in response or "five" in resp_lower
        has_half = "half" in resp_lower
        # Check if it overthinks (mentions "at least" or "up to")
        overthinks = "at least" in resp_lower or "up to" in resp_lower or "could be" in resp_lower
        if has_five and not overthinks:
            return 3
        elif overthinks:
            return 1
        return 0
    
    if prompt_id == "R5":  # Water jug puzzle
        has_steps = len(resp_lines) >= 4 or response.count("liter") >= 4
        has_correct_logic = ("5" in response and "3" in response and "2" in response and "4" in response)
        mentions_pour = "pour" in resp_lower or "fill" in resp_lower or "empty" in resp_lower
        if has_correct_logic and has_steps and mentions_pour:
            return 3
        elif has_correct_logic or (has_steps and mentions_pour):
            return 2
        return 1
    
    # ═══════════════════════════════════════════════════════════════
    # CODING CATEGORY
    # ═══════════════════════════════════════════════════════════════
    
    if prompt_id == "C1":  # Palindrome function
        has_function = "def " in response and "palindrome" in resp_lower
        has_docstring = '"""' in response or "'''" in response
        has_tests = response.count("assert") >= 2 or response.count("print") >= 2 or "test" in resp_lower
        handles_edge = any(word in resp_lower for word in ["lower", "replace", "strip", "punctuation", "space"])
        
        score = 0
        if has_function:
            score = 1
        if has_function and handles_edge:
            score = 2
        if has_function and handles_edge and has_docstring and has_tests:
            score = 3
        return score
    
    if prompt_id == "C2":  # SQL top 3 customers
        has_sum = "sum(" in resp_lower
        has_group = "group by" in resp_lower
        has_order = "order by" in resp_lower
        has_limit = "limit" in resp_lower or "top 3" in resp_lower
        has_desc = "desc" in resp_lower
        
        correct_parts = sum([has_sum, has_group, has_order, has_limit])
        if correct_parts >= 4 and has_desc:
            return 3
        elif correct_parts >= 3:
            return 2
        elif correct_parts >= 2:
            return 1
        return 0
    
    if prompt_id == "C3":  # List vs Tuple
        mentions_mutable = "mutable" in resp_lower or "immutable" in resp_lower or "change" in resp_lower
        has_examples = response.count("[") >= 1 and response.count("(") >= 1
        mentions_use_case = any(word in resp_lower for word in ["performance", "memory", "constant", "fixed", "hash"])
        
        if mentions_mutable and has_examples and mentions_use_case:
            return 3
        elif mentions_mutable and (has_examples or mentions_use_case):
            return 2
        elif mentions_mutable:
            return 1
        return 0
    
    if prompt_id == "C4":  # Retry with exponential backoff
        has_retry_loop = "for" in resp_lower or "while" in resp_lower or "range(3)" in response
        has_exception = "except" in resp_lower or "try" in resp_lower
        has_backoff = ("sleep" in resp_lower and ("**" in response or "*2" in response or "exponential" in resp_lower))
        has_function = "def " in response
        
        if has_function and has_retry_loop and has_exception and has_backoff:
            return 3
        elif has_retry_loop and has_exception:
            return 2
        elif has_retry_loop or has_exception:
            return 1
        return 0
    
    if prompt_id == "C5":  # Race condition
        explains_concept = "race condition" in resp_lower and ("thread" in resp_lower or "concurrent" in resp_lower)
        has_example = "import threading" in resp_lower or "thread" in resp_lower
        has_fix = "lock" in resp_lower or "mutex" in resp_lower or "semaphore" in resp_lower
        
        if explains_concept and has_example and has_fix:
            return 3
        elif explains_concept and (has_example or has_fix):
            return 2
        elif explains_concept:
            return 1
        return 0
    
    # ═══════════════════════════════════════════════════════════════
    # SUMMARIZATION CATEGORY
    # ═══════════════════════════════════════════════════════════════
    
    if prompt_id == "S1":  # Exactly 2 sentences
        sentences = [s.strip() for s in response.replace("!", ".").replace("?", ".").split(".") if s.strip() and len(s.strip()) > 10]
        mentions_key_facts = sum([
            "webb" in resp_lower or "telescope" in resp_lower,
            "infrared" in resp_lower,
            "early universe" in resp_lower or "big bang" in resp_lower,
            "mirror" in resp_lower or "6.5" in response
        ])
        
        if len(sentences) == 2 and mentions_key_facts >= 2:
            return 3
        elif len(sentences) <= 3 and mentions_key_facts >= 2:
            return 2
        elif mentions_key_facts >= 1:
            return 1
        return 0
    
    if prompt_id == "S2":  # Extract 3 actionable items
        items_found = sum([
            "onboarding" in resp_lower or "8 days" in response,
            "app" in resp_lower or "crash" in resp_lower or "2.3" in response,
            "hire" in resp_lower or "sales" in resp_lower or "4 rep" in response or "understaffed" in resp_lower
        ])
        
        if items_found == 3:
            return 3
        elif items_found == 2:
            return 2
        elif items_found == 1:
            return 1
        return 0
    
    if prompt_id == "S3":  # Main argument
        identifies_main = any(phrase in resp_lower for phrase in [
            "embed", "daily work", "process not event", "continuous", "workflow", "not periodic"
        ])
        mentions_problem = "90%" in response or "forgotten" in resp_lower or "fails" in resp_lower
        
        if identifies_main and mentions_problem:
            return 3
        elif identifies_main:
            return 2
        return 1
    
    if prompt_id == "S4":  # Transformer TL;DR
        sentences = [s.strip() for s in response.replace("!", ".").replace("?", ".").split(".") if s.strip() and len(s.strip()) > 10]
        mentions_attention = "attention" in resp_lower
        is_plain_english = not any(word in resp_lower for word in ["tensor", "gradient", "backprop", "embedding layer"])
        is_concise = len(sentences) <= 3
        
        if mentions_attention and is_plain_english and is_concise:
            return 3
        elif mentions_attention and is_concise:
            return 2
        elif mentions_attention or is_concise:
            return 1
        return 0
    
    if prompt_id == "S5":  # Logical flaw in 4.8 star argument
        identifies_flaw = any(phrase in resp_lower for phrase in [
            "not objective", "subjective", "bias", "sample", "doesn't mean best", 
            "correlation", "other factors", "market", "competition"
        ])
        specific_flaw = any(phrase in resp_lower for phrase in [
            "survivorship", "selection bias", "doesn't cover", "not comprehensive"
        ])
        
        if identifies_flaw and specific_flaw:
            return 3
        elif identifies_flaw:
            return 2
        return 1
    
    # ═══════════════════════════════════════════════════════════════
    # INSTRUCTION FOLLOWING CATEGORY
    # ═══════════════════════════════════════════════════════════════
    
    if prompt_id == "I1":  # Exactly 5 languages, numbered, no descriptions
        numbered_lines = [l for l in resp_lines if l and l[0].isdigit()]
        # Check if descriptions are added (lines are too long)
        has_descriptions = any(len(l) > 30 for l in numbered_lines)
        
        if len(numbered_lines) == 5 and not has_descriptions:
            return 3
        elif len(numbered_lines) == 5:
            return 2
        elif 3 <= len(numbered_lines) <= 7:
            return 1
        return 0
    
    if prompt_id == "I2":  # Grumpy customer service
        is_grumpy = any(word in resp_lower for word in [
            "ugh", "sigh", "whatever", "look", "seriously", "fine", "geez", "god"
        ]) or "..." in response
        answers_question = "hour" in resp_lower or "open" in resp_lower or "am" in resp_lower or "pm" in resp_lower
        
        if is_grumpy and answers_question:
            return 3
        elif is_grumpy or answers_question:
            return 2
        return 1
    
    if prompt_id == "I3":  # 3 versions: formal, casual, emoji
        has_formal = any(phrase in resp_lower for phrase in ["discuss", "meeting", "schedule", "financial"])
        has_casual = any(phrase in resp_lower for phrase in ["talk", "chat", "hey", "yo"])
        has_emoji = any(char in response for char in ["💰", "💵", "💬", "🗣", "📊", "😬", "😅"])
        
        versions_count = sum([has_formal, has_casual, has_emoji])
        if versions_count == 3:
            return 3
        elif versions_count == 2:
            return 2
        elif versions_count == 1:
            return 1
        return 0
    
    if prompt_id == "I4":  # Haiku 5-7-5 syllables
        lines = [l for l in resp_lines if l and not l.startswith("#")]
        # Rough syllable check (not perfect but better than nothing)
        if len(lines) == 3:
            # Check approximate syllable counts by word count (rough heuristic)
            line_lengths = [len(l.split()) for l in lines]
            # 5 syllables ≈ 3-4 words, 7 syllables ≈ 4-5 words
            reasonable_structure = (2 <= line_lengths[0] <= 4 and 
                                   3 <= line_lengths[1] <= 6 and 
                                   2 <= line_lengths[2] <= 4)
            if reasonable_structure:
                return 3
            return 2
        elif len(lines) > 0:
            return 1
        return 0
    
    if prompt_id == "I5":  # YES or NO only
        # Must be EXACTLY yes/no with optional punctuation
        clean_resp = resp_lower.replace(".", "").replace("!", "").strip()
        if clean_resp in ["yes", "no"]:
            return 3
        return 0
    
    # Fallback (should never reach here)
    return 1

def get_system_ram_gb() -> float:
    """System-wide RAM in use (GB) — captures Ollama model footprint."""
    return round(psutil.virtual_memory().used / 1024**3, 2)

def get_ollama_model_ram_gb() -> float:
    """Ask Ollama /api/ps how much RAM the loaded model is using."""
    try:
        resp = requests.get("http://localhost:11434/api/ps", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        if models:
            total = sum(m.get("size_vram") or m.get("size") or 0 for m in models)
            return round(total / 1024**3, 2)
    except Exception:
        pass
    return 0.0

def unload_ollama_model(model_name: str):
    """Unload model from RAM so the next model gets a clean baseline."""
    try:
        requests.post(OLLAMA_URL, json={"model": model_name, "keep_alive": 0}, timeout=10)
        time.sleep(2)
    except Exception:
        pass

def query_model(prompt: str, model_name: str) -> tuple[str, float, float, float]:
    """Query model, return (response, duration_sec, tokens_per_sec, ollama_ram_gb)."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0}
    }
    start = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        data             = resp.json()
        duration         = time.time() - start
        
        # Extract token metrics from response
        eval_count       = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 1)
        tps  = round(eval_count / (eval_duration_ns / 1e9), 1) if eval_duration_ns > 0 else 0
        ram  = get_ollama_model_ram_gb()
        
        response_text = data.get("message", {}).get("content", "")
        return response_text, round(duration, 2), tps, ram
    except Exception as e:
        return f"ERROR: {e}", round(time.time() - start, 2), 0, 0.0

def run_model_benchmark(model_config: dict) -> dict:
    """Run all 20 prompts for one model — tracks quality, speed, and RAM."""
    model_name  = model_config["name"]
    model_label = model_config["label"]

    print(f"\n{'='*60}")
    print(f"🤖 Testing: {model_label}")
    print(f"{'='*60}")

    # RAM baseline before model loads
    ram_before = get_system_ram_gb()
    print(f"  💾 System RAM before load : {ram_before} GB used")

    prompt_results  = []
    category_scores = {}
    all_tps         = []
    all_ram         = []
    peak_ram        = 0.0

    for i, test in enumerate(TEST_PROMPTS, 1):
        print(f"  [{i:02d}/20] {test['id']} ({test['category']})...", end=" ", flush=True)
        response, duration, tps, ram_gb = query_model(test["prompt"], model_name)
        score = score_response(response, test["rubric"], test["id"])

        if ram_gb > peak_ram:
            peak_ram = ram_gb
        all_ram.append(ram_gb)

        prompt_results.append({
            "id":               test["id"],
            "category":         test["category"],
            "score":            score,
            "duration_sec":     duration,
            "tokens_per_sec":   tps,
            "ram_gb":           ram_gb,
            "response_preview": response.strip()[:100]
        })

        category_scores.setdefault(test["category"], []).append(score)
        all_tps.append(tps)

        score_icon = ["❌", "⚠️ ", "✅ ", "🎯"][score]
        print(f"{score_icon} score={score}/3.0  {tps} tok/s  RAM={ram_gb}GB")

    # RAM after all queries
    ram_after    = get_system_ram_gb()
    ram_delta    = round(ram_after - ram_before, 2)
    avg_ram      = round(sum(all_ram) / len(all_ram), 2) if all_ram else 0.0
    ollama_ram   = max(all_ram) if all_ram else 0.0

    print(f"\n  💾 System RAM after  load : {ram_after} GB used")
    print(f"  💾 Model RAM footprint    : +{ram_delta} GB delta | peak={peak_ram} GB | avg={avg_ram} GB")
    print(f"  💾 Ollama reported RAM    : {ollama_ram} GB")

    # Unload before next model for clean baseline
    print(f"  🔄 Unloading {model_name} ...")
    unload_ollama_model(model_name)
    ram_unloaded = get_system_ram_gb()
    print(f"  💾 System RAM after unload: {ram_unloaded} GB used")

    all_scores   = [r["score"] for r in prompt_results]
    avg_score    = round(sum(all_scores) / len(all_scores), 2)
    avg_tps      = round(sum(all_tps) / len(all_tps), 1)
    cat_averages = {cat: round(sum(s)/len(s), 2) for cat, s in category_scores.items()}

    print(f"\n  📊 {model_label} Summary:")
    print(f"     Quality : {avg_score}/3.0 ({int(avg_score/3*100)}%)")
    print(f"     Speed   : {avg_tps} tokens/sec")
    print(f"     RAM     : {ollama_ram} GB (Ollama) | +{ram_delta} GB system delta | peak {peak_ram} GB")
    for cat, avg in cat_averages.items():
        print(f"     {cat:<16}: {avg}/3.0")

    return {
        "model":              model_name,
        "label":              model_label,
        "params":             model_config["params"],
        "active_params":      model_config["active"],
        "avg_quality_score":  avg_score,
        "quality_pct":        int(avg_score/3*100),
        "avg_tokens_per_sec": avg_tps,
        "memory": {
            "ram_before_gb":      ram_before,
            "ram_after_gb":       ram_after,
            "ram_delta_gb":       ram_delta,
            "peak_ram_gb":        peak_ram,
            "avg_ram_gb":         avg_ram,
            "ollama_model_ram_gb": ollama_ram,
            "ram_after_unload_gb": ram_unloaded
        },
        "category_scores":    cat_averages,
        "prompt_results":     prompt_results
    }

def print_comparison_table(all_results: list):
    """Print the final comparison table"""
    print("\n" + "="*80)
    print("📊 FINAL COMPARISON TABLE")
    print("="*80)

    header = f"{'Model':<25} {'Params':>7} {'Active':>7} {'Quality':>9} {'Speed':>12} {'RAM (GB)':>10}"
    print(header)
    print("-"*80)

    for r in all_results:
        name        = r["label"][:24]
        params      = r["params"]
        active      = r["active_params"]
        qual        = f"{r['quality_pct']}%"
        speed       = f"{r['avg_tokens_per_sec']} tok/s"
        ram         = r.get("memory", {})
        ollama_ram  = ram.get("ollama_model_ram_gb", 0)
        delta_ram   = ram.get("ram_delta_gb", 0)
        ram_str     = f"{ollama_ram}GB" if ollama_ram > 0 else f"+{delta_ram}GB"
        print(f"{name:<25} {params:>7} {active:>7} {qual:>9} {speed:>12} {ram_str:>10}")

    print("\n  RAM = Ollama-reported model size (or system delta if unavailable)")
    
    print("\n📈 Quality by Category:")
    categories = list(all_results[0]["category_scores"].keys())
    cat_header = f"{'Model':<25}" + "".join(f"{c[:12]:>14}" for c in categories)
    print(cat_header)
    print("-"*70)
    for r in all_results:
        row = f"{r['label'][:24]:<25}"
        for cat in categories:
            score = r["category_scores"].get(cat, 0)
            row += f"{score:>14.2f}"
        print(row)
    
    # Key insight
    print("\n💡 THE KEY INSIGHT:")
    if len(all_results) >= 2:
        moe  = all_results[0]
        base = all_results[1]
        qual_diff  = moe["quality_pct"] - base["quality_pct"]
        speed_diff = moe["avg_tokens_per_sec"] - base["avg_tokens_per_sec"]
        print(f"   Gemma 4 MoE vs {base['label']}:")
        print(f"   Quality diff: {'+' if qual_diff >= 0 else ''}{qual_diff}%")
        print(f"   Speed diff:   {'+' if speed_diff >= 0 else ''}{speed_diff} tok/s")
        if qual_diff > 0 and speed_diff > 0:
            print(f"\n   ✅ MoE wins on BOTH quality AND speed.")
        elif qual_diff > 0 and speed_diff <= 0:
            print(f"\n   ⚠️  MoE is better quality but slower.")
        elif qual_diff <= 0 and speed_diff > 0:
            print(f"\n   ⚠️  MoE is faster but not better quality — the hype may be overstated.")
        else:
            print(f"\n   ❌ MoE wins on neither metric vs a smaller model.")

def main():
    print("🚀 Gemma 4 Experiment 2: MoE Efficiency Benchmark")
    print(f"Testing {len(MODELS)} models × {len(TEST_PROMPTS)} prompts = {len(MODELS)*len(TEST_PROMPTS)} total queries\n")
    print("Make sure these models are pulled in Ollama:")
    for m in MODELS:
        print(f"  ollama pull {m['name']}")
    print()
    
    all_results = []
    for model_config in MODELS:
        result = run_model_benchmark(model_config)
        all_results.append(result)
    
    print_comparison_table(all_results)
    
    output = {
        "experiment": "MoE Efficiency Benchmark",
        "timestamp": datetime.now().isoformat(),
        "models_tested": [r["label"] for r in all_results],
        "prompts_count": len(TEST_PROMPTS),
        "results": all_results
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Raw results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()