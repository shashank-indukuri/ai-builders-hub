"""
EXPERIMENT 1: Lost in the Middle — Context Window Reliability Test
With full debug logging to trace exactly where needles are inserted.
"""

import requests
import json
import time
from datetime import datetime

OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL        = "gemma4"
DOCUMENT_LEN = 15000
OUTPUT_FILE  = "exp1_results.json"
DEBUG        = True   # set False to silence verbose logs

NEEDLES = [
    {
        "position": 5,
        "key": "CEO_NAME",
        "value": "Marisol Quentin-Blackwood",
        "sentence": "The company Chief Executive Officer is Marisol Quentin-Blackwood, who assumed the role in January 2021 after relocating from the London office.",
        "question": "What is the full name of the company Chief Executive Officer mentioned in the document?"
    },
    {
        "position": 25,
        "key": "REVENUE_2023",
        "value": "$4.87 billion",
        "sentence": "According to the annual financial summary, the company reported total revenue of $4.87 billion for fiscal year 2023, surpassing analyst expectations.",
        "question": "What was the company total revenue figure reported for fiscal year 2023 in the document?"
    },
    {
        "position": 50,
        "key": "PROJECT_CODE",
        "value": "ZEPHYR-7749",
        "sentence": "All cross-departmental requests related to the initiative must reference the internal tracking code ZEPHYR-7749 in the subject line of all communications.",
        "question": "What internal tracking code must be referenced in cross-departmental communications for the initiative?"
    },
    {
        "position": 75,
        "key": "LAUNCH_DATE",
        "value": "March 3rd 2027",
        "sentence": "After extensive deliberation, the executive committee confirmed that the global product launch is scheduled for March 3rd 2027 across all primary markets.",
        "question": "What is the confirmed global product launch date mentioned in the document?"
    },
    {
        "position": 95,
        "key": "VAULT_CODE",
        "value": "aurora-delta-9",
        "sentence": "The temporary override passphrase for the secure document vault has been set to aurora-delta-9 and will remain active until the next security rotation.",
        "question": "What is the temporary override passphrase for the secure document vault mentioned in the document?"
    },
]

FILLER_TOPICS = [
    "quarterly performance review", "supply chain logistics", "employee satisfaction survey",
    "market expansion strategy", "product roadmap planning", "customer feedback analysis",
    "budget allocation review", "risk management framework", "competitive landscape report",
    "technology infrastructure audit", "vendor contract negotiations", "compliance review",
    "sustainability initiatives", "talent acquisition strategy", "operational efficiency audit",
    "digital transformation roadmap", "customer retention programs", "data governance policy",
    "international market entry", "merger integration planning", "brand positioning review",
    "investor relations update", "internal audit findings", "procurement process review"
]

def dbg(msg):
    if DEBUG:
        print(f"  [DEBUG] {msg}")

def generate_filler_paragraph(topic: str, index: int) -> str:
    body = (
        f"The working group conducted a detailed review of {topic} this quarter. "
        f"Findings were presented to senior leadership with a focus on measurable outcomes. "
        f"Several action items were identified and assigned to department leads with clear deadlines. "
        f"Progress will be tracked via monthly check-ins and reported to the steering committee. "
        f"The cross-functional team emphasized the importance of aligning {topic} with company objectives. "
        f"Stakeholders raised concerns about resource allocation which were noted for the next planning cycle. "
        f"External consultants provided benchmarking data to support the internal assessment. "
        f"A follow-up workshop has been scheduled to address the gaps identified during this review."
    )
    return f"\nSection {index} — {topic.title()}\n{body}\n"

def build_document_with_needles():
    paragraphs_needed = DOCUMENT_LEN // 100
    dbg(f"Total paragraphs planned: {paragraphs_needed}")

    # Build needle map: paragraph_index -> needle
    needle_map = {}
    for n in NEEDLES:
        para_idx = int(n["position"] / 100 * paragraphs_needed)
        needle_map[para_idx] = n
        dbg(f"Needle '{n['key']}' scheduled at paragraph {para_idx}/{paragraphs_needed} ({n['position']}%)")

    parts = [
        "CONFIDENTIAL INTERNAL REPORT — ACME DYNAMICS INC.\n"
        + "=" * 60 + "\n"
        "Prepared by: Office of the Chief Strategy Officer\n"
        "Distribution: Executive Leadership Team Only\n\n"
    ]

    inserted = []
    for i in range(paragraphs_needed):
        if i in needle_map:
            needle = needle_map[i]
            topic  = FILLER_TOPICS[i % len(FILLER_TOPICS)]
            para   = (
                f"\nSection {i} — {topic.title()}\n"
                f"The following update pertains to {topic}. "
                f"{needle['sentence']} "
                f"Leadership acknowledged this update and requested further analysis.\n"
            )
            parts.append(para)
            inserted.append((i, needle["key"], needle["value"]))
        else:
            topic = FILLER_TOPICS[i % len(FILLER_TOPICS)]
            parts.append(generate_filler_paragraph(topic, i))

    doc = "".join(parts)
    words  = len(doc.split())
    chars  = len(doc)
    tokens = int(words * 0.75)

    print(f"\n{'='*60}")
    print(f"DOCUMENT BUILD REPORT")
    print(f"{'='*60}")
    print(f"  Words  : {words:,}")
    print(f"  Chars  : {chars:,}")
    print(f"  Tokens : ~{tokens:,}")
    print(f"  Paras  : {paragraphs_needed}")

    # ── CRITICAL SANITY CHECK ──────────────────────────────────────────
    # Verify each needle value is actually present in the final document.
    # If NOT found here, it's a build bug, not a model retrieval bug.
    print(f"\n  NEEDLE SANITY CHECK (searching raw document string):")
    for para_idx, key, value in inserted:
        found_in_doc = value in doc
        icon = "✅ PRESENT" if found_in_doc else "🚨 MISSING — BUILD BUG"
        print(f"    [{para_idx:>3}] {key:<20} → {icon}")
        if not found_in_doc:
            print(f"         Expected to find: '{value}'")

    # ── SNIPPET PREVIEW ───────────────────────────────────────────────
    print(f"\n  NEEDLE CONTEXT SNIPPETS (100 chars around each needle):")
    for para_idx, key, value in inserted:
        idx = doc.find(value)
        if idx >= 0:
            snippet = doc[max(0, idx-60):idx+len(value)+60].replace("\n", " ")
            print(f"    {key}: ...{snippet}...")
        else:
            print(f"    {key}: *** NOT FOUND IN DOCUMENT ***")

    return doc

def query_ollama(prompt, needle_key):
    char_count = len(prompt)
    dbg(f"Sending prompt for {needle_key}: {char_count:,} chars to Ollama")

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": 32768   # explicitly set context window — increase if needed
        }
    }
    start = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        duration = round(time.time() - start, 2)

        # Log Ollama's reported token counts
        prompt_eval = data.get("prompt_eval_count", "?")
        eval_count  = data.get("eval_count", "?")
        dbg(f"Ollama processed {prompt_eval} prompt tokens, generated {eval_count} tokens in {duration}s")

        return data.get("response", "").strip(), duration
    except Exception as e:
        return f"ERROR: {e}", round(time.time() - start, 2)

def main():
    print(f"\n{'='*60}")
    print(f"Gemma 4 Experiment 1 — Lost in the Middle")
    print(f"Model: {MODEL}  |  Debug: {DEBUG}")
    print(f"{'='*60}\n")

    document = build_document_with_needles()

    print(f"\n{'='*60}")
    print(f"QUERYING MODEL FOR EACH NEEDLE")
    print(f"{'='*60}\n")

    results = []
    for needle in NEEDLES:
        prompt = (
            f"Read the document below carefully from start to finish. "
            f"Find the specific fact asked about and reply with the exact value only — "
            f"no explanation, no preamble, no 'I found' or 'The answer is'.\n\n"
            f"DOCUMENT:\n{document}\n\n"
            f"--- END OF DOCUMENT ---\n\n"
            f"QUESTION: {needle['question']}\n\n"
            f"ANSWER (exact value only):"
        )

        print(f"📍 Position {needle['position']:>3}% — {needle['key']}")
        dbg(f"Question: {needle['question']}")
        dbg(f"Expected: {needle['value']}")

        response, duration = query_ollama(prompt, needle["key"])
        correct = needle["value"].lower() in response.lower()
        icon    = "✅ CORRECT" if correct else "❌ WRONG"

        print(f"   {icon} ({duration}s)")
        print(f"   Expected : {needle['value']}")
        print(f"   Got      : {response[:120]}\n")

        results.append({
            "position_pct": needle["position"],
            "key":          needle["key"],
            "expected":     needle["value"],
            "response":     response[:200],
            "correct":      correct,
            "duration_sec": duration
        })

    # Summary
    correct_count = sum(1 for r in results if r["correct"])
    print(f"{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    for r in results:
        icon = "✅ FOUND" if r["correct"] else "❌ LOST"
        print(f"  {r['position_pct']:>3}% — {r['key']:<20} {icon}")
    print(f"\nOverall accuracy: {correct_count}/{len(results)} ({int(correct_count/len(results)*100)}%)")

    output = {
        "experiment":   "Lost in the Middle",
        "model":        MODEL,
        "timestamp":    datetime.now().isoformat(),
        "doc_words":    len(document.split()),
        "results":      results,
        "accuracy_pct": round(correct_count / len(results) * 100, 1)
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()