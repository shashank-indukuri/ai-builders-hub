import asyncio
import json
from typing import List, Dict

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from transformers import pipeline

START_URL = "https://blog.scrapinghub.com/"  # change to any site you control/allow
MAX_DEPTH = 2
MAX_PAGES = 50  # optional cap to keep runs bounded
HEADLESS = True

# Candidate labels for zero-shot topic classification (edit to your domain)
CANDIDATE_LABELS = [
    "web scraping", "data privacy", "cookies and tracking", "terms of service",
    "product announcement", "tutorial", "case study", "engineering blog", "company news"
]

def get_doc_text(result) -> str:
    # Prefer structured markdown if present; else fall back to cleaned_html text
    text = getattr(result, "markdown", None)
    if isinstance(text, str) and text.strip():
        return text
    html = getattr(result, "cleaned_html", "") or ""
    # quick HTML tag strip for fallback
    import re
    return re.sub(r"<[^>]+>", " ", html)

async def main():
    # Configure browser and run settings per Crawl4AI Quick Start and Deep Crawling
    browser_conf = BrowserConfig(headless=HEADLESS)  # headless browser session
    run_conf = CrawlerRunConfig(                     # per-page run configuration
        cache_mode=CacheMode.BYPASS,                 # bypass cache to fetch fresh content
        scraping_strategy=LXMLWebScrapingStrategy(), # reliable HTML-to-markdown extraction
        deep_crawl_strategy=BFSDeepCrawlStrategy(    # breadth-first deep crawling
            max_depth=MAX_DEPTH,                     # crawl start + MAX_DEPTH levels
            include_external=False,                  # stay within the same domain
            max_pages=MAX_PAGES                      # optional page limit to bound the crawl
        ),
        verbose=True
    )

    # Run deep crawl: returns a list of results;
    async with AsyncWebCrawler(config=browser_conf) as crawler:
        results = await crawler.arun(url=START_URL, config=run_conf)  # list of Result objects
        print(f"[crawl] collected {len(results)} pages")

    # Load a robust zero-shot classifier
    # BART-MNLI is a strong baseline for zero-shot text classification pipelines
    clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")  # local CPU ok

    classified: List[Dict] = []
    for r in results:
        text = get_doc_text(r)
        if not text.strip():
            continue
        # Truncate to a practical context size for latency while preserving signal
        snippet = text[:2000]
        pred = clf(snippet, CANDIDATE_LABELS, multi_label=True)  # multi-label so content can fit more than one topic
        classified.append({
            "url": getattr(r, "url", ""),
            "depth": (getattr(r, "metadata", {}) or {}).get("depth", None),
            "labels": pred["labels"],
            "scores": [round(float(s), 4) for s in pred["scores"]],
            "preview": snippet[:280]
        })

    # Persist locally for inspection / downstream pipelines
    with open("deepcrawl_zero_shot_results.json", "w", encoding="utf-8") as f:
        json.dump(classified, f, ensure_ascii=False, indent=2)

    print(f"[done] wrote {len(classified)} classified pages to deepcrawl_zero_shot_results.json")

if __name__ == "__main__":
    asyncio.run(main())