# Web Content Analysis with crawl4ai and Zero-Shot Classification 🕷️🤖

Automatically crawl, extract, and classify web content using crawl4ai's powerful web scraping capabilities combined with zero-shot classification.

## Why This Matters
- **🌐 Smart Web Crawling**: Efficiently navigate and extract content from websites
- **🧠 Zero-Shot Learning**: Classify content without prior training
- **⚡ High Performance**: Async processing for fast data collection
- **📊 Actionable Insights**: Get structured data from unstructured web content

## Key Features
- **Breadth-First Crawling** with configurable depth and page limits
- **Modern Web Support**: Handles JavaScript-rendered content
- **Zero-Shot Classification** using Facebook's BART-MNLI model
- **Clean Data Extraction**: Removes noise and extracts relevant content
- **Structured Output**: JSON format for easy integration

## Installation and Setup

### Prerequisites
- Python 3.12 or later
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### 1. Install Dependencies

Using uv (recommended):
```bash
uv sync
(or)
uv pip install -r requirements.txt
```
Using pip:
```bash
pip install -r requirements.txt
```
2. Run the Crawler
```bash
python main.py
```

### Project Structure
```
crawl4ai-with-machine-learning/
├── main.py                     # Main crawling and classification script
├── requirements.txt            # Project dependencies
├── pyproject.toml             # Project configuration
├── README.md                  # This file
└── deepcrawl_zero_shot_results.json  # Sample output
```

### Configuration
Edit 
main.py
 to customize:
```
START_URL: The website to crawl
MAX_DEPTH: How many levels deep to crawl
MAX_PAGES: Maximum number of pages to process
CANDIDATE_LABELS: Categories for classification
```

### Example Output
```json
{
  "url": "https://docs.zyte.com",
  "depth": 1,
  "labels": ["web scraping", "tutorial", "documentation"],
  "scores": [0.9885, 0.7306, 0.5839],
  "preview": "Documentation for Zyte's web scraping services and tools..."
}
```

### Contributing
Contributions are welcome! Please feel free to submit a Pull Request.