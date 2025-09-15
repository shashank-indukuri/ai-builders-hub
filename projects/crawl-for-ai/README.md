# Web Crawling for AI with Crawl4AI 🕷️

Extract clean, LLM-ready content from any website using Crawl4AI - transform web pages into structured Markdown perfect for RAG systems, AI agents, and data pipelines.

## Why This Matters

- 🌐 **LLM-Ready Output**: Converts messy HTML into clean Markdown for AI consumption
- ⚡ **Fast & Efficient**: Async crawling with intelligent content extraction
- 🔒 **Privacy Focused**: Run locally without sending data to external services
- 🛡️ **Undetected Browsing**: Advanced browser fingerprinting protection
- 💰 **Cost Free**: No API costs - crawl unlimited pages locally

## Key Features

- **Clean Markdown Output**: Perfect for RAG systems and AI processing
- **Smart Content Extraction**: Removes ads, navigation, and focuses on main content
- **Async Processing**: Handle multiple URLs simultaneously
- **Browser Automation**: Full JavaScript rendering support
- **Customizable Extraction**: Configure what content to extract

## Installation and Setup

### Prerequisites
- Python 3.8 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### 1. Install Dependencies

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
pip install -U crawl4ai
crawl4ai-setup
```

### 2. Verify Installation

```bash
crawl4ai-doctor
```

If you encounter browser issues:
```bash
python -m playwright install --with-deps chromium
```

### 3. Run the App

```bash
uv run python main.py
# or with pip: python main.py
```

## Project Structure

```
crawl-for-ai/
├── main.py              # Main crawling application
└── README.md
```

## Usage

1. Run the application
2. The crawler will extract content from NBC Sports
3. View the clean Markdown output in your terminal
4. Modify the URL in main.py to crawl different websites

## Example Output

The crawler converts complex HTML into clean Markdown:

```markdown
# NBC Sports

## Latest Sports News

- Basketball: Lakers win championship
- Football: Season highlights
- Baseball: World Series updates
```

Perfect for feeding into AI models, RAG systems, or data analysis pipelines.

---

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.