# LinkedIn Post Analysis with FastEmbed 🔍

Analyze and find similar LinkedIn posts using semantic embeddings - perfect for content creators, marketers, and professionals looking to understand content similarity and engagement patterns.

## Why This Matters

- 🧠 **Semantic Understanding**: Goes beyond keywords to understand actual meaning
- ⚡ **Lightning Fast**: Optimized embedding generation with minimal overhead
- 🔒 **Privacy First**: All processing happens locally - your content stays private
- 💼 **Professional Focus**: Tailored for LinkedIn and business content analysis
- 💰 **Cost Free**: No API costs - analyze unlimited content locally

## Key Features

- **Semantic Similarity**: Find posts with similar meaning, not just keywords
- **Content Clustering**: Group similar posts for trend analysis
- **Engagement Prediction**: Identify content patterns that perform well
- **Fast Processing**: Generate embeddings in milliseconds
- **Easy Integration**: Simple API for any text analysis workflow

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
pip install fastembed numpy
```

### 2. Run the App

```bash
uv run python main.py
# or with pip: python main.py
```

## Project Structure

```
embeddings-with-fastembed/
├── main.py              # Main embedding analysis application
└── README.md
```

## Usage

1. Run the application
2. The system analyzes 5 sample LinkedIn posts
3. Query for similar content using semantic search
4. View similarity scores and matching posts
5. Modify the documents list to analyze your own content

## Example Output

```
FastEmbed model ready for LinkedIn content analysis!
Generated 5 embeddings with 384 dimensions each

Query: 'Building an AI company with venture capital funding'

Most similar LinkedIn posts:
   Similarity: 0.847 - Just launched my new AI startup focused on revolutionizing...
   Similarity: 0.723 - Thrilled to announce our Series A funding round led by...
```

Perfect for content strategy, competitor analysis, and understanding what resonates with your professional network.

---

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.