# Run GPT-OSS Locally with Thinking UI 🧠

Experience the power of GPT-OSS reasoning locally - see exactly how the model thinks through problems before giving you answers.

## Why This Matters

- 🌐 **Groq Integration**: Powered by Groq's fast inference with GPT-OSS model
- 🧠 **See The Thinking**: Watch the model's reasoning process unfold in expandable panels
- 🔒 **Privacy First**: Secure API-based inference with reasoning transparency
- ⚡ **GPT-OSS:20B**: Advanced reasoning capabilities with native thinking support
- 💰 **Cost Effective**: Groq's competitive pricing for high-performance inference

## Key Features

- **Thinking UI**: Expandable panels show model's reasoning process
- **Real-time Streaming**: See responses generate as the model thinks
- **Chat History**: Full conversation history with thinking processes preserved
- **Clean Interface**: Professional UI with OpenAI and Groq logos
- **Environment Variables**: Secure API key management with .env support

## Installation and Setup

### Prerequisites
- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Groq API key

### 1. Get Groq API Key

1. Sign up at [Groq Console](https://console.groq.com/)
2. Create an API key
3. Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

### 2. Install Dependencies

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
pip install streamlit groq python-dotenv
```

### 3. Run the App

```bash
uv run streamlit run main.py
# or with pip: streamlit run main.py
```

The app will be available at `http://localhost:8501` (or 8502 if 8501 is busy).

## Project Structure

```
gpt-oss-reasoning-os/
├── main.py              # Main Streamlit application
├── .env                 # Environment variables (create this)
├── assets/              # Logo assets
│   ├── openai.png
│   └── ollama.png
└── README.md
```

## Usage

1. Start the application
2. Type your message in the chat input
3. Watch the model's reasoning process in the "🧠 Thinking process" expandable section
4. View the final response in the main chat area
5. All conversations are preserved with thinking processes intact

---

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.