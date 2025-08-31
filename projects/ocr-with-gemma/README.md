# OCR with Gemma Vision 📄

Extract text from images locally using Gemma's powerful vision capabilities - transform any image containing text into structured, readable content.

## Why This Matters

- 🏠 **Local Processing**: Powered by Ollama's local inference with Gemma3:12b model
- 👁️ **Vision Intelligence**: Advanced OCR capabilities that understand context and structure
- 🔒 **Privacy First**: All processing happens locally - your images never leave your machine
- ⚡ **Gemma3:12b**: State-of-the-art vision model with superior text recognition
- 💰 **Cost Free**: No API costs - run unlimited OCR tasks locally

## Key Features

- **Smart Text Extraction**: Preserves original formatting and document structure
- **Multi-format Support**: Works with PNG, JPG, JPEG, and WebP images
- **Clean Interface**: Professional two-column layout for easy workflow
- **Instant Download**: Export extracted text as TXT files
- **Real-time Processing**: See extraction progress with visual feedback

## Installation and Setup

### Prerequisites
- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [Ollama](https://ollama.ai/) installed and running
- Gemma3:12b model downloaded

### 1. Install Ollama and Gemma Model

1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull the Gemma3:12b model:

```bash
ollama pull gemma3:12b
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
ocr-with-gemma/
├── main.py              # Main Streamlit application
└── README.md
```

## Usage

1. Start the application
2. Upload an image containing text using the file uploader
3. Click "🔍 Extract Text" to process the image
4. View the extracted text in the right panel
5. Copy the text or download it as a TXT file

---

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.