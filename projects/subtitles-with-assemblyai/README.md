# AI Subtitle Generator with AssemblyAI 🎬

Transform any audio or video into professional subtitles instantly - upload, process, and download in seconds with advanced AI speech recognition.

## Why This Matters

- 🎤 **AssemblyAI Integration**: Powered by AssemblyAI's advanced speech recognition API
- 👥 **Speaker Detection**: Automatically identifies and labels different speakers
- 🔒 **Privacy First**: Secure API-based processing with local file handling
- ⚡ **Lightning Fast**: Get professional subtitles in under 60 seconds
- 💰 **Cost Effective**: AssemblyAI's competitive pricing for high-quality transcription

## Key Features

- **Multi-Format Support**: Upload MP3, WAV, MP4, MOV, AVI, M4A, FLAC files
- **Speaker Identification**: Distinguishes between different speakers automatically
- **Multiple Export Formats**: Download as SRT, VTT, or plain text
- **Real-time Processing**: Visual feedback during AI transcription
- **Dark Theme Compatible**: Beautiful UI that works in light and dark modes
- **Drag & Drop Interface**: Intuitive file upload experience

## Installation and Setup

### Prerequisites
- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- AssemblyAI API key

### 1. Get AssemblyAI API Key

1. Sign up at [AssemblyAI Console](https://www.assemblyai.com/)
2. Create an API key
3. Create a `.env` file in the project root:

```bash
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
```

### 2. Install Dependencies

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
pip install streamlit assemblyai python-dotenv
```

### 3. Run the App

```bash
uv run streamlit run main.py
# or with pip: streamlit run main.py
```

The app will be available at `http://localhost:8501` (or 8502 if 8501 is busy).

## Project Structure

```
subtitles-with-assemblyai/
├── main.py              # Main Streamlit application
├── .env                 # Environment variables (create this)
├── pyproject.toml       # Project dependencies
├── uv.lock             # Dependency lock file
└── README.md
```

## Usage

1. Start the application
2. Upload your audio or video file using the drag & drop interface
3. Toggle speaker detection on/off based on your needs
4. Select your preferred export format (SRT, VTT, or TXT)
5. Click "Generate Subtitles" and wait for AI processing
6. Preview your subtitles and download the file
7. View processing statistics (word count, duration, speakers detected)

## Supported File Formats

**Audio Files:**
- MP3, WAV, M4A, FLAC

**Video Files:**
- MP4, MOV, AVI

## Export Formats

- **SRT**: Standard subtitle format for video editing software
- **VTT**: Web-compatible subtitle format for HTML5 video
- **TXT**: Plain text transcript without timestamps

---

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.