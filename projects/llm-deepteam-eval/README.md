# LLM DeepTeam Eval 🛡️

Test your LLMs for vulnerabilities and adversarial attacks using DeepTeam and Gemini, with a simple Python workflow.

## Why This Matters

- 🛡️ **LLM Red Teaming**: Simulate real-world attacks and uncover hidden vulnerabilities
- 🤖 **Gemini Integration**: Uses Google Gemini for LLM evaluation
- 🔍 **DeepTeam Framework**: State-of-the-art adversarial testing and guardrails
- ⚡ **Local or Cloud**: Works with local or cloud LLMs
- 📊 **Risk Assessment**: Automated reporting of vulnerabilities

## Key Features

- **40+ Vulnerabilities**: Bias, PII leakage, misinformation, robustness, and more
- **10+ Attack Methods**: Prompt injection, jailbreaking, leetspeak, and more
- **Easy Integration**: Just define your model callback and run
- **OWASP & NIST Support**: Out-of-the-box compliance checks

## Installation and Setup

### Prerequisites
- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Google Gemini API key ([get one here](https://aistudio.google.com/))

### 1. Install Dependencies

**Using uv (recommended):**

**Using pip:**
```bash
pip install deepeval deepteam google-generativeai
```

### 2. Set Up Gemini API Key

Set your API key in your environment:
```bash
export GOOGLE_API_KEY="your-gemini-api-key"
```
Or add it to a `.env` file:
```
GOOGLE_API_KEY=your-gemini-api-key
```

## Project Structure
```
llm-deepteam-eval/
├── main.py              # Main red teaming script
├── pyproject.toml       # Project dependencies
└── README.md            # Project documentation
### 3. Configure Gemini Model for DeepEval

You can set the Gemini model and API key using the DeepEval CLI:
```bash
deepeval set-gemini --model-name="gemini-2.5-flash-lite" --google-api-key="your-gemini-api-key"
```
Replace `your-gemini-api-key` with your actual key.
```

## Usage

1. Edit `main.py` to define your LLM callback and select vulnerabilities/attacks
2. Run the script:
   ```bash
   python main.py
   ```
3. View the printed risk assessment results


## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.
