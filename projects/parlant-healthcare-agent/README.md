# Parlant Healthcare Agent 🏥

Build reliable AI agents that follow business rules every single time - demonstrated through an end-to-end patient appointment scheduling system with natural-language guidelines and journey flows.

## Why This Matters

- 📋 **Guidelines Over Prompts**: Rules are enforced, not suggested - no prompt hacks required
- 🎯 **Predictable Behavior**: Eliminate "roll of the dice" LLM responses with structured journeys
- 🔧 **Tool-Ready**: Seamlessly integrate APIs and backend services at any step
- 🗺️ **Journey Mapping**: Define step-by-step conversational flows for customers
- 🛡️ **Built-in Guardrails**: Eliminate hallucinations and off-topic replies
- 🏥 **Healthcare-Ready**: Designed for regulated environments requiring compliance

## Key Features

- **Structured Appointment Scheduling**: Complete patient journey from inquiry to confirmation
- **Conditional Flow Logic**: Handle multiple scenarios (available slots, no availability, urgent cases)
- **Domain Glossary**: Define healthcare-specific terms and office information
- **Tool Integration**: Connect to scheduling systems and databases
- **Empathetic Responses**: Agent personality designed for healthcare interactions
- **Compliance-Ready**: Built for regulated healthcare communications

## Who Uses Parlant?

Parlant delivers complex conversational agents that reliably follow protocols in:
- 🏥 **Healthcare Communications** - Patient scheduling, triage, follow-ups
- 💰 **Regulated Financial Services** - Compliance-driven customer interactions
- ⚖️ **Legal Assistance** - Structured legal consultations and intake
- 🎯 **Brand-Sensitive Customer Service** - Consistent brand voice and policies
- 🏛️ **Government and Civil Services** - Standardized citizen interactions
- 🤝 **Personal Advocacy** - Structured representation workflows

## Installation and Setup

### Prerequisites
- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Google Gemini API key

### 1. Environment Setup

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Add your Google Gemini API key to `.env`:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 2. Install Dependencies

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
pip install parlant google-generativeai google-genai python-dotenv
```

### 3. Run the Agent

```bash
uv run python main.py
# or with pip: python main.py
```

## Project Structure

```
parlant-healthcare-agent/
├── main.py              # Main agent implementation
├── parlant-data/        # Agent data and conversation logs
├── .env.example         # Environment template
├── pyproject.toml       # Project dependencies
└── README.md
```

## How It Works

### 1. Journey Definition
The agent follows a structured appointment scheduling journey:
- **Determine Visit Reason** → Understand patient needs
- **Load Available Slots** → Fetch upcoming appointments
- **Present Options** → Show available times to patient
- **Handle Responses** → Process patient selection or alternatives
- **Confirm Details** → Verify appointment information
- **Schedule Appointment** → Complete the booking

### 2. Conditional Logic
The system handles multiple scenarios:
- ✅ Patient selects available time → Direct scheduling
- ❌ No suitable times → Show later availability
- 🚨 Urgent visit → Direct to office phone
- ❌ No times work → Escalate to phone scheduling

### 3. Domain Knowledge
Built-in glossary includes:
- Office contact information
- Business hours
- Staff details (Dr. Charles Xavier)
- Healthcare-specific terminology

## Usage Example

The agent can handle conversations like:

```
Patient: "I need to schedule an appointment"
Agent: "I'd be happy to help you schedule an appointment. Could you tell me the reason for your visit?"

Patient: "I need a follow-up for my headaches"
Agent: "I understand. Let me check our available appointment times. I have these slots available: Monday 10 AM, Tuesday 2 PM, Wednesday 1 PM. Which of these works best for you?"

Patient: "Tuesday 2 PM works"
Agent: "Perfect! Let me confirm - Tuesday at 2 PM for a headache follow-up. Is this correct?"

Patient: "Yes, that's right"
Agent: "Excellent! Your appointment has been scheduled for Tuesday at 2 PM. You'll receive a confirmation shortly."
```

## Extending the Agent

### Adding New Tools
```python
@p.tool
async def check_insurance(context: p.ToolContext, insurance_id: str) -> p.ToolResult:
    # Verify patient insurance coverage
    return p.ToolResult(data={"covered": True, "copay": "$25"})
```

### Adding Guidelines
```python
await journey.create_guideline(
    condition="The patient mentions pain level above 8",
    action="Prioritize urgent scheduling and suggest calling office immediately"
)
```

### Expanding the Journey
Add new states for insurance verification, symptom assessment, or post-appointment follow-up.

---

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.