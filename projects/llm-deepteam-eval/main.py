from deepteam import red_team
from deepteam.vulnerabilities import PIILeakage
from deepteam.attacks.single_turn import PromptInjection
import google.generativeai as genai
import dotenv

dotenv.load_dotenv()

async def model_callback(input: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(input)
    try:
        return response.text
    except Exception:
        return "Error: Unable to get response from Gemini."

# Define vulnerability and attack
pii_leakage = PIILeakage(types=["direct disclosure"])
prompt_injection = PromptInjection()

# Run red teaming
risk_assessment = red_team(
    model_callback=model_callback,
    vulnerabilities=[pii_leakage],
    attacks=[prompt_injection]
)

print(risk_assessment)