import parlant.sdk as p
import asyncio
import os
from datetime import datetime
import dotenv

dotenv.load_dotenv()

# Set Ollama environment variables
# os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
# os.environ["OLLAMA_MODEL"] = "gemma3:4b"
# os.environ["OLLAMA_EMBEDDING_MODEL"] = "nomic-embed-text:latest"
# os.environ["OLLAMA_API_TIMEOUT"] = "300"


@p.tool
async def get_upcoming_slots(context: p.ToolContext) -> p.ToolResult:
  # Simulate fetching available times from a database or API
  return p.ToolResult(data=["Monday 10 AM", "Tuesday 2 PM", "Wednesday 1 PM"])

@p.tool
async def get_later_slots(context: p.ToolContext) -> p.ToolResult:
  # Simulate fetching later available times
  return p.ToolResult(data=["November 3, 11:30 AM", "November 12, 3 PM"])

@p.tool
async def schedule_appointment(context: p.ToolContext, datetime: datetime) -> p.ToolResult:
  # Simulate scheduling the appointment
  return p.ToolResult(data=f"Appointment scheduled for {datetime}")

async def add_domain_glossary(agent: p.Agent) -> None:
  await agent.create_term(
    name="Office Phone Number",
    description="The phone number of our office, at +1-234-567-8900",
  )

  await agent.create_term(
    name="Office Hours",
    description="Office hours are Monday to Friday, 9 AM to 5 PM",
  )

  await agent.create_term(
    name="Charles Xavier",
    synonyms=["Professor X"],
    description="The renowned doctor who specializes in neurology",
  )


# Journey to schedule an appointment
async def create_scheduling_journey(server: p.Server, agent: p.Agent) -> p.Journey:
  # Create the journey
  journey = await agent.create_journey(
    title="Schedule an Appointment",
    description="Helps the patient find a time for their appointment.",
    conditions=["The patient wants to schedule an appointment"],
  )

  # First, determine the reason for the appointment
  t0 = await journey.initial_state.transition_to(chat_state="Determine the reason for the visit")

  # Load upcoming appointment slots into context
  t1 = await t0.target.transition_to(tool_state=get_upcoming_slots)

  # Ask which one works for them
  # We will transition conditionally from here based on the patient's response
  t2 = await t1.target.transition_to(chat_state="List available times and ask which ones works for them")

  # We'll start with the happy path where the patient picks a time
  t3 = await t2.target.transition_to(
    chat_state="Confirm the details with the patient before scheduling",
    condition="The patient picks a time",
  )

  t4 = await t3.target.transition_to(
    tool_state=schedule_appointment,
    condition="The patient confirms the details",
  )
  t5 = await t4.target.transition_to(chat_state="Confirm the appointment has been scheduled")
  await t5.target.transition_to(state=p.END_JOURNEY)

  # Otherwise, if they say none of the times work, ask for later slots
  t6 = await t2.target.transition_to(
    tool_state=get_later_slots,
    condition="None of those times work for the patient",
  )
  t7 = await t6.target.transition_to(chat_state="List later times and ask if any of them works")

  # Transition back to our happy-path if they pick a time
  await t7.target.transition_to(state=t3.target, condition="The patient picks a time")

  # Otherwise, ask them to call the office
  t8 = await t7.target.transition_to(
    chat_state="Ask the patient to call the office to schedule an appointment",
    condition="None of those times work for the patient either",
  )
  await t8.target.transition_to(state=p.END_JOURNEY)

  await journey.create_guideline(
    condition="The patient says their visit is urgent",
    action="Tell them to call the office immediately",
  )

  return journey

async def main() -> None:
    async with p.Server(nlp_service=p.NLPServices.gemini) as server:
        agent = await server.create_agent(
            name="Healthcare Agent",
            description="Is empathetic and calming to the patient.",
        )

        await add_domain_glossary(agent)

        scheduling_journey = await create_scheduling_journey(server, agent)


if __name__ == "__main__":
    asyncio.run(main())