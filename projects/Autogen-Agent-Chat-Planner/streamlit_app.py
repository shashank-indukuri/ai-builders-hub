import os
import io
import datetime
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

try:
    import autogen
except Exception as e:
    autogen = None
    _import_error = e
else:
    _import_error = None

# Load environment variables from .env if present
load_dotenv()


# -------------------------
# Page
# -------------------------
st.set_page_config(
    page_title="AutoGen Agent Chat Planner",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AutoGen Agent Chat: Assistant + Planner")


# -------------------------
# Helpers
# -------------------------
def str2bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_config_list_from_env() -> list:
    """Gemini-only config list from environment variables.

    Required: GEMINI_API_KEY
    Optional: GEMINI_MODEL (default: gemini-1.5-pro), GEMINI_BASE_URL
    """
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        return []
    model = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
    base_url = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    return [
        {
            "model": model,
            "api_key": gemini_key,
            "base_url": base_url,
        }
    ]


def build_agents(config_list, temperature: float, max_auto_replies: int, use_docker: bool):
    """Create planner, assistant, and user_proxy agents (Gemini-compatible, no function-calling)."""
    # Planner
    planner = autogen.AssistantAgent(
        name="planner",
        llm_config={"config_list": config_list},
        system_message=(
            "You are a helpful AI assistant. You suggest coding and reasoning steps for another AI assistant to "
            "accomplish a task. Do not suggest concrete code. For any action beyond writing code or reasoning, "
            "convert it to a step that can be implemented by writing code. For example, browsing the web can be "
            "implemented by writing code that reads and prints the content of a web page. Finally, inspect the "
            "execution result. If the plan is not good, suggest a better plan. If the execution is wrong, analyze "
            "the error and suggest a fix."
        ),
    )
    planner_user = autogen.UserProxyAgent(
        name="planner_user",
        max_consecutive_auto_reply=0,
        human_input_mode="NEVER",
        code_execution_config={"use_docker": use_docker},
    )

    # Assistant (no functions)
    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config={
            "temperature": temperature,
            "timeout": 600,
            "cache_seed": 42,
            "config_list": config_list,
        },
    )

    # User Proxy that executes code suggested by Assistant
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=max_auto_replies,
        code_execution_config={"work_dir": "planning", "use_docker": use_docker},
    )

    return planner, planner_user, assistant, user_proxy


def run_conversation(planner, planner_user, assistant, user_proxy, task_message: str) -> Tuple[str, Optional[str]]:
    """Two-step run: (1) ask planner for a plan, (2) run assistant with the plan as context."""
    buf_out, buf_err = io.StringIO(), io.StringIO()
    plan_text: str = ""
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        # Step 1: Planner suggests a plan
        planner_prompt = (
            "Propose a concise plan to accomplish the following task. "
            "Focus on coding steps and reasoning.\n\n"
            f"Task: {task_message}"
        )
        planner_user.initiate_chat(planner, message=planner_prompt)
        try:
            plan_text = planner_user.last_message().get("content") or ""
        except Exception:
            plan_text = ""

        # Step 2: Assistant executes with plan context
        assistant_prompt = (
            f"Task: {task_message}\n\n"
            "Plan from planner (for your reference):\n"
            f"{plan_text}\n\n"
            "Please implement the plan by writing Python code blocks when needed. "
            "After executing the code, analyze results and iterate if necessary."
        )
        user_proxy.initiate_chat(assistant, message=assistant_prompt)

    logs = buf_out.getvalue() + ("\n" + buf_err.getvalue() if buf_err.getvalue() else "")
    try:
        last = user_proxy.last_message().get("content")
    except Exception:
        last = None
    return logs, last


# -------------------------
# Main UI
# -------------------------
st.subheader("Task")
task = st.text_area(
    "Describe the task for the Assistant (it may write code and consult the Planner)",
    height=120,
    value="Suggest a fix to an open good first issue of flaml",
)
run_btn = st.button("Run Multi-Agent Chat", type="primary")

st.markdown("---")

if run_btn:
    if autogen is None:
        st.error("autogen is not installed. Please install dependencies and try again.")
        st.stop()

    # Read runtime knobs from env only
    temperature = float(os.environ.get("TEMPERATURE", "0"))
    max_auto_replies = int(os.environ.get("MAX_AUTO_REPLIES", "10"))
    use_docker = str2bool(os.environ.get("USE_DOCKER", "false"))

    cfg = build_config_list_from_env()
    if not cfg:
        st.error("No model configuration found. Set GEMINI_API_KEY (and optionally GEMINI_MODEL/GEMINI_BASE_URL) in .env.")
        st.stop()

    with st.spinner("Setting up agents..."):
        try:
            planner, planner_user, assistant, user_proxy = build_agents(cfg, temperature, max_auto_replies, use_docker)
        except Exception as e:
            st.exception(e)
            st.stop()

    with st.spinner("Running conversation. This can take a minute..."):
        logs, last = run_conversation(planner, planner_user, assistant, user_proxy, task)

    st.success("Conversation finished.")

    st.subheader("Conversation Log")
    st.code(logs or "(no logs)", language="text")

    if last:
        st.subheader("Assistant Final Message")
        st.markdown(last)

else:
    st.info("This minimal app uses Gemini via environment variables only. Set GEMINI_API_KEY before running.")
