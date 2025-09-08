"""
Example: Query your repository using MCP-discovered file tools.

Tools provided by examples/servers/files_server.py:
- list_dir(path, glob, max_items)
- read_file(path, start_line, end_line)
- grep(query, path, regex, case_insensitive, include, max_matches)

Run the server first:
  python examples/servers/files_server.py  # http://127.0.0.1:8001/mcp

Then run this agent:
  python examples/use_files_agent.py

It will choose Gemini if GEMINI_API_KEY/GOOGLE_API_KEY is set, otherwise OpenAI.
"""

import asyncio
import os
import json
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from deepmcpagent import HTTPServerSpec, build_deep_agent


def _extract_tool_text(content) -> str:
    """Best-effort extraction of readable text from ToolMessage.content.
    Handles str or list[dict-like] with possible 'text' or 'content' fields.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            try:
                if isinstance(part, dict):
                    if "text" in part and isinstance(part["text"], str):
                        parts.append(part["text"])
                    elif "content" in part:
                        val = part["content"]
                        if isinstance(val, (dict, list)):
                            parts.append(json.dumps(val, ensure_ascii=False))
                        else:
                            parts.append(str(val))
                    else:
                        parts.append(json.dumps(part, ensure_ascii=False))
                else:
                    parts.append(str(part))
            except Exception:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def _maybe_pretty_json(text: str) -> str:
    try:
        obj = json.loads(text)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return text


async def run_queries(console: Console, graph, queries: List[str]) -> None:
    for q in queries:
        console.print(Panel.fit(q, title="User Query", style="bold magenta"))
        result = await graph.ainvoke({"messages": [{"role": "user", "content": q}]})

        console.print("\n[bold yellow]Agent Trace:[/bold yellow]")
        for msg in result["messages"]:
            role = msg.__class__.__name__
            if role == "AIMessage" and getattr(msg, "tool_calls", None):
                for call in msg.tool_calls:
                    console.print(
                        f"[cyan]→ Invoking tool:[/cyan] [bold]{call['name']}[/bold] with {call['args']}"
                    )
            elif role == "ToolMessage":
                raw = _extract_tool_text(getattr(msg, "content", ""))
                pretty = _maybe_pretty_json(raw)
                console.print(
                    f"[green]✔ Tool result from {getattr(msg, 'name', '-') or '-'}:[/green]\n{pretty}"
                )
            elif role == "AIMessage" and msg.content:
                console.print(Panel(msg.content, title="Final LLM Answer", style="bold green"))
        console.print("\n")


async def main() -> None:
    console = Console()
    load_dotenv()

    # Discover file tools over HTTP from our local server (port 8001)
    servers = {
        "files": HTTPServerSpec(
            url="http://127.0.0.1:8001/mcp",
            transport="http",
        ),
    }

    # Select model: prefer Gemini if GEMINI_API_KEY/GOOGLE_API_KEY is set, else OpenAI
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        os.environ.setdefault("GOOGLE_API_KEY", gemini_key)
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=gemini_key)
    else:
        model = ChatOpenAI(model="gpt-4.1")

    graph, loader = await build_deep_agent(
        servers=servers,
        model=model,
        instructions=(
            "You are a helpful repo assistant. Use the MCP file tools precisely.\n"
        ),
    )

    # Show discovered tools
    infos = await loader.list_tool_info()
    table = Table(title="Discovered MCP Tools (Files Server)", show_lines=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")
    for t in infos:
        table.add_row(t.name, t.description or "-")
    console.print(table)

    # Sample queries that demonstrate the tools
    sample_queries = [
        "Search for examples/servers/files_server.py",
        "Show the first 60 lines of examples/servers/files_server.py"
    ]

    await run_queries(console, graph, sample_queries)


if __name__ == "__main__":
    asyncio.run(main())
