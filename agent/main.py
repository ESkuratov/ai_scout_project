import os
import re
from typing import Any, Dict, List
from dotenv import load_dotenv
import asyncio
from datetime import datetime
from rich.console import Console
from rich.progress import track

from context import Context
from langgraph.runtime import Runtime
from graph import react_graph
from langchain_core.messages import HumanMessage

from data.case_repository import CaseRepository
from data.database import SessionLocal
from services.case_parser import _persist_cases,parse_agent_output




async def main() -> None:
    # Load environment variables first
    load_dotenv()

    # Configure the agent's runtime context
    ctx = Context(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("OPENROUTER_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1",
        max_search_results=10,
    )

    console = Console()
    console.print("[bold cyan]Agent is thinking...[/bold cyan]")

    # Using track for a simulated progress bar
    # In a real scenario, you'd integrate this with actual steps of your agent's work
    for step in track(range(10), description="Generating scenarios..."):
        if step == 0:  # Only call the ainvoke once
            result = await react_graph.ainvoke(
                {
                    "messages": [
                        (
                            "user",
                            "Find 10 interesting or best new scenarios for the use and implementation of Generative AI in Russia, China, USA and worldwide in 2025, considering the economic effect when selecting cases. Focus on the following sectors: manufacturing, finance, medicine, education, government, and cybersecurity. Please provide separate lists for Russia, China, USA, and the worldwide group. Focus only on Generative AI and scenarios specifically for 2025.",
                        )
                    ]
                },
                context=ctx,
            )
        await asyncio.sleep(0.5)  # Simulate work between "steps"

    console.print(":sparkles: Done!", style="green")

    agent_output = result["messages"][-1].content
    print("Agent output:\n", agent_output)

    # Save the output to a file
    with open("agent_output.txt", "w") as f:
        f.write("\n\n--------------------\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("--------------------\n\n")
        f.write(agent_output)
        f.write("--------------------\n\n")
    print("\nAgent output saved to agent_output.txt")

    _persist_cases(agent_output)
    payloads = parse_agent_output(agent_output)
    console.print(f"[bold green]Parsed {len(payloads)} cases[/bold green]")
    for case in payloads[:3]:
        console.print(f"- {case['case']['title']}")
    _persist_cases(agent_output)


if __name__ == "__main__":
    asyncio.run(main())
