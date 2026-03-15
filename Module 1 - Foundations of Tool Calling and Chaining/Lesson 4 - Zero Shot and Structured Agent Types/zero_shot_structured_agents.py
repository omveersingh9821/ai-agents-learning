"""
Lesson 4 - Zero Shot and Structured Agent Types
=================================================

This module demonstrates two types of LangChain agents:
1. zero-shot-react-description       — works with simple string-input tools
2. structured-chat-zero-shot-react-description — works with structured (multi-input) tools

Both agents use the ReAct (Reason + Act) pattern and are invoked with
verbose=True so you can see every reasoning step the LLM takes.
"""

# ──────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────────────
import os
import re
from dotenv import load_dotenv  # pyre-ignore[21]

# LangChain core utilities
from langchain.agents import initialize_agent, AgentType  # pyre-ignore[21]
from langchain_core.tools import tool, StructuredTool  # pyre-ignore[21]
from pydantic import BaseModel, Field  # pyre-ignore[21]

# LLM provider – using Google Generative AI (Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI  # pyre-ignore[21]

# ──────────────────────────────────────────────────────────────────────
# 2. ENVIRONMENT SETUP
# ──────────────────────────────────────────────────────────────────────
# Load API keys from .env (expects GOOGLE_API_KEY to be set)
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY not found. "
        "Please set it in your .env file or environment variables."
    )

# ──────────────────────────────────────────────────────────────────────
# 3. LLM INITIALIZATION
# ──────────────────────────────────────────────────────────────────────
# We'll use Google Gemini as the backbone LLM for both agents.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,            # deterministic output for reproducibility
    convert_system_message_to_human=True,
)

# ======================================================================
#  PART A — ZERO-SHOT-REACT-DESCRIPTION AGENT
# ======================================================================
# This agent type works with *simple* tools that accept a single string
# input.  The agent reads each tool's description and decides, in a
# zero-shot manner, which tool to call and what input to pass.
# ======================================================================

# ──────────────────────────────────────────────────────────────────────
# 4A. TOOL DEFINITION — Simple (string-input) tool
# ──────────────────────────────────────────────────────────────────────
@tool
def add_numbers(input_text: str) -> str:
    """
    Extracts all numbers (integers and decimals) from the input text
    and returns their sum.  Pass a comma-separated list of numbers or
    a natural-language sentence containing numbers.

    Example inputs:
        "10, 20, 30"
        "What is 5 plus 3.5?"
    """
    # Use regex to find all integers and floating-point numbers
    numbers = re.findall(r"-?\d+\.?\d*", input_text)
    if not numbers:
        return "No numbers found in the input."

    float_numbers = [float(n) for n in numbers]
    total = sum(float_numbers)
    return f"The sum of {float_numbers} is {total}"


# ──────────────────────────────────────────────────────────────────────
# 5A. AGENT INITIALIZATION — zero-shot-react-description
# ──────────────────────────────────────────────────────────────────────
# The agent is given a list of tools and uses the tool descriptions
# to decide which tool to use (zero-shot, no few-shot examples).
zero_shot_agent = initialize_agent(
    tools=[add_numbers],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,              # show reasoning steps in the console
    handle_parsing_errors=True,  # gracefully handle LLM output parsing issues
)


# ======================================================================
#  PART B — STRUCTURED-CHAT-ZERO-SHOT-REACT-DESCRIPTION AGENT
# ======================================================================
# This agent type can handle *structured* tools — tools that accept
# multiple typed inputs (not just a single string).  It uses a chat-
# based prompt format and can call tools with complex argument schemas.
# ======================================================================

# ──────────────────────────────────────────────────────────────────────
# 4B. TOOL DEFINITION — Structured (multi-input) tool
# ──────────────────────────────────────────────────────────────────────

# First, define a Pydantic schema describing the tool's inputs.
class AddNumbersInput(BaseModel):
    """Input schema for the structured add_numbers_with_options tool."""
    numbers: list[float] = Field(
        description="A list of numbers to sum together, e.g. [1, 2, 3]."
    )
    absolute: bool = Field(
        default=False,
        description=(
            "If True, take the absolute value of each number before summing. "
            "Useful when you want to ignore negative signs."
        ),
    )


def _add_numbers_with_options(numbers: list[float], absolute: bool = False) -> str:
    """
    Sums a list of numbers.  If `absolute` is True, each number is
    converted to its absolute value before summing.
    """
    if absolute:
        processed = [abs(n) for n in numbers]
        total = sum(processed)
        return (
            f"Taking absolute values: {processed}\n"
            f"The sum is {total}"
        )
    else:
        total = sum(numbers)
        return f"The sum of {list(numbers)} is {total}"


# Wrap the function as a StructuredTool so LangChain knows the schema.
add_numbers_with_options = StructuredTool.from_function(
    func=_add_numbers_with_options,
    name="add_numbers_with_options",
    description=(
        "Adds a list of numbers together. "
        "Accepts two arguments: `numbers` (a list of floats) and "
        "`absolute` (a boolean — set to True to use absolute values)."
    ),
    args_schema=AddNumbersInput,
)

# ──────────────────────────────────────────────────────────────────────
# 5B. AGENT INITIALIZATION — structured-chat-zero-shot-react-description
# ──────────────────────────────────────────────────────────────────────
structured_agent = initialize_agent(
    tools=[add_numbers_with_options],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)


# ──────────────────────────────────────────────────────────────────────
# 6. RUNNING THE AGENTS
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Agent 1: Zero-Shot ReAct Agent ──────────────────────────────
    print("=" * 70)
    print("  AGENT 1 — Zero-Shot ReAct Description Agent")
    print("=" * 70)

    prompt_1 = "What is the sum of 10, 20 and 30?"
    print(f"\n📝 Prompt: {prompt_1}\n")
    response_1 = zero_shot_agent.invoke({"input": prompt_1})
    print(f"\n✅ Final Answer: {response_1['output']}\n")

    # ── Agent 2: Structured Chat Zero-Shot Agent ────────────────────
    print("=" * 70)
    print("  AGENT 2 — Structured Chat Zero-Shot ReAct Agent")
    print("=" * 70)

    prompt_2 = "Add the numbers [-3, -5, -7] using absolute values."
    print(f"\n📝 Prompt: {prompt_2}\n")
    response_2 = structured_agent.invoke({"input": prompt_2})
    print(f"\n✅ Final Answer: {response_2['output']}\n")

    # ── Bonus: Structured agent WITHOUT absolute values ─────────────
    print("=" * 70)
    print("  BONUS — Structured Agent (absolute=False)")
    print("=" * 70)

    prompt_3 = "What is the sum of 100, 200, and 300?"
    print(f"\n📝 Prompt: {prompt_3}\n")
    response_3 = structured_agent.invoke({"input": prompt_3})
    print(f"\n✅ Final Answer: {response_3['output']}\n")
