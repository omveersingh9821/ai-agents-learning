"""
===============================================================================
 MODULE 1 — LESSON 1: Introduction to AI Agents
===============================================================================
 Course : Fundamentals of Building AI Agents (IBM · Coursera)
 Topics : What is an AI agent, Perception → Reasoning → Action loop,
          Difference between LLMs and Agents, Types of agents

 This lesson introduces the core concepts:
   • An AI Agent = LLM + Tools + Memory + Orchestration
   • The Perception → Reasoning → Action (PRA) loop
   • Simple demo: a "no-tools" agent vs. a "tool-augmented" agent
===============================================================================
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─────────────────────────────────────────────
# 1. A plain LLM call (no tools — just text in, text out)
# ─────────────────────────────────────────────
def plain_llm_call(question: str) -> str:
    """
    A standard LLM call — the model can ONLY use its training data.
    It cannot look things up, run code, or call APIs.
    """
    print("=" * 60)
    print("🧠  PLAIN LLM (no tools)")
    print("=" * 60)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. If you don't know "
                    "something, say so honestly."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    answer = response.choices[0].message.content
    print(f"Q: {question}")
    print(f"A: {answer}\n")
    return answer


# ─────────────────────────────────────────────
# 2. A simple "agent" that has a tool (calculator)
# ─────────────────────────────────────────────

# Define a basic tool — a calculator
def calculator(expression: str) -> str:
    """Safely evaluate a math expression and return the result."""
    try:
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# Define the tool schema for OpenAI
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Examples: '2+2', '15*7', '100/4'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]

TOOL_MAP = {"calculator": calculator}


def agent_with_tools(question: str) -> str:
    """
    An AI Agent that follows the Perception → Reasoning → Action loop:
      1. PERCEIVE  — read the user's question
      2. REASON    — decide whether to use a tool
      3. ACT       — call the tool (or respond directly)
      4. OBSERVE   — feed the tool result back to the LLM
      5. RESPOND   — generate the final answer
    """
    print("=" * 60)
    print("🤖  AI AGENT (with tools)")
    print("=" * 60)
    print(f"Q: {question}\n")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI agent. You have access to a calculator tool. "
                "Use it whenever the user asks a math question. "
                "Always show your work."
            ),
        },
        {"role": "user", "content": question},
    ]

    # Step 1: Send the question to the LLM (with tool definitions)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0,
    )

    assistant_msg = response.choices[0].message

    # Step 2: Check if the model wants to call a tool
    if assistant_msg.tool_calls:
        print("📋  Agent decided to use tool(s):\n")
        messages.append(assistant_msg)

        for tool_call in assistant_msg.tool_calls:
            func_name = tool_call.function.name
            func_args = eval(tool_call.function.arguments)

            print(f"   🔧 Tool   : {func_name}")
            print(f"   📥 Input  : {func_args}")

            # Step 3: Execute the tool
            result = TOOL_MAP[func_name](**func_args)
            print(f"   📤 Output : {result}\n")

            # Step 4: Feed the result back to the LLM
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        # Step 5: Get the final answer
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
        )
        answer = final_response.choices[0].message.content
    else:
        answer = assistant_msg.content

    print(f"A: {answer}\n")
    return answer


# ─────────────────────────────────────────────
# 3. Compare: LLM vs Agent
# ─────────────────────────────────────────────
def main():
    question = "What is 4567 * 8923 + 12345?"

    print("\n" + "━" * 60)
    print("  COMPARING: Plain LLM vs. AI Agent")
    print("━" * 60 + "\n")

    plain_llm_call(question)
    agent_with_tools(question)
    agent_with_tools("What is the capital of France?")


if __name__ == "__main__":
    main()
