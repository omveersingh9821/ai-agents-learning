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
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


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

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(question)

    answer = response.text
    print(f"Q: {question}")
    print(f"A: {answer}\n")
    return answer


# ─────────────────────────────────────────────
# 2. A simple "agent" that has a tool (calculator)
# ─────────────────────────────────────────────

def calculator(expression: str) -> str:
    """Safely evaluate a math expression and return the result.

    Args:
        expression: The math expression to evaluate, e.g. '2+2', '15*7'
    """
    try:
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


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

    # Gemini can take Python functions directly as tools!
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        tools=[calculator],
    )

    chat = model.start_chat()
    response = chat.send_message(question)

    # Check if the model wants to call a tool
    function_call = None
    for part in response.parts:
        if hasattr(part, "function_call") and part.function_call.name:
            function_call = part.function_call
            break

    if function_call:
        func_name = function_call.name
        func_args = dict(function_call.args)

        print(f"📋  Agent decided to use a tool:\n")
        print(f"   🔧 Tool   : {func_name}")
        print(f"   📥 Input  : {func_args}")

        # Step 3: Execute the tool
        result = calculator(**func_args)
        print(f"   📤 Output : {result}\n")

        # Step 4: Feed the result back to the LLM
        response = chat.send_message(
            genai.protos.Content(
                parts=[genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=func_name,
                        response={"result": result},
                    )
                )]
            )
        )

    answer = response.text
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
