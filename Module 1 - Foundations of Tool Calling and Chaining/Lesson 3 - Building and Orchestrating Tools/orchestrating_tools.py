"""
===============================================================================
 MODULE 1 — LESSON 3: Building and Orchestrating Tools
===============================================================================
 Course : Fundamentals of Building AI Agents (IBM · Coursera)
 Topics : Tool chaining, multi-step workflows, agentic loop,
          building a research assistant that chains tool calls

 Key concepts:
   • Tool Chaining   — output of one tool becomes input to another
   • Agentic Loop    — the agent keeps calling tools until the task is done
   • Orchestration   — coordinating multiple tools in sequence
===============================================================================
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv  # pyre-ignore[21]
# pyre-ignore[21]: google.generativeai is installed in venv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# ─────────────────────────────────────────────
# 1. Define a richer set of tools
# ─────────────────────────────────────────────

KNOWLEDGE_BASE = {
    "ai agents": "AI agents are autonomous systems that perceive their environment, reason about it, and take actions to achieve goals. They combine LLMs with tools, memory, and planning capabilities. Key types include reactive agents, deliberative agents, and hybrid agents.",
    "langchain": "LangChain is a framework for building applications powered by large language models. It provides abstractions for chains, agents, memory, and tool use. Key components include LCEL, agents, retrievers, and vector stores.",
    "tool calling": "Tool calling (or function calling) allows LLMs to interact with external systems by generating structured function calls. The LLM outputs the function name and arguments; the application executes the function and returns the result.",
    "rag": "Retrieval Augmented Generation (RAG) enhances LLM responses by first retrieving relevant documents from a knowledge base, then using those documents as context for generation. This reduces hallucination and keeps responses grounded.",
}

notebook = []


def search_knowledge(query: str) -> dict:
    """Search the knowledge base for information on AI/tech topics.

    Args:
        query: Search query string
    """
    query_lower = query.lower()
    results = []
    for topic, content in KNOWLEDGE_BASE.items():
        if any(word in topic for word in query_lower.split()):
            results.append({"topic": topic, "content": content})
    if results:
        return {"results": results, "count": len(results)}
    return {"results": [], "count": 0, "message": "No results found"}


def summarize_text(text: str, max_sentences: int = 2) -> dict:
    """Summarize a given text to a specified number of sentences.

    Args:
        text: Text to summarize
        max_sentences: Max sentences in the summary (default 2)
    """
    sentences = text.replace(". ", ".\n").split("\n")
    summary = ". ".join(sentences[:max_sentences]).strip()  # pyre-ignore[6]
    if not summary.endswith("."):
        summary += "."
    return {"summary": summary, "original_length": len(text), "summary_length": len(summary)}


def save_note(title: str, content: str) -> dict:
    """Save a research note with a title and content.

    Args:
        title: Note title
        content: Note content
    """
    note = {"title": title, "content": content, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    notebook.append(note)
    return {"status": "saved", "note_number": len(notebook), "title": title}


def list_notes() -> dict:
    """List all saved research notes."""
    if not notebook:
        return {"notes": [], "message": "No notes saved yet"}
    return {"notes": [{"number": i + 1, "title": n["title"], "timestamp": n["timestamp"]}
                      for i, n in enumerate(notebook)], "total": len(notebook)}


ALL_TOOLS = [search_knowledge, summarize_text, save_note, list_notes]
TOOL_MAP = {fn.__name__: fn for fn in ALL_TOOLS}


# ─────────────────────────────────────────────
# 2. The Agentic Loop — keeps going until done
# ─────────────────────────────────────────────

def run_agent_loop(question: str, max_iterations: int = 5) -> str:
    """
    An agentic loop that keeps calling tools until the agent is done.
    This is the key pattern for tool CHAINING:
      1. User asks a complex question
      2. Agent calls tool #1 → gets result
      3. Agent reasons, calls tool #2 → gets result
      4. Agent continues until it has enough info
      5. Agent provides the final answer
    """
    print("\n" + "━" * 60)
    print(f"  🗣️  User: {question}")
    print("━" * 60)

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        tools=ALL_TOOLS,
        system_instruction=(
            "You are a research assistant with access to a knowledge base, "
            "a summarizer, and a notebook. When the user asks you to research "
            "a topic:\n1. Search the knowledge base\n"
            "2. Summarize the findings\n3. Save important notes\n"
            "4. Provide a comprehensive answer\n"
            "You can chain multiple tool calls to complete complex tasks."
        ),
    )

    chat = model.start_chat()
    response = chat.send_message(question)

    for iteration in range(max_iterations):
        print(f"\n  ⚙️  Iteration {iteration + 1}/{max_iterations}")

        # Check for function calls
        function_calls = [part.function_call for part in response.parts if hasattr(part, "function_call") and part.function_call.name]  # pyre-ignore[16]

        # If no tool calls → agent is done
        if not function_calls:
            print(f"\n  ✅ Agent finished after {iteration + 1} iteration(s)")
            print(f"\n  🤖 Agent: {response.text}")
            return response.text

        # Execute all function calls
        function_responses = []
        for fc in function_calls:
            func_name = fc.name
            func_args = dict(fc.args)
            print(f"     🔧 {func_name}({json.dumps(func_args)[:80]}...)")  # pyre-ignore[6]

            if func_name in TOOL_MAP:
                result = TOOL_MAP[func_name](**func_args)  # pyre-ignore[6]
            else:
                result = {"error": f"Unknown tool: {func_name}"}
            print(f"     📤 → {str(result)[:100]}...")  # pyre-ignore[6]

            function_responses.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=func_name, response={"result": result}
                    )
                )
            )

        # Send results back and get next response
        response = chat.send_message(
            genai.protos.Content(parts=function_responses)
        )

    return "Max iterations reached."


# ─────────────────────────────────────────────
# 3. Demo
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  MODULE 1 · LESSON 3: Building and Orchestrating Tools")
    print("=" * 60)

    run_agent_loop("What are AI agents?")
    run_agent_loop("Research the topic of 'tool calling' in AI. Summarize what you find and save it as a note titled 'Tool Calling Summary'.")
    run_agent_loop("I need to study both RAG and LangChain. Search for each, summarize both, and save notes for each topic.")
    run_agent_loop("Show me all my saved research notes.")


if __name__ == "__main__":
    main()
