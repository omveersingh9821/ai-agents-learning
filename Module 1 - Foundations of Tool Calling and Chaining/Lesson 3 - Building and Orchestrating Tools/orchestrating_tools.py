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
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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


def search_knowledge(query: str) -> str:
    """Search the knowledge base for information on a topic."""
    query_lower = query.lower()
    results = []
    for topic, content in KNOWLEDGE_BASE.items():
        if any(word in topic for word in query_lower.split()):
            results.append({"topic": topic, "content": content})
    if results:
        return json.dumps({"results": results, "count": len(results)})
    return json.dumps({"results": [], "count": 0, "message": "No results found"})


def summarize_text(text: str, max_sentences: int = 2) -> str:
    """Summarize a given text to a specified number of sentences."""
    sentences = text.replace(". ", ".\n").split("\n")
    summary = ". ".join(sentences[:max_sentences]).strip()
    if not summary.endswith("."):
        summary += "."
    return json.dumps({"summary": summary, "original_length": len(text), "summary_length": len(summary)})


def save_note(title: str, content: str) -> str:
    """Save a research note to the notebook."""
    note = {"title": title, "content": content, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    notebook.append(note)
    return json.dumps({"status": "saved", "note_number": len(notebook), "title": title})


def list_notes() -> str:
    """List all saved research notes."""
    if not notebook:
        return json.dumps({"notes": [], "message": "No notes saved yet"})
    return json.dumps({"notes": [{"number": i + 1, "title": n["title"], "timestamp": n["timestamp"]} for i, n in enumerate(notebook)], "total": len(notebook)})


# ─────────────────────────────────────────────
# 2. Tool schemas
# ─────────────────────────────────────────────

TOOLS = [
    {"type": "function", "function": {"name": "search_knowledge", "description": "Search the knowledge base for information on AI/tech topics.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "summarize_text", "description": "Summarize a piece of text to fewer sentences.", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "Text to summarize"}, "max_sentences": {"type": "integer", "description": "Max sentences in summary (default 2)"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "save_note", "description": "Save a research note with a title and content.", "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "Note title"}, "content": {"type": "string", "description": "Note content"}}, "required": ["title", "content"]}}},
    {"type": "function", "function": {"name": "list_notes", "description": "List all saved research notes.", "parameters": {"type": "object", "properties": {}}}},
]

TOOL_MAP = {"search_knowledge": search_knowledge, "summarize_text": summarize_text, "save_note": save_note, "list_notes": list_notes}


# ─────────────────────────────────────────────
# 3. The Agentic Loop — keeps going until done
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

    messages = [
        {"role": "system", "content": "You are a research assistant with access to a knowledge base, a summarizer, and a notebook. When the user asks you to research a topic:\n1. Search the knowledge base\n2. Summarize the findings\n3. Save important notes\n4. Provide a comprehensive answer\nYou can chain multiple tool calls to complete complex tasks."},
        {"role": "user", "content": question},
    ]

    for iteration in range(max_iterations):
        print(f"\n  ⚙️  Iteration {iteration + 1}/{max_iterations}")

        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=TOOLS, tool_choice="auto", temperature=0)
        assistant_msg = response.choices[0].message

        if not assistant_msg.tool_calls:
            print(f"\n  ✅ Agent finished after {iteration + 1} iteration(s)")
            print(f"\n  🤖 Agent: {assistant_msg.content}")
            return assistant_msg.content

        messages.append(assistant_msg)
        for tool_call in assistant_msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            print(f"     🔧 {func_name}({json.dumps(func_args)[:80]}...)")
            if func_name in TOOL_MAP:
                result = TOOL_MAP[func_name](**func_args) if func_args else TOOL_MAP[func_name]()
            else:
                result = json.dumps({"error": f"Unknown tool: {func_name}"})
            print(f"     📤 → {result[:100]}...")
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    return "Max iterations reached."


# ─────────────────────────────────────────────
# 4. Demo
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
