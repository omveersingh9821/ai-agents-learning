"""
===============================================================================
 MODULE 2 — LESSON 1: Introduction to Chaining and LCEL Basics
===============================================================================
 Course : Fundamentals of Building AI Agents (IBM · Coursera)
 Topics : LangChain Expression Language (LCEL), chains, Runnables,
          the pipe (|) operator, prompt templates, output parsers

 Key concepts:
   • LCEL is LangChain's declarative way to compose chains
   • Chains are built using the pipe (|) operator: prompt | llm | parser
   • RunnablePassthrough, RunnableLambda, RunnableParallel
   • StrOutputParser and JsonOutputParser
===============================================================================
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

load_dotenv()


# ─────────────────────────────────────────────
# 1. Basic LCEL Chain: Prompt | LLM | Parser
# ─────────────────────────────────────────────

def basic_chain_example():
    """The simplest LCEL chain: prompt_template | chat_model | output_parser"""
    print("\n" + "=" * 60)
    print("  1️⃣  Basic LCEL Chain")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that explains tech concepts simply."),
        ("human", "Explain {topic} in 2-3 sentences for a beginner."),
    ])
    output_parser = StrOutputParser()

    # Build the chain using the pipe operator
    chain = prompt | llm | output_parser

    result = chain.invoke({"topic": "AI agents"})
    print(f"\n  Topic: AI agents")
    print(f"  Answer: {result}")

    result2 = chain.invoke({"topic": "tool calling"})
    print(f"\n  Topic: tool calling")
    print(f"  Answer: {result2}")


# ─────────────────────────────────────────────
# 2. Chaining multiple steps
# ─────────────────────────────────────────────

def multi_step_chain():
    """Chain multiple LLM calls: explain → generate quiz question."""
    print("\n" + "=" * 60)
    print("  2️⃣  Multi-Step Chain")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    explain_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a tech educator."),
        ("human", "Explain the concept of '{topic}' in 3 sentences."),
    ])

    quiz_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a quiz creator."),
        ("human", "Based on this explanation:\n{explanation}\n\nCreate one multiple-choice question with 4 options (A-D) and indicate the correct answer."),
    ])

    explain_chain = explain_prompt | llm | StrOutputParser()

    full_chain = (
        {"explanation": explain_chain, "topic": RunnablePassthrough()}
        | quiz_prompt | llm | StrOutputParser()
    )

    result = full_chain.invoke({"topic": "RAG (Retrieval Augmented Generation)"})
    print(f"\n  Generated Quiz:\n{result}")


# ─────────────────────────────────────────────
# 3. RunnableParallel — run chains in parallel
# ─────────────────────────────────────────────

def parallel_chains():
    """RunnableParallel runs multiple chains simultaneously."""
    print("\n" + "=" * 60)
    print("  3️⃣  Parallel Chains (RunnableParallel)")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    parallel_chain = RunnableParallel(
        pros=ChatPromptTemplate.from_messages([("human", "List 3 advantages of {topic}. Be concise.")]) | llm | StrOutputParser(),
        cons=ChatPromptTemplate.from_messages([("human", "List 3 disadvantages of {topic}. Be concise.")]) | llm | StrOutputParser(),
        summary=ChatPromptTemplate.from_messages([("human", "Give a one-sentence summary of {topic}.")]) | llm | StrOutputParser(),
    )

    result = parallel_chain.invoke({"topic": "using AI agents in production"})
    print(f"\n  📋 Summary: {result['summary']}")
    print(f"\n  ✅ Pros:\n{result['pros']}")
    print(f"\n  ❌ Cons:\n{result['cons']}")


# ─────────────────────────────────────────────
# 4. RunnableLambda — custom transformation steps
# ─────────────────────────────────────────────

def lambda_chain():
    """RunnableLambda lets you insert custom Python functions into a chain."""
    print("\n" + "=" * 60)
    print("  4️⃣  RunnableLambda (Custom Functions in Chains)")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def format_as_bullet_points(text: str) -> str:
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        return "\n".join(f"  • {line}" for line in lines)

    def add_header(text: str) -> str:
        return f"\n{'─' * 40}\n📝 Key Points:\n{'─' * 40}\n{text}\n{'─' * 40}"

    prompt = ChatPromptTemplate.from_messages([
        ("human", "List 5 key facts about {topic}. Put each fact on its own line, no numbering."),
    ])

    chain = prompt | llm | StrOutputParser() | RunnableLambda(format_as_bullet_points) | RunnableLambda(add_header)
    result = chain.invoke({"topic": "LangChain framework"})
    print(result)


# ─────────────────────────────────────────────
# 5. JSON Output Parsing
# ─────────────────────────────────────────────

def json_output_chain():
    """Use JsonOutputParser to get structured JSON output from the LLM."""
    print("\n" + "=" * 60)
    print("  5️⃣  JSON Output Parsing")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    json_parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that always responds in JSON format."),
        ("human", "Analyze the programming language '{language}' and return a JSON object with: name, paradigm, year_created, popular_frameworks (list), difficulty (1-10)."),
    ])

    chain = prompt | llm | json_parser
    result = chain.invoke({"language": "Python"})
    print(f"\n  Structured output:")
    for key, value in result.items():
        print(f"    {key}: {value}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("\n" + "━" * 60)
    print("  MODULE 2 · LESSON 1: LCEL Basics")
    print("━" * 60)

    basic_chain_example()
    multi_step_chain()
    parallel_chains()
    lambda_chain()
    json_output_chain()


if __name__ == "__main__":
    main()
