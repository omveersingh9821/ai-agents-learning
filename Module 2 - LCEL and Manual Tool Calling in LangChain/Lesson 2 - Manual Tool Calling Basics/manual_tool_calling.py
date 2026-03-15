"""
===============================================================================
 MODULE 2 — LESSON 2: Manual Tool Calling Basics
===============================================================================
 Course : Fundamentals of Building AI Agents (IBM · Coursera)
 Topics : Binding tools to LLMs in LangChain, manual tool invocation,
          ToolMessage, ChatModel.bind_tools(), tool execution flow

 Key concepts:
   • bind_tools() attaches tool schemas to a chat model
   • The LLM returns AIMessage with tool_calls (not direct execution)
   • YOU manually execute the tool and send back a ToolMessage
   • This gives you full control over tool execution
===============================================================================
"""

import os
import json
from dotenv import load_dotenv  # pyre-ignore[21]

from langchain_google_genai import ChatGoogleGenerativeAI  # pyre-ignore[21]
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage  # pyre-ignore[21]
from langchain_core.tools import tool  # pyre-ignore[21]

load_dotenv()


# ─────────────────────────────────────────────
# 1. Define tools using the @tool decorator
# ─────────────────────────────────────────────

@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a given ticker symbol.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, GOOGL, MSFT)
    """
    prices = {
        "AAPL": {"price": 178.50, "change": "+1.2%", "volume": "52M"},
        "GOOGL": {"price": 141.80, "change": "-0.5%", "volume": "28M"},
        "MSFT": {"price": 378.90, "change": "+0.8%", "volume": "35M"},
        "TSLA": {"price": 248.20, "change": "+2.1%", "volume": "95M"},
        "AMZN": {"price": 185.60, "change": "-0.3%", "volume": "41M"},
    }
    ticker_upper = ticker.upper()
    if ticker_upper in prices:
        data = prices[ticker_upper]
        return json.dumps({"ticker": ticker_upper, "price": f"${data['price']}",
                           "change": data["change"], "volume": data["volume"]})
    return json.dumps({"error": f"Unknown ticker: {ticker}"})


@tool
def calculate_portfolio_value(holdings: str) -> str:
    """Calculate total portfolio value from a comma-separated list of ticker:shares pairs.

    Args:
        holdings: Comma-separated holdings like 'AAPL:10,GOOGL:5,MSFT:3'
    """
    prices = {"AAPL": 178.50, "GOOGL": 141.80, "MSFT": 378.90, "TSLA": 248.20, "AMZN": 185.60}
    total = 0.0
    breakdown = []
    for holding in holdings.split(","):
        parts = holding.strip().split(":")
        if len(parts) == 2:
            ticker, shares = parts[0].strip().upper(), int(parts[1].strip())
            if ticker in prices:
                value = prices[ticker] * shares
                total += value
                breakdown.append({"ticker": ticker, "shares": shares, "price": prices[ticker], "value": f"${value:.2f}"})
    return json.dumps({"portfolio": breakdown, "total_value": f"${total:.2f}"})


@tool
def get_company_info(company: str) -> str:
    """Get basic information about a company.

    Args:
        company: Company name or ticker symbol
    """
    companies = {
        "AAPL": {"name": "Apple Inc.", "sector": "Technology", "ceo": "Tim Cook", "founded": 1976},
        "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology", "ceo": "Sundar Pichai", "founded": 1998},
        "MSFT": {"name": "Microsoft Corp.", "sector": "Technology", "ceo": "Satya Nadella", "founded": 1975},
    }
    company_upper = company.upper()
    if company_upper in companies:
        return json.dumps(companies[company_upper])
    return json.dumps({"error": f"Company info not found for: {company}"})


all_tools = [get_stock_price, calculate_portfolio_value, get_company_info]
tool_map = {t.name: t for t in all_tools}


# ─────────────────────────────────────────────
# 2. Bind tools to the LLM
# ─────────────────────────────────────────────

def demo_bind_tools():
    """bind_tools() attaches tool definitions to the chat model."""
    print("\n" + "=" * 60)
    print("  1️⃣  Binding Tools to the LLM")
    print("=" * 60)

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    llm_with_tools = llm.bind_tools(all_tools)

    response = llm_with_tools.invoke("What's the stock price of Apple?")
    print(f"\n  Response type: {type(response).__name__}")
    print(f"  Content: {response.content or '(empty — tool call requested)'}")
    print(f"  Tool calls: {response.tool_calls}")

    if response.tool_calls:
        tc = response.tool_calls[0]
        print(f"\n  📋 Tool requested:")
        print(f"     Name: {tc['name']}")
        print(f"     Args: {tc['args']}")
        print(f"     ID:   {tc['id']}")


# ─────────────────────────────────────────────
# 3. Manual tool execution flow
# ─────────────────────────────────────────────

def manual_tool_calling():
    """
    The full manual tool-calling flow:
      1. Send message to LLM (with bound tools)
      2. LLM returns AIMessage with tool_calls
      3. Execute each tool call manually
      4. Send ToolMessage results back to LLM
      5. LLM generates the final response
    """
    print("\n" + "=" * 60)
    print("  2️⃣  Manual Tool Execution Flow")
    print("=" * 60)

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    llm_with_tools = llm.bind_tools(all_tools)

    question = "What's the current stock price of Tesla and Microsoft?"
    print(f"\n  🗣️ User: {question}\n")

    messages = [HumanMessage(content=question)]
    ai_response = llm_with_tools.invoke(messages)
    messages.append(ai_response)

    print(f"  📋 LLM requested {len(ai_response.tool_calls)} tool call(s):\n")

    for tool_call in ai_response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        print(f"  🔧 Executing: {tool_name}({tool_args})")
        tool_fn = tool_map[tool_name]
        result = tool_fn.invoke(tool_args)
        print(f"  📤 Result: {result}\n")

        messages.append(ToolMessage(content=result, tool_call_id=tool_id))

    final_response = llm_with_tools.invoke(messages)
    print(f"  🤖 Agent: {final_response.content}")


# ─────────────────────────────────────────────
# 4. Multi-turn conversation with tools
# ─────────────────────────────────────────────

def multi_turn_with_tools():
    """Multi-turn conversation using different tools across turns."""
    print("\n" + "=" * 60)
    print("  3️⃣  Multi-Turn Conversation with Tools")
    print("=" * 60)

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    llm_with_tools = llm.bind_tools(all_tools)
    messages = []

    questions = [
        "Tell me about Apple as a company.",
        "What's their current stock price?",
        "Now calculate the value if I own 50 shares of AAPL and 20 shares of MSFT.",
    ]

    for q in questions:
        print(f"\n  {'─' * 50}")
        print(f"  🗣️ User: {q}")
        messages.append(HumanMessage(content=q))
        ai_response = llm_with_tools.invoke(messages)
        messages.append(ai_response)

        if ai_response.tool_calls:
            for tool_call in ai_response.tool_calls:
                tool_fn = tool_map[tool_call["name"]]
                result = tool_fn.invoke(tool_call["args"])
                print(f"  🔧 {tool_call['name']} → {result[:80]}...")
                messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
            final = llm_with_tools.invoke(messages)
            messages.append(final)
            print(f"  🤖 Agent: {final.content}")
        else:
            print(f"  🤖 Agent: {ai_response.content}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("\n" + "━" * 60)
    print("  MODULE 2 · LESSON 2: Manual Tool Calling Basics")
    print("━" * 60)
    demo_bind_tools()
    manual_tool_calling()
    multi_turn_with_tools()


if __name__ == "__main__":
    main()
