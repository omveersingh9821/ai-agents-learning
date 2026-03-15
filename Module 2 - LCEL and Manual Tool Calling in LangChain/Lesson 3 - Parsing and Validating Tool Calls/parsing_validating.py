"""
===============================================================================
 MODULE 2 — LESSON 3: Parsing and Validating Tool Calls
===============================================================================
 Course : Fundamentals of Building AI Agents (IBM · Coursera)
 Topics : Robust tool call parsing, input validation, error handling,
          tool_choice parameter, forcing specific tools, fallbacks

 Key concepts:
   • Validating tool arguments before execution
   • Handling errors gracefully
   • Building a robust agent loop with error recovery
===============================================================================
"""

import os
import json
from typing import Optional
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

load_dotenv()


# ─────────────────────────────────────────────
# 1. Tools with input validation
# ─────────────────────────────────────────────

@tool
def book_flight(origin: str, destination: str, date: str, passengers: int = 1) -> str:
    """Book a flight between two cities.

    Args:
        origin: Departure city (3-letter airport code, e.g. JFK, LAX)
        destination: Arrival city (3-letter airport code, e.g. LHR, NRT)
        date: Travel date in YYYY-MM-DD format
        passengers: Number of passengers (1-9)
    """
    errors = []
    if len(origin) != 3 or not origin.isalpha():
        errors.append(f"Invalid origin airport code: '{origin}'. Must be 3 letters.")
    if len(destination) != 3 or not destination.isalpha():
        errors.append(f"Invalid destination code: '{destination}'. Must be 3 letters.")
    if origin.upper() == destination.upper():
        errors.append("Origin and destination cannot be the same.")

    try:
        from datetime import datetime
        travel_date = datetime.strptime(date, "%Y-%m-%d")
        if travel_date < datetime.now():
            errors.append(f"Date {date} is in the past.")
    except ValueError:
        errors.append(f"Invalid date format: '{date}'. Use YYYY-MM-DD.")

    if not 1 <= passengers <= 9:
        errors.append(f"Passengers must be 1-9, got {passengers}.")

    if errors:
        return json.dumps({"success": False, "errors": errors})

    return json.dumps({"success": True, "booking": {
        "confirmation": f"BK-{origin.upper()}{destination.upper()}-{date.replace('-', '')}",
        "origin": origin.upper(), "destination": destination.upper(),
        "date": date, "passengers": passengers,
        "price_per_person": "$450.00", "total": f"${450 * passengers:.2f}"}})


@tool
def search_hotels(city: str, checkin: str, checkout: str, max_price: Optional[float] = None) -> str:
    """Search for available hotels in a city.

    Args:
        city: City to search for hotels
        checkin: Check-in date (YYYY-MM-DD)
        checkout: Check-out date (YYYY-MM-DD)
        max_price: Optional maximum nightly price in USD
    """
    hotels = [
        {"name": "Grand Plaza Hotel", "price": 250, "rating": 4.5, "city": "london"},
        {"name": "Budget Inn Express", "price": 89, "rating": 3.8, "city": "london"},
        {"name": "Luxury Suites", "price": 520, "rating": 4.9, "city": "london"},
        {"name": "Sakura Hotel", "price": 180, "rating": 4.2, "city": "tokyo"},
        {"name": "Capsule Stay", "price": 45, "rating": 3.5, "city": "tokyo"},
        {"name": "Imperial Grand", "price": 380, "rating": 4.7, "city": "tokyo"},
    ]
    results = [h for h in hotels if h["city"] == city.lower()]
    if max_price is not None:
        results = [h for h in results if h["price"] <= max_price]
    if not results:
        return json.dumps({"hotels": [], "message": f"No hotels found in {city}"})
    return json.dumps({"city": city, "checkin": checkin, "checkout": checkout, "hotels": results, "count": len(results)})


all_tools = [book_flight, search_hotels]
tool_map = {t.name: t for t in all_tools}


# ─────────────────────────────────────────────
# 2. Robust agent loop with error handling
# ─────────────────────────────────────────────

def robust_agent(question: str, max_iterations: int = 5) -> str:
    """A robust agent loop that handles tool errors, validation, and unknown tools."""
    print("\n" + "=" * 60)
    print(f"  🗣️ User: {question}")
    print("=" * 60)

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    llm_with_tools = llm.bind_tools(all_tools)

    messages = [
        SystemMessage(content="You are a travel booking assistant. Help users book flights and find hotels. If a tool returns errors, inform the user and ask for corrections."),
        HumanMessage(content=question),
    ]

    for iteration in range(max_iterations):
        print(f"\n  ⚙️ Iteration {iteration + 1}")
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print(f"\n  🤖 Agent: {response.content}")
            return response.content

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"\n  🔧 Tool: {tool_name}")
            print(f"  📥 Args: {json.dumps(tool_args, indent=2)}")

            try:
                if tool_name not in tool_map:
                    result = json.dumps({"error": f"Unknown tool: {tool_name}. Available: {list(tool_map.keys())}"})
                else:
                    result = tool_map[tool_name].invoke(tool_args)
                print(f"  📤 Result: {result[:120]}...")
            except Exception as e:
                result = json.dumps({"error": f"Tool execution failed: {str(e)}", "suggestion": "Please check arguments."})
                print(f"  ❌ Error: {e}")

            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

    return "Max iterations reached."


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("\n" + "━" * 60)
    print("  MODULE 2 · LESSON 3: Parsing and Validating Tool Calls")
    print("━" * 60)

    robust_agent("Book a flight from JFK to LHR on 2026-06-15 for 2 passengers. Also find hotels in London under $300 per night for Jun 15-20.")
    robust_agent("Book a flight from XX to LHR yesterday for 15 passengers.")


if __name__ == "__main__":
    main()
