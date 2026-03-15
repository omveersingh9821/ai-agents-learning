"""
===============================================================================
 MODULE 1 — LESSON 2: Getting Started with Tool Calling
===============================================================================
 Course : Fundamentals of Building AI Agents (IBM · Coursera)
 Topics : Function calling with Gemini, defining tool schemas,
          handling function_call responses, single & parallel tool calls

 Key concepts:
   • Tools are Python functions that Gemini can call automatically
   • The LLM decides WHEN and HOW to call tools based on user input
   • We execute the tool locally and pass results back to the LLM
   • The LLM can request multiple tool calls in a single response
===============================================================================
"""

import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# ─────────────────────────────────────────────
# 1. Define multiple tools (Python functions)
# ─────────────────────────────────────────────

def get_weather(city: str) -> dict:
    """Get the current weather for a given city.

    Args:
        city: The city name, e.g. 'London', 'New York'
    """
    weather_data = {
        "new york": {"temp": "22°C", "condition": "Partly cloudy", "humidity": "65%"},
        "london": {"temp": "15°C", "condition": "Rainy", "humidity": "80%"},
        "tokyo": {"temp": "28°C", "condition": "Sunny", "humidity": "55%"},
        "paris": {"temp": "18°C", "condition": "Overcast", "humidity": "70%"},
        "mumbai": {"temp": "33°C", "condition": "Humid", "humidity": "85%"},
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        return {"city": city, "temperature": data["temp"],
                "condition": data["condition"], "humidity": data["humidity"]}
    return {"error": f"Weather data not available for {city}"}


def get_time(timezone: str) -> dict:
    """Get the current time in a given timezone.

    Args:
        timezone: Timezone abbreviation, e.g. 'EST', 'GMT', 'IST', 'JST'
    """
    from datetime import datetime, timedelta, timezone as tz
    offsets = {"EST": -5, "CST": -6, "PST": -8, "GMT": 0,
               "IST": 5.5, "JST": 9, "CET": 1, "AEST": 11}
    offset = offsets.get(timezone.upper())
    if offset is not None:
        utc_now = datetime.now(tz.utc)
        local_time = utc_now + timedelta(hours=offset)
        return {"timezone": timezone.upper(),
                "current_time": local_time.strftime("%I:%M %p"),
                "date": local_time.strftime("%B %d, %Y")}
    return {"error": f"Unknown timezone: {timezone}"}


def unit_converter(value: float, from_unit: str, to_unit: str) -> dict:
    """Convert a value from one unit to another.

    Args:
        value: The numeric value to convert
        from_unit: Source unit (km, miles, kg, lbs, celsius, fahrenheit)
        to_unit: Target unit (km, miles, kg, lbs, celsius, fahrenheit)
    """
    conversions = {
        ("km", "miles"): lambda v: v * 0.621371,
        ("miles", "km"): lambda v: v * 1.60934,
        ("kg", "lbs"): lambda v: v * 2.20462,
        ("lbs", "kg"): lambda v: v * 0.453592,
        ("celsius", "fahrenheit"): lambda v: (v * 9 / 5) + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return {"original": f"{value} {from_unit}", "converted": f"{result:.2f} {to_unit}"}
    return {"error": f"Cannot convert from {from_unit} to {to_unit}"}


# Tool registry
ALL_TOOLS = [get_weather, get_time, unit_converter]
TOOL_MAP = {fn.__name__: fn for fn in ALL_TOOLS}


# ─────────────────────────────────────────────
# 2. The tool-calling agent loop
# ─────────────────────────────────────────────

def run_agent(question: str) -> str:
    """
    Complete tool-calling flow with Gemini:
      1. Send user question + tools to the model
      2. Check if model wants to call tool(s)
      3. Execute the tool(s) locally
      4. Send results back to model
      5. Get final answer
    """
    print("\n" + "━" * 60)
    print(f"  🗣️  User: {question}")
    print("━" * 60)

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        tools=ALL_TOOLS,
        system_instruction="You are a helpful assistant with access to weather, time, and unit conversion tools. Use them when needed.",
    )

    chat = model.start_chat()
    response = chat.send_message(question)

    # Process function calls (may be multiple)
    function_calls = [part.function_call for part in response.parts
                      if hasattr(part, "function_call") and part.function_call.name]

    if function_calls:
        print(f"\n  📋 Agent wants to call {len(function_calls)} tool(s):\n")

        # Build all function responses
        function_responses = []
        for i, fc in enumerate(function_calls, 1):
            func_name = fc.name
            func_args = dict(fc.args)
            print(f"  [{i}] 🔧 {func_name}({func_args})")

            result = TOOL_MAP[func_name](**func_args)
            print(f"      📤 Result: {result}")

            function_responses.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=func_name, response={"result": result}
                    )
                )
            )

        # Send all results back at once
        response = chat.send_message(
            genai.protos.Content(parts=function_responses)
        )

    answer = response.text
    print(f"\n  🤖 Agent: {answer}")
    return answer


# ─────────────────────────────────────────────
# 3. Demo
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  MODULE 1 · LESSON 2: Getting Started with Tool Calling")
    print("=" * 60)

    run_agent("What's the weather like in Tokyo right now?")
    run_agent("Convert 100 kilometers to miles.")
    run_agent("I'm planning a trip. What's the weather in London and Paris? Also, what time is it in GMT and CET?")
    run_agent("What does API stand for?")


if __name__ == "__main__":
    main()
