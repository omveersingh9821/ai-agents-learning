"""
===============================================================================
 MODULE 1 — LESSON 2: Getting Started with Tool Calling
===============================================================================
 Course : Fundamentals of Building AI Agents (IBM · Coursera)
 Topics : Function calling with OpenAI, defining tool schemas,
          handling tool_calls response, single & parallel tool calls

 Key concepts:
   • Tools are defined as JSON schemas that describe function signatures
   • The LLM decides WHEN and HOW to call tools based on user input
   • We execute the tool locally and pass results back to the LLM
   • The LLM can request multiple tool calls in a single response
===============================================================================
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────────────────────
# 1. Define multiple tools (Python functions)
# ─────────────────────────────────────────────

def get_weather(city: str) -> str:
    """Simulate fetching weather data for a city."""
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
        return json.dumps({"city": city, "temperature": data["temp"],
                           "condition": data["condition"], "humidity": data["humidity"]})
    return json.dumps({"error": f"Weather data not available for {city}"})


def get_time(timezone: str) -> str:
    """Simulate getting current time in a timezone."""
    from datetime import datetime, timedelta, timezone as tz

    offsets = {"EST": -5, "CST": -6, "PST": -8, "GMT": 0,
               "IST": 5.5, "JST": 9, "CET": 1, "AEST": 11}
    offset = offsets.get(timezone.upper())
    if offset is not None:
        utc_now = datetime.now(tz.utc)
        local_time = utc_now + timedelta(hours=offset)
        return json.dumps({"timezone": timezone.upper(),
                           "current_time": local_time.strftime("%I:%M %p"),
                           "date": local_time.strftime("%B %d, %Y")})
    return json.dumps({"error": f"Unknown timezone: {timezone}"})


def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between common units."""
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
        return json.dumps({"original": f"{value} {from_unit}",
                           "converted": f"{result:.2f} {to_unit}"})
    return json.dumps({"error": f"Cannot convert from {from_unit} to {to_unit}"})


# ─────────────────────────────────────────────
# 2. Define tool schemas for OpenAI
# ─────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name, e.g. 'London'"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time in a given timezone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Timezone abbreviation, e.g. 'EST', 'GMT', 'IST'"}
                },
                "required": ["timezone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": "Convert a value from one unit to another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "The numeric value to convert"},
                    "from_unit": {"type": "string", "description": "Source unit (km, miles, kg, lbs, celsius, fahrenheit)"},
                    "to_unit": {"type": "string", "description": "Target unit (km, miles, kg, lbs, celsius, fahrenheit)"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
]

TOOL_MAP = {"get_weather": get_weather, "get_time": get_time, "unit_converter": unit_converter}


# ─────────────────────────────────────────────
# 3. The tool-calling agent loop
# ─────────────────────────────────────────────

def run_agent(question: str) -> str:
    """
    Complete tool-calling flow:
      1. Send user question + tool schemas to the LLM
      2. Check if LLM wants to call tool(s)
      3. Execute the tool(s) locally
      4. Send results back to LLM
      5. Get final answer
    """
    print("\n" + "━" * 60)
    print(f"  🗣️  User: {question}")
    print("━" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to weather, time, and unit conversion tools. Use them when needed."},
        {"role": "user", "content": question},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=TOOLS, tool_choice="auto", temperature=0)

    assistant_msg = response.choices[0].message

    if assistant_msg.tool_calls:
        print(f"\n  📋 Agent wants to call {len(assistant_msg.tool_calls)} tool(s):\n")
        messages.append(assistant_msg)

        for i, tool_call in enumerate(assistant_msg.tool_calls, 1):
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            print(f"  [{i}] 🔧 {func_name}({func_args})")
            result = TOOL_MAP[func_name](**func_args)
            print(f"      📤 Result: {result}")
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

        final_response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=0)
        answer = final_response.choices[0].message.content
    else:
        answer = assistant_msg.content

    print(f"\n  🤖 Agent: {answer}")
    return answer


# ─────────────────────────────────────────────
# 4. Demo
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
