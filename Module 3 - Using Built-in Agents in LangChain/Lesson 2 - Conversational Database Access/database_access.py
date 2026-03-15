"""
===============================================================================
 MODULE 3 — LESSON 2: Conversational Database Access
===============================================================================
 Course : Fundamentals of Building AI Agents (IBM · Coursera)
 Topics : SQL Database Agent, natural language → SQL queries,
          create_sql_agent, SQLDatabase toolkit, multi-turn DB chat

 Key concepts:
   • LangChain's SQL agent translates natural language to SQL
   • It inspects the database schema automatically
   • It can handle multi-turn conversations about the data
===============================================================================
"""

import os
import json
from dotenv import load_dotenv  # pyre-ignore[21]

import pandas as pd  # pyre-ignore[21]
from sqlalchemy import create_engine, text  # pyre-ignore[21]

from langchain_google_genai import ChatGoogleGenerativeAI  # pyre-ignore[21]
from langchain_community.utilities import SQLDatabase  # pyre-ignore[21]
from langchain_community.agent_toolkits import create_sql_agent  # pyre-ignore[21]

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "sales.db")


# ─────────────────────────────────────────────
# 1. Create a SQLite database from CSV
# ─────────────────────────────────────────────

def setup_database() -> str:
    """Create a SQLite database from the sales CSV and add extra tables."""
    print("\n" + "=" * 60)
    print("  🗄️  Setting Up SQLite Database")
    print("=" * 60)

    db_uri = f"sqlite:///{DB_PATH}"
    engine = create_engine(db_uri)

    csv_path = os.path.join(SCRIPT_DIR, "..", "sample_data", "sales.csv")
    df = pd.read_csv(csv_path)
    df.to_sql("sales", engine, if_exists="replace", index=False)
    print(f"  ✅ Created 'sales' table ({len(df)} rows)")

    customers = pd.DataFrame({
        "customer_id": range(1, 11),
        "name": ["Alice Johnson", "Bob Smith", "Carol White", "David Brown",
                 "Eva Martinez", "Frank Lee", "Grace Kim", "Henry Davis",
                 "Iris Wilson", "Jack Taylor"],
        "region": ["North", "South", "North", "South", "North",
                    "South", "North", "South", "North", "South"],
        "tier": ["Gold", "Silver", "Gold", "Bronze", "Silver",
                 "Gold", "Bronze", "Silver", "Gold", "Bronze"],
        "total_purchases": [15200, 8900, 12400, 4500, 9800,
                            18300, 3200, 7600, 14100, 5400],
    })
    customers.to_sql("customers", engine, if_exists="replace", index=False)
    print(f"  ✅ Created 'customers' table ({len(customers)} rows)")

    orders = pd.DataFrame({
        "order_id": range(1001, 1021),
        "customer_id": [1, 2, 3, 1, 4, 5, 6, 3, 7, 8, 2, 9, 10, 1, 5, 6, 3, 8, 7, 4],
        "product": ["Widget A", "Widget B", "Gadget X", "Widget A", "Widget B",
                     "Gadget X", "Widget A", "Widget B", "Gadget X", "Widget A",
                     "Widget B", "Gadget X", "Widget A", "Widget B", "Gadget X",
                     "Widget A", "Widget B", "Gadget X", "Widget A", "Widget B"],
        "quantity": [5, 3, 10, 8, 2, 15, 4, 6, 12, 3, 7, 9, 5, 4, 11, 6, 3, 8, 7, 2],
        "order_date": ["2025-01-05", "2025-01-10", "2025-01-15", "2025-01-20", "2025-01-25",
                        "2025-02-01", "2025-02-05", "2025-02-10", "2025-02-15", "2025-02-20",
                        "2025-03-01", "2025-03-05", "2025-03-10", "2025-03-15", "2025-03-20",
                        "2025-04-01", "2025-04-05", "2025-04-10", "2025-04-15", "2025-04-20"],
    })
    orders.to_sql("orders", engine, if_exists="replace", index=False)
    print(f"  ✅ Created 'orders' table ({len(orders)} rows)")

    with engine.connect() as conn:
        tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
        print(f"\n  📋 Tables: {[t[0] for t in tables]}")

    return db_uri


# ─────────────────────────────────────────────
# 2. Explore the database manually (reference)
# ─────────────────────────────────────────────

def manual_queries(db_uri: str):
    """Run SQL queries manually — this is what the AI agent does behind the scenes!"""
    print("\n" + "=" * 60)
    print("  🔍 Manual SQL Queries (Reference)")
    print("=" * 60)

    engine = create_engine(db_uri)
    queries = [
        ("Total revenue by product", "SELECT product, SUM(revenue) as total_revenue FROM sales GROUP BY product ORDER BY total_revenue DESC"),
        ("Top 5 customers by purchases", "SELECT name, tier, total_purchases FROM customers ORDER BY total_purchases DESC LIMIT 5"),
        ("Orders per customer", "SELECT c.name, COUNT(o.order_id) as num_orders, SUM(o.quantity) as total_items FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.name ORDER BY total_items DESC"),
    ]

    for title, query in queries:
        print(f"\n  {'─' * 50}")
        print(f"  📋 {title}")
        result = pd.read_sql(query, engine)
        print(f"\n{result.to_string(index=False)}")


# ─────────────────────────────────────────────
# 3. Create the SQL Agent
# ─────────────────────────────────────────────

def create_db_agent(db_uri: str):
    """Create a SQL agent that translates natural language → SQL queries."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    db = SQLDatabase.from_uri(db_uri)

    print("\n" + "=" * 60)
    print("  🤖 SQL Agent — Database Schema Discovery")
    print("=" * 60)
    print(f"\n  Tables: {db.get_usable_table_names()}")
    print(f"\n  Schema info:\n{db.get_table_info()}")

    agent = create_sql_agent(llm, db=db, verbose=True, agent_type="openai-tools")
    return agent


# ─────────────────────────────────────────────
# 4. Natural language database queries
# ─────────────────────────────────────────────

def conversational_queries(agent):
    """Ask questions in plain English — the agent writes and executes SQL."""
    print("\n" + "=" * 60)
    print("  💬 Conversational Database Access")
    print("=" * 60)

    questions = [
        "What tables are in this database and how many rows does each have?",
        "What is the total revenue by product? Sort by highest first.",
        "Who are the Gold tier customers and what is their total purchase amount?",
        "Which customer placed the most orders? Show their name and order count.",
        "What is the total revenue for the North region in March 2025?",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n  {'━' * 50}")
        print(f"  Q{i}: {q}")
        print(f"  {'━' * 50}")
        try:
            result = agent.invoke({"input": q})
            print(f"\n  💡 Answer: {result['output']}")
        except Exception as e:
            print(f"\n  ❌ Error: {e}")


# ─────────────────────────────────────────────
# 5. Interactive chat mode (bonus)
# ─────────────────────────────────────────────

def interactive_mode(agent):
    """Start an interactive chat session with the database. Type 'quit' to exit."""
    print("\n" + "=" * 60)
    print("  🗣️  Interactive Database Chat")
    print("  Type 'quit' to exit, 'help' for example questions")
    print("=" * 60)

    examples = ["How many products are there?", "What was the best-selling month?",
                 "Compare revenue between regions", "List all Gold customers"]

    while True:
        user_input = input("\n  You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("  👋 Goodbye!")
            break
        if user_input.lower() == "help":
            print("\n  Example questions:")
            for ex in examples:
                print(f"    • {ex}")
            continue
        if not user_input:
            continue
        try:
            result = agent.invoke({"input": user_input})
            print(f"\n  🤖 Agent: {result['output']}")
        except Exception as e:
            print(f"\n  ❌ Error: {e}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("\n" + "━" * 60)
    print("  MODULE 3 · LESSON 2: Conversational Database Access")
    print("━" * 60)

    db_uri = setup_database()
    manual_queries(db_uri)

    try:
        agent = create_db_agent(db_uri)
        conversational_queries(agent)
        # Uncomment to start interactive mode:
        # interactive_mode(agent)
    except Exception as e:
        print(f"\n  ⚠️  SQL Agent requires a Google Gemini API key.")
        print(f"     Error: {e}")
        print(f"     Manual queries above show the same data.")


if __name__ == "__main__":
    main()
