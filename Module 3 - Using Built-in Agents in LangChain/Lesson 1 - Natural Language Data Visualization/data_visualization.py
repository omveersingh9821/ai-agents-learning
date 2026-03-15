"""
===============================================================================
 MODULE 3 — LESSON 1: Natural Language Data Visualization
===============================================================================
 Course : Fundamentals of Building AI Agents (IBM · Coursera)
 Topics : Using LangChain's built-in agents for data analysis,
          Pandas DataFrame agent, natural language → charts

 Key concepts:
   • Built-in agents come pre-configured with tools and prompts
   • The Pandas DataFrame agent can analyze CSVs using natural language
   • It writes and executes Python/Pandas code behind the scenes
   • You can ask it to create visualizations (matplotlib charts)
===============================================================================
"""

import os
import pandas as pd  # pyre-ignore[21]
import matplotlib  # pyre-ignore[21]
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pyre-ignore[21]
from dotenv import load_dotenv  # pyre-ignore[21]

from langchain_google_genai import ChatGoogleGenerativeAI  # pyre-ignore[21]
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent  # pyre-ignore[21]

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "..", "sample_data", "sales.csv")


# ─────────────────────────────────────────────
# 1. Load the data
# ─────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load and display the sales dataset."""
    df = pd.read_csv(CSV_PATH)
    print("\n" + "=" * 60)
    print("  📊 Sales Dataset")
    print("=" * 60)
    print(f"\n  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Columns: {', '.join(df.columns)}")
    print(f"\n{df.head(10).to_string(index=False)}")
    return df


# ─────────────────────────────────────────────
# 2. Create the Pandas DataFrame Agent
# ─────────────────────────────────────────────

def create_data_agent(df: pd.DataFrame):
    """Create a Pandas DataFrame agent that can analyze data via natural language."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    agent = create_pandas_dataframe_agent(
        llm, df, verbose=True, allow_dangerous_code=True, agent_type="openai-tools")
    return agent


# ─────────────────────────────────────────────
# 3. Natural Language Data Analysis
# ─────────────────────────────────────────────

def analyze_data(agent):
    """Ask the agent questions about the data in plain English."""
    print("\n" + "=" * 60)
    print("  🔍 Natural Language Data Analysis")
    print("=" * 60)

    questions = [
        "What is the total revenue across all months?",
        "Which product has the highest total units sold?",
        "What is the profit (revenue - cost) for each product?",
        "Compare the performance of North vs South region. Which generates more revenue?",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n  {'─' * 50}")
        print(f"  Q{i}: {q}")
        print(f"  {'─' * 50}")
        try:
            result = agent.invoke({"input": q})
            print(f"\n  📊 Answer: {result['output']}")
        except Exception as e:
            print(f"\n  ❌ Error: {e}")


# ─────────────────────────────────────────────
# 4. Natural Language → Chart Generation
# ─────────────────────────────────────────────

def generate_charts(agent):
    """Ask the agent to create visualizations."""
    print("\n" + "=" * 60)
    print("  📈 Natural Language Data Visualization")
    print("=" * 60)

    output_dir = os.path.join(SCRIPT_DIR, "charts")
    os.makedirs(output_dir, exist_ok=True)

    chart_requests = [
        f"Create a bar chart showing total revenue by product. Use nice colors. Save to '{output_dir}/revenue_by_product.png'. Use plt.savefig() and plt.close().",
        f"Create a line chart showing monthly revenue trends for each product. Add a legend. Save to '{output_dir}/monthly_trends.png'. Use plt.savefig() and plt.close().",
    ]

    for i, prompt in enumerate(chart_requests, 1):
        print(f"\n  Chart {i}: {prompt[:80]}...")
        try:
            result = agent.invoke({"input": prompt})
            print(f"  📊 Agent: {result['output']}")
        except Exception as e:
            print(f"  ❌ Error: {e}")


# ─────────────────────────────────────────────
# 5. Manual visualization (fallback/reference)
# ─────────────────────────────────────────────

def manual_visualization(df: pd.DataFrame):
    """Create charts manually with Pandas + Matplotlib (works without API key)."""
    print("\n" + "=" * 60)
    print("  📊 Manual Visualization (Reference)")
    print("=" * 60)

    output_dir = os.path.join(SCRIPT_DIR, "charts")
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    revenue_by_product = df.groupby("product")["revenue"].sum().sort_values(ascending=False)
    colors = ["#4CAF50", "#2196F3", "#FF9800"]
    revenue_by_product.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Total Revenue by Product", fontsize=16, fontweight="bold")
    ax.set_xlabel("Product", fontsize=12)
    ax.set_ylabel("Revenue ($)", fontsize=12)
    ax.tick_params(axis="x", rotation=0)
    for i, v in enumerate(revenue_by_product):
        ax.text(i, v + 200, f"${v:,.0f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "manual_revenue_by_product.png"), dpi=150)
    plt.close()
    print("  ✅ Saved: charts/manual_revenue_by_product.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    monthly = df.groupby(["month", "product"])["revenue"].sum().unstack()
    monthly.plot(ax=ax, marker="o", linewidth=2)
    ax.set_title("Monthly Revenue Trends by Product", fontsize=16, fontweight="bold")
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Revenue ($)", fontsize=12)
    ax.legend(title="Product")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "manual_monthly_trends.png"), dpi=150)
    plt.close()
    print("  ✅ Saved: charts/manual_monthly_trends.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    df["profit"] = df["revenue"] - df["cost"]
    profit = df.groupby("product")["profit"].sum().sort_values(ascending=False)
    profit.plot(kind="bar", ax=ax, color=["#66BB6A", "#42A5F5", "#FFA726"], edgecolor="white")
    ax.set_title("Total Profit by Product", fontsize=16, fontweight="bold")
    ax.set_ylabel("Profit ($)", fontsize=12)
    ax.tick_params(axis="x", rotation=0)
    for i, v in enumerate(profit):
        ax.text(i, v + 100, f"${v:,.0f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "manual_profit_by_product.png"), dpi=150)
    plt.close()
    print("  ✅ Saved: charts/manual_profit_by_product.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("\n" + "━" * 60)
    print("  MODULE 3 · LESSON 1: Natural Language Data Visualization")
    print("━" * 60)

    df = load_data()
    manual_visualization(df)

    try:
        agent = create_data_agent(df)
        analyze_data(agent)
        generate_charts(agent)
    except Exception as e:
        print(f"\n  ⚠️  Agent-based analysis requires a Google Gemini API key.")
        print(f"     Error: {e}")
        print(f"     Manual charts have been generated as a fallback.")


if __name__ == "__main__":
    main()
