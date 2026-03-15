# 🤖 Fundamentals of Building AI Agents

> Code companion for the **IBM Coursera** course: *Fundamentals of Building AI Agents*
> 
> Powered by **Google Gemini** (gemini-2.0-flash)

## 📁 Course Structure

```
ai-agents-fundamentals/
│
├── Module 1 - Foundations of Tool Calling and Chaining/
│   ├── Lesson 1 - Introduction to AI Agents/
│   │   └── intro_agents.py
│   ├── Lesson 2 - Getting Started with Tool Calling/
│   │   └── tool_calling.py
│   └── Lesson 3 - Building and Orchestrating Tools/
│       └── orchestrating_tools.py
│
├── Module 2 - LCEL and Manual Tool Calling in LangChain/
│   ├── Lesson 1 - Introduction to Chaining and LCEL Basics/
│   │   └── lcel_basics.py
│   ├── Lesson 2 - Manual Tool Calling Basics/
│   │   └── manual_tool_calling.py
│   └── Lesson 3 - Parsing and Validating Tool Calls/
│       └── parsing_validating.py
│
├── Module 3 - Using Built-in Agents in LangChain/
│   ├── Lesson 1 - Natural Language Data Visualization/
│   │   └── data_visualization.py
│   ├── Lesson 2 - Conversational Database Access/
│   │   └── database_access.py
│   └── sample_data/
│       └── sales.csv
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Setup

### 1. Create virtual environment
```bash
cd ai-agents-fundamentals
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API key
```bash
cp .env.example .env
# Edit .env and add your Google Gemini API key
# Get one for free at https://aistudio.google.com/apikey
```

### 4. Run any lesson
```bash
python "Module 1 - Foundations of Tool Calling and Chaining/Lesson 1 - Introduction to AI Agents/intro_agents.py"
```

## 📖 Module Overview

| Module | Topic | Key Concepts |
|--------|-------|--------------|
| **1** | Foundations of Tool Calling & Chaining | AI agents, function calling, tool orchestration, agentic loop |
| **2** | LCEL & Manual Tool Calling | LangChain Expression Language, bind_tools, ToolMessage, validation |
| **3** | Built-in Agents in LangChain | Pandas DataFrame agent, SQL database agent, data visualization |

## 🔑 API Key

This project uses **Google Gemini** (free tier available). Get your API key at:

👉 [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

## ⚙️ Prerequisites

- Python 3.10+
- Google Gemini API key
- Basic Python knowledge
