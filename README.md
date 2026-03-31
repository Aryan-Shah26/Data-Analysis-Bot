# 🤖 Autonomous EDA & Data Cleaning Agent

An agentic pipeline that automatically cleans messy CSV files and generates
a full exploratory data analysis dashboard. Built with Groq, Llama 3.3 70B,
and Streamlit.

## What it does

- Profiles any CSV file — dtypes, nulls, skewness, outliers
- Sends the profile to an LLM which builds a structured cleaning plan
- Executes the plan using pre-coded, statistically correct tool functions
- Generates an EDA dashboard — distributions, correlation heatmap, boxplots
- Lets you ask follow-up questions about the data in natural language

## Architecture
```
Upload CSV
    ↓
profile_dataset()        ← statistical profiling
    ↓
_pre_analyze_profile()   ← deterministic Python pre-analysis
    ↓
LLM (Llama 3.3 70B)      ← builds cleaning plan as JSON
    ↓
validate_plan()          ← hallucination check
    ↓
Tool execution           ← dtype fixes, null handling, outlier handling
    ↓
generate_all_visualizations()  ← Plotly dashboard
```

## Key design decision

Rather than having the LLM generate raw Pandas code, the LLM acts purely
as an orchestrator — deciding which pre-coded tool functions to call and
in what order. This dramatically reduces hallucinations and makes every
cleaning decision auditable.

## Setup

1. Clone the repo
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and add your Groq API key
6. Run: `streamlit run app.py`

## Tech stack

- **LLM**: Llama 3.3 70B via Groq API
- **Frontend**: Streamlit
- **Data**: Pandas, NumPy, SciPy
- **Visualization**: Plotly Express, Missingno
- **Orchestration**: Plain Python agentic loop

## Limitations

- Optimized for CSV files up to ~50MB (Pandas in-memory)
- Agent decisions are based on column names and statistics only — no domain knowledge
- Logical bugs in data (e.g. negative ages) are not caught — only statistical anomalies