import logging
import pandas as pd
from groq import  Groq
import os
import json
from dotenv import load_dotenv
from config import LLM_MODEL, LLM_TEMPERATURE_PLAN, LLM_MAX_TOKENS_PLAN

load_dotenv()
logger = logging.getLogger(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

NL_QUERY_SYSTEM_PROMPT = NL_QUERY_SYSTEM_PROMPT = """
You are a data analysis assistant. The user will ask a question about a pandas DataFrame called `df`.
Your job is to return a JSON object with exactly two fields:

{
    "code": "single line of valid pandas code that answers the question",
    "chart_type": "bar" | "line" | "scatter" | "histogram" | "none"
}

RULES:
- code must be a single executable Python expression using `df`
- code must return either a DataFrame, Series, scalar value, or dict
- Never use print(), never use multiple lines, never import anything
- chart_type should reflect the best visualization for the result
- Use "none" for chart_type if the result is a scalar or doesn't benefit from a chart
- Respond ONLY with valid JSON. No markdown, no explanation.
- Pay close attention to comparison operators:
  * "older than X" → > X (NOT == X)
  * "younger than X" → < X
  * "more than X" → > X
  * "less than X" → < X
  * "at least X" → >= X
  * "exactly X" or "equal to X" → == X

EXAMPLES:
User: "show average fare by passenger class"
Response: {"code": "df.groupby('pclass')['fare'].mean()", "chart_type": "bar"}

User: "show all passengers older than 60"
Response: {"code": "df[df['age'] > 60]", "chart_type": "none"}

User: "how many rows are there"
Response: {"code": "len(df)", "chart_type": "none"}

User: "show distribution of age"
Response: {"code": "df['age'].value_counts().sort_index()", "chart_type": "histogram"}

User: "what percentage of passengers survived"
Response: {"code": "df['survived'].mean() * 100", "chart_type": "none"}

User: "survival rate by passenger class"
Response: {"code": "df.groupby('pclass')['survived'].mean()", "chart_type": "bar"}

User: "correlation between age and fare"
Response: {"code": "df[['age', 'fare']].corr()", "chart_type": "none"}

User: "how many missing values in each column"
Response: {"code": "df.isnull().sum()", "chart_type": "bar"}
"""

def run_nl_query(question : str, df: pd.DataFrame, query_history: list) -> dict :
    """
    Translates a natural language question into Pandas code, executes it safely, and returns the result with chart if applicable.

    Returns dict with:
        query: original question
        code: generated pandas code
        result: execution result as string or dict
        chart: Plotly figure or None
        error: error message if failed, else None
    """

    columns_info = {col : str(df[col].dtype) for col in df.columns}

    history_context = ""
    if query_history :
        recent = query_history[:-3]
        history_context = "\nRecent queries for context:\n" + "\n".join(
            f"- Q: {h.get('user', h.get('query', ''))} → code: {h.get('code', '')}"
            for h in recent
            if h.get('code')
        )
    user_message = f"""
DataFrame columns and dtypes: {json.dumps(columns_info)}
DataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns
{history_context}

Question: {question}
"""
    
    try :
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": NL_QUERY_SYSTEM_PROMPT},
                {"role": "user","content": user_message}
            ],
            temperature= LLM_TEMPERATURE_PLAN,
            max_tokens= 256
        )

        raw = response.choices[0].message.content.strip()

        if raw.startswith("```") :
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        code = parsed.get("code", "")
        chart_type = parsed.get("chart_type", "none")

    except Exception as e :
        return {
            "query": question,
            "code": "",
            "result": None,
            "chart": None,
            "error": f"LLM failed to generate query: {e}"
        }
    
    result, error = _safe_execute(code, df)

    if error :
        return {
            "query": question,
            "code": code,
            "result": None,
            "chart": None,
            "error": error
        }
    
    chart = _generate_chart(result, chart_type, question)

    return {
        "query": question,
        "code": code,
        "result": _format_result(result),
        "chart": chart,
        "error": None
    }

def _safe_execute(code : str, df : pd.DataFrame) :
    """
    Executes a single pandas expression in a restricted scope.
    Returns (result, error).
    """

    blocked = ["import", "open(", "os.", "sys.", "exec", "eval",
               "subprocess", "__", "write", "delete", "drop("]
    
    for term in blocked :
        if term in code :
            return None, f"Blocked operation detected: '{term}'"
        
    try :
        result = eval(code, {"df" : df, "pd" : pd})
        return result, None
    except Exception as e :
        return None, f"Execution error {e}"
    
def _format_result(result) -> str :
    """
    Converts execution result to a readable string. 
    """
    if isinstance(result, pd.DataFrame):
        return result.to_string()
    elif isinstance(result, pd.Series) :
        return result.to_string()
    elif isinstance(result, dict):
        return json.dumps(result, indent = 2, default=str)
    else :
        return str(result)
    
def _generate_chart(result, chart_type: str, question: str) :
    """
    Generates a Plotly chart from the query result if applicable.
    Returns Plotly figure or None.
    """
    import plotly.express as px

    if chart_type.strip().lower() == "none" or result is None :
        return None
    
    try :
        if isinstance(result, pd.Series) :
            df_plot = result.reset_index()
            df_plot.columns = ["category", "value"]

            if chart_type == "bar" :
                return px.bar(
                    df_plot, x="category", y="value",
                    title=question, template="plotly_white"
                )
            elif chart_type == "line":
                return px.line(
                    df_plot, x="category", y="value",
                    title=question, template="plotly_white"
                )
            elif chart_type == "histogram":
                return px.histogram(
                    df_plot, x="category",
                    title=question, template="plotly_white"
                )
        
        if isinstance(result, pd.DataFrame) :
            if chart_type == "scatter" and result.shape[1] >= 2:
                cols = result.columns.tolist()
                return px.scatter(
                    result, x=cols[0], y=cols[1],
                    title=question, template="plotly_white"
                )
    except Exception :
        return None
    
    return None

def route_question(question: str, df_columns: list, chat_history: list) -> str:
    """
    Uses the LLM to decide whether a question should be routed
    to the NL query engine or the conversational agent.
    Returns: "analytical" or "conversational"
    """
    history_context = ""
    if chat_history:
        recent = chat_history[-3:]
        history_context = "\nRecent conversation:\n" + "\n".join(
            f"- {h['user']}" for h in recent
        )

    prompt = f"""You are a router. Decide if the user's question requires executing a pandas query on a DataFrame, or if it's a conversational/explanatory question.

Available DataFrame columns: {', '.join(df_columns)}
{history_context}

User question: "{question}"

Rules:
- Return "analytical" if the question asks to filter, aggregate, count, compare, show, display, or compute something FROM the data
- Return "conversational" if the question asks for explanations, reasons, recommendations, or opinions ABOUT the data or cleaning decisions
- Return ONLY the single word: analytical or conversational

Examples:
"show passengers older than 60" → analytical
"give me all rows where fare > 100" → analytical  
"what is the average age" → analytical
"why was cabin column dropped" → conversational
"what model should I use" → conversational
"explain the cleaning decisions" → conversational
"how many survived" → analytical
"what does high skewness mean" → conversational"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        decision = response.choices[0].message.content.strip().lower()
        return "analytical" if "analytical" in decision else "conversational"
    except Exception:
        # Default to conversational if routing fails
        return "conversational"
    
def narrate_result(question: str, code: str, result: str) -> str:
    """
    Takes a pandas query result and narrates it in plain English.
    Makes analytical responses feel conversational.
    """
    prompt = f"""You are a data analyst assistant. A user asked a question and a pandas query was executed.
Summarize the result in 1-2 sentences of plain English. Be specific with numbers.
Do not mention pandas, DataFrames, or code. Just answer the question naturally.
Highlight important parts of the output like numbers, relations or insights clearly in your answer.

User question: {question}
Query executed: {code}
Result: {result[:500]}

Answer in plain English:"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return result