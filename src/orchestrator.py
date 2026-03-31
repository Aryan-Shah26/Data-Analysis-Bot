import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"


# ── Prompt Templates ────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a data cleaning agent. You will receive a pre-analyzed summary of a dataset's problems.
Your ONLY job is to convert that analysis into the exact JSON format below.
Every problem mentioned MUST appear in your JSON output.
Respond ONLY with valid JSON. No explanations, no markdown fences.

OUTPUT FORMAT:
{
    "cleaning_plan": {
        "drop_duplicates": true or false,
        "standardize_column_names": true,
        "dtype_fixes": {
            "column_name": "should_be_numeric" | "should_be_datetime" | "should_be_categorical"
        },
        "null_handling": {
            "column_name": "fill_mean" | "fill_median" | "fill_mode" | "fill_forward" | "drop_rows" | "drop_column"
        },
        "outlier_handling": {
            "column_name": "cap" | "drop"
        }
    },
    "reasoning": {
        "column_name": "one sentence explaining the decision"
    }
}
"""

FOLLOWUP_SYSTEM_PROMPT = """
You are an expert data scientist assistant.
The user has uploaded a CSV file that has been cleaned and analyzed.
You have access to the dataset profile and the cleaning decisions that were made.
Answer the user's questions clearly and concisely.
If asked to explain a decision, refer to the statistical reasoning.
If asked to suggest further analysis, be specific and practical.
Do not write code. Speak plainly.
"""


# ── Core Orchestrator Functions ──────────────────────────────────────
def _pre_analyze_profile(profile: dict) -> str:
    """
    Pre-analyzes the profile in Python and produces explicit instructions
    for the LLM. This removes ambiguity and prevents empty responses.
    """
    lines = ["Based on the dataset profile, here are the findings per column:"]
    lines.append(f"- Dataset has {profile['duplicate_rows']} duplicate rows.")

    for col, info in profile["columns"].items():
        type_cat = info.get("type_category", "")
        null_pct = info.get("null_percent", 0)
        outlier_count = info.get("outlier_count", 0)
        skew_label = info.get("skew_label", "symmetric")

        col_lines = [f"\nColumn '{col}' (type: {type_cat}):"]

        # Dtype issues
        if type_cat == "should_be_numeric":
            col_lines.append(f"  - Stored as string but contains numbers. Needs dtype fix.")
        elif type_cat == "should_be_datetime":
            col_lines.append(f"  - Stored as string but contains dates. Needs dtype fix.")

        # Null issues
        if null_pct > 0:
            if null_pct > 60:
                col_lines.append(f"  - {null_pct}% missing — should be dropped.")
            elif type_cat in ("numeric", "should_be_numeric"):
                strategy = "fill_median" if skew_label in ("moderate_skew", "high_skew") else "fill_mean"
                col_lines.append(f"  - {null_pct}% missing, skew={skew_label} — use {strategy}.")
            elif type_cat in ("should_be_datetime", "datetime"):
                col_lines.append(f"  - {null_pct}% missing — use fill_forward.")
            else:
                col_lines.append(f"  - {null_pct}% missing — use fill_mode.")

        # Outlier issues
        # Outlier issues
        if outlier_count > 0:
            total_rows = profile["shape"]["rows"]
            outlier_pct = outlier_count / total_rows
            # Only drop if outliers are very few (<2% of rows) AND dataset is large enough
            # Otherwise always cap — dropping rows is too destructive
            strategy = "drop" if (outlier_pct < 0.02 and total_rows > 100) else "cap"
            col_lines.append(f"  - {outlier_count} outliers ({round(outlier_pct*100,1)}% of rows) detected — use {strategy}.")

        if len(col_lines) > 1:
            lines.extend(col_lines)

    lines.append("\nNow produce the cleaning plan JSON exactly as specified.")
    return "\n".join(lines)


def build_cleaning_plan(profile: dict) -> dict:
    """
    Sends the dataset profile to the LLM.
    Returns a structured cleaning plan as a Python dict.
    """
    pre_analysis = _pre_analyze_profile(profile)

    print(pre_analysis)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pre_analysis}
        ],
        temperature=0.1,
        max_tokens=2048
    )

    raw = response.choices[0].message.content.strip()

    print(raw)

    # Strip markdown code fences if the LLM adds them anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON.\nRaw output:\n{raw}\nError: {e}")

    return plan

def answer_followup(
        question : str,
        profile : dict,
        cleaning_log : list,
        chat_history : list) -> str :
    """
    Handles follow-up questions from the user after cleaning is done.
    Maintains a chat history to provide context to the LLM.
    """

    context = f"""
                DATASET PROFILE:
                {json.dumps(profile, indent=2, default=str)}

                CLEANING DECISIONS MADE:
                {chr(10).join(cleaning_log)}
                """
    
    messages = [
        {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
        {"role": "user", "content": context},
        {"role": "assistant", "content": "Understood. I have reviewed the dataset profile and cleaning decisions. What would you like to know?"},
    ]

    # Append conversation history
    for turn in chat_history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    # Add the new question
    messages.append({"role" : "user", "content" : question})

    reponse = client.chat.completions.create(
        model = MODEL,
        messages= messages,
        temperature= 0.3,
        max_tokens= 1024
    )

    return reponse.choices[0].message.content.strip()

def validate_plan(plan : dict, profile : dict) -> dict :
    """
    Sanity checks the LLM's cleaning plan against the actual profile.
    Returns a list of warnings if anything looks off.
    Prevents the LLM from hallucinating column names.
    """

    warnings = []
    actual_columns = set(profile["columns"].keys())
    cleaning_plan = plan.get("cleaning_plan", {})

    # Check dtype_fixes
    for col in cleaning_plan.get("dtype_fixes", {}).keys() :
        if col not in actual_columns :
            warnings.append(f"dtype_fixes : '{col}' does not exist in dataset - skipped.")

    # Check null_handling
    for col in cleaning_plan.get("null_handling", {}).keys() :
        if col not in actual_columns :
            warnings.append(f"null_handling : '{col}' does not exist in dataset - skipped.")

    # Check outlier_handling
    for col in cleaning_plan.get("outlier_handling", {}).keys() :
        if col not in actual_columns :
            warnings.append(f"outlier_handling : '{col}' does not exist in dataset - skipped.")

    return warnings