import pandas as pd
from src.tools.eda_tools import (
    profile_dataset,
    generate_summary_text,
    drop_duplicate_rows,
    standardize_column_names
)
from src.tools.dtype_tools import apply_dtype_fixes
from src.tools.null_tools import apply_null_fixes
from src.tools.outlier_tools import apply_outlier_fixes
from src.tools.viz_tools import generate_all_visualizations
from src.orchestrator import build_cleaning_plan, validate_plan

def  run_agent(df : pd.DataFrame) -> dict :
    """
    Master agent loop. Takes a raw dataframe, runs the full cleaning and EDA pipeline, returns everything the frontend needs.

    Returns a dict with:
        - cleaned_df: the cleaned dataframe
        - original_df : the original dataframe (for before/after comparison)
        - profile : the dataset profile
        - plan : the LLM cleaning plan
        - log : list of every descision made
        - warnings : list of any validation warnings
        - summary_text : plain-english dataset summary
        - visualizations : dict of Plotly figures and nulity matrix
    """

    result = {
        "cleaned_df" : None,
        "original_df" : df.copy(),
        "profile" : None,
        "plan" : None,
        "log" : [],
        "warnings" : [],
        "summary_text" : "",
        "visualizations" : {}
    }

    log = result["log"]
    

    # ── Step 1: Profile the raw dataset ─────────────────────────────
    log.append("▶ Profiling dataset...")
    profile = profile_dataset(df)
    result["profile"] = profile
    result["summary_text"] = generate_summary_text(profile)
    log.append(f" Dataset shape: {profile['shape']['rows']} rows x {profile['shape']['columns']} columns.")
    log.append(f" Duplicate rows found: {profile['duplicate_rows']}.")


    # ── Step 2: Get cleaning plan from LLM ──────────────────────────
    log.append("▶ Sending profile to LLM for cleaning plan...")
    plan = build_cleaning_plan(profile)
    result["plan"] = plan
    warnings = validate_plan(plan, profile)
    result["warnings"] = warnings

    if warnings :
        for w in warnings :
            log.append(f"  ⚠ Warning: {w}")

    cleaning_plan = plan.get("cleaning_plan", {})
    reasoning = plan.get("reasoning", {})

    # ── Step 3: Drop duplicates ──────────────────────────────────────
    if cleaning_plan.get("drop_duplicates") :
        df, dropped = drop_duplicate_rows(df)
        log.append(f"✔ Dropped {dropped} duplicate rows.")

    # ── Step 4: Standardize column names ─────────────────────────────
    if cleaning_plan.get("standardize_column_names") :
        old_cols = df.columns.tolist()
        df = standardize_column_names(df)
        new_cols = df.columns.tolist()
        changes = [
            f"'{old}' → '{new}'"
            for old, new in zip(old_cols, new_cols)
            if old != new
        ]

        if changes:
            log.append(f"✔ Standardized column names: {', '.join(changes)}.")
        else:
            log.append("✔ Column names already clean — no changes needed.")

        # Remap plan keys to new column names
        col_mapping =dict(zip(old_cols, new_cols))
        cleaning_plan = _remap_plan_keys(cleaning_plan, col_mapping)
        reasoning = _remap_plan_keys(reasoning, col_mapping)

    # ── Step 5: Fix dtypes ───────────────────────────────────────────
    dtype_fixes = cleaning_plan.get("dtype_fixes", {})
    if dtype_fixes :
        dtype_fixes = {
            col: fix for col, fix in dtype_fixes.items() if col in df.columns
        }
        df, dtype_log = apply_dtype_fixes(df, dtype_fixes)
        for entry in dtype_log :
            reason = reasoning.get(entry.split("'")[1], "")
            log.append(f"✔ Dtype fix — {entry}" + (f" Reason: {reason}" if reason else ""))

    # ── Step 6: Re-profile after dtype fixes ────────────────────────
    # Re-profile so null/outlier tools work on correct dtypes
    profile_after_dtype = profile_dataset(df)

    # ── Step 7: Handle nulls ─────────────────────────────────────────
    null_handling = cleaning_plan.get("null_handling" , {})
    if null_handling :
        # Override profile strategies with LLM decisions
        for col, strategy in null_handling.items() :
            if col in profile_after_dtype["columns"] :
                profile_after_dtype["columns"][col]["_override_strategy"] = strategy
        
        df, null_log = _apply_null_with_override(df, profile_after_dtype, null_handling)
        for entry in null_log :
            log.append(f"✔ Null handling — {entry}")

    # ── Step 8: Handle outliers ──────────────────────────────────────
    outlier_handling = cleaning_plan.get("outlier_handling", {})
    if outlier_handling :
        profile_for_outliers = profile_dataset(df)
        df, outlier_log = _apply_outlier_with_override(df, profile_for_outliers, outlier_handling)
        for entry in outlier_log :
            log.append(f"✔ Outlier handling — {entry}")

    # ── Step 9: Generate visualizations ─────────────────────────────
    log.append("▶ Generating visualizations...")
    final_profile = profile_dataset(df)
    visualizations = generate_all_visualizations(df, final_profile)
    result["visualizations"] = visualizations
    log.append(f"  Generated {len(visualizations)} charts.")

    # ── Done ─────────────────────────────────────────────────────────
    result["cleaned_df"] = df
    log.append("✅ Cleaning complete.")

    return result


# ── Helper Functions ─────────────────────────────────────────────────
def _remap_plan_keys(plan_section : dict, col_mapping : dict) -> dict :
    """
    After column names are standardized, remap any plan keys that referred the old column names to new ones.
    """
    if not isinstance(plan_section, dict) :
        return plan_section
    
    remapped = {}
    for key, value in plan_section.items() :
        new_key = col_mapping.get(key, key)
        if isinstance(value, dict) :
            remapped[new_key] = _remap_plan_keys(value, col_mapping)
        else :
            remapped[new_key] = value
    
    return remapped

def _apply_null_with_override(df : pd.DataFrame, profile : dict, null_handling : dict) -> tuple[pd.DataFrame, list] :
    """
    Applies null hadling using LLM decisions.
    Falls back to auto-strategy for columns not in plan.
    """

    from src.tools.null_tools import (
        fill_with_mean, fill_with_median, fill_with_mode,
        fill_forward, drop_column, drop_rows_with_nulls
    )

    log = []
    
    for col, strategy in null_handling.items() :
        if col not in df.columns : 
            continue
        if df[col].isnull().sum == 0 :
            continue

        if strategy == "fill_mean" :
            df = fill_with_mean(df, col)
            log.append(f"'{col}': filled nulls with mean.")
        
        elif strategy == "fill_median" :
            df = fill_with_median(df, col)
            log.append(f"'{col}': filled nulls with median.")

        elif strategy == "fill_mode" :
            df = fill_with_mode(df, col)
            log.append(f"'{col}': filled nulls with mode.")

        elif strategy == "fill_forward" :
            df = fill_forward(df, col)
            log.append(f"'{col}': filled nulls with forward fill.")

        elif strategy == "drop_column" :
            df = drop_column(df, col)
            log.append(f"'{col}': dropped column due to too many nulls.")

        elif strategy == "drop_rows" :
            before = len(df)
            df = drop_rows_with_nulls(df, col)
            log.append(f"'{col}': {before - len(df)} rows dropped.")

    return df, log

def _apply_outlier_with_override(df : pd.DataFrame, profile : dict, outlier_handling : dict) -> tuple[pd.DataFrame, list] :
    """
    Applies outlier handling using LLM decisions.
    """

    from src.tools.outlier_tools import cap_outliers_iqr, drop_outliers_iqr

    log = []

    for col, startegy in outlier_handling.items() :
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        if startegy == "cap" :
            df, count = cap_outliers_iqr(df, col)
            log.append(f"'{col}': {count} outliers capped.")

        elif startegy == "drop" :
            df, count = drop_outliers_iqr(df, col)
            log.append(f"'{col}': {count} outlier rows dropped.")

    return df, log

