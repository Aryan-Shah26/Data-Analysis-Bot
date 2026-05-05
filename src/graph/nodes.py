import logging 
import pandas as pd
from config import LARGE_FILE_THRESHOLD_MB, FILE_SIZE_WARNING_MB, FILE_SIZE_LIMIT_MB

from src.graph.state import AgentState
from src.tools.eda_tools import (
    profile_dataset,
    profile_dataset_chunked,
    generate_summary_text,
    drop_duplicate_rows,
    standardize_column_names
)
from src.tools.dtype_tools import apply_dtype_fixes
from src.tools.null_tools import apply_null_fixes
from src.tools.outlier_tools import apply_outlier_fixes
from src.tools.viz_tools import generate_all_visualizations
from src.tools.analysis_tools import analyze_target_column, suggest_feature_engineering
from src.tools.rag_tools import build_rag_index, retrieve_context
from src.orchestrator import build_cleaning_plan, validate_plan

logger = logging.getLogger(__name__)

#NODE 1: Load & Validate
def node_load(state : AgentState) -> dict :
    """
    Validate the uploaded files.
    Checks file  size and sets chunked flag.
    """

    logger.info("NODE : load")
    df = state["df_raw"]
    file_size_mb = state["file_size_mb"]
    log = []

    if file_size_mb > FILE_SIZE_LIMIT_MB :
        raise ValueError(
            f"File too large ({file_size_mb:.2f} MB). Maximum allowed size is {FILE_SIZE_LIMIT_MB} MB."
        )
    if file_size_mb > FILE_SIZE_WARNING_MB :
        log.append(f"⚠ Large file ({file_size_mb:.1f}MB) — chunked profiling will be used.")

    chunked = file_size_mb > LARGE_FILE_THRESHOLD_MB
    log.append(f"▶ File loaded: {df.shape[0]:,} rows × {df.shape[1]} columns ({file_size_mb:.1f}MB).")

    return {
        "chunked": chunked,
        "cleaning_log": log,
        "warnings": [],
        "plan_approved": False,
        "plan_overrides": {},
        "target_analysis": {},
        "feature_suggestions": [],
        "nl_query_history": [],
        "visualizations": {},
    }

#NODE 2: Profile
def node_profile(state: AgentState) -> dict:
    """
    Profiles the dataset — chunked if large.
    Merges RAG context if provided.
    """
    logger.info("NODE: profile")
    df = state["df_raw"]
    log = list(state["cleaning_log"])

    # Initialize RAG variables before conditional block
    rag_index = None
    rag_chunks = []
    domain_context = state.get("domain_context", "")

    if domain_context and domain_context.strip():
        log.append("▶ Building RAG index from domain context...")
        rag_index, rag_chunks = build_rag_index(domain_context)
        log.append(f"  RAG index built: {len(rag_chunks)} chunks.")

    # Profile the dataset
    log.append("▶ Profiling dataset...")
    if state["chunked"]:
        log.append("  Using chunked profiling (large file).")
        profile = profile_dataset_chunked(df)
    else:
        profile = profile_dataset(df)

    summary_text = generate_summary_text(profile)

    log.append(f"  Shape: {profile['shape']['rows']:,} rows × {profile['shape']['columns']} columns.")
    log.append(f"  Duplicate rows: {profile['duplicate_rows']}.")

    return {
        "profile": profile,
        "summary_text": summary_text,
        "cleaning_log": log,
        "rag_index": rag_index,
        "rag_chunks": rag_chunks,
    }

#NODE 3: Plan
def node_plan(state : AgentState) -> dict :
    """
    Sends profile to LLM; gets structured cleaning plan.
    Validates plan against actual column names.
    Retrieves RAG context per column if available.
    """

    logger.info("NODE : plan")
    log = list(state["cleaning_log"])
    profile = state["profile"]

    #Retrieve RAG context for each column if index is available
    rag_context  = {}
    rag_index = state.get("rag_index")
    rag_chunks = state.get("rag_chunks", []) 

    if rag_index is not None :
        log.append("▶ Retrieving domain context per column...")
        for col in profile["columns"].keys() :
            context = retrieve_context(
                query = f"Column {col} desscription constraints valid range",
                index = rag_index,
                chunks = rag_chunks
            )
            if context :
                rag_context[col] = context

    log.append("▶ Sending profile to LLM for cleaning plan...")
    plan = build_cleaning_plan(profile, rag_context=rag_context)
    warnings = validate_plan(plan, profile)

    if warnings :
        for w in warnings : 
            log.append(f"⚠ Plan warning: {w}")

    return {
        "plan" : plan,
        "warnings" : warnings,
        "cleaning_log" : log,
    }

# NODE 4: HITL
def node_human_approval(state : AgentState) -> dict :
    """
    Pauses the graph for human review of the cleaning plan.
    In streamlit, the frontend reads plan_approved from state and resumes the graph when user clicks Approve.
    This node itself just merges any overrides into the plan.
    """

    logger.info("NODE: human_approval")
    log = list(state["cleaning_log"])
    plan = state["plan"]
    overrides = state.get("plan_overrides", {})

    if overrides :
        log.append(f"▶ Applying {len(overrides)} user override(s) to cleaning plan.")
        cleaning_plan = plan.get("cleaning_plan", {})

        # Apply null handling overrides
        for col, strategy in overrides.items() :
            if col in cleaning_plan.get("null_handling", {}) :
                cleaning_plan["null_handling"][col] = strategy
                log.append(f"  Override: '{col}' null strategy → {strategy}")
        plan["cleaning_plan"] = cleaning_plan

    log.append("✔ Cleaning plan approved.")

    return {
        "plan": plan,
        "cleaning_log": log,
    }

#NODE 5: Execute
def node_execute(state : AgentState) -> dict :
    """
    Executes the cleaning plan on the dataset using pre-coded tools.
    """

    logger.info("NODE: execute")
    df = state["df_raw"].copy()
    log = list(state["cleaning_log"])
    plan = state["plan"]
    profile = state["profile"]
    cleaning_plan = plan.get("cleaning_plan", {})
    reasoning = plan.get("reasoning", {})

    #Drop duplicates
    if cleaning_plan.get("drop_duplicates"):
        df, dropped = drop_duplicate_rows(df)
        log.append(f"✔ Dropped {dropped} duplicate rows.")

    # Standardize column names
    if cleaning_plan.get("standardize_column_names"):
        old_cols = df.columns.tolist()
        df = standardize_column_names(df)
        new_cols = df.columns.tolist()
        col_mapping = dict(zip(old_cols, new_cols))
        changes = [f"'{o}' → '{n}'" for o, n in col_mapping.items() if o != n]
        if changes:
            log.append(f"✔ Standardized column names: {', '.join(changes)}.")
        cleaning_plan = _remap_keys(cleaning_plan, col_mapping)
        reasoning = _remap_keys(reasoning, col_mapping)

    # Fix dtypes
    dtype_fixes = {
        col: fix for col, fix in cleaning_plan.get("dtype_fixes", {}).items()
        if col in df.columns
    }
    if dtype_fixes:
        df, dtype_log = apply_dtype_fixes(df, dtype_fixes)
        for entry in dtype_log:
            log.append(f"✔ Dtype fix — {entry}")

    # Re-profile after dtype fixes
    from src.tools.eda_tools import profile_dataset
    profile_after_dtype = profile_dataset(df)

    # Handle nulls
    null_handling = {
        col: s for col, s in cleaning_plan.get("null_handling", {}).items()
        if col in df.columns
    }
    if null_handling:
        df, null_log = _apply_nulls(df, null_handling)
        for entry in null_log:
            log.append(f"✔ Null handling — {entry}")

    # Handle outliers
    outlier_handling = {
        col: s for col, s in cleaning_plan.get("outlier_handling", {}).items()
        if col in df.columns
    }
    if outlier_handling:
        profile_for_outliers = profile_dataset(df)
        df, outlier_log = _apply_outliers(df, outlier_handling)
        for entry in outlier_log:
            log.append(f"✔ Outlier handling — {entry}")

    log.append("✅ Cleaning complete.")

    return {
        "df_cleaned": df,
        "cleaning_log": log,
    }

#NODE 6: Analyze
def node_analyze(state: AgentState) -> dict :
    """
    Runs target variable analysis and feature engineering suggestions if target column specificed.
    """
    logger.info("NODE: analyze")
    log = list(state["cleaning_log"])
    df = state["df_cleaned"]
    target_column = state.get("target_column")
    target_analysis = {}
    feature_suggestions = []

    if target_column and target_column in df.columns : 
        log.append(f"▶ Analyzing target column: '{target_column}'...")
        target_analysis = analyze_target_column(df, target_column)
        log.append(f"  Target type detected: {target_analysis.get('task_type', 'unknown')}.")

        log.append("▶ Generating feature engineering suggestions...")
        feature_suggestions = suggest_feature_engineering(df, target_column)
        log.append(f"  {len(feature_suggestions)} suggestions generated.")

    else :
        log.append("▶ No target column specified — skipping target analysis.")
    return {
        "target_analysis": target_analysis,
        "feature_suggestions": feature_suggestions,
        "cleaning_log": log,
    }

#NODE 7: Visualize
def node_visualize(state: AgentState) -> dict :
    """
    Generates visualizations based on the cleaned dataset and target analysis.
    """
    logger.info("NODE: visualize")
    log = list(state["cleaning_log"])
    df = state["df_cleaned"]

    log.append("▶ Generating visualizations...")
    from src.tools.eda_tools import profile_dataset
    final_profile = profile_dataset(df) 
    visualizations = generate_all_visualizations(df, final_profile)
    
    log.append(f"  Generated {len(visualizations)} charts.")

    return {
        "visualizations": visualizations,
        "cleaning_log": log,
    }

# Helper functions
def _remap_keys(plan_section: dict, col_mapping: dict) -> dict :
    if not isinstance(plan_section, dict) :
        return plan_section
    return{
        col_mapping.get(k,k): (
            _remap_keys(v, col_mapping) if isinstance(v, dict) else v
        )
        for k, v in plan_section.items()  
    }

def _apply_nulls(df : pd.DataFrame, null_handling : dict) -> tuple :
    from src.tools.null_tools import fill_with_mean, fill_with_median, fill_with_mode, fill_forward, drop_column, drop_rows_with_nulls
    log = []
    strategies = {
        "fill_mean": (fill_with_mean, "filled nulls with mean."),
        "fill_median": (fill_with_median, "filled nulls with median."),
        "fill_mode": (fill_with_mode, "filled nulls with mode."),
        "fill_forward": (fill_forward, "filled nulls with forward fill."),
    }

    for col, strategy in null_handling.items() :
        if col not in df.columns or df[col].isnull().sum() == 0 :
            continue
        if strategy in strategies :
            fn, msg = strategies[strategy]
            df = fn(df,col)
            log.append(f"'{col}': {msg}")
        elif strategy == "drop_column" :
            df = drop_column(df, col)
            log.append(f"'{col}': column dropped (too many nulls).")
        elif strategy == "drop_rows" :
            before = len(df)
            df = drop_rows_with_nulls(df, col)
            log.append(f"'{col}': dropped {before - len(df)} rows with nulls.")
    
    return df, log

def _apply_outliers(df : pd.DataFrame, outlier_handling : dict) -> tuple :
    from src.tools.outlier_tools import cap_outliers_iqr, drop_outliers_iqr
    log = []
    for col, strategy in outlier_handling.items() :
        if col not in df.columns :
            continue
        if not pd.api.types.is_numeric_dtype(df[col]) :
            continue
        if strategy == "cap" :
            df, count = cap_outliers_iqr(df, col)
            log.append(f"'{col}': {count} outliers capped.")
        elif strategy == "drop" :
            df, count = drop_outliers_iqr(df, col)
            log.append(f"'{col}': {count} outlier rows dropped.")
    return df, log