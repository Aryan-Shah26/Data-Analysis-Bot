from typing import TypedDict, Optional
import pandas as pd

class AgentState(TypedDict) :
    # Input
    df_raw : pd.DataFrame
    file_size_mb : float
    domain_context : Optional[str]
    target_column : Optional[str]

    #Profiling
    profile : dict
    summary_text : str
    chunked : bool

    #Planning
    plan : dict
    plan_approved : bool
    plan_overrides : dict

    #Execution
    df_cleaned : pd.DataFrame
    cleaning_log : dict
    warnings : list

    #Analysis
    target_analysis : dict
    feature_suggestions : list

    #Visualization
    visualizations : dict

    #NL Query
    nl_query_history : list

    #RAG
    rag_index : object
    rag_chunks : list
