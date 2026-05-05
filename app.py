import streamlit as st
import pandas as pd
import base64
import logging
from src.graph.graph import agent_graph, approval_graph
from src.graph.state import AgentState
from src.orchestrator import answer_followup
from src.nl_query import run_nl_query, route_question, narrate_result

logging.basicConfig(level=logging.INFO)

# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA Agent V2",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .stChatFloatingInputContainer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: var(--background-color);
        padding: 1rem 1rem 1.5rem 1rem;
        z-index: 999;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    .stChatMessage { margin-bottom: 0.5rem; }
    section[data-testid="stChatMessageContainer"] { padding-bottom: 100px; }
    [data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────
defaults = {
    "graph_state": None,
    "stage": "upload",
    "chat_history": [],
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


def get_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def get_file_size_mb(uploaded_file) -> float:
    uploaded_file.seek(0, 2)
    size = uploaded_file.tell() / (1024 * 1024)
    uploaded_file.seek(0)
    return round(size, 2)


# ── Header ─────────────────────────────────────────────────────────
st.title("🤖 Autonomous EDA & Data Cleaning Agent")
st.divider()

# ══════════════════════════════════════════════════════════════════
# STAGE: UPLOAD
# ══════════════════════════════════════════════════════════════════
if st.session_state.stage == "upload":

    # ── File upload + options ──────────────────────────────────────
    col_upload, col_context, col_target = st.columns([2, 2, 2])

    with col_upload:
        st.markdown("#### 📂 Upload Dataset")
        st.caption("Upload a .csv dataset.")
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            label_visibility="collapsed"
        )

    with col_context:
        st.markdown("#### 📄 Domain Context *(optional)*")
        st.caption("Upload a .txt data dictionary to help the agent make smarter decisions.")
        context_file = st.file_uploader(
            "Upload context",
            type=["txt"],
            label_visibility="collapsed",
            key="context_uploader"
        )
        domain_context = ""
        if context_file is not None:
            try:
                domain_context = context_file.read().decode("utf-8")
                st.success(f"Loaded: {len(domain_context):,} characters.")
            except Exception as e:
                st.error(f"Could not read file: {e}")

    with col_target:
        st.markdown("#### 🎯 Target Column *(optional)*")
        st.caption("For modeling recommendations.")
        target_column_input = st.text_input(
            "Target column",
            placeholder="e.g. survived",
            label_visibility="collapsed"
        )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            file_size_mb = get_file_size_mb(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        st.divider()
        st.markdown("#### 👀 Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
        st.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns · {file_size_mb}MB")

        st.divider()
        if st.button("🚀 Run Agent", type="primary", use_container_width=True):
            initial_state: AgentState = {
                "df_raw": df,
                "file_size_mb": file_size_mb,
                "domain_context": domain_context,
                "target_column": target_column_input.strip() or None,
                "profile": {},
                "summary_text": "",
                "chunked": False,
                "plan": {},
                "plan_approved": False,
                "plan_overrides": {},
                "df_cleaned": None,
                "cleaning_log": [],
                "warnings": [],
                "target_analysis": {},
                "feature_suggestions": [],
                "visualizations": {},
                "nl_query_history": [],
                "rag_index": None,
                "rag_chunks": [],
            }

            with st.spinner("Agent is profiling and planning..."):
                try:
                    for event in agent_graph.stream(
                        initial_state, stream_mode="values"
                    ):
                        graph_state = event

                    st.session_state.graph_state = graph_state
                    st.session_state.stage = "awaiting_approval"
                    st.rerun()
                except Exception as e:
                    st.error(f"Agent failed: {e}")
                    st.exception(e)
    # else:
    #     st.markdown("### What this agent does")
    #     c1, c2, c3 = st.columns(3)
    #     with c1:
    #         st.markdown("**🔍 Profiling**")
    #         st.caption("Schema, nulls, skewness, outliers — automatic.")
    #     with c2:
    #         st.markdown("**🧹 Cleaning**")
    #         st.caption("Statistically correct cleaning with human approval.")
    #     with c3:
    #         st.markdown("**📊 Dashboard**")
    #         st.caption("Full EDA + target analysis + NL querying.")

# ══════════════════════════════════════════════════════════════════
# STAGE: AWAITING APPROVAL
# ══════════════════════════════════════════════════════════════════
elif st.session_state.stage == "awaiting_approval":
    graph_state = st.session_state.graph_state
    plan = graph_state.get("plan", {})
    cleaning_plan = plan.get("cleaning_plan", {})
    reasoning = plan.get("reasoning", {})
    profile = graph_state.get("profile", {})

    st.subheader("🧠 Review Cleaning Plan")
    st.caption("Review and approve the cleaning plan before execution.")
    st.info(graph_state.get("summary_text", ""))

    for w in graph_state.get("warnings", []):
        st.warning(w)

    st.divider()
    overrides = {}

    dtype_fixes = cleaning_plan.get("dtype_fixes", {})
    if dtype_fixes:
        st.markdown("**Dtype Fixes**")
        for col, fix in dtype_fixes.items():
            st.markdown(f"&nbsp;&nbsp;`{col}` → convert to **{fix.replace('should_be_', '')}**")

    null_handling = cleaning_plan.get("null_handling", {})
    if null_handling:
        st.markdown("**Null Handling**")
        null_options = ["fill_mean", "fill_median", "fill_mode",
                        "fill_forward", "drop_rows", "drop_column"]
        for col, strategy in null_handling.items():
            c1, c2, c3 = st.columns([2, 2, 3])
            with c1:
                st.markdown(f"`{col}`")
            with c2:
                new_strategy = st.selectbox(
                    f"strategy_{col}",
                    null_options,
                    index=null_options.index(strategy) if strategy in null_options else 0,
                    label_visibility="collapsed",
                    key=f"null_{col}"
                )
            with c3:
                reason = reasoning.get(col, "")
                st.caption(reason[:100] + "..." if len(reason) > 100 else reason)
            if new_strategy != strategy:
                overrides[col] = new_strategy

    outlier_handling = cleaning_plan.get("outlier_handling", {})
    if outlier_handling:
        st.markdown("**Outlier Handling**")
        for col, strategy in outlier_handling.items():
            col_info = profile.get("columns", {}).get(col, {})
            outlier_count = col_info.get("outlier_count", "?")
            st.markdown(
                f"&nbsp;&nbsp;`{col}` → **{strategy}** "
                f"({outlier_count} outliers detected)"
            )

    st.divider()
    col_approve, col_cancel = st.columns(2)

    with col_approve:
        if st.button("✅ Approve & Execute", type="primary", use_container_width=True):
            with st.spinner("Executing cleaning plan..."):
                try:
                    current_state = st.session_state.graph_state
                    current_state["plan_approved"] = True
                    current_state["plan_overrides"] = overrides

                    for event in approval_graph.stream(
                        current_state, stream_mode="values"
                    ):
                        graph_state = event

                    st.session_state.graph_state = graph_state
                    st.session_state.stage = "done"
                    st.rerun()
                except Exception as e:
                    st.error(f"Execution failed: {e}")
                    st.exception(e)

    with col_cancel:
        if st.button("✖ Cancel", use_container_width=True):
            for key in defaults:
                st.session_state[key] = defaults[key]
            st.rerun()

# ══════════════════════════════════════════════════════════════════
# STAGE: DONE
# ══════════════════════════════════════════════════════════════════
elif st.session_state.stage == "done":
    graph_state = st.session_state.graph_state
    result_df = graph_state.get("df_cleaned")
    original_df = graph_state.get("df_raw")
    log = graph_state.get("cleaning_log", [])
    visualizations = graph_state.get("visualizations", {})
    target_analysis = graph_state.get("target_analysis", {})
    feature_suggestions = graph_state.get("feature_suggestions", [])

    # Reset button
    if st.button("🔄 Upload New File", use_container_width=False):
        for key in defaults:
            st.session_state[key] = defaults[key]
        st.rerun()

    st.divider()

    # ── Tabs ───────────────────────────────────────────────────────
    tab_dashboard, tab_target, tab_chat, tab_log = st.tabs([
        "📊 Dashboard",
        "🎯 Target Analysis",
        "💬 Chat",
        "🗂 Agent Log"
    ])

    # ── Tab 1: Dashboard ───────────────────────────────────────────
    with tab_dashboard:
        # Summary
        st.info(graph_state.get("summary_text", ""))

        # Before / After
        col_before, col_after = st.columns(2)
        with col_before:
            st.markdown("**Original Data**")
            st.dataframe(original_df.head(20), use_container_width=True)
            st.caption(
                f"{original_df.shape[0]:,} rows · "
                f"{original_df.shape[1]} cols · "
                f"{original_df.isnull().sum().sum()} nulls"
            )
        with col_after:
            st.markdown("**Cleaned Data**")
            st.dataframe(result_df.head(20), use_container_width=True)
            st.caption(
                f"{result_df.shape[0]:,} rows · "
                f"{result_df.shape[1]} cols · "
                f"{result_df.isnull().sum().sum()} nulls"
            )

        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=get_csv_download(result_df),
            file_name="cleaned_data.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.divider()

        if not visualizations:
            st.info("No visualizations generated.")
        else:
            if "nullity_matrix" in visualizations:
                st.markdown("**Missing Value Matrix**")
                st.image(
                    base64.b64decode(visualizations["nullity_matrix"]),
                    use_container_width=True,
                    caption="White lines = missing values"
                )

            if "correlation_heatmap" in visualizations:
                st.plotly_chart(
                    visualizations["correlation_heatmap"],
                    use_container_width=True
                )

            numeric_cols = [
                k.replace("dist_", "")
                for k in visualizations
                if k.startswith("dist_")
            ]
            if numeric_cols:
                st.markdown("**Numeric Columns**")
                for col in numeric_cols:
                    c1, c2 = st.columns(2)
                    with c1:
                        if f"dist_{col}" in visualizations:
                            st.plotly_chart(
                                visualizations[f"dist_{col}"],
                                use_container_width=True
                            )
                    with c2:
                        if f"box_{col}" in visualizations:
                            st.plotly_chart(
                                visualizations[f"box_{col}"],
                                use_container_width=True
                            )

            cat_cols = [
                k.replace("bar_", "")
                for k in visualizations
                if k.startswith("bar_")
            ]
            if cat_cols:
                st.markdown("**Categorical Columns**")
                cat_cols_layout = st.columns(min(len(cat_cols), 2))
                for i, col in enumerate(cat_cols):
                    with cat_cols_layout[i % 2]:
                        if f"bar_{col}" in visualizations:
                            st.plotly_chart(
                                visualizations[f"bar_{col}"],
                                use_container_width=True
                            )

    # ── Tab 2: Target Analysis ─────────────────────────────────────
    with tab_target:
        # Target column selector
        col_options = ["None"] + result_df.columns.tolist()
        selected_target = st.selectbox(
            "Select target column",
            col_options,
            key="target_col_select"
        )

        if selected_target != "None":
            if st.button("🎯 Run Target Analysis", type="primary"):
                from src.tools.analysis_tools import (
                    analyze_target_column,
                    suggest_feature_engineering
                )
                with st.spinner("Analyzing..."):
                    ta = analyze_target_column(result_df, selected_target)
                    fs = suggest_feature_engineering(result_df, selected_target)
                    st.session_state.graph_state["target_analysis"] = ta
                    st.session_state.graph_state["feature_suggestions"] = fs
                    target_analysis = ta
                    feature_suggestions = fs
                st.success(f"Analysis complete for '{selected_target}'.")

        if not target_analysis:
            st.info("Select a target column above and click Run Target Analysis.")
        else:
            task_type = target_analysis.get("task_type", "unknown")
            st.subheader(f"Task Type: {task_type.title()}")

            if task_type == "classification":
                st.markdown(f"**Classes:** {target_analysis.get('n_classes')}")
                st.markdown(f"**Class Balance:** {target_analysis.get('class_balance', '').replace('_', ' ').title()}")

                class_dist = target_analysis.get("class_distribution", {})
                if class_dist:
                    import plotly.express as px
                    dist_df = pd.DataFrame(
                        list(class_dist.items()),
                        columns=["class", "count"]
                    )
                    fig = px.bar(
                        dist_df, x="class", y="count",
                        title="Class Distribution",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            elif task_type == "regression":
                m1, m2, m3 = st.columns(3)
                m1.metric("Mean", target_analysis.get("mean"))
                m2.metric("Median", target_analysis.get("median"))
                m3.metric("Skewness", target_analysis.get("skewness"))

            st.divider()
            corrs = target_analysis.get("feature_correlations", {})
            if corrs:
                st.markdown("**Feature Correlations with Target**")
                corr_data = [
                    {
                        "Feature": col,
                        "Correlation": info["correlation"],
                        "P-Value": info["pvalue"],
                        "Significant": "✅" if info["significant"] else "❌"
                    }
                    for col, info in corrs.items()
                ]
                st.dataframe(pd.DataFrame(corr_data), use_container_width=True)

            st.divider()
            recs = target_analysis.get("model_recommendations", [])
            if recs:
                st.markdown("**Model Recommendations**")
                for rec in recs:
                    priority = rec.get("priority", "")
                    icon = "🟢" if priority == "start here" else "🔵" if priority == "recommended" else "🟡"
                    with st.expander(f"{icon} {rec['model']} — {priority}"):
                        st.markdown(rec.get("reason", ""))
                        if rec.get("note"):
                            st.warning(rec["note"])

            st.divider()
            if feature_suggestions:
                st.markdown("**Feature Engineering Suggestions**")
                for s in feature_suggestions:
                    with st.expander(f"`{s['column']}` → {s['suggestion']}"):
                        st.caption(s.get("reason", ""))
                        st.code(s.get("code_hint", ""), language="python")

    # ── Tab 3: Chat ────────────────────────────────────────────────
    with tab_chat:
        st.caption("Ask anything — analytical questions get executed as queries, conversational questions get answered by the agent.")

        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(turn["user"])
            with st.chat_message("assistant"):
                if turn.get("chart"):
                    st.plotly_chart(turn["chart"], use_container_width=True)
                if turn.get("code"):
                    with st.expander("View generated code", expanded=False):
                        st.code(turn["code"], language="python")
                if turn.get("response"):
                    st.markdown(turn["response"])

        user_input = st.chat_input("Ask something about your dataset...")

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                route = route_question(
                    user_input,
                    df_columns=result_df.columns.tolist(),
                    chat_history=st.session_state.chat_history
                )

                if route == "analytical":
                    with st.spinner("Running query..."):
                        query_result = run_nl_query(
                            question=user_input,
                            df=result_df,
                            query_history=[
                                h for h in st.session_state.chat_history
                                if h.get("code")
                            ]
                        )

                    if query_result.get("error"):
                        st.error(query_result["error"])
                        turn = {
                            "user": user_input,
                            "code": query_result.get("code", ""),
                            "chart": None,
                            "response": f"❌ {query_result['error']}"
                        }
                    else:
                        narration = narrate_result(
                            question=user_input,
                            code=query_result.get("code", ""),
                            result=query_result.get("result", "")
                        )
                        if query_result.get("chart"):
                            st.plotly_chart(
                                query_result["chart"],
                                use_container_width=True
                            )
                        st.markdown(narration)
                        with st.expander("View generated code", expanded=False):
                            st.code(query_result.get("code", ""), language="python")
                        turn = {
                            "user": user_input,
                            "code": query_result.get("code", ""),
                            "chart": query_result.get("chart"),
                            "response": narration
                        }
                else:
                    with st.spinner("Thinking..."):
                        response = answer_followup(
                            question=user_input,
                            profile=graph_state.get("profile", {}),
                            cleaning_log=log,
                            chat_history=[
                                {"user": h["user"], "assistant": h.get("response", "")}
                                for h in st.session_state.chat_history
                            ]
                        )
                    st.markdown(response)
                    turn = {
                        "user": user_input,
                        "code": None,
                        "chart": None,
                        "response": response
                    }

            st.session_state.chat_history.append(turn)
            st.rerun()

    # ── Tab 4: Agent Log ───────────────────────────────────────────
    with tab_log:
        st.subheader("📋 Dataset Summary")
        st.info(graph_state.get("summary_text", ""))

        st.divider()
        st.subheader("🔀 Before & After")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original**")
            st.dataframe(original_df.head(20), use_container_width=True)
            st.caption(
                f"{original_df.shape[0]:,} rows · "
                f"{original_df.shape[1]} cols · "
                f"{original_df.isnull().sum().sum()} nulls"
            )
        with c2:
            st.markdown("**Cleaned**")
            st.dataframe(result_df.head(20), use_container_width=True)
            st.caption(
                f"{result_df.shape[0]:,} rows · "
                f"{result_df.shape[1]} cols · "
                f"{result_df.isnull().sum().sum()} nulls"
            )

        st.divider()
        st.subheader("🧠 Cleaning Log")
        for entry in log:
            if entry.startswith("✅"):
                st.success(entry)
            elif entry.startswith("▶"):
                st.markdown(f"**{entry}**")
            elif entry.startswith("✔"):
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{entry}")
            elif entry.startswith("⚠"):
                st.warning(entry)
            else:
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{entry}")

        st.divider()
        st.subheader("💡 LLM Reasoning")
        reasoning = graph_state.get("plan", {}).get("reasoning", {})
        if reasoning:
            for col, reason in reasoning.items():
                st.markdown(f"**`{col}`** — {reason}")
        else:
            st.caption("No reasoning available.")

        if graph_state.get("warnings"):
            st.divider()
            st.subheader("⚠️ Warnings")
            for w in graph_state["warnings"]:
                st.warning(w)