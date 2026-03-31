import streamlit as st
import pandas as pd
import io
import base64
from src.agent_loop import run_agent
from src.orchestrator import answer_followup

# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA Agent",
    page_icon="🤖",
    layout="wide"
)

# ── Session State Init ─────────────────────────────────────────────
# Session state persists data between Streamlit reruns
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_ran" not in st.session_state:
    st.session_state.agent_ran = False


# ── Helper: Download Button ────────────────────────────────────────
def get_csv_download(df: pd.DataFrame) -> str:
    return df.to_csv(index=False).encode("utf-8")


# ── Header ─────────────────────────────────────────────────────────
st.title("🤖 Autonomous EDA & Data Cleaning Agent")
st.caption("Upload a messy CSV. The agent will clean it, explain every decision, and generate a full EDA dashboard.")

st.divider()

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        st.success(f"Loaded: {uploaded_file.name}")

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Upload a CSV\n2. Agent profiles your data\n3. LLM builds a cleaning plan\n4. Tools execute the plan\n5. Dashboard is generated")
    st.divider()
    st.caption("Built with Groq · Llama 3.3 70B · Streamlit · Plotly")


# ── Main Area ──────────────────────────────────────────────────────
if uploaded_file is None:
    # Landing state
    st.info("👈 Upload a CSV file from the sidebar to get started.")

    st.markdown("### What this agent does")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔍 Profiling**")
        st.caption("Analyzes schema, dtypes, nulls, skewness, and outliers automatically.")
    with col2:
        st.markdown("**🧹 Cleaning**")
        st.caption("Fixes dtypes, fills nulls with statistically correct strategies, handles outliers.")
    with col3:
        st.markdown("**📊 Visualization**")
        st.caption("Generates distributions, correlation heatmaps, boxplots, and more.")

else:
    # Load the dataframe
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    # Show raw data preview
    with st.expander("📄 Raw Data Preview", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
        st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Run agent button
    if not st.session_state.agent_ran:
        if st.button("🚀 Run Agent", type="primary", use_container_width=True):
            with st.spinner("Agent is analyzing and cleaning your data..."):
                try:
                    result = run_agent(df)
                    st.session_state.result = result
                    st.session_state.agent_ran = True
                    st.session_state.chat_history = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Agent failed: {e}")
                    st.exception(e)
    else:
        if st.button("🔄 Reset & Upload New File", use_container_width=True):
            st.session_state.result = None
            st.session_state.agent_ran = False
            st.session_state.chat_history = []
            st.rerun()

    # ── Results ────────────────────────────────────────────────────
    if st.session_state.agent_ran and st.session_state.result:
        result = st.session_state.result

        st.divider()

        # ── Summary Card ───────────────────────────────────────────
        st.subheader("📋 Dataset Summary")
        st.info(result["summary_text"])

        # ── Before / After ─────────────────────────────────────────
        st.subheader("🔀 Before & After")
        col_before, col_after = st.columns(2)

        with col_before:
            st.markdown("**Original Data**")
            st.dataframe(result["original_df"].head(20), use_container_width=True)
            orig = result["original_df"]
            st.caption(f"{orig.shape[0]:,} rows · {orig.shape[1]} columns · {orig.isnull().sum().sum()} nulls")

        with col_after:
            st.markdown("**Cleaned Data**")
            st.dataframe(result["cleaned_df"].head(20), use_container_width=True)
            clean = result["cleaned_df"]
            st.caption(f"{clean.shape[0]:,} rows · {clean.shape[1]} columns · {clean.isnull().sum().sum()} nulls")

        # Download button
        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=get_csv_download(result["cleaned_df"]),
            file_name="cleaned_data.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.divider()

        # ── Agent Decision Log ─────────────────────────────────────
        st.subheader("🧠 Agent Decision Log")
        with st.expander("View full decision log", expanded=True):
            for entry in result["log"]:
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

        # ── LLM Reasoning ──────────────────────────────────────────
        st.subheader("💡 LLM Reasoning")
        reasoning = result["plan"].get("reasoning", {})
        if reasoning:
            with st.expander("View column-by-column reasoning", expanded=False):
                for col, reason in reasoning.items():
                    st.markdown(f"**`{col}`** — {reason}")
        else:
            st.caption("No reasoning returned.")

        # ── Warnings ───────────────────────────────────────────────
        if result["warnings"]:
            st.subheader("⚠️ Warnings")
            for w in result["warnings"]:
                st.warning(w)

        st.divider()

        # ── EDA Dashboard ──────────────────────────────────────────
        st.subheader("📊 EDA Dashboard")

        charts = result["visualizations"]

        # Nullity matrix (base64 image)
        if "nullity_matrix" in charts:
            st.markdown("**Missing Value Matrix**")
            img_b64 = charts["nullity_matrix"]
            st.image(
                base64.b64decode(img_b64),
                use_column_width=True,
                caption="Nullity matrix — white lines indicate missing values"
            )

        # Correlation heatmap
        if "correlation_heatmap" in charts:
            st.plotly_chart(
                charts["correlation_heatmap"],
                use_container_width=True
            )

        # Distribution + boxplot pairs
        numeric_cols = [
            key.replace("dist_", "")
            for key in charts
            if key.startswith("dist_")
        ]

        if numeric_cols:
            st.markdown("**Numeric Column Analysis**")
            for col in numeric_cols:
                c1, c2 = st.columns(2)
                with c1:
                    if f"dist_{col}" in charts:
                        st.plotly_chart(
                            charts[f"dist_{col}"],
                            use_container_width=True
                        )
                with c2:
                    if f"box_{col}" in charts:
                        st.plotly_chart(
                            charts[f"box_{col}"],
                            use_container_width=True
                        )

        # Categorical bar charts
        cat_cols = [
            key.replace("bar_", "")
            for key in charts
            if key.startswith("bar_")
        ]

        if cat_cols:
            st.markdown("**Categorical Column Analysis**")
            cat_chart_cols = st.columns(min(len(cat_cols), 2))
            for i, col in enumerate(cat_cols):
                with cat_chart_cols[i % 2]:
                    if f"bar_{col}" in charts:
                        st.plotly_chart(
                            charts[f"bar_{col}"],
                            use_container_width=True
                        )

        st.divider()

        # ── Follow-up Chat ─────────────────────────────────────────
        st.subheader("💬 Ask the Agent")
        st.caption("Ask follow-up questions about the data, cleaning decisions, or request further analysis.")

        # Display chat history
        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(turn["user"])
            with st.chat_message("assistant"):
                st.markdown(turn["assistant"])

        # Chat input
        user_question = st.chat_input("Ask something about your dataset...")

        if user_question:
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = answer_followup(
                        question=user_question,
                        profile=result["profile"],
                        cleaning_log=result["log"],
                        chat_history=st.session_state.chat_history
                    )
                st.markdown(response)

            st.session_state.chat_history.append({
                "user": user_question,
                "assistant": response
            })
            st.rerun()