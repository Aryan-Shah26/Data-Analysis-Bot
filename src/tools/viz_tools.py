import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import missingno as msno
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, required for Streamlit
import matplotlib.pyplot as plt
import io
import base64
from config import MIN_ROWS_FOR_KDE, CATEGORICAL_TOP_N, HEATMAP_MIN_COLUMNS

def plot_distribution(df: pd.DataFrame, column: str) -> go.Figure:
    """Histogram for a numeric column."""
    try:
        fig = px.histogram(
            df, x=column,
            marginal = "kde" if df[column].dropna().shape[0] >= MIN_ROWS_FOR_KDE else None,
            title=f"Distribution of {column}",
            template="plotly_white",
            color_discrete_sequence=["#636EFA"]
        )
        fig.update_layout(bargap=0.1)
        return fig
    except Exception:
        # Fall back to plain histogram if KDE fails
        fig = px.histogram(
            df, x=column,
            title=f"Distribution of {column}",
            template="plotly_white",
            color_discrete_sequence=["#636EFA"]
        )
        fig.update_layout(bargap=0.1)
        return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Correlation heatmap for all numeric columns.
    """
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        title="Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1
    )
    return fig


def plot_categorical_bar(df: pd.DataFrame, column: str) -> go.Figure:
    """
    Bar chart of top N value counts for a categorical column.
    """
    counts = df[column].value_counts().head(CATEGORICAL_TOP_N).reset_index()
    counts.columns = [column, "count"]
    fig = px.bar(
        counts, x=column, y="count",
        title=f"Top {CATEGORICAL_TOP_N} values in '{column}'",
        template="plotly_white",
        color_discrete_sequence=["#EF553B"]
    )
    return fig


def plot_nullity_matrix(df: pd.DataFrame) -> str:
    """
    Generates a missingno nullity matrix.
    Returns as a base64 PNG string for Streamlit display.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    msno.matrix(df, ax=ax, sparkline=False)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)

    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_b64


def plot_boxplot(df: pd.DataFrame, column: str) -> go.Figure:
    """
    Box plot to visualize outlier spread for a numeric column.
    """
    fig = px.box(
        df, y=column,
        title=f"Boxplot of {column}",
        template="plotly_white",
        color_discrete_sequence=["#00CC96"]
    )
    return fig


def generate_all_visualizations(df: pd.DataFrame, profile: dict) -> dict:
    """
    Master function — generates all relevant charts based on the profile.
    Returns a dict of chart name -> Plotly figure (or base64 string for nullity).
    """
    charts = {}

    # Nullity matrix
    charts["nullity_matrix"] = plot_nullity_matrix(df)

    # Correlation heatmap
    heatmap = plot_correlation_heatmap(df)
    if heatmap:
        charts["correlation_heatmap"] = heatmap

    # Per-column charts
    for col, col_profile in profile["columns"].items():
        if col not in df.columns:
            continue

        type_cat = col_profile.get("type_category", "")

        if type_cat in ("numeric", "should_be_numeric"):
            if pd.api.types.is_numeric_dtype(df[col]):
                charts[f"dist_{col}"] = plot_distribution(df, col)
                charts[f"box_{col}"] = plot_boxplot(df, col)

        elif type_cat in ("categorical",):
            charts[f"bar_{col}"] = plot_categorical_bar(df, col)

    return charts