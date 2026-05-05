import pandas as pd
import numpy as np
from scipy import stats
from config import SKEW_SYMMETRIC_MAX, SKEW_MODERATE_MAX

# Target column analysis

def analyze_target_column(df: pd.DataFrame, target_column: str) -> dict:
    """
    Analyze the target column to determine the task type and produce statistics relevant to modeling decisions.
     """
    result = {}
    col = df[target_column]

    n_unique = col.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(col)

    if not is_numeric or n_unique <= 10 :
        result["task_type"] = "classification"
        result["n_classes"] = n_unique
        result["class_distribution"] = col.value_counts().to_dict()
        result['class_balance'] = _asses_class_balance(col)

    else :
        result["task_type"] = "regression"
        result["mean"] = round(float(col.mean()), 4)
        result["median"] = round(float(col.median()), 4)
        result["std"] = round(float(col.std()), 4)
        result["skewness"] = round(float(col.skew()), 4)
        skew = abs(result["skewness"])
        result["skew_label"] = (
            "symmetric" if skew < SKEW_SYMMETRIC_MAX
            else "moderate_skew" if skew < SKEW_MODERATE_MAX
            else "high_skew"
        )

    # ── Feature correlations with target ──────────────────────────
    result["feature_correlations"] = _compute_correlations(df, target_column)

    # ── Model recommendations ──────────────────────────────────────
    result["model_recommendations"] = _recommend_models(result, df)

    return result

def _asses_class_balance(col: pd.Series) -> str :
    """
    Return a plain english assessment of class balance based on the distribution of classes.
    """
    counts = col.value_counts()
    if len(counts) < 2 :
        return "Single class"
    
    majority = counts.iloc[0]
    minority = counts.iloc[-1]
    ratio = minority / majority

    if ratio >= 0.8 :
        return "Balanced"
    elif ratio >= 0.4 :
        return "Slightly imbalanced"
    elif ratio >= 0.1 :
        return "Imbalanced"
    else :
        return "Severly imbalanced"
    
def _compute_correlations(df: pd.DataFrame, target_column: str) -> dict :
    """
    Computes correlation between each feature and the target.
    Uses Pearson for numeric-numeric and Point-Biserial for numeric-categorical.
    Returns sorted dict of {column: correlation_value}.
    """
    correlations = {}
    target = df[target_column]

    for col in df.columns :
        if col == target_column :
            continue
        if not pd.api.types.is_numeric_dtype(df[col]) :
            continue

        try :
            #Drop rows where either column is null
            valid = df[[col, target_column]].dropna()
            if len(valid) < 10 :
                continue

            corr, pvalue = stats.pearsonr(valid[col], valid[target_column])
            correlations[col] = {
                "correlation" : round(float(corr), 4),
                "pvalue" : round(float(pvalue), 4),
                "significant" : pvalue < 0.05
            }
        except Exception as e :
            continue

    #Sort by absolute correlation value
    correlations = dict(sorted(
        correlations.items(),
        key= lambda x: abs(x[1]["correlation"]),
        reverse=True
    ))

    return correlations

def _recommend_models(analysis: dict, df: pd.DataFrame) -> list :
    """
    Recommend models based on task types, dataset size and target distribution.
    """

    recommendations = []
    n_rows = len(df)
    n_features = len(df.columns) - 1
    task_type = analysis.get("task_type")

    if task_type == "classification":
        balance = analysis.get("class_balance", "balanced")
        n_classes = analysis.get("n_classes", 2)

        # Always recommend logistic regression as baseline
        recommendations.append({
            "model": "Logistic Regression",
            "priority": "start here",
            "reason": "Fast, interpretable baseline. Always run this first to establish a benchmark.",
            "note": "Use class_weight='balanced'" if balance in ("imbalanced", "severely_imbalanced") else ""
        })

        recommendations.append({
                "model": "Random Forest",
                "priority": "recommended",
                "reason": f"Handles {n_features} features well, robust to outliers, provides feature importance.",
                "note": "Use class_weight='balanced'" if balance in ("imbalanced", "severely_imbalanced") else ""
            })
        
        if n_rows > 1000:
                recommendations.append({
                    "model": "XGBoost",
                    "priority": "recommended",
                    "reason": "Strong performer on tabular data. Often best results with tuning.",
                    "note": f"Use scale_pos_weight parameter for imbalanced classes." if balance in ("imbalanced", "severely_imbalanced") else ""
                })

        if balance in ("imbalanced", "severely_imbalanced"):
                recommendations.append({
                    "model": "Note on class imbalance",
                    "priority": "important",
                    "reason": f"Class balance is '{balance}'. Consider SMOTE oversampling or adjusting class weights before modeling.",
                    "note": ""
                })

    elif task_type == "regression":
        skew_label = analysis.get("skew_label", "symmetric")

        recommendations.append({
            "model": "Linear Regression",
            "priority": "start here",
            "reason": "Interpretable baseline. Check residuals for normality after fitting.",
            "note": "Apply log transform to target first." if skew_label == "high_skew" else ""
        })

        recommendations.append({
            "model": "Random Forest Regressor",
            "priority": "recommended",
            "reason": "Non-linear relationships, robust to outliers, no scaling required.",
            "note": ""
        })

        if n_rows > 1000:
            recommendations.append({
                "model": "XGBoost Regressor",
                "priority": "recommended",
                "reason": "Best performance on tabular regression with proper tuning.",
                "note": ""
            })

        if skew_label == "high_skew":
            recommendations.append({
                "model": "Note on target skewness",
                "priority": "important",
                "reason": "Target is highly skewed. Consider log1p transform before modeling and inverse transform predictions.",
                "note": ""
            })

    return recommendations

#Feature engineering suggestions
def suggest_feature_engineering(df: pd.DataFrame, target_column: str) -> list :
    """
    Analyze feature distributions and relationships with target to suggest potential feature engineering steps.
    Returns a list of {column, suggestion, reason} dicts.
    """

    suggestions = []
    for col in df.columns :
        if col == target_column :
            continue

        col_suggestions = _analyze_column_for_engineering(df, col)
        suggestions.extend(col_suggestions)

    return suggestions

def _analyze_column_for_engineering(df: pd.DataFrame, col: str) -> list :
    """
    Returns feature engineering suggestions for a single column.
    """
    suggestions = []
    series = df[col].dropna()

    if pd.api.types.is_numeric_dtype(df[col]):
        skew = abs(float(series.skew()))

        # Log transform for high skew positive columns
        if skew > SKEW_MODERATE_MAX and series.min() >= 0:
            suggestions.append({
                "column": col,
                "suggestion": "log1p transform",
                "code_hint": f"df['{col}_log'] = np.log1p(df['{col}'])",
                "reason": f"High skewness ({round(skew, 2)}) — log transform will normalize the distribution for linear models."
            })

        # Binning for age-like columns
        if series.nunique() > 20 and series.max() - series.min() > 10:
            suggestions.append({
                "column": col,
                "suggestion": "binning",
                "code_hint": f"df['{col}_bin'] = pd.cut(df['{col}'], bins=5, labels=False)",
                "reason": "High range numeric column — binning can help tree models find better splits."
            })

        # Standardization suggestion
        if series.std() > 10 * series.mean() if series.mean() != 0 else False:
            suggestions.append({
                "column": col,
                "suggestion": "standardize",
                "code_hint": f"df['{col}_scaled'] = (df['{col}'] - df['{col}'].mean()) / df['{col}'].std()",
                "reason": "Large scale variance — standardize before using distance-based models (KNN, SVM)."
            })
    
    elif pd.api.types.is_object_dtype(df[col]) or str(df[col].dtype) == "category":
        n_unique = series.nunique()

        # One-hot encoding for low cardinality
        if n_unique <= 10:
            suggestions.append({
                "column": col,
                "suggestion": "one-hot encoding",
                "code_hint": f"df = pd.get_dummies(df, columns=['{col}'], drop_first=True)",
                "reason": f"Low cardinality ({n_unique} unique values) — one-hot encoding is appropriate."
            })

        # Label encoding for high cardinality
        elif n_unique <= 50:
            suggestions.append({
                "column": col,
                "suggestion": "label encoding or target encoding",
                "code_hint": f"from sklearn.preprocessing import LabelEncoder\ndf['{col}_enc'] = LabelEncoder().fit_transform(df['{col}'].astype(str))",
                "reason": f"Medium cardinality ({n_unique} unique values) — consider label or target encoding."
            })

         # Drop suggestion for very high cardinality
        else:
            suggestions.append({
                "column": col,
                "suggestion": "consider dropping",
                "code_hint": f"df = df.drop(columns=['{col}'])",
                "reason": f"Very high cardinality ({n_unique} unique values) — unlikely to generalize as a feature."
            })

    return suggestions