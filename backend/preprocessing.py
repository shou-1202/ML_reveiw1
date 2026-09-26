"""
preprocessing.py

Converts the preprocessing workflow from `housing_data_analysis.ipynb`
(House Prices - Advanced Regression Techniques) into reusable functions.

IMPORTANT: This module intentionally mirrors the notebook's own preprocessing
decisions (which columns are filled, which are dropped, the exact
OneHotEncoder / StandardScaler / ColumnTransformer configuration, the IQR
outlier logic, and the train_test_split configuration). It does not change
or "improve" the methodology - it only restructures the notebook's code into
functions that also record what happened at each stage, so the website can
show every step instead of just a final result.

Only the "Dataset Preprocessing" part of the notebook is implemented here.
The later regression modelling cells (Linear/Lasso/Ridge) are out of scope
for this task and are deliberately not included.
"""

import os
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Exact notebook constants - kept as named constants so the "exact
# configuration" the notebook uses is visible in one place.
# ---------------------------------------------------------------------------

# Columns dropped during the notebook's EDA-driven "Feature Removal" section
# (cells that call df.drop(...) for low-signal / redundant columns), plus
# 'Id', which is dropped because it has no relationship with price.
FEATURE_REMOVAL_COLUMNS = [
    "Id",
    "Street",
    "Alley",
    "Utilities",
    "LotShape",
    "LandContour",
    "LotConfig",
    "LandSlope",
    "Condition1",
    "Condition2",
    "BldgType",
    "OverallCond",
    "RoofStyle",
    "RoofMatl",
    "BsmtCond",
    "BsmtFinType2",
    "Heating",
    "CentralAir",
    "Electrical",
    "Functional",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "SaleType",
    "SaleCondition",
]

# Columns dropped after encoding, once the correlation heatmap revealed they
# were redundant with other retained (already-encoded) features.
REDUNDANT_FEATURE_COLUMNS = ["num__GarageArea", "num__1stFlrSF", "num__TotRmsAbvGrd"]

RANDOM_STATE = 43  # train_test_split(..., random_state=43) in the notebook

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train.csv")

PREVIEW_ROWS = 10


# ---------------------------------------------------------------------------
# JSON-safety helpers
# ---------------------------------------------------------------------------

def _json_safe_value(value: Any) -> Any:
    """Convert a single pandas/numpy scalar into a plain JSON-safe value."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if (np.isnan(value) or np.isinf(value)) else value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_json_safe_value(v) for v in value.tolist()]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if pd.isna(value) if not isinstance(value, (list, dict)) else False:
        return None
    return value


def df_preview(df: pd.DataFrame, n: int = PREVIEW_ROWS) -> list[dict]:
    """Return the first n rows of df as a list of JSON-safe dicts."""
    records = df.head(n).to_dict(orient="records")
    return [{k: _json_safe_value(v) for k, v in row.items()} for row in records]


def rows_preview(df: pd.DataFrame, n: int = PREVIEW_ROWS) -> list[dict]:
    """Same as df_preview but does not assume ordering (used for outlier rows, etc.)."""
    return df_preview(df, n)


# ---------------------------------------------------------------------------
# Pipeline stage functions
# ---------------------------------------------------------------------------

def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the real House Prices train.csv from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"train.csv not found at {path}. The real House Prices train.csv "
            "is required and must be placed at backend/data/train.csv."
        )
    return pd.read_csv(path)


def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Reproduces the notebook's missing-value handling, in order:
      1. LotFrontage -> filled with the column mean
      2. Alley -> filled with 'None'
      3. MasVnrType -> filled with 'None'
      4. MasVnrArea -> filled with 0
      5. BsmtQual -> filled with 'None'
      6. All remaining object columns -> filled with 'None'
      7. All remaining numeric columns -> filled with 0
    """
    df = df.copy()

    lot_frontage_mean = float(df["LotFrontage"].mean())
    df["LotFrontage"] = df["LotFrontage"].replace(np.nan, lot_frontage_mean)

    df["Alley"] = df["Alley"].replace(np.nan, "None")
    df["MasVnrType"] = df["MasVnrType"].replace(np.nan, "None")
    df["MasVnrArea"] = df["MasVnrArea"].replace(np.nan, 0)
    df["BsmtQual"] = df["BsmtQual"].replace(np.nan, "None")

    obj_col = df.select_dtypes(include=["object"]).columns
    df[obj_col] = df[obj_col].fillna("None")

    num_col = df.select_dtypes(exclude=["object"]).columns
    df[num_col] = df[num_col].fillna(0)

    strategy = {
        "LotFrontage": f"Filled with the column mean ({lot_frontage_mean:.2f})",
        "categorical_columns": "Filled with 'None' (missing means the amenity/feature is absent)",
        "numeric_columns": "Filled with 0",
    }
    return df, strategy


def remove_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Drops the columns identified during the notebook's EDA as low-signal or
    redundant (plus 'Id'). Returns the trimmed dataframe and the list of
    columns actually removed (in case any are already absent).
    """
    to_remove = [c for c in FEATURE_REMOVAL_COLUMNS if c in df.columns]
    df = df.drop(columns=to_remove)
    return df, to_remove


def create_log_target(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Log_SalePrice = np.log1p(SalePrice), exactly as in the notebook."""
    df = df.copy()
    df["Log_SalePrice"] = np.log1p(df["SalePrice"])
    return df


def encode_and_scale(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Reproduces the notebook's ColumnTransformer:
      - StandardScaler() on numeric feature columns (excluding SalePrice / Log_SalePrice)
      - OneHotEncoder(drop='first', handle_unknown='ignore') on categorical columns
    Returns the encoded+scaled dataframe (with Log_SalePrice re-attached as the
    last column, exactly like the notebook's df_final) plus metadata about the
    transform.
    """
    target = ["SalePrice", "Log_SalePrice"]
    numeric_cols = [c for c in df.select_dtypes(exclude=["object"]).columns if c not in target]
    categorical_cols = list(df.select_dtypes(include=["object"]).columns)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ]
    )

    X = preprocess.fit_transform(df)
    if sp.issparse(X):
        X = X.toarray()
    names = preprocess.get_feature_names_out()

    transformed = pd.DataFrame(X, columns=names, index=df.index)
    df_final = pd.concat([transformed, df[["Log_SalePrice"]]], axis=1)

    meta = {
        "numeric_columns_scaled": numeric_cols,
        "categorical_columns_encoded": categorical_cols,
        "scaling_method": "StandardScaler()",
        "encoding_method": "OneHotEncoder(drop='first', handle_unknown='ignore')",
    }
    return df_final, meta


def remove_redundant_features(df_final: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drops the columns identified as redundant via the post-encoding correlation heatmap."""
    to_remove = [c for c in REDUNDANT_FEATURE_COLUMNS if c in df_final.columns]
    df_final = df_final.drop(columns=to_remove)
    return df_final, to_remove


def detect_outliers(df_final: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Reproduces the notebook's IQR-based outlier detection on Log_SalePrice:
      Q1 = 25th percentile, Q3 = 75th percentile, IQR = Q3 - Q1
      bounds = [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    """
    q1 = float(df_final["Log_SalePrice"].quantile(0.25))
    q3 = float(df_final["Log_SalePrice"].quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df_final[
        (df_final["Log_SalePrice"] < lower_bound) | (df_final["Log_SalePrice"] > upper_bound)
    ]

    info = {
        "method": "IQR (Interquartile Range) on Log_SalePrice",
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outliers_detected": int(len(outliers)),
    }
    return outliers, info


def remove_outliers(df_final: pd.DataFrame, outliers: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drops the flagged outlier rows from df_final."""
    rows_before = int(len(df_final))
    df_clean = df_final.drop(outliers.index)
    rows_after = int(len(df_clean))
    info = {
        "rows_before": rows_before,
        "rows_removed": rows_before - rows_after,
        "rows_after": rows_after,
    }
    return df_clean, info


def prepare_train_test_data(df_final_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits df_final_clean into feature matrix (x_data) and target (y_data),
    exactly like the notebook: y_data is the last column (Log_SalePrice),
    x_data is everything else.
    """
    y_data = df_final_clean.iloc[:, -1:]
    x_data = df_final_clean.iloc[:, :-1]
    return x_data, y_data


def split_data(x_data: pd.DataFrame, y_data: pd.DataFrame):
    """train_test_split(x_data, y_data, random_state=43), same as the notebook."""
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=RANDOM_STATE)
    return x_train, x_test, y_train, y_test


# ---------------------------------------------------------------------------
# Orchestration: run the full pipeline and build the frontend-shaped response
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline() -> dict:
    """
    Runs the entire notebook-derived preprocessing pipeline against the real
    train.csv and returns a JSON-safe dict shaped for the existing React
    frontend (see frontend/src/pages/Preprocessing.jsx and
    frontend/src/components/StageDetails.jsx for the exact fields consumed).
    """
    raw_df = load_dataset()

    raw_rows, raw_columns = raw_df.shape
    missing_by_col_before = raw_df.isnull().sum()
    missing_by_col_before = missing_by_col_before[missing_by_col_before > 0]
    before_missing_values_total = int(missing_by_col_before.sum())

    stages = []

    # 1. Raw Dataset
    stages.append(
        {
            "name": "Raw Dataset",
            "rows": int(raw_rows),
            "columns": int(raw_columns),
            "total_missing_values": before_missing_values_total,
            "column_names": list(raw_df.columns),
            "preview": df_preview(raw_df),
        }
    )

    # 2. Missing Value Handling
    filled_df, strategy = handle_missing_values(raw_df)
    after_missing_values_total = int(filled_df.isnull().sum().sum())
    stages.append(
        {
            "name": "Missing Value Handling",
            "columns": int(filled_df.shape[1]),
            "before_missing_values_total": before_missing_values_total,
            "after_missing_values_total": after_missing_values_total,
            "before_missing_values_by_column": {
                col: int(count) for col, count in missing_by_col_before.items()
            },
            "strategy": strategy,
            "preview": df_preview(filled_df),
        }
    )

    # 3. Feature Removal
    reduced_df, columns_removed = remove_features(filled_df)
    stages.append(
        {
            "name": "Feature Removal",
            "columns": int(reduced_df.shape[1]),
            "columns_removed_count": len(columns_removed),
            "columns_removed": columns_removed,
            "preview": df_preview(reduced_df),
        }
    )

    # 4. Log Target Transformation
    log_df = create_log_target(reduced_df)
    sale_price_stats = {
        "mean": float(log_df["SalePrice"].mean()),
        "min": float(log_df["SalePrice"].min()),
        "max": float(log_df["SalePrice"].max()),
    }
    log_sale_price_stats = {
        "mean": float(log_df["Log_SalePrice"].mean()),
        "min": float(log_df["Log_SalePrice"].min()),
        "max": float(log_df["Log_SalePrice"].max()),
    }
    stages.append(
        {
            "name": "Log Target Transformation",
            "columns": int(log_df.shape[1]),
            "transformation": "Log_SalePrice = np.log1p(SalePrice)",
            "sale_price_stats": sale_price_stats,
            "log_sale_price_stats": log_sale_price_stats,
            "preview": df_preview(log_df),
        }
    )

    # 5. Encoding + Scaling
    encoded_df, encode_meta = encode_and_scale(log_df)
    stages.append(
        {
            "name": "Encoding + Scaling",
            "columns": int(encoded_df.shape[1]),
            "numeric_columns_scaled_count": len(encode_meta["numeric_columns_scaled"]),
            "categorical_columns_encoded_count": len(encode_meta["categorical_columns_encoded"]),
            "numeric_columns_scaled": encode_meta["numeric_columns_scaled"],
            "categorical_columns_encoded": encode_meta["categorical_columns_encoded"],
            "scaling_method": encode_meta["scaling_method"],
            "encoding_method": encode_meta["encoding_method"],
            "preview": df_preview(encoded_df),
        }
    )

    # 6. Redundant Feature Removal
    trimmed_df, redundant_removed = remove_redundant_features(encoded_df)
    stages.append(
        {
            "name": "Redundant Feature Removal",
            "columns": int(trimmed_df.shape[1]),
            "columns_removed": redundant_removed,
            "reason": (
                "These features were highly correlated with other retained features "
                "in the post-encoding correlation heatmap, so they were dropped to "
                "reduce multicollinearity."
            ),
            "preview": df_preview(trimmed_df),
        }
    )

    # 7. Outlier Detection
    outliers, outlier_info = detect_outliers(trimmed_df)
    stages.append(
        {
            "name": "Outlier Detection",
            **outlier_info,
            "preview": df_preview(outliers, n=len(outliers) if len(outliers) else PREVIEW_ROWS),
        }
    )

    # 8. Outlier Removal
    clean_df, removal_info = remove_outliers(trimmed_df, outliers)
    stages.append(
        {
            "name": "Outlier Removal",
            **removal_info,
            "preview": df_preview(clean_df),
        }
    )

    # 9. Final Dataset (x_data / y_data ready for modelling)
    x_data, y_data = prepare_train_test_data(clean_df)
    stages.append(
        {
            "name": "Final Dataset",
            "rows": int(clean_df.shape[0]),
            "feature_columns_count": int(x_data.shape[1]),
            "feature_columns": list(x_data.columns),
            "target_column": "Log_SalePrice",
            "preview": df_preview(clean_df),
        }
    )

    # 10. Train/Test Split
    x_train, x_test, y_train, y_test = split_data(x_data, y_data)
    train_df = pd.concat([x_train, y_train], axis=1)
    test_df = pd.concat([x_test, y_test], axis=1)
    stages.append(
        {
            "name": "Train/Test Split",
            "random_state": RANDOM_STATE,
            "test_size": 0.25,
            "feature_columns_count": int(x_data.shape[1]),
            "train_rows": int(x_train.shape[0]),
            "test_rows": int(x_test.shape[0]),
            "train_preview": df_preview(train_df),
            "test_preview": df_preview(test_df),
        }
    )

    return {
        "dataset": {"rows": int(raw_rows), "columns": int(raw_columns)},
        "stages": stages,
    }
