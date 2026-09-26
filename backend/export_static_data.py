"""
export_static_data.py

Precomputes the results of the preprocessing pipeline and all three regression
models (Linear, Lasso, Ridge) from the dataset, and exports them as JSON files
into `frontend/public/data/`.

This allows the frontend website to run completely standalone on static hosting
platforms like GitHub Pages without requiring an active Python server.
"""

import json
import os
import sys

# Ensure this script can import modules from backend
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import numpy as np
import models
from pipeline import prepare_ml_data
from preprocessing import run_preprocessing_pipeline

OUTPUT_DIR = os.path.abspath(os.path.join(backend_dir, "..", "frontend", "public", "data"))


def _num(value):
    """Convert numpy / python scalar to JSON-safe float or None."""
    if value is None:
        return None
    val = float(value)
    return None if (np.isnan(val) or np.isinf(val)) else val


def _num_list(values):
    """Convert array / iterable into JSON-safe list of floats."""
    return [_num(v) for v in np.asarray(values).ravel().tolist()]


def _build_regression_response(result, alpha_search=None):
    model_info = {
        "name": result["model_name"],
        "polynomial_degree": result["polynomial_degree"],
    }
    if alpha_search is not None:
        model_info["alpha_search"] = alpha_search
        model_info["cv_folds"] = 4

    metrics = {
        "mse": _num(result["mse"]),
        "r2": _num(result["r2"]),
        "cv_scores": _num_list(result["cv_scores"]) if "cv_scores" in result else None,
        "cv_mean_r2": _num(result.get("cv_mean_r2")) if "cv_mean_r2" in result else None,
        "best_alpha": _num(result.get("best_alpha")) if "best_alpha" in result else None,
        "best_cv_r2": _num(result.get("best_cv_r2")) if "best_cv_r2" in result else None,
    }

    model_details = {
        "coefficient_count": result["coefficient_count"],
        "intercept": _num(result["intercept"]),
        "polynomial_degree": result["polynomial_degree"],
        "best_params": result.get("best_params"),
    }

    return {
        "model": model_info,
        "data": {
            "target_column": "Log_SalePrice",
            "feature_count": result["feature_count"],
            "train_rows": result["train_rows"],
            "test_rows": result["test_rows"],
            "random_state": 43,
            "test_size": 0.25,
        },
        "metrics": metrics,
        "model_details": model_details,
        "predictions": _num_list(result["predictions"]),
        "actual_values": _num_list(result["actual"]),
        "residuals": _num_list(result["residuals"]),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Exporting static data to {OUTPUT_DIR}...")

    # 1. Preprocessing pipeline
    print("1/4. Running preprocessing pipeline...")
    prep = run_preprocessing_pipeline()
    with open(os.path.join(OUTPUT_DIR, "preprocessing.json"), "w", encoding="utf-8") as f:
        json.dump(prep, f)
    print("   [OK] preprocessing.json created.")

    # Prepare data for regression models
    print("Preparing ML training & test data...")
    data = prepare_ml_data()

    # 2. Linear Regression
    print("2/4. Running Linear Regression...")
    lin_res = models.run_linear_regression(
        data["x_train"],
        data["x_test"],
        data["y_train"],
        data["y_test"],
        data["x_data"],
        data["y_data"],
    )
    lin = _build_regression_response(lin_res)
    with open(os.path.join(OUTPUT_DIR, "regression_linear.json"), "w", encoding="utf-8") as f:
        json.dump(lin, f)
    print("   [OK] regression_linear.json created.")

    # 3. Lasso Regression
    print("3/4. Running Lasso Regression (GridSearchCV)...")
    lasso_res = models.run_lasso_regression(
        data["x_train"],
        data["x_test"],
        data["y_train"],
        data["y_test"],
        data["x_data"],
        data["y_data"],
    )
    lasso = _build_regression_response(
        lasso_res, alpha_search={"min": 1e-6, "max": 1e4, "count": 20}
    )
    with open(os.path.join(OUTPUT_DIR, "regression_lasso.json"), "w", encoding="utf-8") as f:
        json.dump(lasso, f)
    print("   [OK] regression_lasso.json created.")

    # 4. Ridge Regression
    print("4/4. Running Ridge Regression (GridSearchCV)...")
    ridge_res = models.run_ridge_regression(
        data["x_train"],
        data["x_test"],
        data["y_train"],
        data["y_test"],
        data["x_data"],
        data["y_data"],
    )
    ridge = _build_regression_response(
        ridge_res, alpha_search={"min": 1e-6, "max": 1e4, "count": 20}
    )
    with open(os.path.join(OUTPUT_DIR, "regression_ridge.json"), "w", encoding="utf-8") as f:
        json.dump(ridge, f)
    print("   [OK] regression_ridge.json created.")

    print("\nAll datasets exported successfully for GitHub Pages static hosting!")


if __name__ == "__main__":
    main()
