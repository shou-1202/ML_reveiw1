"""
FastAPI backend for the Housing Price Regression Lab.

  GET /                  -> health check
  GET /preprocessing     -> runs the real notebook-derived preprocessing
                            pipeline against backend/data/train.csv and
                            returns the stage-by-stage result the React
                            frontend renders. Unchanged from before.
  GET /regression/linear -> trains/evaluates the notebook's Linear Regression
  GET /regression/lasso  -> trains/evaluates the notebook's Lasso Regression
  GET /regression/ridge  -> trains/evaluates the notebook's Ridge Regression

The regression endpoints reuse the exact same processed dataset
(pipeline.prepare_ml_data(), which itself calls preprocessing.py) that
GET /preprocessing is built on - there is no second/duplicate preprocessing
implementation. Model fitting happens in models.py; this file's only job is
to call it and convert the result into a JSON-safe response (no sklearn
Pipeline/GridSearchCV objects, no numpy types, no NaN/Infinity).
"""

from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import models
from pipeline import prepare_ml_data
from preprocessing import run_preprocessing_pipeline

app = FastAPI(title="Housing Price Regression Backend")

# Allow the existing Vite React frontend (running on 5173) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok", "service": "Housing Price Regression Backend"}


@app.get("/preprocessing")
def preprocessing():
    try:
        return run_preprocessing_pipeline()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surface real backend errors to the UI
        raise HTTPException(
            status_code=500, detail=f"Preprocessing pipeline failed: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# JSON-safety helpers for the regression endpoints
# ---------------------------------------------------------------------------

def _num(value: Any) -> Optional[float]:
    """Convert a numpy/python scalar to a JSON-safe float (or None for NaN/Inf)."""
    if value is None:
        return None
    value = float(value)
    return None if (np.isnan(value) or np.isinf(value)) else value


def _num_list(values: Any) -> list:
    """Convert a numpy array / iterable of numbers into a JSON-safe list of floats."""
    return [_num(v) for v in np.asarray(values).ravel().tolist()]


def _build_regression_response(result: dict, alpha_search: Optional[dict] = None) -> dict:
    """
    Turns a models.run_*_regression() result dict into the JSON-safe shape the
    frontend's RegressionLab page expects. Deliberately drops "pipeline" and
    "grid_search" (the raw sklearn objects) - only plain values leave this
    function.
    """
    model_info: dict = {
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


def _run_regression_endpoint(model_fn, alpha_search: Optional[dict] = None) -> dict:
    try:
        data = prepare_ml_data()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        result = model_fn(
            data["x_train"],
            data["x_test"],
            data["y_train"],
            data["y_test"],
            data["x_data"],
            data["y_data"],
        )
    except Exception as exc:  # pragma: no cover - surface real model errors to the UI
        raise HTTPException(status_code=500, detail=f"Model execution failed: {exc}") from exc

    try:
        return _build_regression_response(result, alpha_search=alpha_search)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=500, detail=f"Failed to serialize model results: {exc}"
        ) from exc


@app.get("/regression/linear")
def regression_linear():
    return _run_regression_endpoint(models.run_linear_regression)


@app.get("/regression/lasso")
def regression_lasso():
    return _run_regression_endpoint(
        models.run_lasso_regression,
        alpha_search={"min": 1e-6, "max": 1e4, "count": 20},
    )


@app.get("/regression/ridge")
def regression_ridge():
    return _run_regression_endpoint(
        models.run_ridge_regression,
        alpha_search={"min": 1e-6, "max": 1e4, "count": 20},
    )
