"""
models.py

Implements the three regression models from housing_data_analysis(2).ipynb:
  - run_linear_regression: Pipeline(PolynomialFeatures(degree=2) -> LinearRegression())
  - run_lasso_regression:  Pipeline(PolynomialFeatures() -> Lasso(max_iter=10000))
                           tuned with GridSearchCV over PF__degree=[1],
                           model__alpha=np.geomspace(1e-6, 1e4, 20), cv=4
  - run_ridge_regression:  Pipeline(PolynomialFeatures() -> Ridge())
                           tuned the same way, cv=4 (no n_jobs, matching the notebook)

Each function fits/evaluates against the real x_train/x_test/y_train/y_test
produced by pipeline.prepare_ml_data() - nothing here is hardcoded from the
notebook's own printed results (those were only used earlier to sanity-check
that this implementation reproduces the same methodology).

Each function returns a plain dict. Some values in that dict (the fitted
"pipeline" / "grid_search" objects, and raw numpy arrays) are NOT JSON-safe
on their own - main.py is responsible for extracting only the serializable
fields before sending an HTTP response. This module never talks to FastAPI.

Note on the notebook's "stale pipe" bug: in the notebook, cell 137 creates a
fresh `pipe1 = Pipeline(...)` for the post-outlier-removal retraining, but
then mistakenly calls `pipe.fit(...)` again (the pipeline object from the
*pre-outlier-removal* run) instead of `pipe1.fit(...)`. Because both objects
share an identical, unfit configuration this happens not to change the
notebook's own numbers, but it's still a bug. This implementation always
creates and fits exactly one pipeline instance per call, so that mistake
cannot occur here.
"""

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

CV_FOLDS = 4
ALPHA_GRID = np.geomspace(1e-6, 1e4, 20)


def _flatten(y) -> np.ndarray:
    """y_train/y_test arrive as single-column DataFrames; flatten to 1D for sklearn."""
    return np.asarray(y).ravel()


def run_linear_regression(x_train, x_test, y_train, y_test, x_data, y_data) -> dict:
    """
    Pipeline([('pf', PolynomialFeatures(degree=2)), ('lr', LinearRegression())])
    trained on x_train/y_train, evaluated on x_test/y_test, exactly like the
    notebook's post-outlier-removal Linear Regression cell. The 4-fold
    cross-validation R^2 is computed with a freshly-constructed, identically
    configured pipeline over the full x_data/y_data (post outlier removal,
    pre-split), matching the notebook's own cross_val_score call.
    """
    y_train_flat = _flatten(y_train)
    y_test_flat = _flatten(y_test)
    y_data_flat = _flatten(y_data)

    pipe = Pipeline([("pf", PolynomialFeatures(degree=2)), ("lr", LinearRegression())])
    pipe.fit(x_train, y_train_flat)

    predictions = pipe.predict(x_test)
    mse = mean_squared_error(y_test_flat, predictions)
    r2 = r2_score(y_test_flat, predictions)

    cv_pipe = Pipeline([("pf", PolynomialFeatures(degree=2)), ("lr", LinearRegression())])
    cv_scores = cross_val_score(cv_pipe, x_data, y_data_flat, cv=CV_FOLDS, scoring="r2")

    lr_step = pipe.named_steps["lr"]

    return {
        "pipeline": pipe,
        "model_name": "Linear Regression",
        "polynomial_degree": 2,
        "train_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "feature_count": int(x_train.shape[1]),
        "mse": float(mse),
        "r2": float(r2),
        "cv_scores": cv_scores,
        "cv_mean_r2": float(cv_scores.mean()),
        "predictions": predictions,
        "actual": y_test_flat,
        "residuals": y_test_flat - predictions,
        "coefficient_count": int(np.asarray(lr_step.coef_).size),
        "intercept": float(np.asarray(lr_step.intercept_).ravel()[0]),
    }


def run_lasso_regression(x_train, x_test, y_train, y_test, x_data, y_data) -> dict:
    """
    Pipeline([('PF', PolynomialFeatures()), ('model', Lasso(max_iter=10000))])
    tuned with GridSearchCV(cv=4, n_jobs=-1) over
    PF__degree=[1], model__alpha=np.geomspace(1e-6, 1e4, 20),
    exactly like the notebook's Lasso Regression section.
    """
    y_train_flat = _flatten(y_train)
    y_test_flat = _flatten(y_test)

    pipe_las = Pipeline([("PF", PolynomialFeatures()), ("model", Lasso(max_iter=10000))])
    params = {"PF__degree": [1], "model__alpha": ALPHA_GRID}
    grid = GridSearchCV(pipe_las, params, cv=CV_FOLDS, n_jobs=-1, verbose=1)
    grid.fit(x_train, y_train_flat)

    predictions = grid.predict(x_test)
    mse = mean_squared_error(y_test_flat, predictions)
    r2 = r2_score(y_test_flat, predictions)

    model_step = grid.best_estimator_.named_steps["model"]

    return {
        "grid_search": grid,
        "model_name": "Lasso Regression",
        "polynomial_degree": int(grid.best_params_["PF__degree"]),
        "train_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "feature_count": int(x_train.shape[1]),
        "mse": float(mse),
        "r2": float(r2),
        "best_alpha": float(grid.best_params_["model__alpha"]),
        "best_cv_r2": float(grid.best_score_),
        "best_params": {
            "PF__degree": int(grid.best_params_["PF__degree"]),
            "model__alpha": float(grid.best_params_["model__alpha"]),
        },
        "predictions": predictions,
        "actual": y_test_flat,
        "residuals": y_test_flat - predictions,
        "coefficient_count": int(np.asarray(model_step.coef_).size),
        "intercept": float(np.asarray(model_step.intercept_).ravel()[0]),
    }


def run_ridge_regression(x_train, x_test, y_train, y_test, x_data, y_data) -> dict:
    """
    Pipeline([('PF', PolynomialFeatures()), ('model', Ridge())])
    tuned with GridSearchCV(cv=4) (no n_jobs, matching the notebook) over
    PF__degree=[1], model__alpha=np.geomspace(1e-6, 1e4, 20),
    exactly like the notebook's Ridge Regression section.
    """
    y_train_flat = _flatten(y_train)
    y_test_flat = _flatten(y_test)

    pipe_ridge = Pipeline([("PF", PolynomialFeatures()), ("model", Ridge())])
    params = {"PF__degree": [1], "model__alpha": ALPHA_GRID}
    grid1 = GridSearchCV(pipe_ridge, params, cv=CV_FOLDS)
    grid1.fit(x_train, y_train_flat)

    predictions = grid1.predict(x_test)
    mse = mean_squared_error(y_test_flat, predictions)
    r2 = r2_score(y_test_flat, predictions)

    model_step = grid1.best_estimator_.named_steps["model"]

    return {
        "grid_search": grid1,
        "model_name": "Ridge Regression",
        "polynomial_degree": int(grid1.best_params_["PF__degree"]),
        "train_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "feature_count": int(x_train.shape[1]),
        "mse": float(mse),
        "r2": float(r2),
        "best_alpha": float(grid1.best_params_["model__alpha"]),
        "best_cv_r2": float(grid1.best_score_),
        "best_params": {
            "PF__degree": int(grid1.best_params_["PF__degree"]),
            "model__alpha": float(grid1.best_params_["model__alpha"]),
        },
        "predictions": predictions,
        "actual": y_test_flat,
        "residuals": y_test_flat - predictions,
        "coefficient_count": int(np.asarray(model_step.coef_).size),
        "intercept": float(np.asarray(model_step.intercept_).ravel()[0]),
    }
