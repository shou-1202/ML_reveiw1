"""
pipeline.py

Thin orchestration layer that produces the ML-ready data the regression
models consume. It deliberately reuses every function already defined in
preprocessing.py (load_dataset, handle_missing_values, remove_features,
create_log_target, encode_and_scale, remove_redundant_features,
detect_outliers, remove_outliers, prepare_train_test_data, split_data)
rather than re-implementing any preprocessing logic. This guarantees the
Regression Lab trains on exactly the same processed dataset the
Dataset Preprocessing page displays (down to the same 28 IQR outliers
removed and the same random_state=43 split).
"""

from preprocessing import (
    RANDOM_STATE,
    create_log_target,
    detect_outliers,
    encode_and_scale,
    handle_missing_values,
    load_dataset,
    prepare_train_test_data,
    remove_features,
    remove_outliers,
    remove_redundant_features,
    split_data,
)

TEST_SIZE = 0.25  # sklearn's default, matching the notebook's unparameterized train_test_split


def prepare_ml_data() -> dict:
    """
    Runs the full notebook-derived preprocessing pipeline (identical to what
    GET /preprocessing runs) and returns the post-outlier-removal x_data/y_data
    plus the train_test_split(..., random_state=43) result, ready to be fed
    into any of the three regression models.
    """
    raw_df = load_dataset()
    filled_df, _ = handle_missing_values(raw_df)
    reduced_df, _ = remove_features(filled_df)
    log_df = create_log_target(reduced_df)
    encoded_df, _ = encode_and_scale(log_df)
    trimmed_df, _ = remove_redundant_features(encoded_df)
    outliers, _ = detect_outliers(trimmed_df)
    clean_df, _ = remove_outliers(trimmed_df, outliers)

    x_data, y_data = prepare_train_test_data(clean_df)
    x_train, x_test, y_train, y_test = split_data(x_data, y_data)

    return {
        "x_data": x_data,
        "y_data": y_data,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(x_data.columns),
        "target_column": "Log_SalePrice",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }
