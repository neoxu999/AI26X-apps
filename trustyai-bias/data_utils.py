"""
Data utilities for server failure prediction.
This module provides functions for data loading, preprocessing, and splitting.
"""

# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
# ]
# ///

import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_SERVER_METRICS_FILE = Path(__file__).parent / "server_metrics.csv"
DEFAULT_SCALER_FILE = Path(__file__).parent / "data_scaler.pkl"

# Only the numeric columns are scaled. The workload_type one-hot columns stay
# 0/1, so that TrustyAI can use them as protected attributes for bias metrics.
NUMERIC_FEATURES = [
    "server_age_months",
    "cpu_temp_celsius",
    "cpu_utilization_percent",
    "memory_usage_percent",
    "disk_io_ops_per_sec",
    "network_throughput_mbps",
    "fan_speed_rpm",
    "power_draw_watts",
    "disk_read_errors_24h",
    "memory_errors_24h",
]
WORKLOAD_TYPES = ["compute", "database", "storage", "web"]
WORKLOAD_FEATURES = [f"workload_type_{w}" for w in WORKLOAD_TYPES]

# Fixed feature order for training and inference (TrustyAI input index 0..13)
FEATURES = NUMERIC_FEATURES + WORKLOAD_FEATURES


def load_data(filepath: Path | str = DEFAULT_SERVER_METRICS_FILE):
    """Load the server metrics raw data."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    return df


def encode_features(X):
    """
    One-hot encode workload_type as 0.0/1.0 values and return the
    columns in the fixed FEATURES order (missing workload columns are 0.0).
    """
    X_encoded = pd.get_dummies(X, columns=["workload_type"], dtype=float)
    for col in FEATURES:
        if col not in X_encoded.columns:
            X_encoded[col] = 0.0
    return X_encoded[FEATURES].astype(float)


def preprocess_features(df, scaler_path=DEFAULT_SCALER_FILE):
    """
    Preprocess the features for model training.
    Saves the scaler to be used later during inference

    Args:
        df: DataFrame with server metrics
        scaler_path: Path to save the scaler

    Returns:
        Tuple of (X_scaled, y) where:
            - X_scaled: Preprocessed features
            - y: Target variable
    """
    print("Preprocessing features...")

    # Separate features and target
    X = df.drop(["failure_within_48h", "server_id", "timestamp"], axis=1)
    y = df["failure_within_48h"]

    # One-hot encode categorical features (kept as 0/1, not scaled)
    X_scaled = encode_features(X)

    # Scale numeric features only
    scaler = StandardScaler()
    X_scaled[NUMERIC_FEATURES] = scaler.fit_transform(X_scaled[NUMERIC_FEATURES])

    # Save scaler for later use in inference
    print(f"Saving scaler to {scaler_path}...")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Preprocessed features shape: {X_scaled.shape}")
    return X_scaled, y


def preprocess_for_inference(df):
    """
    Preprocess data for model inference using saved scaler.

    Args:
        df: DataFrame with server metrics (without target variable)

    Returns:
        Preprocessed features ready for prediction
    """
    # For inference, these columns may not exist
    cols_to_drop = [
        col
        for col in ["server_id", "timestamp", "failure_within_48h"]
        if col in df.columns
    ]
    X = df.drop(cols_to_drop, axis=1)

    # One-hot encode categorical features, in the training column order
    X_scaled = encode_features(X)

    # Scale numeric features only, using the scaler saved during training
    scaler = load_scaler()
    X_scaled[NUMERIC_FEATURES] = scaler.transform(X_scaled[NUMERIC_FEATURES])

    return X_scaled


def load_scaler(scaler_path=DEFAULT_SCALER_FILE):
    """Load a saved StandardScaler from disk."""
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and test sets."""
    print(
        f"Splitting data: {int((1 - test_size) * 100)}% train, {int(test_size * 100)}% test..."
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test


def calculate_class_weights(y_train):
    """
    Calculate class weights to handle imbalanced datasets.

    The weights are inversely proportional to class frequencies.
    This helps the model pay more attention to the minority class.

    Args:
        y_train: Training labels (pandas Series or numpy array)

    Returns:
        Dictionary mapping class labels to weights
    """
    # Count samples per class
    n_samples = len(y_train)
    n_classes = len(y_train.unique())

    # Calculate weight for each class
    class_weights = {}
    for class_label in sorted(y_train.unique()):
        n_class_samples = (y_train == class_label).sum()
        weight = n_samples / (n_classes * n_class_samples)
        class_weights[class_label] = weight

    print("Class weights calculated:")
    for class_label, weight in class_weights.items():
        class_name = "No failure" if class_label == 0 else "Failure"
        print(f"  Class {class_label} ({class_name}): {weight:.4f}")

    return class_weights


def save_prepared_data(X_train, X_test, y_train, y_test):
    """Save the prepared datasets to CSV files."""
    print("Saving prepared datasets...")

    # Combine features and target for saving
    train_df = X_train.copy()
    train_df["failure_within_48h"] = y_train.values

    test_df = X_test.copy()
    test_df["failure_within_48h"] = y_test.values

    train_df.to_csv("train_data.csv", index=False)
    test_df.to_csv("test_data.csv", index=False)

    print(f"Saved train_data.csv ({len(train_df)} samples)")
    print(f"Saved test_data.csv ({len(test_df)} samples)")


def prepare_data(input_file=DEFAULT_SERVER_METRICS_FILE):
    """
    Complete data preparation pipeline.

    Loads raw data, preprocesses it, splits into train/test,
    and saves everything including the scaler.
    """
    print("=" * 60)
    print("Server Failure Prediction - Data Preparation")
    print("=" * 60)

    # Load data
    df = load_data(input_file)

    # Preprocess features (fit and save scaler)
    X, y = preprocess_features(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Save prepared data
    save_prepared_data(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("Files created:")
    print("  - train_data.csv")
    print("  - test_data.csv")
    print("  - data_scaler.pkl")
    print("=" * 60)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Run the complete data preparation pipeline
    prepare_data()
