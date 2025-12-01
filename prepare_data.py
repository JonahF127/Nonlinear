import numpy as np
import pandas as pd

def load_email_data(csv_path: str = "email_data.csv") -> pd.DataFrame:
    """
    Load the engineered email dataset (output of svm.py).
    Assumes columns: Category, Message, plus feature columns created in svm.py.
    """
    df = pd.read_csv(csv_path)
    return df


def encode_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Convert the Category column into numeric labels:
      ham  -> +1
      spam -> -1
    """
    label_map = {"ham": 1, "spam": -1}
    if "Category" not in df.columns:
        raise ValueError("Expected a 'Category' column in the dataframe.")

    labels = df["Category"].map(label_map)
    if labels.isnull().any():
        bad_values = df["Category"][labels.isnull()].unique()
        raise ValueError(f"Unrecognized Category values: {bad_values}")

    return labels.to_numpy(dtype=float)


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Extract the numeric feature matrix X from the dataframe.
    These columns must match the ones created in svm.py.
    """
    feature_cols = [
        "word_count",
        "uppercase_word_count",
        "unique_word_count",
        "character_count",
        "letter_count",
        "flagged_word_count",
        "flagged_bigrams_count",
        "digit_count",
        "links_count",
        "free_count",
        "special_count"
    ]

    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Expected feature column '{col}' not found in dataframe.")

    X = df[feature_cols].to_numpy(dtype=float)
    return X


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.75,
    random_seed: int = 42
):
    """
    Shuffle and split X, y into training and test sets.
    train_ratio is the proportion of data used for training.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    n_samples = X.shape[0]
    rng = np.random.default_rng(seed=random_seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_train = int(train_ratio * n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


def prepare_datasets(csv_path: str = "email_data.csv"):
    """
    High-level helper:
      1. Load email_data.csv
      2. Encode labels
      3. Build feature matrix
      4. Split into train/test (75%/25%)
    Returns: X_train, X_test, y_train, y_test
    """
    df = load_email_data(csv_path)
    y = encode_labels(df)
    X = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Simple sanity check
    X_train, X_test, y_train, y_test = prepare_datasets()
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
