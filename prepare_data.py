import numpy as np
import pandas as pd


def load_email_data(csv_path: str = "email_data.csv") -> pd.DataFrame:
    """
    Load the engineered email dataset (output of svm.py).
    Assumes columns: Category, Message, plus feature columns created in svm.py.
    """
    df = pd.read_csv(csv_path)
    return df


def encode_labels(df: pd.DataFrame):
    """
    Clean and encode the Category column into numeric labels:
      ham  -> +1
      spam -> -1
    Returns:
        df_clean: dataframe with only valid Category rows
        labels:  numpy array of labels (+1, -1)
    """
    if "Category" not in df.columns:
        raise ValueError("Expected a 'Category' column in the dataframe.")

    # Normalize category strings: lower-case, strip whitespace
    cats = df["Category"].astype(str).str.strip().str.lower()

    # Keep only rows that are exactly 'ham' or 'spam'
    valid_mask = cats.isin(["ham", "spam"])
    bad_values = cats[~valid_mask].unique()

    if len(bad_values) > 0:
        print("Warning: Dropping rows with unrecognized Category values:", bad_values)

    df_clean = df[valid_mask].copy()
    cats_clean = cats[valid_mask]

    label_map = {"ham": 1, "spam": -1}
    labels = cats_clean.map(label_map).to_numpy(dtype=float)

    return df_clean, labels


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Extract the numeric feature matrix X from the dataframe.
    These columns must match the ones created in add_features.py. For analysis purposes, the data points are numbered below
        List of Data Points
        0) Category (spam/ham)
        0) Message
        1) word_count
        2) uppercase_word_count
        3) unique_word_count
        4) character_count
        5) letter_count
        6) flagged_word_count
        7) flagged_bigrams_count
        8) digit_count
        9) links_count
        10) free_count
        11) special_count
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
        # scale data with min-max normalization
        min = df[col].min()
        max = df[col].max()
        if max != min:
            df[col] = (df[col] - min) / (max - min)

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
      2. Clean and encode labels
      3. Build feature matrix
      4. Split into train/test (75%/25%)
    Returns: X_train, X_test, y_train, y_test
    """
    df = load_email_data(csv_path)

    # encode_labels now returns a cleaned df and label array
    df_clean, y = encode_labels(df)

    X = build_feature_matrix(df_clean)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Simple sanity check
    X_train, X_test, y_train, y_test = prepare_datasets()
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)