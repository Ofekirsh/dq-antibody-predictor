import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_test(df, test_size=0.2):
    """
    Splits the dataset into train and test sets while ensuring that all rows
    with the same ID are only in one of the sets.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing the dataset.
    - test_size (float): Proportion of unique IDs to include in the test set.

    Returns:
    - X_train (pd.DataFrame): Training feature set.
    - X_test (pd.DataFrame): Test feature set.
    - y_train (pd.Series): Training labels.
    - y_test (pd.Series): Test labels.
    """
    if "ID" not in df.columns:
        raise ValueError("[split_train_test] Error: The input DataFrame must have an 'ID' column.")

    # Get unique IDs and split them
    unique_ids = df["ID"].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=test_size)

    # Assign samples based on unique ID split
    train_df = df[df["ID"].isin(train_ids)].copy()
    test_df = df[df["ID"].isin(test_ids)].copy()

    # Separate features and labels
    X_train, y_train = train_df.drop(columns=["Label"]), train_df["Label"]
    X_test, y_test = test_df.drop(columns=["Label"]), test_df["Label"]

    return X_train, X_test, y_train, y_test
