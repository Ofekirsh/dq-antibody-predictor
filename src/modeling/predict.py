import numpy as np
import pandas as pd
import joblib


def predict_with_model(df, model_path):
    """
    Loads a trained model and predicts probabilities per ID.

    Args:
        df (pd.DataFrame): The dataset including "ID" column and features. May optionally
                          include "Label" and "MM" columns for evaluation.
        model_path (str): Path to the saved model file.

    Returns:
        dict: ID-wise predictions with format {ID: [(probability, label, mm_value), ...]}
              where each ID maps to a list of prediction tuples.

    Raises:
        ValueError: If the DataFrame doesn't contain the required "ID" column.
    """
    # Ensure 'ID' column exists
    if "ID" not in df.columns:
        raise ValueError("The input DataFrame must have an 'ID' column for grouping.")

    # Extract Labels and MM values before dropping
    labels = df["Label"].values if "Label" in df.columns else np.full(len(df), np.nan)
    mm_values = df["MM"].values if "MM" in df.columns else np.full(len(df), np.nan)

    # Create a copy to avoid modifying the original dataframe
    df_features = df.copy()

    # Drop unnecessary columns before prediction
    drop_columns = [col for col in ["MM", "Label", "Time"] if col in df_features.columns]
    if drop_columns:
        df_features = df_features.drop(columns=drop_columns)
        print(f"[INFO] Dropped columns for prediction: {drop_columns}")

    try:
        # Load trained model
        model = joblib.load(model_path)
        print(f"[INFO] Model loaded from: {model_path}")

        # Extract features (excluding ID)
        X = df_features.drop(columns=["ID"])

        # Predict probabilities
        probabilities = model.predict_proba(X)[:, 1]  # Probability for class 1

        # Prepare predictions per ID
        id_predictions = {}
        ids = df_features["ID"].values

        for i, id_ in enumerate(ids):
            if id_ not in id_predictions:
                id_predictions[id_] = []

            # Store as (probability, label, mm_value) tuple
            id_predictions[id_].append((
                float(probabilities[i]),
                float(labels[i]) if not np.isnan(labels[i]) else np.nan,
                int(mm_values[i]) if not np.isnan(mm_values[i]) else None
            ))

        # Validate that prediction structure matches what's expected by ROC functions
        incomplete_ids = [id_ for id_, preds in id_predictions.items() if len(preds) != 2]
        if incomplete_ids:
            print(f"[WARNING] Some IDs don't have exactly 2 predictions: {incomplete_ids[:5]}...")

        return id_predictions

    except Exception as e:
        print(f"[ERROR] Failed to predict with model: {str(e)}")
        raise