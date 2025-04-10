import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression


def train_and_save_model(df, model_type="lgbm", k=5, save_model=False, model_path="output/best_model.pkl"):
    """
    Trains a model using GroupKFold cross-validation, ensuring all samples with the same ID stay together.
    Optionally saves the best-performing model based on AUC.

    Args:
        df (pd.DataFrame): The dataset, including an "ID" column, features, and "Label".
        model_type (str): Model type ("lgbm", "xgb", "logreg"). Default is "lgbm".
        k (int): Number of cross-validation folds. Default is 5.
        save_model (bool): If True, saves the trained model. Default is False.
        model_path (str): Path to save the model.

    Returns:
        tuple: (avg_auc, auc_se, best_model, id_predictions)
            - avg_auc (float): The average AUC score across all folds.
            - auc_se (float): The standard error of the AUC scores.
            - best_model (object): The selected model based on median AUC.
            - predictions (dict): Aggregated predictions per ID in format {ID: [(prob, label, mm_value), ...]}

    Raises:
        ValueError: If required columns are missing or model_type is invalid.
    """
    try:
        # Ensure required columns exist
        required_columns = ["ID", "MM", "Label"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The input DataFrame is missing required columns: {missing_columns}")

        # Separate features and target
        drop_cols = ["ID", "MM", "Label"]
        if "Time" in df.columns:
            drop_cols.append("Time")

        X = df.drop(columns=drop_cols)
        y = df["Label"]
        mm_values = df["MM"].values
        groups = df["ID"].values  # Ensure that the same ID stays in the same split

        # Group K-Fold setup
        gkf = GroupKFold(n_splits=k, shuffle=True)

        # Lists to store results
        auc_scores = []
        models = []
        fold_predictions = {}  # Store predictions for each fold

        # Initialize id_predictions dictionary with all unique IDs
        unique_ids = np.unique(groups).tolist()
        id_predictions = {int(id_): [] for id_ in unique_ids}

        print(f"[INFO] Starting {k}-fold cross-validation with {model_type} model")
        print(f"[INFO] Dataset has {len(unique_ids)} unique IDs and {len(df)} samples")

        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            # Create train and validation sets
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            val_mm = mm_values[val_idx]
            val_groups = groups[val_idx]  # IDs in the validation set

            # Initialize model with appropriate parameters
            if model_type == "lgbm":
                model = lgb.LGBMClassifier(verbose=-1)
            elif model_type == "xgb":
                model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            elif model_type == "logreg":
                model = LogisticRegression(max_iter=1000)
            else:
                raise ValueError("Invalid model_type. Choose 'lgbm', 'xgb', or 'logreg'.")

            # Train model
            model.fit(X_train, y_train)
            models.append(model)

            # Predict probabilities
            val_preds = model.predict_proba(X_val)[:, 1]  # Probability of class 1

            # Compute AUC
            auc_score = roc_auc_score(y_val, val_preds)
            auc_scores.append(auc_score)
            print(f"[INFO] Fold {fold + 1}/{k} - AUC: {auc_score:.4f}")

            # Store predictions and corresponding labels for this fold
            fold_predictions[fold] = {
                'indices': val_idx,
                'predictions': val_preds,
                'labels': y_val.values,
                'mm_values': val_mm,
                'groups': val_groups
            }

            # Store predictions per ID with corresponding labels
            for id_ in np.unique(val_groups):
                # Get indices where this ID appears in the validation set
                id_indices = val_groups == id_
                # Convert to int ID to ensure consistent key type
                int_id = int(id_)

                # Add predictions as tuples (probability, label, mm_value)
                for prob, label, mm in zip(val_preds[id_indices], y_val.iloc[id_indices], val_mm[id_indices]):
                    id_predictions[int_id].append((float(prob), int(label), mm))

        # Compute average AUC and standard error
        avg_auc = np.mean(auc_scores)
        auc_se = np.std(auc_scores) / np.sqrt(k)  # Standard Error = std(AUC scores) / sqrt(k)

        # Select the model from the fold with the median AUC
        median_auc = np.median(auc_scores)
        median_fold_idx = np.argmin(np.abs(np.array(auc_scores) - median_auc))  # Find closest AUC to median
        selected_model = models[median_fold_idx]  # Pick the model from that fold

        print(f"[INFO] Cross-validation results:")
        print(f"[INFO] - Average AUC: {avg_auc:.4f}")
        print(f"[INFO] - AUC Standard Error: {auc_se:.4f}")
        print(f"[INFO] - Selected model from fold {median_fold_idx + 1} with AUC: {auc_scores[median_fold_idx]:.4f}")

        # Check if any IDs have unexpected number of predictions
        for id_, preds in id_predictions.items():
            if len(preds) != 2:
                print(f"[WARNING] ID {id_} has {len(preds)} predictions (expected 2)")

        # Save the selected model if required
        if save_model:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(selected_model, model_path)
            print(f"[INFO] Selected model saved to: {model_path}")

            # Save model metadata
            metadata_path = os.path.join(os.path.dirname(model_path),
                                         f"{os.path.splitext(os.path.basename(model_path))[0]}_metadata.pkl")
            model_metadata = {
                'model_type': model_type,
                'avg_auc': avg_auc,
                'auc_se': auc_se,
                'fold_aucs': auc_scores,
                'selected_fold': median_fold_idx,
                'features': list(X.columns),
                'timestamp': pd.Timestamp.now()
            }
            joblib.dump(model_metadata, metadata_path)
            print(f"[INFO] Model metadata saved to: {metadata_path}")

        return avg_auc, auc_se, selected_model, id_predictions

    except Exception as e:
        print(f"[ERROR] Failed to train model: {str(e)}")
        raise


if __name__ == "__main__":
    """
        Example usage of the training function with a synthetic dataset.
        """
    # Create a small dataset with grouped IDs
    np.random.seed(42)
    n_samples = 200
    n_ids = 10

    data = {
        "ID": np.repeat(np.arange(1, n_ids + 1), n_samples // n_ids),
        "Feature1": np.random.rand(n_samples),
        "Feature2": np.random.rand(n_samples),
        "Feature3": np.random.rand(n_samples),
        "MM": np.random.randint(0, 2, n_samples),
        "Label": np.random.randint(0, 2, n_samples)
    }

    df = pd.DataFrame(data)
    print(f"Created synthetic dataset with {n_samples} samples and {n_ids} unique IDs")

    # Train with cross-validation
    avg_auc, auc_se, best_model, id_predictions = train_and_save_model(
        df,
        model_type="lgbm",
        k=5,
        save_model=True,
        model_path="trained_models/example_model.pkl"
    )

    # Print feature importances if available
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_names = df.drop(columns=["ID", "MM", "Label"]).columns
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        print("\nFeature Importances:")
        print(feature_importance)

    # Print summary of the predictions
    print("\nPrediction Summary:")
    print(f"Number of IDs with predictions: {len(id_predictions)}")

    # Get an example ID
    example_id = list(id_predictions.keys())[0]
    print(f"Example predictions for ID {example_id}: {id_predictions[example_id]}")
