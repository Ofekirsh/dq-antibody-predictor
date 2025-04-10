import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc


def compute_roc(id_predictions, filename="../output/roc_curve.png"):
    """
    Calculate ROC and Precision-Recall curves from prediction data and create visualization.

    Args:
        id_predictions (dict): Dictionary with IDs as keys and lists of tuples (probability, label, metadata)
                              as values. Each ID should have exactly 2 predictions.
        filename (str): Name of the output file for the plot

    Returns:
        tuple: (roc_auc_max, roc_auc_regular, pr_auc_max, pr_auc_regular) - AUC scores for different methods
    """
    try:

        os.makedirs(os.path.dirname(filename), exist_ok=True)


        # Lists to store probability values and labels
        max_probs, max_labels = [], []
        all_probs, all_labels = [], []

        for id_, values in id_predictions.items():
            # Unpack values - ignore metadata for calculations
            prob1, label1, _ = values[0]
            prob2, label2, _ = values[1]

            # Take the max probability and max label per ID
            max_probs.append(max(prob1, prob2))
            max_labels.append(max(label1, label2))

            # Store all probabilities and labels for regular ROC
            all_probs.extend([prob1, prob2])
            all_labels.extend([label1, label2])

        # Compute ROC AUC (Method 1: Taking max per ID)
        roc_auc_max = roc_auc_score(max_labels, max_probs)

        # Compute ROC AUC (Method 2: Regular ROC with all samples)
        roc_auc_regular = roc_auc_score(all_labels, all_probs)

        # Compute ROC Curve for max method
        fpr, tpr, _ = roc_curve(max_labels, max_probs)

        # Compute ROC Curve for regular method
        fpr_reg, tpr_reg, _ = roc_curve(all_labels, all_probs)

        # Compute Precision-Recall Curve for max method
        precision, recall, _ = precision_recall_curve(max_labels, max_probs)
        pr_auc_max = auc(recall, precision)

        # Compute Precision-Recall Curve for regular method
        precision_reg, recall_reg, _ = precision_recall_curve(all_labels, all_probs)
        pr_auc_regular = auc(recall_reg, precision_reg)

        # Create figure with two subplots (ROC Curve and Precision-Recall Curve)
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot ROC Curve
        axs[0].plot(fpr, tpr, color="blue", lw=2, label=f"Full Donor ROC AUC = {roc_auc_max:.4f}")
        axs[0].plot(fpr_reg, tpr_reg, color="red", lw=2, linestyle="dashed",
                    label=f"Single DQ ROC AUC = {roc_auc_regular:.4f}")
        axs[0].plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line
        axs[0].set_xlabel("FPR")
        axs[0].set_ylabel("TPR")
        axs[0].set_title("ROC Curve")
        axs[0].legend(loc="lower right")

        # Plot Precision-Recall Curve
        axs[1].plot(recall, precision, color="blue", lw=2, label=f"Full Donor PR AUC = {pr_auc_max:.4f}")
        axs[1].plot(recall_reg, precision_reg, color="red", lw=2, linestyle="dashed",
                    label=f"Single DQ PR AUC = {pr_auc_regular:.4f}")
        axs[1].set_xlabel("Recall")
        axs[1].set_ylabel("Precision")
        axs[1].set_title("Precision-Recall Curve")
        axs[1].legend(loc="lower left")

        # Adjust layout and save plot
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"[INFO] ROC & PR curves saved to: {filename}")

        return roc_auc_max, roc_auc_regular, pr_auc_max, pr_auc_regular

    except Exception as e:
        print(f"[ERROR] An error occurred while computing ROC curves: {str(e)}")
        return None, None, None, None


def evaluate_and_plot_roc_auc(id_predictions, filename="../output/roc_curve.png"):
    """
    Processes predictions per ID, computes ROC AUC scores, and plots the ROC/PR curves.

    Args:
        id_predictions (dict): Dictionary where keys are IDs, and values are lists of tuples
                              (probability, label, metadata)
        filename (str): Name of the output file for the plot

    Returns:
        tuple: (roc_auc_max, roc_auc_regular, pr_auc_max, pr_auc_regular) - AUC scores for different methods
    """
    # Compute and plot ROC for the dataset
    return compute_roc(id_predictions, filename=filename)


if __name__ == "__main__":
    # Example Predictions Format: ID -> [(probability1, label1, metadata1), (probability2, label2, metadata2)]
    example_predictions = {
        1: [(0.61, 0, None), (0.74, 1, None)],
        2: [(0.52, 0, None), (0.49, 0, None)],
        3: [(0.88, 1, None), (0.85, 1, None)],
        4: [(0.36, 0, None), (0.31, 0, None)],
        5: [(0.81, 1, None), (0.78, 1, None)]
    }

    # Set output directory
    output_dir = "output"

    # Evaluate and plot
    roc_auc_max, roc_auc_regular, pr_auc_max, pr_auc_regular = evaluate_and_plot_roc_auc(
        example_predictions,
        output_dir=output_dir,
        filename="example_roc.png"
    )

    # Print results
    print("\nResults Summary:")
    print(f"ROC AUC (Full Donor Method): {roc_auc_max:.4f}")
    print(f"ROC AUC (Single DQ Method): {roc_auc_regular:.4f}")
    print(f"PR AUC (Full Donor Method): {pr_auc_max:.4f}")
    print(f"PR AUC (Single DQ Method): {pr_auc_regular:.4f}")