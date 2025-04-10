#!/usr/bin/env python3
"""
Inference Pipeline for Trained ML Models

This script implements an inference pipeline for applying trained models to new data,
extracting features, making predictions, and evaluating results.
"""

import os
import sys
import logging
import argparse
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

# Feature extraction imports
from utils.data.xlsx_to_dataframe import load_xlsx_to_dataframe
from src.feature_sets.one_hot_eplet import OneHotEpletFeature
from src.feature_sets.biological import BiologicalFeature
from src.feature_sets.bio_eplet import BioEpletFeature

# Model imports
from src.modeling.predict import predict_with_model
from src.modeling.evaluate import evaluate_and_plot_roc_auc


class InferencePipeline:
    """
    End-to-end inference pipeline for feature extraction, prediction, and evaluation.
    Configurable through YAML configuration files.
    """

    def __init__(self, config_path: str, log_level: str = "INFO"):
        """
        Initialize the inference pipeline with configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = self._setup_logger(log_level)
        self.logger.info(f"Initializing inference pipeline with config: {config_path}")

        # Load configuration
        self.config = self._load_config(config_path)

        # Ensure output directories exist
        self._ensure_directories()

        # Track pipeline execution state
        self.feature_file = None
        self.predictions = None

    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Set up and configure logger."""
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }

        logger = logging.getLogger("InferencePipeline")
        logger.setLevel(log_levels.get(log_level.upper(), logging.INFO))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary with configuration parameters

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file has invalid YAML syntax
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.logger.debug("Configuration loaded successfully")
                return config
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration: {e}")
            raise

    def _ensure_directories(self) -> None:
        """Create output directories if they don't exist."""
        # Get output paths from config
        output_dirs = set()

        # Feature output directories
        for output in [
            self.config['features']['eplet']['output_file'],
            self.config['features']['biological']['output_file'],
            self.config['features']['combined']['output_file'],
            self.config['evaluation']['roc_plot_path']
        ]:
            if output:
                output_dirs.add(os.path.dirname(output))

        # Create directories
        for directory in output_dirs:
            if directory:  # Skip empty directory names
                Path(directory).mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Ensured directory exists: {directory}")

    def load_data(self) -> pd.DataFrame:
        """
        Load test data from configured file.

        Returns:
            DataFrame containing the test data
        """
        input_file = self.config['data']['test_file']
        self.logger.info(f"Loading test data from {input_file}")

        # Check if file exists
        if not os.path.exists(input_file):
            error_msg = f"Test data file not found: {input_file}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Load based on file extension
        if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
            df = load_xlsx_to_dataframe(input_file)
        elif input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        else:
            error_msg = f"Unsupported file format for {input_file}. Use .csv, .xlsx or .xls."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Add Time column for model compatibility if it doesn't exist
        if "Time" not in df.columns:
            df["Time"] = 0
            self.logger.debug("Added 'Time' column with default value 0")

        return df

    def extract_features(self, df: pd.DataFrame, mode: str) -> str:
        """
        Extract features based on specified mode.

        Args:
            df: Input DataFrame
            mode: Feature extraction mode ('eplet', 'biological', or 'combined')

        Returns:
            Path to the output feature file

        Raises:
            ValueError: If an invalid mode is specified
        """
        self.logger.info(f"Starting feature extraction - Mode: {mode}")

        if mode == "eplet":
            return self._extract_one_hot_eplet(df)
        elif mode == "biological":
            return self._extract_biological(df)
        elif mode == "combined":
            return self._extract_combined(df)
        else:
            error_msg = f"Invalid feature extraction mode: {mode}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _extract_one_hot_eplet(self, df: pd.DataFrame) -> str:
        """Extract One-Hot Eplet features."""
        config = self.config['features']['eplet']
        output_file = config['output_file']

        self.logger.info("Extracting One-Hot Eplet Features")
        extractor = OneHotEpletFeature(
            config['dqa_to_sequence_file'],
            config['dqb_to_sequence_file'],
            config['allele_descriptions_file']
        )
        extractor.extract_features(df, output_file=output_file)
        extractor.save_missing_keys()
        return output_file

    def _extract_biological(self, df: pd.DataFrame) -> str:
        """Extract Biological features."""
        config = self.config['features']['biological']
        output_file = config['output_file']

        self.logger.info("Extracting Biological Features")
        extractor = BiologicalFeature(
            ems_file=config['ems_distances_file'],
            pam_file=config['pam_distances_file'],
            g12_file=config['g12_values_file'],
            features_to_extract=config['features_to_extract']
        )
        extractor.extract_features(df, output_file=output_file)
        extractor.save_missing_keys()
        return output_file

    def _extract_combined(self, df: pd.DataFrame) -> str:
        """Extract Combined (One-Hot + Biological) features."""
        oh_config = self.config['features']['eplet']
        bio_config = self.config['features']['biological']
        combined_config = self.config['features']['combined']
        output_file = combined_config['output_file']

        self.logger.info("Extracting Combined Features")
        extractor = BioEpletFeature(
            dqa_to_seq=oh_config['dqa_to_sequence_file'],
            dqb_to_seq=oh_config['dqb_to_sequence_file'],
            allele_desc=oh_config['allele_descriptions_file'],
            ems_file=bio_config['ems_distances_file'],
            pam_file=bio_config['pam_distances_file'],
            g12_file=bio_config['g12_values_file'],
            bio_features_to_extract=bio_config['features_to_extract']
        )
        extractor.extract_features(df, output_file=output_file)
        return output_file

    def make_predictions(self, feature_file: Optional[str] = None) -> Dict:
        """
        Make predictions using the trained model.

        Args:
            feature_file: Path to feature file (uses the last extracted features if None)

        Returns:
            Dictionary of predictions keyed by ID

        Raises:
            FileNotFoundError: If model file or feature file doesn't exist
        """
        # Use provided feature file or the one from previous extraction step
        if feature_file:
            self.feature_file = feature_file

        if not self.feature_file:
            error_msg = "No feature file specified for predictions"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        model_path = self.config['model']['path']

        # Check if files exist
        if not os.path.exists(model_path):
            error_msg = f"Model file not found: {model_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not os.path.exists(self.feature_file):
            error_msg = f"Feature file not found: {self.feature_file}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        self.logger.info(f"Loading features from {self.feature_file}")
        test_df = pd.read_csv(self.feature_file)

        self.logger.info(f"Making predictions using model from {model_path}")
        predictions = predict_with_model(test_df, model_path)

        # Store predictions for later use
        self.predictions = predictions

        return predictions

    def evaluate(self, predictions: Optional[Dict] = None) -> None:
        """
        Evaluate model performance and generate ROC curve.

        Args:
            predictions: Dictionary with ID-prediction mappings (uses the last predictions if None)

        Raises:
            ValueError: If no predictions are available
        """
        # Use provided predictions or the ones from previous step
        if predictions:
            self.predictions = predictions

        if not self.predictions:
            error_msg = "No predictions available for evaluation"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        output_plot = self.config['evaluation']['roc_plot_path']

        self.logger.info("Evaluating model and generating ROC AUC plot")
        evaluate_and_plot_roc_auc(self.predictions, filename=output_plot)
        self.logger.info(f"ROC AUC plot saved to {output_plot}")

    def save_predictions(self, output_file: Optional[str] = None) -> None:
        """
        Save predictions to a CSV file.

        Args:
            output_file: Path to save predictions (uses config value if None)

        Raises:
            ValueError: If no predictions are available
        """
        if not self.predictions:
            error_msg = "No predictions available to save"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if not output_file:
            output_file = self.config['output']['predictions_file']

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Convert dictionary to DataFrame
        pred_df = pd.DataFrame([
            {"ID": id_, "Probability": prob}
            for id_, prob in self.predictions.items()
        ])

        # Save to CSV
        pred_df.to_csv(output_file, index=False)
        self.logger.info(f"Predictions saved to {output_file}")

    def run(self, feature_mode: str, skip_features: bool = False,
            skip_evaluation: bool = False) -> Dict:
        """
        Execute the complete inference pipeline.

        Args:
            feature_mode: Feature extraction mode ('eplet', 'biological', 'combined')
            skip_features: Skip feature extraction if True (use existing files)
            skip_evaluation: Skip evaluation step if True

        Returns:
            Dictionary of predictions
        """
        self.logger.info(f"Starting inference pipeline - Feature mode: {feature_mode}")

        # Feature extraction
        if skip_features:
            self.logger.info("Skipping feature extraction as requested")
            if feature_mode == "eplet":
                self.feature_file = self.config['features']['eplet']['output_file']
            elif feature_mode == "biological":
                self.feature_file = self.config['features']['biological']['output_file']
            else:  # combined
                self.feature_file = self.config['features']['combined']['output_file']
        else:
            # Load data
            df = self.load_data()
            # Extract features
            self.feature_file = self.extract_features(df, feature_mode)
            self.logger.info(f"Feature extraction completed: {self.feature_file}")

        # Make predictions
        predictions = self.make_predictions()
        self.logger.info(f"Predictions completed for {len(predictions)} samples")

        # Save predictions
        self.save_predictions()

        # Model evaluation (if not skipped)
        if not skip_evaluation:
            self.evaluate()
        else:
            self.logger.info("Skipping evaluation as requested")

        self.logger.info("Inference pipeline execution completed successfully")
        return predictions


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ML Model Inference Pipeline"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="../configs/inference_pipeline_config.yml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["eplet", "biological", "combined"],
        default="combined",
        help="Feature extraction mode"
    )
    parser.add_argument(
        "--skip-features", "-s",
        action="store_true",
        help="Skip feature extraction step (use existing files)"
    )
    parser.add_argument(
        "--skip-evaluation", "-e",
        action="store_true",
        help="Skip evaluation step"
    )
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    try:
        # Initialize and run pipeline
        pipeline = InferencePipeline(args.config, args.log_level)
        predictions = pipeline.run(
            args.mode,
            skip_features=args.skip_features,
            skip_evaluation=args.skip_evaluation
        )
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        sys.exit(1)
