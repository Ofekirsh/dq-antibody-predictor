"""
Feature Engineering and Model Training Pipeline

This script implements a complete ML pipeline for feature extraction,
model training, evaluation, and prediction with configurable components.
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
from src.modeling.train import train_and_save_model
from src.modeling.predict import predict_with_model
from src.modeling.evaluate import evaluate_and_plot_roc_auc


class Pipeline:
    """
    End-to-end ML pipeline for feature engineering, training, and evaluation.
    Configurable through YAML configuration files.
    """

    def __init__(self, config_path: str, log_level: str = "INFO"):
        """
        Initialize the pipeline with configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = self._setup_logger(log_level)
        self.logger.info(f"Initializing pipeline with config: {config_path}")

        # Load configuration
        self.config = self._load_config(config_path)

        # Ensure output directories exist
        self._ensure_directories()

    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Set up and configure logger."""
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }

        logger = logging.getLogger("Pipeline")
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
                self.logger.debug(f"Configuration loaded successfully")
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
            self.config['features']['combined']['output_file']
        ]:
            output_dirs.add(os.path.dirname(output))

        # Model and evaluation directories
        output_dirs.add(os.path.dirname(self.config['model']['path']))
        output_dirs.add(os.path.dirname(self.config['evaluation']['roc_plot_path']))

        # Create directories
        for directory in output_dirs:
            if directory:  # Skip empty directory names
                Path(directory).mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Ensured directory exists: {directory}")

    def extract_features(self, mode: str) -> str:
        """
        Extract features based on specified mode.

        Args:
            mode: Feature extraction mode ('one_hot', 'biological', or 'combined')

        Returns:
            Path to the output feature file

        Raises:
            ValueError: If an invalid mode is specified
        """
        self.logger.info(f"Starting feature extraction - Mode: {mode}")

        # Load input data
        input_file = self.config['data']['input_file']
        self.logger.info(f"Loading data from {input_file}")
        df = load_xlsx_to_dataframe(input_file)

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

    def train_model(self, feature_file: str) -> tuple:
        """
        Train model using extracted features.

        Args:
            feature_file: Path to the feature file

        Returns:
            Tuple containing (average_auc, auc_standard_error, best_model, id_predictions)
        """
        self.logger.info(f"Loading features from {feature_file}")
        df = pd.read_csv(feature_file)

        model_config = self.config['model']
        model_type = model_config['type']
        k_folds = model_config['k_folds']
        save_model = model_config['save']
        model_path = model_config['path']

        self.logger.info(f"Training {model_type} model using {k_folds}-fold cross-validation")
        return train_and_save_model(
            df,
            model_type=model_type,
            k=k_folds,
            save_model=save_model,
            model_path=model_path
        )

    def predict(self, test_data: Optional[str] = None) -> Dict:
        """
        Make predictions using the trained model.

        Args:
            test_data: Path to test data file (uses training data if None)

        Returns:
            Dictionary of predictions
        """
        model_path = self.config['model']['path']

        if test_data is None:
            test_data = self.config['features']['combined']['output_file']

        self.logger.info(f"Loading test data from {test_data}")
        test_df = pd.read_csv(test_data)

        self.logger.info(f"Making predictions using model from {model_path}")
        return predict_with_model(test_df, model_path)

    def evaluate(self, id_predictions: Dict) -> None:
        """
        Evaluate model performance and generate ROC curve.

        Args:
            id_predictions: Dictionary with ID-prediction mappings
        """
        output_plot = self.config['evaluation']['roc_plot_path']

        self.logger.info("Evaluating model and generating ROC AUC plot")
        evaluate_and_plot_roc_auc(id_predictions, filename=output_plot)
        self.logger.info(f"ROC AUC plot saved to {output_plot}")

    def run(self, feature_mode: str, skip_features: bool = False) -> None:
        """
        Execute the complete pipeline.

        Args:
            feature_mode: Feature extraction mode ('one_hot', 'biological', 'combined')
            skip_features: Skip feature extraction if True
        """
        self.logger.info(f"Starting pipeline run - Feature mode: {feature_mode}")

        # Feature extraction
        if skip_features:
            self.logger.info("Skipping feature extraction as requested")
            if feature_mode == "eplet":
                feature_file = self.config['features']['eplet']['output_file']
            elif feature_mode == "biological":
                feature_file = self.config['features']['biological']['output_file']
            else:  # combined
                feature_file = self.config['features']['combined']['output_file']
        else:
            feature_file = self.extract_features(feature_mode)
            self.logger.info(f"Feature extraction completed: {feature_file}")

        # Model training
        avg_auc, auc_se, best_model, id_predictions = self.train_model(feature_file)
        self.logger.info(f"Training completed - Avg AUC: {avg_auc:.4f}, SE: {auc_se:.4f}")

        # Model evaluation
        self.evaluate(id_predictions)

        self.logger.info("Pipeline execution completed successfully")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Feature Engineering and Model Training Pipeline"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="../configs/train_pipeline_config.yml",
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
        help="Skip feature extraction step"
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
        pipeline = Pipeline(args.config, args.log_level)
        pipeline.run(args.mode, args.skip_features)
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        sys.exit(1)