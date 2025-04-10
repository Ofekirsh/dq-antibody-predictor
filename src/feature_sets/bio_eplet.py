import pandas as pd
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from src.feature_sets.one_hot_eplet import OneHotEpletFeature
from src.feature_sets.biological import BiologicalFeature


class BioEpletFeature:
    """
    Combines One-Hot Eplet and Biological Feature extraction while ensuring row consistency.

    This extractor merges features from both approaches to create comprehensive feature vectors
    for immunological compatibility analysis. It handles data validation, logging, and ensures
    consistent output formatting.
    """

    def __init__(
            self,
            dqa_to_seq: str,
            dqb_to_seq: str,
            allele_desc: str,
            ems_file: str,
            pam_file: str,
            g12_file: str,
            bio_features_to_extract: List[str],
            log_level: int = logging.INFO
    ):
        """
        Initialize the combined feature extractor with paths to required data files.

        Args:
            dqa_to_seq: Path to DQA1 sequence mapping file
            dqb_to_seq: Path to DQB1 sequence mapping file
            allele_desc: Path to allele descriptions file
            ems_file: Path to EMS distances file
            pam_file: Path to PAM distances file
            g12_file: Path to G12 classification file
            bio_features_to_extract: List of biological features to extract (e.g., ["PAM", "EMS", "G12"])
            log_level: Logging level (default: logging.INFO)

        Raises:
            FileNotFoundError: If any of the required files don't exist
        """
        # Set up logging
        self.logger = self._setup_logger(log_level)

        # Validate input files
        self._validate_input_files(
            dqa_to_seq, dqb_to_seq, allele_desc,
            ems_file, pam_file, g12_file
        )

        # Initialize feature extractors
        self.logger.info("Initializing feature extractors...")
        self.one_hot_extractor = OneHotEpletFeature(dqa_to_seq, dqb_to_seq, allele_desc)
        self.bio_extractor = BiologicalFeature(
            ems_file=ems_file,
            pam_file=pam_file,
            g12_file=g12_file,
            features_to_extract=bio_features_to_extract
        )
        self.bio_features_to_extract = bio_features_to_extract
        self.logger.info("Feature extractors initialized successfully")

    def _setup_logger(self, log_level: int) -> logging.Logger:
        """Configure and return a logger for this class."""
        logger = logging.getLogger("BioEpletFeatureExtractor")
        logger.setLevel(log_level)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _validate_input_files(self, *file_paths: str) -> None:
        """
        Ensure all required input files exist.

        Args:
            *file_paths: Variable number of file paths to check

        Raises:
            FileNotFoundError: If any file doesn't exist
        """
        for path in file_paths:
            if not Path(path).is_file():
                self.logger.error(f"File not found: {path}")
                raise FileNotFoundError(f"Required file not found: {path}")

    def extract_features(
            self,
            df: pd.DataFrame,
            output_file: Optional[str] = "vectors_model/combined_features.csv"
    ) -> Optional[pd.DataFrame]:
        """
        Extract and combine One-Hot Eplet and Biological features.

        Args:
            df: Input DataFrame containing allele data
            output_file: Path to save extracted features (set to None to skip saving)

        Returns:
            DataFrame with combined features or None if processing failed

        Raises:
            ValueError: If input DataFrame doesn't have required columns
        """
        self.logger.info(f"Beginning feature extraction for {len(df)} rows")

        # Validate input DataFrame
        if "ID" not in df.columns:
            error_msg = "Input DataFrame must have an 'ID' column"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Track extraction statistics
        combined_features = []
        valid_ids = set()
        error_counts = {"one_hot": 0, "biological": 0}

        # Process each row
        for index, row in df.iterrows():
            row_id = row["ID"]

            # Extract One-Hot Eplet Features
            one_hot_features = self.one_hot_extractor.compute(row)
            if one_hot_features is None:
                self.logger.warning(f"Row {index} (ID={row_id}): One-Hot Eplet extraction failed")
                error_counts["one_hot"] += 1
                continue

            # Extract Biological Features
            bio_features = self.bio_extractor.compute(row)
            if bio_features is None:
                self.logger.warning(f"Row {index} (ID={row_id}): Biological extraction failed")
                error_counts["biological"] += 1
                continue

            # Remove redundant columns (Label & Time) from one_hot_features
            one_hot_features = one_hot_features[:-3]  # Exclude last 3 elements

            # Combine features and store
            combined_vector = [row_id] + list(one_hot_features) + list(bio_features)
            combined_features.append(combined_vector)
            valid_ids.add(row_id)

        # Handle empty results
        if not combined_features:
            self.logger.error("No valid rows were processed. Output will be empty.")
            return None

        # Create result DataFrame with appropriate columns
        features_df = self._create_features_dataframe(combined_features)

        # Save results if output_file is specified
        if output_file:
            self._save_results(features_df, output_file)

        # Log statistics
        self._log_extraction_statistics(df, features_df, error_counts)

        return features_df

    def _create_features_dataframe(
            self,
            combined_features: List[List[Any]],
    ) -> pd.DataFrame:
        """
        Create a DataFrame from the extracted features with appropriate column names.

        Args:
            combined_features: List of feature vectors

        Returns:
            DataFrame with properly named columns
        """
        # Define column names dynamically based on selected features
        bio_feature_columns = []
        if "PAM" in self.bio_features_to_extract:
            bio_feature_columns.extend(["PAM_Low", "PAM_High"])
        if "EMS" in self.bio_features_to_extract:
            bio_feature_columns.extend(["EMS_Low", "EMS_High"])
        if "G12" in self.bio_features_to_extract:
            bio_feature_columns.extend(["G12_R1", "G12_R2", "G12_D"])

        columns = (
                ["ID"] +
                [f'{desc}' for desc in self.one_hot_extractor.list_of_all_descriptions] +
                ["mm_DQA_Low", "mm_DQB_Low", "mm_DQA_High", "mm_DQB_High"] +
                bio_feature_columns +
                ["MM", "Label", "Time"]
        )

        # Create DataFrame with appropriate columns
        features_df = pd.DataFrame(combined_features, columns=columns)

        # Ensure pairs of rows with the same ID exist (important for compatibility analysis)
        features_df = features_df[features_df["ID"].duplicated(keep=False)].reset_index(drop=True)

        return features_df

    def _save_results(self, features_df: pd.DataFrame, output_file: str) -> None:
        """
        Save the features DataFrame to a CSV file and log missing keys.

        Args:
            features_df: DataFrame containing the extracted features
            output_file: Path to save the CSV file
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        features_df.to_csv(output_file, index=False)
        self.logger.info(f"Features saved to: {output_file}")

        # Save information about missing keys for future analysis
        self.bio_extractor.save_missing_keys()
        self.one_hot_extractor.save_missing_keys()
        self.logger.info("Missing biological and alleles keys saved")

    def _log_extraction_statistics(
            self,
            input_df: pd.DataFrame,
            output_df: pd.DataFrame,
            error_counts: Dict[str, int]
    ) -> None:
        """
        Log statistics about the extraction process.

        Args:
            input_df: Original input DataFrame
            output_df: Resulting features DataFrame
            error_counts: Dictionary with counts of different error types
        """
        input_rows = len(input_df)
        output_rows = len(output_df)
        success_rate = (output_rows / input_rows * 100) if input_rows > 0 else 0

        self.logger.info(
            f"Extraction complete: {output_rows}/{input_rows} rows processed successfully ({success_rate:.1f}%)")
        self.logger.info(
            f"Error breakdown: One-Hot failures: {error_counts['one_hot']}, Biological failures: {error_counts['biological']}")