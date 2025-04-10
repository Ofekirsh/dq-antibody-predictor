import json
import logging
import os
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.feature_sets.base_feature import BaseFeature

# Constants
DQA1_ROTATION = 22  # Fixed rotation value for DQA1
DQB1_ROTATION = 31  # Fixed rotation value for DQB1


class OneHotEpletFeature(BaseFeature):
    """
    Extracts One-Hot Eplet features based on allele sequences.

    This class generates feature vectors by identifying matching eplet descriptions
    between donor and recipient alleles and calculating mismatches.
    """

    def __init__(
            self,
            dqa_to_sequence_file: str,
            dqb_to_sequence_file: str,
            description_file: str = "data/jsons/allele_descriptions.json",
            log_level: int = logging.INFO
    ):
        """
        Initialize the One-Hot Eplet feature extractor.

        Args:
            dqa_to_sequence_file: Path to the JSON file containing DQA1 allele mappings
            dqb_to_sequence_file: Path to the JSON file containing DQB1 allele mappings
            description_file: Path to the JSON file containing allele descriptions
            log_level: Logging level

        Raises:
            ValueError: If any required file is missing or contains invalid JSON
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Load allele-to-sequence mappings
        self.dqa_to_seq = self._load_allele_mapping(dqa_to_sequence_file, "DQA1")
        self.dqb_to_seq = self._load_allele_mapping(dqb_to_sequence_file, "DQB1")

        # Apply rotation to sequences
        self.dqa_to_seq = self._rotate_values(self.dqa_to_seq, DQA1_ROTATION)
        self.dqb_to_seq = self._rotate_values(self.dqb_to_seq, DQB1_ROTATION)

        # Load allele descriptions
        self.list_of_all_descriptions = self._generate_all_descriptions(description_file)

        # Sets to track missing alleles
        self.missing_dqa_keys: Set[str] = set()
        self.missing_dqb_keys: Set[str] = set()

        self.logger.info(
            f"Initialized OneHotEpletFeature with {len(self.list_of_all_descriptions)} descriptions"
        )

    def _load_allele_mapping(self, file_path: str, allele_type: str) -> Dict[str, str]:
        """
        Load the allele-to-sequence mapping from a JSON file.

        Args:
            file_path: Path to the JSON file
            allele_type: Either 'DQA1' or 'DQB1' for error reporting

        Returns:
            Mapping of alleles to sequences

        Raises:
            ValueError: If the file is missing or contains invalid JSON
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            error_msg = f"Unable to load {allele_type} allele-to-sequence file ({file_path}): {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    @staticmethod
    def _rotate_values(dictionary: Dict[str, str], rotation: int) -> Dict[str, str]:
        """
        Rotate string values in a dictionary by the specified number of positions.

        Args:
            dictionary: Dictionary with keys and string values to be rotated
            rotation: Number of positions to rotate the string values

        Returns:
            Dictionary with rotated string values
        """
        return {
            key: value[rotation % len(value):] if isinstance(value, str) else value
            for key, value in dictionary.items()
        }

    def _generate_all_descriptions(self, description_file: str) -> List[str]:
        """
        Load all predefined allele descriptions from a JSON file.

        Args:
            description_file: Path to the JSON file

        Returns:
            List of allele descriptions

        Raises:
            ValueError: If the file is missing or contains invalid JSON
        """
        try:
            with open(description_file, 'r') as file:
                descriptions = json.load(file)
            self.logger.info(f"Loaded {len(descriptions)} descriptions from {description_file}")
            return descriptions
        except (FileNotFoundError, json.JSONDecodeError) as e:
            error_msg = f"Unable to load description file ({description_file}): {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def check_descriptions(self, seq: Optional[str]) -> np.ndarray:
        """
        Check which descriptions from a predefined list match the given sequence.

        Args:
            seq: The sequence to check

        Returns:
            Boolean mask where matching descriptions are marked as 1
        """
        if seq is None:
            return np.zeros(len(self.list_of_all_descriptions), dtype=int)

        matches = np.zeros(len(self.list_of_all_descriptions), dtype=int)

        for i, desc in enumerate(self.list_of_all_descriptions):
            try:
                desc_parts = desc.split(' ')
                len_conditions = len(desc_parts) // 2

                # Count matching conditions
                counter = 0
                for j in range(0, len(desc_parts), 2):
                    pos = int(desc_parts[j])
                    expected_char = desc_parts[j + 1]
                    if pos < len(seq) and seq[pos] == expected_char:
                        counter += 1

                if counter == len_conditions:
                    matches[i] = 1

            except (IndexError, ValueError):
                continue

        return matches

    @staticmethod
    def get_one_hot_patient_donor_description(
            patient_desc: np.ndarray,
            donor_desc: np.ndarray
    ) -> np.ndarray:
        """
        Generate a one-hot encoded vector for patient-donor description comparison.

        Args:
            patient_desc: One-hot encoding for patient descriptions
            donor_desc: One-hot encoding for donor descriptions

        Returns:
            One-hot vector indicating mismatches (1 if patient has 1 and donor has 0, otherwise 0)
        """
        return np.where((patient_desc == 1) & (donor_desc == 0), 1, 0)

    def check_label(self, row: pd.Series) -> Optional[float]:
        """
        Check if the Label value is valid.

        Args:
            row: A row from the DataFrame

        Returns:
            Validated Label (0.0 or 1.0) or None if invalid
        """
        label = row.get("Label", np.nan)

        # Handle NaN/NaT values
        if pd.isna(label) or isinstance(label, pd._libs.tslibs.nattype.NaTType):
            self.logger.warning(f"Row {row.name}: Label is missing (NaN/NaT)")
            return None

        try:
            label = float(label)
        except (ValueError, TypeError):
            self.logger.warning(f"Row {row.name}: Invalid label format '{label}' (could not convert to float)")
            return None

        if label not in {0.0, 1.0}:
            self.logger.warning(f"Row {row.name}: Unexpected label value '{label}' (must be 0.0 or 1.0)")
            return None

        return label

    def _handle_missing_keys(self, row: pd.Series) -> List[str]:
        """
        Identify and store missing allele keys for DQA1 and DQB1 mappings.

        Args:
            row: The input row being processed

        Returns:
            List of missing allele keys
        """
        missing_keys = []

        # Check DQA1 alleles
        for key in ["R_DQA1_1", "R_DQA1_2", "D_DQA1"]:
            try:
                allele = row[key]
                if allele not in self.dqa_to_seq:
                    self.missing_dqa_keys.add(allele)
                    missing_keys.append(f"{key}={allele}")
            except KeyError:
                missing_keys.append(f"{key}=<missing column>")

        # Check DQB1 alleles
        for key in ["R_DQB1_1", "R_DQB1_2", "D_DQB1"]:
            try:
                allele = row[key]
                if allele not in self.dqb_to_seq:
                    self.missing_dqb_keys.add(allele)
                    missing_keys.append(f"{key}={allele}")
            except KeyError:
                missing_keys.append(f"{key}=<missing column>")

        if missing_keys:
            self.logger.debug(f"Row {row.name}: Missing allele(s): {', '.join(missing_keys)}")

        return missing_keys

    def compute(self, row: pd.Series) -> Optional[np.ndarray]:
        """
        Compute One-Hot Eplet features for a given row.

        Args:
            row: A row from the DataFrame

        Returns:
            One-hot encoded feature vector or None if computation fails
        """
        try:
            # Validate Label
            label = self.check_label(row)
            if label is None:
                self.logger.info(f"Row {row.name}: Skipping due to invalid Label")
                return None

            # Get time value (default to NaN if missing)
            time = row.get("Time", np.nan)

            # Check for missing allele keys before proceeding
            missing_keys = self._handle_missing_keys(row)
            if missing_keys:
                self.logger.info(f"Row {row.name}: Skipping due to missing allele keys")
                return None

            # Convert alleles to sequences
            sequences = {
                'B': self.dqa_to_seq[row["R_DQA1_1"]],  # Recipient DQA1 first
                'C': self.dqb_to_seq[row["R_DQB1_1"]],  # Recipient DQB1 first
                'E': self.dqa_to_seq[row["R_DQA1_2"]],  # Recipient DQA1 second
                'F': self.dqb_to_seq[row["R_DQB1_2"]],  # Recipient DQB1 second
                'H': self.dqa_to_seq[row["D_DQA1"]],  # Donor DQA1
                'I': self.dqb_to_seq[row["D_DQB1"]]  # Donor DQB1
            }

            # Generate one-hot encodings for each sequence
            one_hot_encodings = {
                key: self.check_descriptions(seq)
                for key, seq in sequences.items()
            }

            # Compute patient-donor mismatches
            mismatches = {
                'BH': self.get_one_hot_patient_donor_description(one_hot_encodings['B'], one_hot_encodings['H']),
                'CI': self.get_one_hot_patient_donor_description(one_hot_encodings['C'], one_hot_encodings['I']),
                'EH': self.get_one_hot_patient_donor_description(one_hot_encodings['E'], one_hot_encodings['H']),
                'FI': self.get_one_hot_patient_donor_description(one_hot_encodings['F'], one_hot_encodings['I'])
            }

            # Sum one-hot vectors for combined representation
            combined_vector = sum(mismatches.values())

            # Calculate summary features (total mismatches per allele combination)
            summary_features = np.array([
                mismatches['BH'].sum(),  # DQA1 Low
                mismatches['CI'].sum(),  # DQB1 Low
                mismatches['EH'].sum(),  # DQA1 High
                mismatches['FI'].sum()  # DQB1 High
            ])

            # Create a consolidated feature vector
            mismatch_level = np.nan  # Placeholder for mismatch level
            result = np.concatenate((
                combined_vector,
                summary_features,
                np.array([mismatch_level, int(label), int(time)], dtype=object)
            ))

            return result

        except Exception as e:
            self.logger.error(f"Row {row.name}: Error computing features: {str(e)}")
            return None

    def get_feature_columns(self) -> List[str]:
        """
        Get the feature column names for One-Hot Eplet Features.

        Returns:
            List of column names
        """
        return (
                [f'{desc}' for desc in self.list_of_all_descriptions] +
                ["mm_DQA_Low", "mm_DQB_Low", "mm_DQA_High", "mm_DQB_High", "MM", "Label", "Time"]
        )

    def save_missing_keys(self, output_dir: str = "../output/missing_keys") -> None:
        """
        Save missing DQA1 and DQB1 keys to CSV files.

        Args:
            output_dir: Directory to save the CSV files
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save missing DQA1 keys
        if self.missing_dqa_keys:
            dqa_path = os.path.join(output_dir, "missing_dqa_keys.csv")
            pd.DataFrame({"Missing_DQA1_Alleles": sorted(self.missing_dqa_keys)}).to_csv(dqa_path, index=False)
            self.logger.info(f"Missing DQA1 keys ({len(self.missing_dqa_keys)}) saved to {dqa_path}")

        # Save missing DQB1 keys
        if self.missing_dqb_keys:
            dqb_path = os.path.join(output_dir, "missing_dqb_keys.csv")
            pd.DataFrame({"Missing_DQB1_Alleles": sorted(self.missing_dqb_keys)}).to_csv(dqb_path, index=False)
            self.logger.info(f"Missing DQB1 keys ({len(self.missing_dqb_keys)}) saved to {dqb_path}")
