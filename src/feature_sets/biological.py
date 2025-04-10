import csv
import json
import logging
import os
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from src.feature_sets.base_feature import BaseFeature


class BiologicalFeature(BaseFeature):
    """
    Extracts biological distance-based features (PAM, EMS, G12) for each row.

    This class calculates distance metrics between donor and recipient alleles
    based on pre-computed distance matrices.
    """

    def __init__(
            self,
            ems_file: Optional[str] = None,
            pam_file: Optional[str] = None,
            g12_file: Optional[str] = None,
            features_to_extract: Optional[List[str]] = None,
            log_level: int = logging.INFO
    ):
        """
        Initialize the biological feature extractor.

        Args:
            ems_file: Path to the JSON file containing EMS distances
            pam_file: Path to the JSON file containing PAM distances
            g12_file: Path to the JSON file containing G12 values
            features_to_extract: List of features to extract (e.g., ["PAM", "EMS", "G12"])
            log_level: Logging level

        Raises:
            ValueError: If a required file is missing or contains invalid JSON
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Initialize feature selection
        self.features_to_extract = features_to_extract or ["PAM", "EMS", "G12"]

        # Load only required distance matrices
        self.distance_matrices = {
            "EMS": self._load_json(ems_file) if "EMS" in self.features_to_extract else None,
            "PAM": self._load_json(pam_file) if "PAM" in self.features_to_extract else None,
            "G12": self._load_json(g12_file) if "G12" in self.features_to_extract else None
        }

        # Track missing keys for each distance type
        self.missing_keys: Dict[str, Set[str]] = {
            feature_type: set() for feature_type in self.features_to_extract
        }

        self.logger.info(f"Initialized BiologicalFeature with features: {self.features_to_extract}")

    def _load_json(self, file_path: Union[str, Dict]) -> Dict:
        """
        Load a JSON file into a dictionary.

        Args:
            file_path: Path to the JSON file or a dictionary for testing

        Returns:
            Dictionary containing the loaded data

        Raises:
            ValueError: If the file is missing or contains invalid JSON
        """
        if file_path is None:
            return None

        # Allow direct dictionary input for testing
        if isinstance(file_path, dict):
            return file_path

        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            error_msg = f"Error loading JSON file ({file_path}): {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def calculate_mismatch(self, first_pair: float, second_pair: float) -> int:
        """
        Determine the mismatch level (0, 1, or 2) based on distance values.

        Args:
            first_pair: Distance value for the first pair
            second_pair: Distance value for the second pair

        Returns:
            Mismatch level (0, 1, or 2)
        """
        if first_pair == 0 and second_pair == 0:
            return 0
        elif (first_pair == 0 and second_pair != 0) or (first_pair != 0 and second_pair == 0):
            return 1
        else:  # first_pair != 0 and second_pair != 0
            return 2

    def _get_distance(
            self,
            distance_type: str,
            key_standard: str,
            key_reversed: str
    ) -> Optional[float]:
        """
        Retrieve a distance value from the dictionary, checking both key orders.

        Args:
            distance_type: Type of distance (EMS, PAM, G12)
            key_standard: Standard order key
            key_reversed: Reversed order key

        Returns:
            Retrieved distance value or None if missing
        """
        distance_dict = self.distance_matrices[distance_type]
        if distance_dict is None:
            return None  # Feature not selected

        try:
            return distance_dict[key_standard]
        except KeyError:
            try:
                return distance_dict[key_reversed]  # Check reversed order
            except KeyError:
                self.missing_keys[distance_type].add(key_standard)
                self.logger.debug(f"Missing {distance_type} key: '{key_standard}' or '{key_reversed}'")
                return None

    def _extract_allele_keys(self, row) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Extract allele keys from a data row.

        Args:
            row: A data row containing allele information

        Returns:
            Tuple containing:
                - Dictionary of lookup keys
                - Dictionary of individual allele keys
        """
        # Extract recipient and donor alleles
        recipient_dqa1_first = f"DQA1*{row['R_DQA1_1']}"
        recipient_dqb1_first = f"DQB1*{row['R_DQB1_1']}"
        recipient_dqa1_second = f"DQA1*{row['R_DQA1_2']}"
        recipient_dqb1_second = f"DQB1*{row['R_DQB1_2']}"
        donor_dqa1 = f"DQA1*{row['D_DQA1']}"
        donor_dqb1 = f"DQB1*{row['D_DQB1']}"

        # Construct paired keys for lookup
        lookup_keys = {
            "recipient_first_donor": f"{recipient_dqa1_first}_{recipient_dqb1_first}_{donor_dqa1}_{donor_dqb1}",
            "recipient_second_donor": f"{recipient_dqa1_second}_{recipient_dqb1_second}_{donor_dqa1}_{donor_dqb1}",
            "donor_recipient_first": f"{donor_dqa1}_{donor_dqb1}_{recipient_dqa1_first}_{recipient_dqb1_first}",
            "donor_recipient_second": f"{donor_dqa1}_{donor_dqb1}_{recipient_dqa1_second}_{recipient_dqb1_second}"
        }

        # Individual allele keys
        allele_keys = {
            "recipient_first": f"{recipient_dqa1_first}_{recipient_dqb1_first}",
            "recipient_second": f"{recipient_dqa1_second}_{recipient_dqb1_second}",
            "donor": f"{donor_dqa1}_{donor_dqb1}"
        }

        return lookup_keys, allele_keys

    def compute(self, row):
        """
        Compute biological distance-based features for a given row.

        Args:
            row: A row from the DataFrame

        Returns:
            Computed feature vector or None if required features are missing
        """
        try:
            lookup_keys, allele_keys = self._extract_allele_keys(row)
            features = []

            # Extract PAM features if requested
            if "PAM" in self.features_to_extract:
                pam_first_pair = self._get_distance(
                    "PAM",
                    lookup_keys["recipient_first_donor"],
                    lookup_keys["donor_recipient_first"]
                )
                pam_second_pair = self._get_distance(
                    "PAM",
                    lookup_keys["recipient_second_donor"],
                    lookup_keys["donor_recipient_second"]
                )
                features.extend([pam_first_pair, pam_second_pair])

            # Extract EMS features if requested
            if "EMS" in self.features_to_extract:
                ems_first_pair = self._get_distance(
                    "EMS",
                    lookup_keys["recipient_first_donor"],
                    lookup_keys["donor_recipient_first"]
                )
                ems_second_pair = self._get_distance(
                    "EMS",
                    lookup_keys["recipient_second_donor"],
                    lookup_keys["donor_recipient_second"]
                )
                features.extend([ems_first_pair, ems_second_pair])

            # Extract G12 features if requested
            if "G12" in self.features_to_extract:
                g12_features = [
                    self._get_distance("G12", allele_keys["recipient_first"], allele_keys["recipient_first"]),
                    self._get_distance("G12", allele_keys["recipient_second"], allele_keys["recipient_second"]),
                    self._get_distance("G12", allele_keys["donor"], allele_keys["donor"])
                ]
                features.extend(g12_features)

            # Calculate mismatch level if possible
            mismatch = None
            if "EMS" in self.features_to_extract and ems_first_pair is not None and ems_second_pair is not None:
                mismatch = self.calculate_mismatch(ems_first_pair, ems_second_pair)
            elif "PAM" in self.features_to_extract and pam_first_pair is not None and pam_second_pair is not None:
                mismatch = self.calculate_mismatch(pam_first_pair, pam_second_pair)

            # Check for missing features
            if None in features or mismatch is None:
                self.logger.debug(f"Skipping row {row.name} due to missing feature values")
                return None

            # Add mismatch level, label, and time to features
            features.append(mismatch)
            features.append(row["Label"])
            features.append(row["Time"])

            return np.array(features, dtype=np.float32)

        except KeyError as e:
            self.logger.error(f"Missing expected column {e} in input data")
            return None

    def get_feature_columns(self):
        """
        Get the feature column names dynamically based on selected features.

        Returns:
            List of column names
        """
        columns = []

        if "PAM" in self.features_to_extract:
            columns.extend(["PAM Low", "PAM High"])

        if "EMS" in self.features_to_extract:
            columns.extend(["EMS Low", "EMS High"])

        if "G12" in self.features_to_extract:
            columns.extend(["G12_R1", "G12_R2", "G12_D"])

        # Always include these columns
        columns.extend(["MM", "Label", "Time"])

        return columns

    def save_missing_keys(self, output_dir="../output/missing_keys"):
        """
        Save missing keys into separate CSV files.

        Args:
            output_dir: Directory to save the CSV files
        """
        os.makedirs(output_dir, exist_ok=True)

        for feature_type, keys in self.missing_keys.items():
            if not keys:
                continue

            file_path = os.path.join(output_dir, f"missing_{feature_type.lower()}_keys.csv")
            with open(file_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Missing_Key"])
                for key in sorted(keys):
                    writer.writerow([key])

        self.logger.info(f"Missing keys saved to '{output_dir}/'")