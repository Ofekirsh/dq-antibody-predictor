from abc import ABC, abstractmethod
import pandas as pd


class BaseFeature(ABC):
    @abstractmethod
    def compute(self, row):
        """Computes features for a given row."""
        pass

    @abstractmethod
    def get_feature_columns(self):
        """Returns the column names for the extracted features."""
        pass

    def extract_features(self, df, output_file):
        """
        Extracts features for all rows in a DataFrame while maintaining ID consistency.

        Parameters:
        - df (pd.DataFrame): Input data.
        - output_file (str): Path to save the extracted features.

        Returns:
        - pd.DataFrame: Processed feature DataFrame.
        """

        # Ensure ID column exists
        if "ID" not in df.columns:
            raise ValueError(f"[{self.__class__.__name__}] Error: The input DataFrame must have an 'ID' column.")

        # Apply feature extraction row by row
        df["features"] = df.apply(lambda row: self.compute(row), axis=1)

        # Identify IDs where feature extraction returned None
        invalid_ids = df[df["features"].isna()]["ID"].unique()

        # Remove all rows with those IDs
        df_filtered = df[~df["ID"].isin(invalid_ids)].copy()  # Copy to avoid modifying original

        # If all rows were removed, return an empty DataFrame
        if df_filtered.empty:
            print(f"[{self.__class__.__name__}] All rows were removed due to missing data.")

            # Create an empty DataFrame with the correct headers
            empty_df = pd.DataFrame(columns=["ID"] + self.get_feature_columns())

            # Save the empty DataFrame with only headers
            empty_df.to_csv(output_file, index=False)
            print(f"[{self.__class__.__name__}] Empty feature file saved to: {output_file}")

            return empty_df

        # Preserve ID column before resetting index
        expanded_features = pd.DataFrame(df_filtered["features"].tolist(), index=df_filtered["ID"])

        # Get feature column names from subclass
        columns = self.get_feature_columns()

        # Assign column names to the expanded DataFrame
        expanded_features.columns = columns

        # Restore the ID column explicitly
        expanded_features.insert(0, "ID", expanded_features.index)

        # Save the processed DataFrame as a CSV
        expanded_features.to_csv(output_file, index=False)
        print(f"[{self.__class__.__name__}] Features saved to: {output_file}")

        return expanded_features
