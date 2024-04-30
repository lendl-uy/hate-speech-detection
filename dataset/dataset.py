import csv
from typing import Tuple, List

class Dataset:
    """
    A class to handle loading and preprocessing of dataset for machine learning.
    """
    def __init__(self, train_path: str, val_path: str, test_path: str, for_cleaning: bool = False):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

    def build(self) -> None:
        train_data = self.read_csv(self.train_path)
        val_data = self.read_csv(self.val_path)
        test_data = self.read_csv(self.test_path)

        train_data = self.preprocess_data(train_data)
        val_data = self.preprocess_data(val_data)
        test_data = self.preprocess_data(test_data)

        self.X_train, self.Y_train = self.get_features_and_labels(train_data)
        self.X_val, self.Y_val = self.get_features_and_labels(val_data)
        self.X_test, self.Y_test = self.get_features_and_labels(test_data)

    def get_features(self) -> List:
        """Returns a copy of combined features from all sets."""
        return self.X_train + self.X_val + self.X_test
    
    def get_labels(self) -> List:
        """Returns a copy of combined labels from all sets."""
        return self.Y_train + self.Y_val + self.Y_test

    def read_csv(self, file_path: str) -> List[List[str]]:
        """Reads a CSV file and returns the data as a list of rows."""
        try:
            data = []
            with open(file_path, newline="", encoding="utf-8") as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
                    data.append(row)
            return data
        except FileNotFoundError:
            print(f"Error: The file {file_path} does not exist.")
            return []

    def preprocess_data(self, data: List[List[str]]) -> List[List[str]]:
        """Removes empty rows, rows with missing labels, and skips the title row."""
        return [row for row in data if row and len(row) > 1][1:]

    def get_features_and_labels(self, data: List[List[str]]) -> Tuple[List, List]:
        """Extracts features and labels from the dataset."""
        X = [row[0] for row in data]
        Y = [int(row[1]) for row in data]
        return X, Y