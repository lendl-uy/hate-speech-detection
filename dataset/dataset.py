import pandas as pd
import pickle
from typing import Tuple, List

class Dataset:
    """
    A class to handle loading and preprocessing of dataset for machine learning.
    """
    def __init__(self, train_path: str = None, val_path: str = None, test_path: str = None, full_data_path: str = None, split_sizes: List = None, from_scratch=True):
        self.from_scratch = from_scratch
        if from_scratch:
            if (train_path is None) or (val_path is None) or (test_path is None):
                raise ValueError("Paths to the dataset splits were not specified. Please recreate a Dataset object \
                                 with the paths to the dataset splits.")
            self.train_path = train_path
            self.val_path = val_path
            self.test_path = test_path
        else:
            if (full_data_path is None) or (split_sizes is None):
                raise ValueError("Path to the full data and/or dataset splits were not specified. Please recreate a Dataset object with \
                                 the path to the full data and/or the splits of the dataset.")
            self.full_data_path = full_data_path
            self.split_sizes = split_sizes

    def build(self) -> None:
        if self.from_scratch:
            train_data = self.read_csv(self.train_path)
            val_data = self.read_csv(self.val_path)
            test_data = self.read_csv(self.test_path)

            train_data = self.preprocess_data(train_data)
            val_data = self.preprocess_data(val_data)
            test_data = self.preprocess_data(test_data)

            self.X_train, self.Y_train = self.get_features_and_labels(train_data)
            self.X_val, self.Y_val = self.get_features_and_labels(val_data)
            self.X_test, self.Y_test = self.get_features_and_labels(test_data)
        else:
            try:
                with open(self.full_data_path, "rb") as f:
                    X, Y = pickle.load(f)
                    print(f"Data loaded from {self.full_data_path}")
                    train_split = self.split_sizes[0]
                    val_split = self.split_sizes[0] + self.split_sizes[1]
                    test_split = self.split_sizes[0] + self.split_sizes[1] + self.split_sizes[2]
                    self.X_train, self.Y_train = X[:train_split], Y[:train_split]
                    self.X_val, self.Y_val = X[train_split:val_split], Y[train_split:val_split]
                    self.X_test, self.Y_test = X[val_split:test_split], Y[val_split:test_split]
            except FileNotFoundError:
                print("File not found. Please check the path or pull the dataset first.")

    def save_to_file(self, X, Y, file_path):
        with open(file_path, "wb") as f:
            pickle.dump((X, Y), f)
            print(f"Data saved to {file_path}")

    def get_features(self, split_type="all") -> List:
        """Returns a copy of the features."""
        if split_type == "train":
            return self.X_train
        elif split_type == "val":
            return self.X_val
        elif split_type == "test":
            return self.X_test
        else:
            return self.X_train + self.X_val + self.X_test 
    
    def get_labels(self, split_type="all") -> List:
        """Returns a copy of the labels."""
        if split_type == "train":
            return self.Y_train
        elif split_type == "val":
            return self.Y_val
        elif split_type == "test":
            return self.Y_test
        else:
            return self.Y_train + self.Y_val + self.Y_test 

    def read_csv(self, file_path: str) -> List[List[str]]:
        """Reads a CSV file using pandas and returns the data as a list of rows."""
        try:
            # Using pandas to read the CSV file
            df = pd.read_csv(file_path, delimiter=',', quotechar='"', lineterminator='\n', encoding='utf-8')
            
            # Convert dataframe to a list of lists
            data = df.values.tolist()
            return data
        except FileNotFoundError:
            print(f"Error: The file {file_path} does not exist.")
            return []
        except Exception as e:
            print(f"Error reading the file {file_path}: {e}")
            return []

    def preprocess_data(self, data: List[List[str]]) -> List[List[str]]:
        """Removes empty rows and rows with missing labels."""
        return [row for row in data if row and len(row) > 1]

    def get_features_and_labels(self, data: List[List[str]]) -> Tuple[List, List]:
        """Extracts features and labels from the dataset."""
        X = [row[0] for row in data]
        Y = [int(row[1]) for row in data]
        return X, Y