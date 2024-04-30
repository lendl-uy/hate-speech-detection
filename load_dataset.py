import pickle

def load_dataset(file_path):
    try:
        with open(file_path, "rb") as f:
            X, Y = pickle.load(f)
            print(f"Data loaded from {file_path}")
            return X, Y
    except FileNotFoundError:
        print("File not found. Please check the path or pull the dataset first.")
        return None, None