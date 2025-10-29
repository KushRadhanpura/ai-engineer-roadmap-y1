import pandas as pd
from datasets import Dataset

def load_data(train_path, val_path):
    """
    Load training and validation data from CSV files.
    This is a placeholder function. You should modify it
    to fit your specific dataset structure.
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Assuming the CSVs have 'text' and 'label' columns
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    return train_dataset, val_dataset
