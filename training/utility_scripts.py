from typing import List, Tuple, Optional

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score


def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix'):
    """
    Plots a confusion matrix using seaborn heatmap.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    labels (list): List of label names.
    title (str): Title of the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

def preprocess_descriptors(
    df: pd.DataFrame, 
    target_column: str = 'is_cpp', 
    columns_to_use: Optional[List[str]] = None, 
    test_size: float = 0.2, 
    random_state: int = 42, 
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Preprocesses the data by selecting specified columns, encoding categorical variables,
    splitting into train and test sets, and asserting no missing values.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    target_column (str): The name of the target column.
    columns_to_use (list): List of columns to use. If None, default columns will be used.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.
    stratify (bool): Whether to stratify the split based on the target column.

    Returns:
    X_train (pd.DataFrame): Training feature matrix.
    X_test (pd.DataFrame): Testing feature matrix.
    y_train (pd.Series): Training target vector.
    y_test (pd.Series): Testing target vector.
    """
    if columns_to_use is None:
        columns_to_use = [
            'seq_length', 'molecular_weight', 'nh3_tail', 'po3_pos', 
            'biotinylated', 'acylated_n_terminal', 'cyclic', 'amidated', 
            'stearyl_uptake', 'hexahistidine_tagged', 'aromaticity', 
            'instability_index', 'isoelectric_point', 'helix_fraction', 
            'turn_fraction', 'sheet_fraction', 'molar_extinction_coefficient_reduced', 
            'molar_extinction_coefficient_oxidized', 'gravy'
        ]
    
    # Select specified columns
    df_clean = df[columns_to_use + [target_column]]
    
    # Encode boolean columns
    bool_cols = df_clean.columns[df_clean.dtypes == 'bool'].tolist()
    for col in bool_cols:
        df_clean[col] = df_clean[col].astype(int)
    
    # Encode target variable
    label_encoder = LabelEncoder()
    df_clean[target_column] = label_encoder.fit_transform(df_clean[target_column])
    
    # Define features and target
    X = df_clean.drop(target_column, axis=1)
    y = df_clean[target_column]
    
    # Assert no missing values
    assert X.isnull().sum().sum() == 0, "There are missing values in the feature matrix"
    assert y.isnull().sum() == 0, "There are missing values in the target vector"
    
    # Split the data
    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    # Print the shapes of the resulting dataframes
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

# Function to calculate evaluation metrics
def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
    }