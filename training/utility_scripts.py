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

def prepare_descriptors_dataframe(
    df: pd.DataFrame,
    target_column: str = 'is_cpp',
    feature_columns: Optional[List[str]] = None,
    id_column: Optional[str] = None,
    strict_columns: bool = True
) -> pd.DataFrame:
    """
    Prepare a descriptor-based dataframe for downstream tasks.

    This function:
      - selects the requested feature columns,
      - optionally keeps an id column,
      - converts boolean columns to int,
      - encodes the target column to 0/1 using LabelEncoder,
      - checks that there are no missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_column : str, default='is_cpp'
        Target column name.
    feature_columns : list[str] or None
        Feature columns to use.
    id_column : str or None
        Optional id column to keep in the output dataframe.
        This column is preserved for matching rows later, but should not be used as a model feature.
    strict_columns : bool, default=True
        If True, raises an error when required columns are missing.
        If False, uses only available columns and prints a warning.

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe containing:
          - optional id column,
          - selected feature columns,
          - encoded target column.
    """
    if feature_columns is None:
        feature_columns = [
            'seq_length', 'molecular_weight', 'nh3_tail', 'po3_pos', 
            'biotinylated', 'acylated_n_terminal', 'cyclic', 'amidated', 
            'stearyl_uptake', 'hexahistidine_tagged', 'aromaticity', 
            'instability_index', 'isoelectric_point', 'helix_fraction', 
            'turn_fraction', 'sheet_fraction', 'molar_extinction_coefficient_reduced', 
            'molar_extinction_coefficient_oxidized', 'gravy'
        ]

    required_columns = feature_columns.copy()
    if id_column is not None:
        required_columns = [id_column] + required_columns

    missing = [c for c in required_columns + [target_column] if c not in df.columns]
    if missing:
        if strict_columns:
            raise ValueError(f"Missing columns in dataframe: {missing}")
        else:
            print(f"[WARN] Missing columns in dataframe: {missing}. Only available columns will be used.")
            required_columns = [c for c in required_columns if c in df.columns]
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' is missing from dataframe.")

    # Explicit copy to avoid SettingWithCopyWarning
    df_clean = df.loc[:, required_columns + [target_column]].copy()

    # Convert boolean columns to int
    bool_cols = df_clean.columns[df_clean.dtypes == 'bool'].tolist()
    for col in bool_cols:
        df_clean.loc[:, col] = df_clean[col].astype(int)

    # Encode target to 0/1
    label_encoder = LabelEncoder()
    df_clean.loc[:, target_column] = label_encoder.fit_transform(df_clean[target_column])

    # Check missing values in features and target
    feature_part = df_clean.drop(columns=[target_column])
    target_part = df_clean[target_column]

    assert feature_part.isnull().sum().sum() == 0, "There are missing values in the feature matrix"
    assert target_part.isnull().sum() == 0, "There are missing values in the target vector"

    print(
        f"Prepared dataframe shape: {df_clean.shape} | "
        f"features: {len([c for c in df_clean.columns if c not in [target_column, id_column]])} | "
        f"id included: {id_column is not None}"
    )

    return df_clean

def preprocess_descriptors(
    df: pd.DataFrame,
    target_column: str = 'is_cpp',
    columns_to_use: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    strict_columns: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Preprocess descriptor data and split it into train/test subsets.

    This is a wrapper around prepare_descriptors_dataframe(...).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_column : str, default='is_cpp'
        Target column name.
    columns_to_use : list[str] or None
        Feature columns to use.
    test_size : float, default=0.2
        Proportion of the dataset included in the test split.
    random_state : int, default=42
        Random seed for reproducibility.
    stratify : bool, default=True
        Whether to stratify the split by the target column.
    strict_columns : bool, default=True
        If True, raises an error when expected columns are missing.

    Returns
    -------
    X_train : pd.DataFrame
    X_test : pd.DataFrame
    y_train : pd.Series
    y_test : pd.Series
    """
    df_prepared = prepare_descriptors_dataframe(
        df=df,
        target_column=target_column,
        feature_columns=columns_to_use,
        id_column=None,
        strict_columns=strict_columns
    )

    X = df_prepared.drop(columns=[target_column])
    y = df_prepared[target_column]

    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )

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