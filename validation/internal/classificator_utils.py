from typing import List, Tuple, Optional

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import re
import numpy as np

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import molecular_weight

def categorize_sequences(
    df, 
    sequence_col='sequence', 
    output_col='sequence_category', 
    patterns=None
):
    """
    Categorizes peptide sequences based on regex patterns.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        sequence_col (str): Column containing peptide sequences
        output_col (str): Output column for categories
        patterns (dict): Dictionary of {category: regex pattern}
    
    Returns:
        pd.DataFrame: Modified DataFrame with category column
    """
    if patterns is None:
        patterns = {
            "Standard": re.compile(r"^[A-Z]+$"),
            "Enantiomer Mix": re.compile(r"^[A-Za-z]+$"),
            "Chemical Formula": re.compile(r"^[A-Za-z0-9]+-[a-z]+$"),
            "Special Structure": re.compile(r"^[A-Za-z0-9\-]+-BLGTYTQDFNXFHTFPQTAIGVGAP$"),
            "Repeating Segments": re.compile(r"^[A-Za-z0-9\(\)\-]+-NH2$"),
            "Biotinylated": re.compile(r"\(biotin\)-[a-zA-Z0-9]+"),
            "N-terminal Acylated": re.compile(r"^Ac-[A-Z]+.*$"),
            "Cyclic": re.compile(r"^Cyclo\([A-Za-z0-9]+.*\)$"),
            "Amidated": re.compile(r"^.*-NH2$"),
            "Peptide with Substitutions": re.compile(r"[A-Za-z0-9\(\)\-\[\]]+"),
            "Uptake with Stearyl": re.compile(r".*(Stearyl|Myristoyl|Lauroyl)-[A-Za-z0-9]+"),
            "Polypeptide with Repeats": re.compile(r"poly-[A-Za-z0-9]+"),
            "Hexahistidine Tagged": re.compile(r"^(His6|HHHHHH)+.*"),
            "Unknown": re.compile(r".*")
        }

    def _categorizer(sequence):
        for category, pattern in patterns.items():
            if pattern.match(str(sequence)):
                return category
        return "Unknown"

    df[output_col] = df[sequence_col].apply(_categorizer)
    return df

def standardize_sequence(sequence):
    """
    Helper function to extract features from peptide sequences.
    Returns tuple of (cleaned_sequence, features...).
    """
    features = {
        'nh3_tail': False,
        'po3_pos': False,
        'biotinylated': False,
        'acylated_n_terminal': False,
        'cyclic': False,
        'amidated': False,
        'stearyl_uptake': False,
        'hexahistidine_tagged': False,
        'modifications': []
    }

    if not isinstance(sequence, str):
        return (np.nan, *features.values())

    # Feature detection
    features['nh3_tail'] = bool(re.search(r"-NH2$", sequence))
    features['po3_pos'] = bool(re.search(r"\(PO3\)", sequence))
    features['biotinylated'] = bool(re.search(r"\(biotin\)", sequence))
    features['acylated_n_terminal'] = bool(re.search(r"^Ac-", sequence))
    features['cyclic'] = bool(re.search(r"^Cyclo\(", sequence))
    features['amidated'] = bool(re.search(r"-NH2$", sequence))
    features['stearyl_uptake'] = bool(re.search(r"(Stearyl|Myristoyl|Lauroyl)", sequence))
    features['hexahistidine_tagged'] = bool(re.search(r"^(His6|HHHHHH)+", sequence))

    # Sequence cleaning
    cleaned = re.sub(r"[^A-Za-z]", "", sequence)
    cleaned_standard = re.sub(r"[^A-Z]", "", cleaned)
    
    if cleaned_standard.isalpha() and cleaned_standard.isupper():
        features['modifications'] = [
            (char.upper(), i+1) 
            for i, char in enumerate(cleaned) 
            if char.islower()
        ]
        return (cleaned_standard, *features.values())
    
    return (np.nan, *features.values())

def add_sequence_features(
    df,
    sequence_col='sequence',
    output_cols=[
        'standard_sequence', 'nh3_tail', 'po3_pos', 'biotinylated',
        'acylated_n_terminal', 'cyclic', 'amidated', 'stearyl_uptake',
        'hexahistidine_tagged', 'modifications'
    ]
):
    """
    Adds standardized sequence and feature columns to DataFrame.
    """
    df[output_cols] = df[sequence_col].apply(
        lambda x: pd.Series(standardize_sequence(x))
    )
    return df

def validate_sequences(df, sequence_col='standard_sequence'):
    """
    Filters valid sequences using standard amino acid check.
    """
    def _is_valid(seq):
        if not isinstance(seq, str):
            return False
        return bool(re.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", seq))
    
    df[sequence_col] = df[sequence_col].apply(
        lambda s: s if _is_valid(s) else np.nan
    )
    return df

pattern = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

def is_valid_amino_acid_sequence(sequence):

    if isinstance(sequence, str):
        return bool(pattern.match(sequence))
    
    return False

def calculate_protein_descriptors(sequence):

    if not is_valid_amino_acid_sequence(sequence):
        return pd.Series({})

    analysis = ProteinAnalysis(sequence)

    descriptors = {
        'molecular_weight': analysis.molecular_weight(),
        'seq_length': len(sequence),
        'aromaticity': analysis.aromaticity(),
        'instability_index': analysis.instability_index(),
        'isoelectric_point': analysis.isoelectric_point(),
        'helix_fraction': analysis.secondary_structure_fraction()[0],
        'turn_fraction': analysis.secondary_structure_fraction()[1],
        'sheet_fraction': analysis.secondary_structure_fraction()[2],
        'molar_extinction_coefficient_reduced': analysis.molar_extinction_coefficient()[0],
        'molar_extinction_coefficient_oxidized': analysis.molar_extinction_coefficient()[1],
        'gravy': analysis.gravy()
    }
    
    return pd.Series(descriptors)

def add_descriptors_features(
        df: pd.DataFrame,
        sequence_column: str = 'standard_sequence',
):
    
    """
    Adds protein descriptors features to the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing sequences.
    sequence_column (str): The name of the column containing amino acid sequences.

    Returns:
    pd.DataFrame: The DataFrame with added protein descriptors features.
    """
    
    descriptors_df = df[sequence_column].apply(calculate_protein_descriptors)
    
    # Concatenate the original DataFrame with the descriptors DataFrame
    df_with_descriptors = pd.concat([df, descriptors_df], axis=1)
    
    return df_with_descriptors

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
    
    return X_train, X_test, y_train, y_test, label_encoder