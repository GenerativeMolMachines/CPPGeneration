import re

import pandas as pd

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import molecular_weight

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