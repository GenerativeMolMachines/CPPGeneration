import re
import pandas as pd
import numpy as np
from rdkit import Chem
from collections import Counter
from typing import Dict, Set

from rdkit.Chem import Draw
from IPython.display import display

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

class PeptideToSmilesConverter:
    """Converts peptide sequences to SMILES with modification handling."""
    
    def __init__(self):
        self._init_maps()
        self.unrecognized_aa_counter = Counter()
        self.bad_sequences: Set[str] = set()
        
    def _init_maps(self):
        """Initialize SMILES mappings for modifications and amino acids."""
        self.modification_map = {
            "Ac-": "CC(=O)",
            "(Acp)": "CC(=O)",  
            "(biotin)": "C1[C@H]2SC(=S)N[C@H]1CCCCC(=O)",  # Correct biotin structure
            "-NH2": "N",
            "Stearyl": "CCCCCCCCCCCCCCCCCC(=O)",  # 18 carbons (C18)
            "Myristoyl": "CCCCCCCCCCCCCC(=O)",    # Correct (C14)
            "Lauroyl": "CCCCCCCCCCCC(=O)",        # Correct (C12)
            "Nspe": "N[C@H](C(C)O)C(=O)",         # Assume correct stereochemistry
            "Nbtg": "N[C@H](C(C)(C)C)C(=O)",      # N-tert-butylglycine
            "Ac": "CC(=O)",
            "Et": "OCC",                          # Ethyl ester (O-linked)
            "Npm": "N1[C@H](C(C)C)C(=O)",         # Verify context if needed
            "Nssb": "N1[C@H](C(C)CC)C(=O)",       # Verify context
            "Mpa": "SCCC(=O)",                    # Mercaptopropionyl (HS-CH2CH2CO-)
            "Cou": "C1=CC(=O)OC2=CC=CC=C12",      # Coumarin (corrected)
            "Xr": "N[C@H](C(C)C)C(=O)",           # D-valine (matches 'v' in aa_map)
            "NII": "C(C)C",                       # N-isopropyl (assumed)
            "PIC": "C1=CC=NC(=C1)CO",             # 4-picolyl (example)
            "IC": "NC1=NC(=O)NC=N1"               # Isocytosine
        }
        
        self.aa_map = {
            # Standard L/D amino acids
            'A': 'N[C@@H](C)C(=O)', 'a': 'N[C@H](C)C(=O)',
            'C': 'N[C@@H](CS)C(=O)', 'c': 'N[C@H](CS)C(=O)',
            'D': 'N[C@@H](CC(=O)O)C(=O)', 'd': 'N[C@H](CC(=O)O)C(=O)',
            'E': 'N[C@@H](CCC(=O)O)C(=O)', 'e': 'N[C@H](CCC(=O)O)C(=O)',
            'F': 'N[C@@H](CC1=CC=CC=C1)C(=O)', 'f': 'N[C@H](CC1=CC=CC=C1)C(=O)',
            'G': 'NCC(=O)', 'g': 'NCC(=O)',
            'H': 'N[C@@H](CC1=CNC=N1)C(=O)', 'h': 'N[C@H](CC1=CNC=N1)C(=O)',
            'I': 'N[C@@H](C(C)CC)C(=O)', 'i': 'N[C@H](C(C)CC)C(=O)',
            'K': 'N[C@@H](CCCCN)C(=O)', 'k': 'N[C@H](CCCCN)C(=O)',
            'L': 'N[C@@H](CC(C)C)C(=O)', 'l': 'N[C@H](CC(C)C)C(=O)',
            'M': 'N[C@@H](CCSC)C(=O)', 'm': 'N[C@H](CCSC)C(=O)',
            'N': 'N[C@@H](CC(=O)N)C(=O)', 'n': 'N[C@H](CC(=O)N)C(=O)',
            'P': 'N1[C@@H](CCC1)C(=O)', 'p': 'N1[C@H](CCC1)C(=O)',
            'Q': 'N[C@@H](CCC(=O)N)C(=O)', 'q': 'N[C@H](CCC(=O)N)C(=O)',
            'R': 'N[C@@H](CCCNC(=N)N)C(=O)', 'r': 'N[C@H](CCCNC(=N)N)C(=O)',
            'S': 'N[C@@H](CO)C(=O)', 's': 'N[C@H](CO)C(=O)',
            'T': 'N[C@@H](C(O)C)C(=O)', 't': 'N[C@H](C(O)C)C(=O)',
            'V': 'N[C@@H](C(C)C)C(=O)', 'v': 'N[C@H](C(C)C)C(=O)',
            'W': 'N[C@@H](CC1=CNC2=CC=CC=C12)C(=O)', 'w': 'N[C@H](CC1=CNC2=CC=CC=C12)C(=O)',
            'Y': 'N[C@@H](CC1=CC=C(O)C=C1)C(=O)', 'y': 'N[C@H](CC1=CC=C(O)C=C1)C(=O)',
            # Special cases
            'X': '*', '?': '*',
            'O': 'N[C@@H](CCCCN)C(=O)',  # Ornithine (corrected side chain)
            'Aib': 'NC(C)(C)C(=O)',      # Aib without chiral center
            'B': 'N[C@@H](CC(=O)N)C(=O)',  # Asn (standard 'B' ambiguity resolved to Asn)
            # Removed invalid 'b' entry to avoid conflicts
        }
        
    @staticmethod
    def expand_repeats(sequence: str) -> str:
        """Expand notation like R8 to RRRRRRRR."""
        # Use a substitution function to replace each occurrence of <letter(s)><digit>
        def repl(match):
            token = match.group(1)
            count = int(match.group(2))
            return token * count
        return re.sub(r'([A-Za-z]+)(\d+)', repl, sequence)
    
    def sequence_to_smiles(self, sequence: str) -> str:
        """Convert a raw peptide sequence to SMILES."""
        try:
            # First try RDKit's built-in conversion.
            mol = Chem.MolFromSequence(sequence)
            if mol:
                return Chem.MolToSmiles(mol)
        except Exception:
            pass
        
        # Fallback to the custom conversion.
        return self._custom_sequence_conversion(sequence)
    
    def _custom_sequence_conversion(self, sequence: str) -> str:
        """Handle non-standard sequences with modifications."""
        smiles_parts = []
        seq = self.expand_repeats(sequence)
        
        # Process modifications first.
        for mod, smi in self.modification_map.items():
            if mod in seq:
                seq = seq.replace(mod, "")
                smiles_parts.append(smi)
        
        # Remove any non-letter characters (such as dashes).
        seq = re.sub(r'[^A-Za-z]', '', seq)
        
        # Process amino acids one-by-one.
        for aa in seq:
            if aa in self.aa_map:
                smiles_parts.append(self.aa_map[aa])
            else:
                self.unrecognized_aa_counter[aa] += 1
                self.bad_sequences.add(sequence)
        
        final_smiles = "".join(smiles_parts)
        try:
            mol = Chem.MolFromSmiles(final_smiles)
            final_smiles = Chem.MolToSmiles(mol) if mol else None
            if not final_smiles:
                return None
            final_smiles = Chem.CanonSmiles(final_smiles, useChiral=True)
            return final_smiles
        except Exception:
            return None
    
    def process_dataframe(
        self, 
        df: pd.DataFrame, 
        sequence_col: str = 'sequence',
        output_col: str = 'smiles_sequence'
    ) -> pd.DataFrame:
        """Process a dataframe with peptide sequences."""
        df[output_col] = df[sequence_col].apply(self.sequence_to_smiles)
        return df
    
    def get_bad_sequences_df(self) -> pd.DataFrame:
        """Get a dataframe of problematic sequences."""
        return pd.DataFrame({
            'sequence': list(self.bad_sequences),
            'reason': 'Contains unrecognized components'
        })
    
    def report_validity(self, df: pd.DataFrame, smiles_col: str = 'smiles_sequence') -> Dict:
        """Generate a validity report."""
        valid = df[smiles_col].notna()
        return {
            'valid_percentage': valid.mean() * 100,
            'invalid_count': len(df) - valid.sum(),
            'unrecognized_aa': dict(self.unrecognized_aa_counter)
        }

def convert_sequences(df, sequence_col='sequence', smiles_col='smiles_sequence'):
    """
    Convert peptide sequences in a DataFrame to SMILES and return the modified DataFrame.
    Designed to be used with .pipe().
    """
    converter = PeptideToSmilesConverter()
    df[smiles_col] = df[sequence_col].apply(converter.sequence_to_smiles)
    return df

def draw_sampled_smiles(df: pd.DataFrame, smiles_col='smiles_sequence', sequence_col='sequence', N=5, seed=42):
    """
    Selects the first N SMILES from the given DataFrame, draws them individually, 
    and prints their corresponding sequences in high quality.
    
    Args:
        df (pd.DataFrame): DataFrame containing SMILES and sequences.
        smiles_col (str): Column name containing SMILES.
        sequence_col (str): Column name containing peptide sequences.
        N (int): Number of samples to draw (default is 5).
    """
    if df.empty or smiles_col not in df.columns or sequence_col not in df.columns:
        print("Invalid DataFrame or missing columns.")
        return
    
    # Select the first N rows (or fewer if df is smaller)
    sample_df = df.sample(n=min(N, len(df)), random_state=seed)
    
    # Process each row separately
    for _, row in sample_df.iterrows():
        sequence = row[sequence_col]
        smiles = row[smiles_col]
        
        if not smiles:
            continue
        
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue
        
        # Print sequence separately
        print(f"Sequence: {sequence}\n")
        
        # Draw and display in high resolution
        img = Draw.MolToImage(mol, size=(500, 500), dpi=300)
        display(img)

# Example usage:
# draw_sampled_smiles(with_smiles_df, N=5)
