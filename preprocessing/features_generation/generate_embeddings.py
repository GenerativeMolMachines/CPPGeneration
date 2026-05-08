import numpy as np
import pandas as pd
import warnings
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem

def blomap_extra_encode(amino_sequence, shape = None):

    if not isinstance(amino_sequence, str):
        if shape is None:
            raise ValueError("Shape must be provided if amino_sequence is not a string")
        return [0] * shape

    blomap_physicochemical = {
        'A': [-0.57, 0.39, -0.96, -0.61, -0.69, 1, 71, 6.0, 0.806, 0.0, 10.0],
        'R': [-0.40, -0.83, -0.61, 1.26, -0.28, 4, 156, 10.8, 0.000, 52.0, 32.6],
        'N': [-0.70, -0.63, -1.47, 1.02, 1.06, 3, 114, 5.4, 0.448, 3.4, 20.4],
        'D': [-1.62, -0.52, -0.67, 1.02, 1.47, 3, 115, 3.0, 0.417, 49.7, 18.3],
        'C': [0.07, 2.04, 0.65, -1.13, -0.39, 2, 103, 5.0, 0.721, 1.5, 3.8],
        'Q': [-0.05, -1.50, -0.67, 0.49, 0.21, 4, 128, 5.7, 0.430, 3.5, 24.7],
        'E': [-0.64, -1.59, -0.39, 0.69, 1.04, 4, 129, 3.2, 0.458, 49.9, 15.8],
        'G': [-0.90, 0.87, -0.36, 1.08, 1.95, 0, 57, 6.0, 0.770, 0.0, 7.6],
        'H': [0.73, -0.67, -0.42, 1.13, 0.99, 3, 137, 7.6, 0.548, 51.6, 12.3],
        'I': [0.59, 0.79, 1.44, -1.90, -0.93, 3, 113, 6.0, 1.000, 0.2, 7.0],
        'L': [0.65, 0.84, 1.25, -0.99, -1.90, 3, 113, 6.0, 0.918, 0.1, 7.7],
        'K': [-0.64, -1.19, -0.65, 0.68, -0.13, 4, 128, 9.7, 0.263, 49.5, 34.3],
        'M': [0.76, 0.05, 0.06, -0.62, -1.59, 4, 131, 5.7, 0.811, 1.4, 7.2],
        'F': [1.87, 1.04, 1.28, -0.61, -0.16, 3, 147, 5.5, 0.951, 0.4, 8.1],
        'P': [-1.82, -0.63, 0.32, 0.03, 0.68, 1, 97, 6.3, 0.678, 1.6, 12.8],
        'S': [-0.39, -0.27, -1.51, -0.25, 0.31, 2, 87, 5.7, 0.601, 1.7, 12.9],
        'T': [-0.04, -0.30, -0.82, -1.02, -0.04, 2, 101, 5.6, 0.634, 1.6, 15.1],
        'W': [1.38, 1.69, 1.91, 1.07, -0.05, 3, 186, 5.9, 0.854, 2.1, 10.3],
        'Y': [1.75, 0.11, 0.65, 0.21, -0.41, 3, 163, 5.7, 0.714, 1.6, 18.3],
        'V': [-0.02, 0.30, 0.97, -1.55, -1.16, 2, 99, 6.0, 0.923, 0.1, 7.2],
        'X': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # unknown amino acids
        'Z': [-0.345, -1.545, -0.53, 0.59, 0.625, 4.0, 128.5, 4.45, 0.444, 26.7, 20.25],  # Glutamic acid or Glutamine,
        'U': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Selenocysteine
        'O': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Pyrrolysine
        'B': [-1.16, -0.575, -1.07, 1.02, 1.265, 3.0, 114.5, 4.2, 0.4325, 26.55, 19.35],  # Aspartic acid or Asparagine
        'J': [0.62, 0.815, 1.345, -1.445, -1.415, 3.0, 113.0, 6.0, 0.959, 0.15, 7.35]  # Leucine or Isoleucine
    }

    translated_sequence = list()
    for amino_acid in amino_sequence:
        if amino_acid in ['J', 'B', 'O', 'U', 'Z']:
            warnings.warn("non-standard amino acid in seqeunce", Warning)
        translated_sequence.extend(blomap_physicochemical[amino_acid.upper()])

    return translated_sequence

def add_padding(string, length):
    while len(string) < length:
        string += 'X'
    return string

def get_morgan_fingerprint(smiles, radius=2, nBits=1024):
    """Generate a Morgan fingerprint from a SMILES string."""
    try:
        molecule = Chem.MolFromSmiles(smiles)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=nBits)
        return np.array(fingerprint)
    except Exception:
        global counter
        counter += 1
        return np.zeros(nBits)  # Return a zero array of the desired shape if the SMILES is invalid

# Blomap-related functions
def add_padding(sequence: str, max_length: int, pad_char: str = 'X') -> str:
    """Pad sequence to specified length with given padding character"""
    return sequence + pad_char * (max_length - len(sequence)) if isinstance(sequence, str) else sequence

def calculate_max_sequence_length(df: pd.DataFrame, seq_col: str = 'sequence') -> int:
    """Calculate maximum sequence length in dataframe"""
    return df[seq_col].apply(lambda x: len(x) if isinstance(x, str) else 0).max()

def generate_blomap_embeddings(
    df: pd.DataFrame, 
    seq_col: str = 'standard_sequence',
    blomap_encoder: callable = blomap_extra_encode
) -> np.ndarray:
    """Generate Blomap embeddings for sequences"""
    max_len = calculate_max_sequence_length(df, seq_col)
    df['adjusted_sequence'] = df[seq_col].apply(
        lambda s: add_padding(s, max_len) if isinstance(s, str) else s
    )
    
    embeddings = np.array([
        blomap_encoder(seq) if isinstance(seq, str) else np.nan 
        for seq in df['adjusted_sequence']
    ])
    
    return embeddings[~np.isnan(embeddings).any(axis=1)]

def protbert_preprocess(seq: str) -> str:
    # 1) uppercase, 2) replace non-standard letters, 3) add spaces
    seq = seq.upper().replace('U', 'X').replace('O', 'X') \
                     .replace('B', 'X').replace('Z', 'X') \
                     .replace('*', '')                  # remove stop codon
    return ' '.join(list(seq))


# ProtBERT-related functions
class ProtBERTEmbedder:
    def __init__(self, model_name: str = "Rostlab/prot_bert_bfd"):
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model      = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _tokenise(self, seq_batch):
        seq_batch = [protbert_preprocess(s) for s in seq_batch]
        tok = self.tokenizer(
                seq_batch,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors='pt')
        return {k: v.to(self.device) for k, v in tok.items()}

    @torch.no_grad()
    def embed_batch(self, sequences, batch_size=32):
        out = []
        for i in tqdm(range(0, len(sequences), batch_size), desc='ProtBERT'):
            tok = self._tokenise(sequences[i:i + batch_size])
            emb = self.model(**tok).last_hidden_state.mean(dim=1)
            out.append(emb.cpu().numpy())
        return np.vstack(out)


# Data processing functions
def process_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Add sequence length and padding columns"""
    df['seq_length'] = df['sequence'].apply(lambda s: len(s) if isinstance(s, str) else 0)
    max_len = df['seq_length'].max()
    df['adjusted_sequence'] = df['standard_sequence'].apply(
        lambda s: add_padding(s, max_len) if isinstance(s, str) else s
    )
    return df

if __name__ == "__main__":

    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Generate protein sequence embeddings')
    parser.add_argument('csv_path', type=str, help='Path to input CSV file')
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv_path)
    GLOBAL_LEN = len(df)
    print("Number of sequences: ", GLOBAL_LEN)
    processed_df = process_sequences(df)

    print("Generating Blomap embeddings...")

    SEQ_LEN = calculate_max_sequence_length(processed_df, 'adjusted_sequence')
    df['adjusted_sequence'] = df['standard_sequence'].apply(lambda s: add_padding(s, SEQ_LEN) if isinstance(s, str) else s)

    # Get a non-NaN sequence
    non_nan_sequence = df['adjusted_sequence'].dropna().iloc[0]

    # Get the Blomap embedding for the sequence
    embedding = blomap_extra_encode(non_nan_sequence)

    # Get the length of the embedding
    SHAPE = len(embedding)
    print("Shape of Blomap embedding: ", SHAPE)

    blomap_embeddings = list(df['adjusted_sequence'].apply(blomap_extra_encode, args=(SHAPE,)))
    blomap_embeddings = np.array(blomap_embeddings)

    assert blomap_embeddings.shape[0] == GLOBAL_LEN

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(SCRIPT_DIR, 'blomap_embeddings.npy')
    np.save(save_path, blomap_embeddings)
    print(f"Saved Blomap embeddings with shape {blomap_embeddings.shape} to:\n{save_path}")

    print("Generating ProtBERT embeddings...")

    protbert_embedder = ProtBERTEmbedder()

    # Use original sequences (unpadded) and handle NaNs
    sequences = df['standard_sequence'].fillna('').tolist()  # Fill NaN with empty strings

    # Generate ProtBERT embeddings
    protbert_embeddings = protbert_embedder.embed_batch(sequences, batch_size=32)
    assert protbert_embeddings.shape[0] == GLOBAL_LEN

    save_path = os.path.join(SCRIPT_DIR, 'protbert_embeddings.npy')
    np.save(save_path, protbert_embeddings)
    print(f"Saved ProtBERT embeddings with shape {protbert_embeddings.shape} to:\n{save_path}")

    # Generate Morgan fingerprints

    print("Generating Morgan fingerprints...")
    counter = 0
    fingerprints = []
    for smiles in tqdm(df['smiles_sequence']):
        fingerprint = get_morgan_fingerprint(smiles)
        fingerprints.append(fingerprint)
    fingerprints_array = np.array(fingerprints)

    assert fingerprints_array.shape[0] == GLOBAL_LEN

    save_path = os.path.join(SCRIPT_DIR, 'morgan_fingerprints.npy')
    np.save(save_path, fingerprints_array)
    print(f"Saved Morgan fingerprints with shape {fingerprints_array.shape} to:\n{save_path}")

    print("All representations have been generated successfully!")

    # Print all the shapes of the generated embeddings
    print("All done!")
    print(f"Blomap embeddings shape: {blomap_embeddings.shape}")
    print(f"ProtBERT embeddings shape: {protbert_embeddings.shape}")
    print(f"Morgan fingerprints shape: {fingerprints_array.shape}")