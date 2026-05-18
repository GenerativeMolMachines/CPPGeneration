# General imports
import os
import json
import math
import re
import csv
import gc # For GPU garbage collection
from itertools import combinations
from multiprocessing import Pool, cpu_count, freeze_support
import argparse
import traceback 

# Data handling and numerical operations
import pandas as pd
import numpy as np

# Plotting
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch and Hugging Face Transformers
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

# SciKit-learn for MMD kernel
from sklearn.metrics.pairwise import rbf_kernel

# Scipy for distance calculations
from scipy.spatial.distance import pdist

# Progress bar
from tqdm import tqdm

import joblib
from pathlib import Path
import sys
from pandarallel import pandarallel

try:
    from aligment import calculate_alignment_similarity 
    from classificator_utils import (
        categorize_sequences,
        add_sequence_features,
        validate_sequences,
        add_descriptors_features
    )
    from regressor_utils import (PeptideRegressorPreprocessor, 
        predict_raw_efficiency)
except ImportError as e:
    print(f"[ERROR] It was not possible to import the necessary utilities: {e}")
    print("Make sure that the files 'aligment.py', 'classificator_utils.py' and 'regressor_utils.py' are located in the same directory.")
    sys.exit(1)

# --- Configuration ---
# Placeholder for model version name - easily changeable for each run
MODEL_VERSION_NAME = "my_finetuned_protgpt2_v1" 

# Paths to input files
GENERATED_SEQUENCES_FILE = "log_generation_cpp_2.txt" # File with generated sequences
NATURAL_PEPTIDES_FILE = "cpps_for_generator.csv"     # File with natural peptides

# Path to fine-tuned ProtGPT2 model
PROTGPT2_MODEL_PATH = ""

# ProtBERT model parameters for MMD
PROTBERT_MODEL_NAME = "Rostlab/prot_bert"
PROTBERT_MAX_SEQ_LEN = 200
PROTBERT_BATCH_SIZE = 16
MMD_RBF_GAMMA_FIXED = 0.0864 # Fixed gamma for MMD RBF kernel
MAX_NATURAL_PEPTIDES_SAMPLES = 500 # Max samples for natural peptides in MMD calculation

# Column name for sequences in CSVs
SEQUENCE_COLUMN = "sequence"

# Output directory base name
METRICS_OUTPUT_DIR_PREFIX = "metrics_"
ALL_METRICS_CSV = "all_metrics.csv" # Consolidated metrics table

CLASSIFIER_ARTIFACTS_PATH = Path(__file__).resolve().parent / 'classifier_artifacts_active_learning.joblib'
REGRESSOR_ARTIFACTS_PATH = Path(__file__).resolve().parent / 'peptide_regressor_artifacts.joblib'

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Functions for Loss and Perplexity ---
def is_valid_fasta_format(seq_content):
    """
    Checks if the given sequence content (without <|endoftext|> tokens) 
    adheres to the expected FASTA-like format (60 characters per line, except last).
    """
    lines = seq_content.strip().split('\n')
    if not lines or (len(lines) == 1 and not lines[0]): # Handle empty or just whitespace content
        return False
    
    # Check all lines except the last one
    for line in lines[:-1]:
        if len(line) != 60:
            return False
    
    # Check the last line
    if len(lines[-1]) == 0 or len(lines[-1]) > 60:
        return False
    return True

def format_fasta_for_protgpt2(clean_seq):
    """
    Formats a clean peptide sequence (without internal newlines) into 
    the ProtGPT2 input format (60-char lines, with <|endoftext|> tokens).
    """
    # Remove any existing newlines in the input sequence to ensure 60-char splitting
    seq_cleaned = clean_seq.replace('\n', '')
    lines = [seq_cleaned[i:i+60] for i in range(0, len(seq_cleaned), 60)]
    return "<|endoftext|>\n" + '\n'.join(lines) + "\n<|endoftext|>"

def calculate_loss_and_perplexity_for_sequence(sequence_text_for_model, model, tokenizer):
    """
    Calculates loss and perplexity for a single sequence formatted for the model.
    """
    input_ids = tokenizer.encode(sequence_text_for_model, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
    perplexity = math.exp(loss)
    return loss, perplexity

def process_generated_sequences_for_lp(generated_sequences_file, protgpt2_model_path, output_dir):
    """
    Parses generated sequences, calculates loss and perplexity, and saves results to CSV.
    
    Returns: avg_loss, avg_perplexity, path_to_sequences_with_metrics_csv
    """
    print("\n--- Step 1: Calculating Loss and Perplexity ---")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(protgpt2_model_path)
        model = AutoModelForCausalLM.from_pretrained(protgpt2_model_path)
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Error loading ProtGPT2 model from {protgpt2_model_path}: {e}")
        return None, None, None

    sequences_to_process = []
    print(f"Loading sequences from {generated_sequences_file}...")
    try:
        with open(generated_sequences_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line.replace("'", '"'))  
                    seq_raw = obj["generated_text"]
                    
                    # Extract the clean peptide sequence without any <|endoftext|> tokens
                    # and remove any leading/trailing newlines or spaces.
                    clean_peptide_content = seq_raw.replace("<|endoftext|>\n", "").replace("\n<|endoftext|>", "").strip()
                    
                    if is_valid_fasta_format(clean_peptide_content):
                        # The sequence stored in CSV should be flat (no internal newlines)
                        clean_seq_flat = clean_peptide_content.replace('\n', '')
                        # The sequence for model input needs to be correctly formatted with newlines and tokens
                        formatted_seq_for_model = format_fasta_for_protgpt2(clean_peptide_content)
                        sequences_to_process.append((clean_seq_flat, formatted_seq_for_model)) 
                except json.JSONDecodeError as jde:
                    print(f"JSON decoding error for line: {line[:100]}... Error: {jde}")
                except Exception as ex:
                    print(f"Error parsing line for LP calculation: {line[:100]}... Error: {ex}")
    except FileNotFoundError:
        print(f"Error: Generated sequences file not found at {generated_sequences_file}")
        return None, None, None
    except Exception as e:
        print(f"Error reading generated sequences file: {e}")
        return None, None, None

    print(f"Amount of valid sequences found for Loss/Perplexity calculation: {len(sequences_to_process)}")

    data = []
    if not sequences_to_process:
        print("No valid sequences to calculate Loss/Perplexity. Skipping.")
        return None, None, os.path.join(output_dir, "sequences_with_metrics.csv")

    for clean_seq_flat, formatted_seq_for_model in tqdm(sequences_to_process, desc="Calculating Loss & Perplexity"):
        try:
            loss, ppl = calculate_loss_and_perplexity_for_sequence(formatted_seq_for_model, model, tokenizer)
            data.append({
                SEQUENCE_COLUMN: clean_seq_flat, 
                "loss": loss,
                "perplexity": ppl
            })
        except Exception as ex:
            print(f"Error calculating metrics for sequence: {clean_seq_flat[:50]}... Error: {ex}")

    df = pd.DataFrame(data)
    df_sorted = df.sort_values(by='perplexity')
    
    output_csv_path = os.path.join(output_dir, "sequences_with_metrics.csv")
    try:
        df_sorted.to_csv(output_csv_path, index=False)
        print(f"Loss and Perplexity results saved to {output_csv_path}")
    except Exception as e:
        print(f"Error saving sequences with metrics CSV: {e}")
        return None, None, None
    
    if not df.empty:
        avg_loss = df['loss'].mean()
        avg_perplexity = df['perplexity'].mean()
        print(f"Average Loss: {avg_loss:.4f}, Average Perplexity: {avg_perplexity:.4f}")
        return avg_loss, avg_perplexity, output_csv_path
    else:
        print("No metrics calculated due to errors or no valid sequences.")
        return None, None, output_csv_path

def process_soft_prompt_csv_for_lp(generated_csv_file, soft_prompt_model_path, base_model_name, output_dir):
    """
    Alternative step 1: processes a CSV from a soft-prompt model,
    correctly computes loss/perplexity accounting for soft prompts,
    and saves the result.
    """
    print("\n--- Step 1 (Soft-Prompt Mode): Calculating Loss and Perplexity ---")
    
    try:
        print(f"Loading base model: {base_model_name}")
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(DEVICE).eval()
        
        print(f"Loading soft-prompt model artifacts from: {soft_prompt_model_path}")
        artifacts = torch.load(soft_prompt_model_path, map_location=DEVICE)
        
        class ConditionPrefixGenerator(nn.Module): # Need to copy the class from the training script
            def __init__(self, cond_dim, prefix_len, embed_dim, hidden=512, dropout=0.0):
                super().__init__()
                self.prefix_len, self.embed_dim = prefix_len, embed_dim
                self.mlp = nn.Sequential(nn.Linear(cond_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, prefix_len * embed_dim))
                self.layernorm = nn.LayerNorm(embed_dim)
            def forward(self, cond):
                x = self.mlp(cond).view(-1, self.prefix_len, self.embed_dim)
                return self.layernorm(x)

        prefix_generator = ConditionPrefixGenerator(
            cond_dim=artifacts['condition_dim'], prefix_len=artifacts['prefix_len'],
            embed_dim=base_model.config.hidden_size, hidden=512
        ).to(DEVICE).eval()
        
        prefix_generator.load_state_dict(artifacts['state_dict'])
        cell_line_mapping = artifacts['cell_line_mapping']
        condition_dim = artifacts['condition_dim']
    except Exception as e:
        print(f"Error loading models or artifacts: {e}")
        return None, None, None
        
    try:
        df_gen = pd.read_csv(generated_csv_file)
    except FileNotFoundError:
        print(f"Error: Generated CSV file not found at {generated_csv_file}")
        return None, None, None
    
    data_for_df = []
    print(f"Calculating metrics for {len(df_gen)} sequences...")
    for _, row in tqdm(df_gen.iterrows(), total=len(df_gen), desc="Calculating Loss & Perplexity (Soft-Prompt)"):
        seq_raw = row['sequence']
        cell_line = row['cell_line']
        
        try:
            cond_vector = [0.0] * condition_dim
            cond_vector[0] = 1.0 # is_cpp
            idx = cell_line_mapping[cell_line]
            cond_vector[1 + idx] = 1.0
            cond_tensor = torch.tensor(cond_vector, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                prefix_emb = prefix_generator(cond_tensor)
            
            input_ids = base_tokenizer.encode(seq_raw, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                input_embeds = base_model.get_input_embeddings()(input_ids)
            
            inputs_embeds = torch.cat([prefix_emb, input_embeds], dim=1)
            labels = torch.cat([torch.full((1, prefix_generator.prefix_len), -100, dtype=torch.long).to(DEVICE), input_ids], dim=1)

            with torch.no_grad():
                outputs = base_model(inputs_embeds=inputs_embeds, labels=labels)
                loss = outputs.loss.item()
            
            perplexity = math.exp(loss)
            clean_seq_flat = seq_raw.replace("<|endoftext|>", "").replace('\n', '').strip()

            # IMPORTANT: Add cell_line to the intermediate file
            data_for_df.append({
                SEQUENCE_COLUMN: clean_seq_flat, 
                "loss": loss, "perplexity": perplexity,
                "cell_line": cell_line
            })
        except Exception as e:
            print(f"Skipping sequence due to error: {seq_raw[:50]}... Error: {e}")

    df = pd.DataFrame(data_for_df)
    df_sorted = df.sort_values(by='perplexity')
    
    output_csv_path = os.path.join(output_dir, "sequences_with_metrics.csv")
    df_sorted.to_csv(output_csv_path, index=False)
    print(f"Loss and Perplexity results saved to {output_csv_path}")

    avg_loss = df['loss'].mean()
    avg_perplexity = df['perplexity'].mean()
    print(f"Average Loss: {avg_loss:.4f}, Average Perplexity: {avg_perplexity:.4f}")
    
    return avg_loss, avg_perplexity, output_csv_path

def process_prefix_tuned_csv_for_lp(generated_csv_file, prefix_model_artifacts_path, base_model_name, output_dir):
    """
    Alternative step 1: processes a CSV from a prefix-tuned model,
    correctly computes loss/perplexity using past_key_values,
    and saves the result.
    """
    print("\n--- Step 1 (Prefix-Tune Mode): Calculating Loss and Perplexity ---")
    
    try:
        # 1. Load the base model and tokenizer
        print(f"Loading base model: {base_model_name}")
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(DEVICE).eval()
        
        # 2. Load prefix-generator artifacts
        print(f"Loading prefix-tuned model artifacts from: {prefix_model_artifacts_path}")
        artifacts = torch.load(prefix_model_artifacts_path, map_location=DEVICE)
        
        # Define the KVPrefixGenerator class (must match the one used during training)
        class KVPrefixGenerator(nn.Module):
            """
            Condition -> past_key_values generator.
            """
            def __init__(self, cond_dim, num_layers, num_heads, prefix_len, head_dim, hidden=1024):
                super().__init__()
                self.cond_dim = cond_dim
                self.num_layers = num_layers
                self.num_heads = num_heads
                self.prefix_len = prefix_len
                self.head_dim = head_dim
                self.out_dim = num_layers * 2 * num_heads * prefix_len * head_dim
                self.mlp = nn.Sequential(
                    nn.Linear(cond_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, self.out_dim)
                )

            def forward(self, cond):
                batch = cond.size(0)
                x = self.mlp(cond)
                x = x.view(batch, self.num_layers, 2, self.num_heads, self.prefix_len, self.head_dim)
                past_key_values = []
                for layer_idx in range(self.num_layers):
                    key = x[:, layer_idx, 0, :, :, :].contiguous()
                    value = x[:, layer_idx, 1, :, :, :].contiguous()
                    past_key_values.append((key, value))
                return tuple(past_key_values)

        # 3. Initialize and load weights into KVPrefixGenerator
        model_config = artifacts['model_config']
        condition_dim = artifacts['condition_dim'] 

        kv_generator = KVPrefixGenerator(
            cond_dim=condition_dim,  # <-- Use variable
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            prefix_len=model_config['prefix_len'],
            head_dim=model_config['head_dim']
        ).to(DEVICE).eval()

        kv_generator.load_state_dict(artifacts['state_dict'])
        cell_line_mapping = artifacts['cell_line_mapping']
        
    except Exception as e:
        print(f"Error loading models or artifacts: {e}")
        traceback.print_exc() # For detailed debugging
        return None, None, None
        
    try:
        # 4. Read generated data
        df_gen = pd.read_csv(generated_csv_file)
    except FileNotFoundError:
        print(f"Error: Generated CSV file not found at {generated_csv_file}")
        return None, None, None
    
    data_for_df = []
    print(f"Calculating metrics for {len(df_gen)} sequences...")

    # 5. Loop over sequences to calculate loss
    for _, row in tqdm(df_gen.iterrows(), total=len(df_gen), desc="Calculating Loss & Perplexity (Prefix-Tune)"):
        seq_raw = row['sequence']
        cell_line = row['cell_line']
        
        try:
            # 5.1. Create the condition vector
            cond_vector = [0.0] * condition_dim
            cond_vector[0] = 1.0 # is_cpp
            if cell_line in cell_line_mapping:
                idx = cell_line_mapping[cell_line]
                cond_vector[1 + idx] = 1.0
            cond_tensor = torch.tensor(cond_vector, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            # 5.2. Generate past_key_values using the KV generator
            with torch.no_grad():
                prefix_past = kv_generator(cond_tensor)
            
            # 5.3. Tokenize the sequence. For loss we need both input_ids and labels
            input_ids = base_tokenizer.encode(seq_raw, return_tensors="pt").to(DEVICE)
            labels = input_ids.clone() # For CausalLM, labels are the same as input_ids

            # 5.4. Call the model with both input_ids and past_key_values
            with torch.no_grad():
                # IMPORTANT: We pass `past_key_values`. The model will understand that
                # this is a cache for the first `prefix_len` positions and start computation from there.
                outputs = base_model(input_ids=input_ids, past_key_values=prefix_past, labels=labels)
                loss = outputs.loss.item()
            
            perplexity = math.exp(loss)
            clean_seq_flat = seq_raw.replace("<|endoftext|>", "").replace('\n', '').strip()

            data_for_df.append({
                SEQUENCE_COLUMN: clean_seq_flat, 
                "loss": loss, "perplexity": perplexity,
                "cell_line": cell_line
            })
        except Exception as e:
            print(f"Skipping sequence due to error: {seq_raw[:50]}... Error: {e}")

    df = pd.DataFrame(data_for_df)
    df_sorted = df.sort_values(by='perplexity')
    
    output_csv_path = os.path.join(output_dir, "sequences_with_metrics.csv")
    df_sorted.to_csv(output_csv_path, index=False)
    print(f"Loss and Perplexity results saved to {output_csv_path}")

    avg_loss = df['loss'].mean()
    avg_perplexity = df['perplexity'].mean()
    print(f"Average Loss: {avg_loss:.4f}, Average Perplexity: {avg_perplexity:.4f}")
    
    return avg_loss, avg_perplexity, output_csv_path

# --- Functions for CPP classifier ---

def load_classifier_artifacts(path: Path) -> dict:
    """Loads classifier artifacts from the joblib file."""
    print(f"Loading classifier artifacts from a file: {path}")
    if not path.exists():
        print(f"Error: The classifier artifact file was not found on the way.: {path}")
        return None
    
    artifacts = joblib.load(path)
    print("    - The classifier model, threshold, and LabelEncoder have been successfully loaded.")
    return artifacts

def classifier_preprocessing_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    """The full preprocessing pipeline for the classifier."""
    print("Loading the preprocessing pipeline for the classifier...")
    
    # Generate features from sequences
    processed_df = (raw_df.copy()
                    .pipe(categorize_sequences)
                    .pipe(add_sequence_features)
                    .pipe(validate_sequences)
                    .pipe(lambda df: df.dropna(subset=['standard_sequence']))
                    .pipe(add_descriptors_features))

    # Convert boolean columns to int
    bool_cols = processed_df.columns[processed_df.dtypes == 'bool'].tolist()
    for col in bool_cols:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].astype(int)
    
    print("Preprocessing for the classifier is complete.")
    return processed_df

def predict_cpp_labels(data: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Returns a DataFrame with probabilities and predicted CPP labels."""
    model = artifacts['model']
    threshold = artifacts['threshold']
    feature_names = model.feature_names_in_
    
    missing_features = set(feature_names) - set(data.columns)
    if missing_features:
        print(f"Error: The data for the classifier lacks the necessary features: {missing_features}")
        # Return an empty DataFrame so as not to interrupt the entire pipeline.
        return pd.DataFrame(columns=['probability_cpp', 'prediction_cpp'])
        
    X_new = data[feature_names]
    
    print("Getting CPP classifier predictions...")
    probas = model.predict_proba(X_new)[:, 1]
    prediction_cpp = (probas >= threshold).astype(int)
    
    results_df = pd.DataFrame({
        'probability_cpp': probas,
        'prediction_cpp': prediction_cpp,
    }, index=data.index)
    
    print("CPP classifier predictions are ready.")
    return results_df


def run_cpp_classification(metrics_csv_path: str) -> float:
    """
    A new pipeline step: starts classification and updates files.
    Returns the percentage of CPPs generated.
    """
    print("\n--- Step X: Predicting CPP classifier ---")
    
    # 1. Load classifier artifacts
    classifier_artifacts = load_classifier_artifacts(CLASSIFIER_ARTIFACTS_PATH)
    if not classifier_artifacts:
        print("Classifier artifacts could not be loaded. Skipping a step.")
        return None

    # 2. Read generated sequences from the intermediate file
    try:
        generated_df = pd.read_csv(metrics_csv_path)
        if generated_df.empty:
            print("The file of generated sequences is empty. Skipping a step.")
            return None
    except FileNotFoundError:
        print(f"File {metrics_csv_path}  not found. Skipping the classification step.")
        return None

    # 3. Preprocess data for the classifier
    preprocessed_df = classifier_preprocessing_pipeline(generated_df)
    
    # 4. Get predictions
    predictions_df = predict_cpp_labels(preprocessed_df, classifier_artifacts)
    
    if predictions_df.empty:
        print("Couldn't get predictions. Skipping a step.")
        return None
    
    # 5. Merge predictions with the main DataFrame
    # Use .join() on the index because it is preserved
    updated_df = generated_df.join(predictions_df)
    
    # 6. Save the updated file, overwriting the old one
    try:
        updated_df.to_csv(metrics_csv_path, index=False)
        print(f"The {metrics_csv_path} file has been updated with CPP predictions.")
    except Exception as e:
        print(f"Error saving the updated CSV file: {e}")

    # 7. Calculate percentage of predicted CPP
    if 'prediction_cpp' in updated_df.columns:
        percent_cpp = updated_df['prediction_cpp'].mean() * 100
        print(f"Percentage of generated sequences marked as CPP: {percent_cpp:.2f}%")
        return percent_cpp
    else:
        return None

def run_efficiency_prediction(metrics_csv_path: str) -> float:
    """
    Executes raw_efficiency prediction for CPP peptides and updates files.
    Returns the mean predicted efficiency.
    """
    print("\n--- Step Y: Predicting Raw Efficiency for CPPs ---")
    
    # 1. Load artifacts
    regressor_artifacts = joblib.load(REGRESSOR_ARTIFACTS_PATH)

    # 2. Read data
    try:
        metrics_df = pd.read_csv(metrics_csv_path)
    except FileNotFoundError:
        return None

    # 3. Filter data
    cpp_df = metrics_df[metrics_df['prediction_cpp'] == 1].copy()
    if cpp_df.empty:
        print("No peptides labeled as CPP were found. Skipping Raw Efficiency prediction.")
        # Initialize empty columns before saving
        metrics_df['cell_line'] = 'HeLa cells'
        metrics_df['prediction_raw_efficiency'] = np.nan
        metrics_df.to_csv(metrics_csv_path, index=False)
        return None
        
    if 'cell_line' in cpp_df.columns:
        df_to_predict = cpp_df[cpp_df['cell_line'] != 'other'].copy()
        print(f"Found {len(cpp_df)} target peptides. Excluding {len(cpp_df) - len(df_to_predict)} peptides with cell_line='other'.")
    else:
        print("Warning: 'cell_line' column not found. Predicting efficiency for all target peptides.")
        df_to_predict = cpp_df

    if df_to_predict.empty:
        print("No peptides left to predict efficiency after filtering the 'other'")
        # Save file with empty values for consistency
        if 'prediction_raw_efficiency' not in metrics_df.columns:
            metrics_df['prediction_raw_efficiency'] = np.nan
        metrics_df.to_csv(metrics_csv_path, index=False)
        return None

    # 4–7. Safe preprocessing and prediction, skipping problematic sequences
    import warnings

    regressor_preprocessor = PeptideRegressorPreprocessor()

    problem_cases = []   # collect problematic sequences here
    batch_ok = True

    try:
        # Try batch processing (faster). Convert RuntimeWarnings to exceptions.
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)
            old_np = np.seterr(all='raise')
            try:
                preprocessed_df = regressor_preprocessor.transform(df_to_predict)
            finally:
                np.seterr(**old_np)
    except Exception as e:
        # Fall back to item-by-item processing
        batch_ok = False
        pre_list = []
        ok_indices = []

        for idx, row in df_to_predict.iterrows():
            row_df = row.to_frame().T
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", category=RuntimeWarning)
                    old_np = np.seterr(all='raise')
                    try:
                        pre = regressor_preprocessor.transform(row_df)
                    finally:
                        np.seterr(**old_np)
                pre_list.append(pre)
                ok_indices.append(idx)
            except Exception as ex:
                problem_cases.append({
                    "index": idx,
                    "sequence": str(row.get("sequence", "")),
                    "reason": str(ex)
                })

        if pre_list:
            preprocessed_df = pd.concat(pre_list, ignore_index=True)

    # 5. Make predictions
    if batch_ok:
        efficiency_predictions = predict_raw_efficiency(preprocessed_df, regressor_artifacts)
        df_to_predict['prediction_raw_efficiency'] = efficiency_predictions
    else:
        # Initialize NaN for all, then fill successful cases
        df_to_predict['prediction_raw_efficiency'] = np.nan
        if 'preprocessed_df' in locals():
            efficiency_predictions = predict_raw_efficiency(preprocessed_df, regressor_artifacts)
            df_to_predict.loc[ok_indices, 'prediction_raw_efficiency'] = efficiency_predictions

    # 6–7. Update and merge results
    df_to_predict['prediction_raw_efficiency'] = efficiency_predictions
    efficiency_series = df_to_predict['prediction_raw_efficiency']
    if 'prediction_raw_efficiency' not in metrics_df.columns:
        metrics_df['prediction_raw_efficiency'] = np.nan
    metrics_df.loc[efficiency_series.index, 'prediction_raw_efficiency'] = efficiency_series
    updated_metrics_df = metrics_df

    # update_data = df_to_predict[['sequence', 'cell_line', 'prediction_raw_efficiency']]
    # updated_metrics_df = pd.merge(metrics_df, update_data, on=['sequence', 'cell_line'], how='left')

    # Output problematic sequences (after calculating for the rest)
    if problem_cases:
        print("\n[DEBUG] Problematic sequences during descriptor calculation (first 120 chars shown):")
        for i, item in enumerate(problem_cases, 1):
            seq = item["sequence"]
            print(f"{i}) idx={item['index']}, len={len(seq)}: '{seq[:120]}' ... | reason={item['reason']}")
    # 8. Save the updated file
    output_dir = os.path.dirname(metrics_csv_path)
    output_csv_path = os.path.join(output_dir, "sequences_with_metrics.csv")

    try:
        updated_metrics_df.to_csv(output_csv_path, index=False)
        print(f"File {output_csv_path} has been updated with raw_efficiency predictions.")
    except Exception as e:
        print(f"Error saving the updated CSV file: {e}")

    # 9. Calculate the mean value (using the updated DataFrame)
    if not updated_metrics_df['prediction_raw_efficiency'].isnull().all():
        mean_efficiency = updated_metrics_df['prediction_raw_efficiency'].mean()
        evaluated_count = updated_metrics_df['prediction_raw_efficiency'].notna().sum()
        print(f"Average predicted raw_efficiency ({evaluated_count} peptides): {mean_efficiency:.4f}")
        return mean_efficiency
    else:
        return None

# --- Functions for Similarity ---
def work_on_pair_sim(args):
    """Helper function for multiprocessing similarity calculation."""
    seq1, seq2, bin_label = args
    sim = calculate_alignment_similarity(
        sequence1=seq1,
        sequence2=seq2,
        alignment_type="global"
    )
    return bin_label, seq1, seq2, sim

def all_pairs_sim(df, sequence_column):
    """Generator for all unique pairs within each length bin."""
    for bin_label, group in df.groupby("len_bin", observed=False):
        seqs = group[sequence_column].tolist()
        if len(seqs) > 1: # Only yield pairs if there are at least two sequences
            for s1, s2 in combinations(seqs, 2):
                yield (s1, s2, str(bin_label))

def calculate_and_save_similarity(metrics_csv_path, output_dir):
    """
    Calculates pairwise similarity for generated peptides and saves results.
    
    Returns: avg_similarity, path_to_similarity_csv
    """
    print("\n--- Step 2: Calculating Similarity ---")
    
    # Support for multiprocessing on Windows
    freeze_support() 

    actual_output_dir = os.path.dirname(metrics_csv_path)
    output_csv_path = os.path.join(actual_output_dir, "sequences_with_metrics_sim.csv")
    
    try:
        df_peptide = pd.read_csv(metrics_csv_path)
    except FileNotFoundError:
        print(f"Error: Input CSV for similarity not found at {metrics_csv_path}. Skipping similarity calculation.")
        return None, None
    except Exception as e:
        print(f"Error reading input CSV for similarity: {e}. Skipping similarity calculation.")
        return None, None
    
    if df_peptide.empty:
        print("Input CSV for similarity is empty. Skipping similarity calculation.")
        return None, output_csv_path

    # If no len_bin column, create a default bin (all sequences in one group)
    if "len_bin" not in df_peptide.columns:
        df_peptide["len_bin"] = 1 

    n_cores = cpu_count()
    print(f"Using {n_cores} CPU cores for similarity calculation.")

    pair_iter = all_pairs_sim(df_peptide, SEQUENCE_COLUMN)
    # Calculate total_pairs by iterating over the generator once (or convert to list if needed for sum)
    # Be careful with very large dataframes as this might be slow or consume memory.
    # A more robust approach would be to calculate it from group sizes.
    total_pairs = sum(
        len(g) * (len(g)-1) // 2
        for _, g in df_peptide.groupby("len_bin", observed=False) if len(g) > 1 # Only groups with >1 sequence
    )
    
    if total_pairs == 0:
        print("Not enough sequences to form pairs for similarity calculation (need at least 2 in a bin). Skipping.")
        return None, output_csv_path

    print(f"Calculating similarity for {total_pairs} pairs...")
    all_similarities = []
    try:
        with Pool(processes=n_cores) as pool, open(output_csv_path, "w", newline="") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(["bin", "sequence_1", "sequence_2", "similarity"])

            # Use chunksize for efficiency with very large iterables
            chunksize = max(1, total_pairs // (n_cores * 10)) 

            for bin_lbl, s1, s2, sim in tqdm(
                pool.imap_unordered(work_on_pair_sim, pair_iter, chunksize=chunksize),
                total=total_pairs,
                desc="Calculating all pairs similarity"
            ):
                writer.writerow([bin_lbl, s1, s2, sim])
                all_similarities.append(sim)
        
        print(f"Similarity results saved to {output_csv_path}")
        
        if all_similarities:
            avg_similarity = np.mean(all_similarities)
            print(f"Average Similarity: {avg_similarity:.2f}%")
            return avg_similarity, output_csv_path
        else:
            print("No similarity values calculated due to errors or insufficient pairs.")
            return None, output_csv_path
    except Exception as e:
        print(f"An error occurred during similarity calculation: {e}")
        return None, None

def visualize_similarity_distribution(similarity_csv_path, output_dir):
    """
    Creates and saves a histogram of similarity distribution.
    """
    print("\n--- Step 3: Visualizing Similarity Distribution ---")
    if not os.path.exists(similarity_csv_path):
        print(f"Similarity CSV not found at {similarity_csv_path}. Skipping visualization.")
        return

    try:
        df_sim = pd.read_csv(similarity_csv_path)
    except Exception as e:
        print(f"Error reading similarity CSV from {similarity_csv_path}: {e}. Skipping visualization.")
        return

    if df_sim.empty or 'similarity' not in df_sim.columns:
        print("No similarity data to plot or 'similarity' column missing. Skipping visualization.")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(df_sim['similarity'], bins=20, kde=True)
    plt.title('Distribution of Pairwise Peptide Similarity')
    plt.xlabel('Similarity Percentage')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    
    plot_path = os.path.join(output_dir, "similarity_distribution.png")
    try:
        plt.savefig(plot_path)
        plt.close() # Close plot to free memory
        print(f"Similarity distribution plot saved to {plot_path}")
    except Exception as e:
        print(f"Error saving similarity distribution plot: {e}")

# --- Functions for MMD ---
def load_peptides_from_csv_mmd(filepath, sequence_column, max_samples=None):
    """
    Loads peptide sequences from a CSV file, cleans them, converts to uppercase.
    Optionally performs random subsampling.
    """
    try:
        df = pd.read_csv(filepath)
        if sequence_column not in df.columns:
            raise ValueError(f"Column '{sequence_column}' not found in file {filepath}")
        
        peptides = df[sequence_column].astype(str).tolist()
        peptides = [seq.strip().upper() for seq in peptides if seq.strip()]
        
        print(f"Loaded {len(peptides)} peptides from {filepath}")

        if max_samples and len(peptides) > max_samples:
            print(f"Number of peptides ({len(peptides)}) exceeds limit ({max_samples}).")
            print(f"Performing random subsampling of {max_samples} peptides.")
            rng = np.random.default_rng(seed=42) # For reproducibility of subsampling
            peptides = rng.choice(peptides, max_samples, replace=False).tolist()
            print(f"After subsampling: {len(peptides)} peptides.")
            
        return peptides
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e:
        print(f"Error loading or processing file {filepath}: {e}")
        return []

def get_protbert_embeddings(sequences, tokenizer, model, max_seq_len, batch_size):
    """
    Obtains embeddings for a list of peptide sequences using ProtBERT.
    """
    if not sequences:
        print("No sequences provided for embedding generation.")
        return np.array([]) # Return empty array if no sequences

    model.eval()
    embeddings = []
    
    # Ensure model is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device for ProtBERT: {device}")

    # Add spaces between amino acids for ProtBERT tokenizer
    processed_sequences = [" ".join(list(seq)) for seq in sequences]

    with torch.no_grad():
        for i in tqdm(range(0, len(processed_sequences), batch_size), desc="Getting embeddings"):
            batch_sequences = processed_sequences[i:i+batch_size]
            
            inputs = tokenizer(batch_sequences,
                               padding=True,
                               truncation=True,
                               max_length=max_seq_len,
                               return_tensors="pt")
            
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            
            # Mean pooling over tokens for sequence embedding
            sequence_embeddings = last_hidden_states.mean(dim=1)
            
            embeddings.append(sequence_embeddings.cpu().numpy())

            # Clear GPU memory
            del inputs, outputs, last_hidden_states, sequence_embeddings
            torch.cuda.empty_cache()
            gc.collect()

    return np.vstack(embeddings)

def calculate_mmd_squared_val(X, Y, gamma=None):
    """
    Calculates Maximum Mean Discrepancy squared (MMD^2) between two datasets X and Y.
    """
    if gamma is None:
        raise ValueError("Parameter 'gamma' cannot be None for MMD RBF kernel.")
    
    if X.shape[0] == 0 or Y.shape[0] == 0:
        print("Warning: One or both datasets for MMD are empty. MMD will be 0.")
        return 0.0

    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)

    n = X.shape[0]
    m = Y.shape[0]

    mmd_sq = (np.sum(K_XX) / (n * n) +
              np.sum(K_YY) / (m * m) -
              2 * np.sum(K_XY) / (n * m))
    
    return mmd_sq

def calculate_mmd(generated_metrics_csv_path, natural_peptides_file, output_dir):
    """
    Calculates MMD between generated and natural peptide embeddings.
    
    Returns: mmd_value
    """
    print("\n--- Step 4: Calculating MMD ---")

    print("Loading generated peptides for MMD...")
    generated_peptides = load_peptides_from_csv_mmd(generated_metrics_csv_path, SEQUENCE_COLUMN)
    print("Loading natural peptides for MMD...")
    natural_peptides = load_peptides_from_csv_mmd(natural_peptides_file, SEQUENCE_COLUMN, MAX_NATURAL_PEPTIDES_SAMPLES)

    if not generated_peptides or not natural_peptides:
        print("Could not load sufficient peptides for MMD calculation. Check file paths and 'sequence' column presence.")
        return None

    try:
        print(f"\nLoading ProtBERT tokenizer and model ({PROTBERT_MODEL_NAME})...")
        protbert_tokenizer = AutoTokenizer.from_pretrained(PROTBERT_MODEL_NAME)
        protbert_model = AutoModel.from_pretrained(PROTBERT_MODEL_NAME)
        print("ProtBERT model loaded.")
    except Exception as e:
        print(f"Error loading ProtBERT model {PROTBERT_MODEL_NAME}: {e}")
        return None

    print("\nGetting embeddings for generated peptides...")
    generated_embeddings = get_protbert_embeddings(generated_peptides, protbert_tokenizer, protbert_model, PROTBERT_MAX_SEQ_LEN, PROTBERT_BATCH_SIZE)
    
    print("\nGetting embeddings for natural peptides...")
    natural_embeddings = get_protbert_embeddings(natural_peptides, protbert_tokenizer, protbert_model, PROTBERT_MAX_SEQ_LEN, PROTBERT_BATCH_SIZE)

    if generated_embeddings.shape[0] == 0 or natural_embeddings.shape[0] == 0:
        print("One or both sets of embeddings are empty. Cannot calculate MMD.")
        return None

    print(f"\nCalculating MMD with RBF kernel (gamma={MMD_RBF_GAMMA_FIXED:.6f})...")
    mmd_val_sq = calculate_mmd_squared_val(natural_embeddings, generated_embeddings, gamma=MMD_RBF_GAMMA_FIXED)
    mmd_val = np.sqrt(max(0, mmd_val_sq)) # MMD is sqrt(MMD^2) and must be non-negative

    print(f"\nMMD^2 between natural and generated peptides: {mmd_val_sq:.6f}")
    print(f"MMD between natural and generated peptides: {mmd_val:.6f}")
    
    return mmd_val

# --- Main Pipeline Orchestrator ---
def run_full_pipeline(model_version_name, 
                      generated_seq_file, 
                      natural_peptides_file, 
                      protgpt2_model_path,
                      input_type,
                      calculate_efficiency,
                      calculate_similarity):
    """
    Executes the full pipeline for calculating various metrics and logging them.
    """
    # Create the output directory for this model version
    output_base_dir = os.path.join(os.getcwd(), f"{METRICS_OUTPUT_DIR_PREFIX}{model_version_name}")
    os.makedirs(output_base_dir, exist_ok=True) 
    print(f"\n--- Starting Pipeline for Model: {model_version_name} ---")
    print(f"All outputs for this run will be saved in: {output_base_dir}")

    # Initialize metrics variables to None
    avg_loss, avg_perplexity, avg_similarity, mmd_value, percent_cpp, mean_raw_efficiency = None, None, None, None, None, None
    generated_metrics_csv_path = None # Path to the intermediate CSV from LP step

    # Step 1: Loss and Perplexity
    try:
        if input_type == "soft_prompt_csv":
            avg_loss, avg_perplexity, generated_metrics_csv_path = process_soft_prompt_csv_for_lp(
                generated_csv_file=generated_seq_file, 
                soft_prompt_model_path=protgpt2_model_path,
                base_model_name="nferruz/ProtGPT2",
                output_dir=output_base_dir
            )
        elif input_type == "prefix_tuned_csv":
            avg_loss, avg_perplexity, generated_metrics_csv_path = process_prefix_tuned_csv_for_lp(
                generated_csv_file=generated_seq_file,
                prefix_model_artifacts_path=protgpt2_model_path, # Here protgpt2_model_path is the path to the .pt file
                base_model_name="nferruz/ProtGPT2", # The base model being used
                output_dir=output_base_dir)
        else: # input_type == "finetuned_txt"
            avg_loss, avg_perplexity, generated_metrics_csv_path = process_generated_sequences_for_lp(
                generated_seq_file, protgpt2_model_path, output_base_dir
            )
    except Exception as e:
        print(f"An unexpected error occurred during Loss & Perplexity calculation: {e}")

    # Step 2: CPP or not prediction

    if generated_metrics_csv_path and os.path.exists(generated_metrics_csv_path):
        try:
            percent_cpp = run_cpp_classification(generated_metrics_csv_path)
        except Exception as e:
            print(f"An unexpected error occurred during CPP classification: {e}")
    else:
        print("No generated metrics CSV found. Skipping CPP classification.")

    # Step 2.5: Raw Efficiency prediction
    if calculate_efficiency:
        print("\nRaw Efficiency prediction is ENABLED.")
        if generated_metrics_csv_path and os.path.exists(generated_metrics_csv_path):
            try:
                mean_raw_efficiency = run_efficiency_prediction(generated_metrics_csv_path)
            except Exception as e:
                print(f"An unexpected error occurred during Raw Efficiency prediction: {e}")
        else:
            print("No generated metrics CSV found. Skipping Raw Efficiency prediction.")
    else:
        print("\nRaw Efficiency prediction is DISABLED by the user.")

    # Step 3: Similarity and Visualization
    # Proceed only if the generated metrics CSV exists and is not empty
    if calculate_similarity:
        print("\nSimilarity calculation is ENABLED.")
        if generated_metrics_csv_path and os.path.exists(generated_metrics_csv_path):
            try:
                # Need at least 2 sequences to calculate pairs for similarity
                df_for_sim_check = pd.read_csv(generated_metrics_csv_path)
                if df_for_sim_check.shape[0] > 1:
                    avg_similarity, similarity_csv_path = calculate_and_save_similarity(
                        generated_metrics_csv_path, output_base_dir
                    )
                    if similarity_csv_path:
                        visualize_similarity_distribution(similarity_csv_path, output_base_dir)
                    else:
                        print("Similarity CSV path is None, skipping visualization.")
                else:
                    print("Less than 2 sequences found in generated data. Skipping similarity calculation and visualization.")
            except Exception as e:
                print(f"An unexpected error occurred during Similarity calculation/visualization: {e}")
        else:
            print("No generated metrics CSV found or it's empty. Skipping similarity calculation and visualization.")
    else:
        print("\nSimilarity calculation is DISABLED by the user.")

    # Step 4: MMD
    # Proceed only if the generated metrics CSV exists and is not empty
    if generated_metrics_csv_path and os.path.exists(generated_metrics_csv_path):
        try:
            mmd_value = calculate_mmd(generated_metrics_csv_path, natural_peptides_file, output_base_dir)
        except Exception as e:
            print(f"An unexpected error occurred during MMD calculation: {e}")
    else:
        print("No generated metrics CSV found or it's empty. Skipping MMD calculation.")

    # Step 5: Update All Metrics CSV
    print("\n--- Updating All Metrics Summary ---")
    metrics_data = {
        "model": model_version_name,
        "loss": f"{avg_loss:.4f}" if avg_loss is not None else None,
        "perplexity": f"{avg_perplexity:.4f}" if avg_perplexity is not None else None,
        "similarity": f"{avg_similarity:.2f}" if avg_similarity is not None else None,
        "mmd": f"{mmd_value:.6f}" if mmd_value is not None else None,
        "%_cpp": f"{percent_cpp:.2f}" if percent_cpp is not None else None,
        "mean_raw_efficiency": f"{mean_raw_efficiency:.4f}" if mean_raw_efficiency is not None else None
    }
    
    print("\n--- Updating All Metrics Summary ---")
    
    # Create a DataFrame for the current run for easier handling
    current_run_df = pd.DataFrame([metrics_data])

    try:
        # Scenario 1: all_metrics.csv does not yet exist
        if not os.path.exists(ALL_METRICS_CSV):
            current_run_df.to_csv(ALL_METRICS_CSV, index=False)
            print(f"Created new {ALL_METRICS_CSV} and added metrics for '{model_version_name}'.")

        # Scenario 2: The file already exists
        else:
            existing_df = pd.read_csv(ALL_METRICS_CSV)
            if model_version_name in existing_df['model'].values:
                # Scenario 2a: The model exists. UPDATE the row.
                print(f"Model '{model_version_name}' found. Updating metrics...")
                idx = existing_df[existing_df['model'] == model_version_name].index
                for col, val in metrics_data.items():
                    existing_df.loc[idx, col] = val
                existing_df.to_csv(ALL_METRICS_CSV, index=False)
                print(f"Metrics updated successfully in {ALL_METRICS_CSV}.")

            else:
                # Scenario 2b: The model is new. APPEND the row.
                print(f"Model '{model_version_name}' is new. Appending metrics...")
                combined_df = pd.concat([existing_df, current_run_df], ignore_index=True)
                combined_df.to_csv(ALL_METRICS_CSV, index=False)
                print(f"Metrics appended successfully to {ALL_METRICS_CSV}.")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while updating {ALL_METRICS_CSV}: {e}")

    print(f"\n--- Pipeline Finished for Model: {model_version_name} ---")

if __name__ == '__main__':
    pandarallel.initialize(progress_bar=False, nb_workers=1)
    parser = argparse.ArgumentParser(description="Analyze generated peptide sequences and evaluate a fine-tuned ProtGPT2 model.")
    
    # Add the arguments we will accept from the command line
    parser.add_argument(
        "--model_version_name", 
        type=str, 
        default="default_model_version", 
        help="Name of the model version, used for output directory naming (e.g., 'my_finetuned_protgpt2_v1')."
    )
    parser.add_argument(
        "--protgpt2_model_path", 
        type=str, 
        default="", 
        help="Path to the directory containing the fine-tuned ProtGPT2 model (tokenizer and model files)."
    )
    parser.add_argument(
        "--generated_sequences_file", 
        type=str, 
        default="log_generation_cpp_2.txt", 
        help="Path to the .txt file containing the generated peptide sequences in JSON format."
    )
    parser.add_argument(
        "--input_type", 
        type=str, 
        default="finetuned_txt", 
        choices=["finetuned_txt", "soft_prompt_csv", "prefix_tuned_csv"],
        help="Specify the format of the input file and the model type."
    )  
    
    parser.add_argument(
        "--calculate_efficiency",
        type=str,
        default="true",
        help="Set to 'false' to disable the raw_efficiency prediction block. Default is 'true'."
    )  

    parser.add_argument(
        "--calculate_similarity",
        type=str,
        default="true",
        help="Set to 'false' to disable the similarity calculation block. Default is 'true'."
    )
    
    args = parser.parse_args()
    do_calculate_efficiency = args.calculate_efficiency.lower() == 'true'
    do_calculate_similarity = args.calculate_similarity.lower() == 'true'

    # The main pipeline
    run_full_pipeline(
        model_version_name=args.model_version_name,
        generated_seq_file=args.generated_sequences_file,
        natural_peptides_file=NATURAL_PEPTIDES_FILE, # NATURAL_PEPTIDES_FILE remains a global constant as before
        protgpt2_model_path=args.protgpt2_model_path,
        input_type=args.input_type,
        calculate_efficiency=do_calculate_efficiency,
        calculate_similarity=do_calculate_similarity
    )