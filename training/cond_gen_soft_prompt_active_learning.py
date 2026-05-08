"""
Soft-prompt generator for conditional generation with ProtGPT2 (HF transformers + PyTorch).

How it works (short):
- Condition vector (binary) -> small MLP (ConditionEncoder) -> outputs `prefix_len` embeddings.
- We prepend those embeddings to the token embeddings of the input sequence (inputs_embeds).
- Model is frozen (PEFT-style); only ConditionEncoder is trained.
- Use inputs_embeds + attention_mask adjusted for generation and training.

Replace MODEL_NAME if you use another HF model than nferruz/ProtGPT2.
"""

import os
import math
import pandas as pd
import numpy as np
from collections import defaultdict
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW 
from tqdm import tqdm
import json

# -----------------------
# Config / hyperparams
# -----------------------
MODEL_NAME = "nferruz/ProtGPT2"  # replace if needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREFIX_LEN = 20          # number of soft tokens to prepend
LEARNING_RATE = 5e-4
BATCH_SIZE = 8
EPOCHS = 3
MAX_SEQ_LEN = 256        # tokenized sequence length (adjust)
EMBEDDING_DROPOUT = 0.1
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)

# -----------------------
# Load tokenizer & model
# -----------------------
print(f"Loading model {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()  # we will freeze it, use it in eval mode for safe inference

# Freeze base model parameters (PEFT-style)
for param in model.parameters():
    param.requires_grad = False

# Extract embedding size from model config
embed_dim = model.config.hidden_size

# -----------------------
# Condition -> prefix generator
# -----------------------
class ConditionPrefixGenerator(nn.Module):
    """
    MLP that converts a binary condition vector into prefix_len * embed_dim embeddings.
    Output shape: (batch, prefix_len, embed_dim)
    """
    def __init__(self, cond_dim, prefix_len, embed_dim, hidden=512, dropout=0.0):
        super().__init__()
        self.prefix_len = prefix_len
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, prefix_len * embed_dim)
        )
        # optional small layernorm to stabilize prefixes
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, cond):  # cond: (batch, cond_dim)
        x = self.mlp(cond)  # (batch, prefix_len * embed_dim)
        x = x.view(-1, self.prefix_len, self.embed_dim)
        # normalize per-vector
        x = self.layernorm(x)
        return x


# -----------------------
# Toy dataset wrapper (replace with your real dataset)
# -----------------------
class PeptideDataset(Dataset):
    """
    dataset_items: list of (condition_vector, tokenized_ids)
    condition_vector is torch.float32 vector of shape (CONDITION_DIM,)
    tokenized_ids is list/torch.tensor of token ids (already tokenized/padded/truncated to MAX_SEQ_LEN)
    """
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cond, ids = self.items[idx]
        return torch.tensor(cond, dtype=torch.float32), torch.tensor(ids, dtype=torch.long)


# -----------------------
# Helpers: batching + collate
# -----------------------
def collate_fn(batch):
    """
    batch: list of (cond_tensor, token_ids)
    We'll pad token_ids to same length (but in our training we use fixed MAX_SEQ_LEN).
    Returns:
      cond_batch: (B, CONDITION_DIM)
      input_ids: (B, L)
      attention_mask: (B, L)
    """
    conds = torch.stack([b[0] for b in batch], dim=0)
    ids = [b[1] for b in batch]
    # simple padding to max length in batch
    maxlen = max([len(x) for x in ids])
    padded = torch.zeros((len(ids), maxlen), dtype=torch.long)
    mask = torch.zeros((len(ids), maxlen), dtype=torch.long)
    for i, seq in enumerate(ids):
        L = len(seq)
        padded[i, :L] = seq
        mask[i, :L] = 1
    return conds, padded, mask


# -----------------------
# Training components
# -----------------------

# Loss: causal LM loss from model itself (we will use model's output logits)
# Training loop: we put prefix embeddings at inputs_embeds and provide labels accordingly.

def tokenize_peptide(seq_str):
    # example: tokenizer specific to ProtGPT2 may need special preprocessing (like spaces between residues).
    # assume user already has sequences tokenized; here we just use tokenizer.encode
    toks = tokenizer.encode(seq_str, add_special_tokens=False)
    if len(toks) > MAX_SEQ_LEN:
        toks = toks[:MAX_SEQ_LEN]
    return toks

def load_dataset_from_csv(csv_path):
    """
    Uploads a CSV file and converts it into a format suitable for training
    """
    print(f"\nLoading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter only CPP peptides for training
    df_cpp = df[df['is_cpp'] == 1]  

    # Mapping cell lines to indices
    unique_cell_lines = sorted(df_cpp['cell_line'].unique())
    cell_line_to_idx = {cl: idx for idx, cl in enumerate(unique_cell_lines)}
    num_cell_lines = len(unique_cell_lines)
    
    print(f"Found {num_cell_lines} unique cell lines: {unique_cell_lines}")
    print(f"Total CPP sequences: {len(df_cpp)}")
    
    CONDITION_DIM = 1 + num_cell_lines

    # Creating training examples
    dataset_items = []
    
    for _, row in df_cpp.iterrows():
        # Creating a binary condition vector
        condition_vector = [0] * CONDITION_DIM
        condition_vector[0] = 1  
        
        # Set 1 for the corresponding cell line
        cell_line_idx = 1 + cell_line_to_idx[row['cell_line']]
        condition_vector[cell_line_idx] = 1
        
        # Tokenizing a sequence
        sequence = row['sequence']
        token_ids = tokenize_peptide(sequence)    
        dataset_items.append((condition_vector, token_ids))
    
    return dataset_items, cell_line_to_idx, num_cell_lines

dataset_items, cell_line_mapping_active_learning, num_cell_lines = load_dataset_from_csv('cpp_only_165.csv')
CONDITION_DIM = 1 + num_cell_lines

# Update the model creation with the correct CONDITION_DIM
prefix_generator = ConditionPrefixGenerator(
    CONDITION_DIM, PREFIX_LEN, embed_dim, 
    hidden=512, dropout=EMBEDDING_DROPOUT
).to(DEVICE)

optimizer = AdamW(prefix_generator.parameters(), lr=LEARNING_RATE)

dataset = PeptideDataset(dataset_items)
g = torch.Generator()
g.manual_seed(SEED)

dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_fn,
    generator=g,  
    worker_init_fn=lambda worker_id: set_seed(SEED + worker_id)
)

# -----------------------
# Training loop
# -----------------------
print("Start training soft-prompt generator on device:", DEVICE)
print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}")
for epoch in range(EPOCHS):
    prefix_generator.train()
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    epoch_loss = 0.0
    for cond_batch, input_ids, att_mask in loop:
        cond_batch = cond_batch.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        att_mask = att_mask.to(DEVICE)

        # Create labels: shift left by 1 (causal LM)
        # We'll compute loss on model's logits for tokens positions (excluding possible prefix positions)
        labels = input_ids.clone()  # (B, L)
        # you could set label tokens that are padding to -100 to ignore in loss:
        labels[att_mask == 0] = -100

        # Build token embeddings for input_ids
        with torch.no_grad():
            input_embeds = model.get_input_embeddings()(input_ids)  # (B, L, embed_dim)

        # Generate prefix embeddings from condition
        prefix_emb = prefix_generator(cond_batch)  # (B, PREFIX_LEN, embed_dim)

        # Concatenate along seq dimension
        inputs_embeds = torch.cat([prefix_emb, input_embeds], dim=1)  # (B, PREFIX_LEN+L, embed_dim)

        # Build extended attention mask (prefix tokens are real tokens now)
        prefix_mask = torch.ones((input_ids.size(0), PREFIX_LEN), dtype=torch.long).to(DEVICE)
        extended_att_mask = torch.cat([prefix_mask, att_mask], dim=1)  # (B, PREFIX_LEN+L)

        # We need labels aligned with logits: shift labels accordingly (we want model logits to predict tokens of original input)
        # Because we prepended PREFIX_LEN tokens, model logits will correspond to positions [0..PREFIX_LEN+L-1], but labels should be
        # - set first PREFIX_LEN positions to -100 (ignore),
        # - next L positions correspond to original tokens we want to predict.
        # If using model(..., labels=...), HF will compute loss automatically.
        labels_with_prefix = torch.cat([
            torch.full((labels.size(0), PREFIX_LEN), -100, dtype=torch.long).to(DEVICE),
            labels
        ], dim=1)  # shape (B, PREFIX_LEN + L)

        # Forward pass through model with inputs_embeds
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=extended_att_mask, labels=labels_with_prefix)
        loss = outputs.loss

        # Backprop only into prefix_generator (model frozen)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

# -----------------------
# Save model and configuration
# -----------------------
save_path = "cpp_soft_prompt_model_active_learning.pt"
torch.save({
    'state_dict': prefix_generator.state_dict(),
    'cell_line_mapping': cell_line_mapping_active_learning,
    'condition_dim': CONDITION_DIM,
    'prefix_len': PREFIX_LEN,
    'seed': SEED,
    'hyperparameters': {
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'embedding_dropout': EMBEDDING_DROPOUT,
    }
}, save_path)
print(f"\nSaved model and configuration to {save_path}")

# Also save cell line mapping as JSON for easy reference
with open('cell_line_mapping_active_learning.json', 'w') as f:
    json.dump(cell_line_mapping_active_learning, f, indent=2)
print("Saved cell line mapping to cell_line_mapping_active_learning.json")