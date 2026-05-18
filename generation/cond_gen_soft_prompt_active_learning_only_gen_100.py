#!/usr/bin/env python3
"""
Soft-prompt generation for ProtGPT2 using a trained ConditionPrefixGenerator.

- Loads: base HF model (nferruz/ProtGPT2 by default) + trained soft-prompt checkpoint (.pt)
- Uses: inputs_embeds = [prefix_emb | token_embeds(prompt)]
- Generates sequences conditioned on cell line (from saved mapping)

Outputs:
- JSONL and/or CSV with sequences and metadata
"""

import os
import json
import math
import argparse
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
)

# -----------------------
# Model components
# -----------------------
class ConditionPrefixGenerator(nn.Module):
    """
    MLP that converts a binary condition vector into prefix_len * embed_dim embeddings.
    Output: (batch, prefix_len, embed_dim)
    """
    def __init__(self, cond_dim: int, prefix_len: int, embed_dim: int, hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        self.prefix_len = prefix_len
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, prefix_len * embed_dim)
        )
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, cond):  # cond: (B, cond_dim)
        x = self.mlp(cond)  # (B, prefix_len * embed_dim)
        x = x.view(-1, self.prefix_len, self.embed_dim)
        x = self.layernorm(x)
        return x


# -----------------------
# Utils
# -----------------------
def set_seed(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def set_seed_light(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_amino_acids(sequence: str) -> int:
    """
    Counts the number of amino acids in the sequence, ignoring newline characters.
    """
    return len(sequence.replace('\n', '').replace('\r', '').replace('<|endoftext|>', '').strip())


def build_logits_processor(args) -> LogitsProcessorList:
    lp = LogitsProcessorList()
    if args.repetition_penalty and args.repetition_penalty != 1.0:
        lp.append(RepetitionPenaltyLogitsProcessor(args.repetition_penalty))
    if args.no_repeat_ngram_size and args.no_repeat_ngram_size > 0:
        lp.append(NoRepeatNGramLogitsProcessor(args.no_repeat_ngram_size))
    return lp


def load_prefix_generator(ckpt_path: str, embed_dim: int, device: str):
    """
    Restore ConditionPrefixGenerator from checkpoint:
    ckpt must contain keys: 'state_dict', 'condition_dim', 'prefix_len', optionally 'cell_line_mapping'
    Hidden size is inferred from state_dict shapes.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    cond_dim = int(ckpt["condition_dim"])
    prefix_len = int(ckpt["prefix_len"])

    # infer hidden size from first linear layer weight: (hidden, cond_dim)
    if "mlp.0.weight" in state_dict:
        hidden = state_dict["mlp.0.weight"].shape[0]
    else:
        hidden = 512  # fallback

    gen = ConditionPrefixGenerator(cond_dim, prefix_len, embed_dim, hidden=hidden, dropout=0.0)
    gen.load_state_dict(state_dict, strict=True)
    gen.to(device).eval()

    cell_line_mapping = ckpt.get("cell_line_mapping", None)
    return gen, cell_line_mapping, cond_dim, prefix_len


def prepare_prompt_ids(tokenizer, prompt_text: str, add_bos_eos: bool = True, device: str = "cpu"):
    """
    Creates prompt token ids. For ProtGPT2 it's often beneficial to start from <|endoftext|>.
    """
    if add_bos_eos:
        prompt = f"<|endoftext|>{prompt_text}"
    else:
        prompt = prompt_text
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    return input_ids


def decode_generated(tokenizer, seq_ids: torch.LongTensor, prompt_len: int) -> Dict[str, str]:
    """
    Returns dict with 'generated' (only new tokens) and 'full' (prompt + new) strings.
    """
    # full (skip special tokens)
    full_text = tokenizer.decode(seq_ids, skip_special_tokens=False)
    # only generated part
    gen_only_ids = seq_ids[prompt_len:]
    generated = tokenizer.decode(gen_only_ids, skip_special_tokens=False)
    return {"generated": generated, "full": full_text}


# -----------------------
# Core generation
# -----------------------
def generate_batch(
    prefix_generator: ConditionPrefixGenerator,
    model: AutoModelForCausalLM,
    tokenizer,
    cond_batch: torch.FloatTensor,               # (B, cond_dim)
    prompt_ids: torch.LongTensor,                # (1, Lp)
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    num_return_sequences: int,
    logits_processor: Optional[LogitsProcessorList],
    device: str,
    batch_seed: Optional[int] = None,
):
    """
    Generates sequences for a batch of conditions. Returns list of dicts (len = B * num_return_sequences)
    """
    if batch_seed is not None:  
        set_seed_light(batch_seed)

    B = cond_batch.size(0)
    with torch.no_grad():
        prefix_emb = prefix_generator(cond_batch.to(device))  # (B, P, H)
        # Prepare prompt embeds for batch
        if prompt_ids is None or prompt_ids.numel() == 0:
            # empty prompt
            prompt_len = 0
            prompt_emb = torch.empty((B, 0, model.config.hidden_size), device=device)
        else:
            prompt_len = prompt_ids.size(1)
            tok_emb_layer = model.get_input_embeddings()
            prompt_emb_one = tok_emb_layer(prompt_ids)  # (1, Lp, H)
            prompt_emb = prompt_emb_one.expand(B, -1, -1).contiguous()  # (B, Lp, H)

        inputs_embeds = torch.cat([prefix_emb, prompt_emb], dim=1)  # (B, P+Lp, H)
        attn_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

        sequences = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=logits_processor,
            renormalize_logits=True
        )  # shape: (B * N, P+Lp + new_len)

        # Decode results
        out = []
        total = sequences.size(0)
        for i in range(total):
            seq_ids = sequences[i].detach().cpu()
            # prompt_len refers to tokens from prompt only; prefix embeddings are not tokens
            dec = decode_generated(tokenizer, seq_ids, prompt_len=prompt_len)
            out.append(dec)
        return out

def clean_sequence(full_sequence: str, tokenizer) -> str:
    """
    Cleans the sequence, leaving only the text up to the first EOS token.
    """
    eos_str = tokenizer.eos_token or "<|endoftext|>"
    pos = full_sequence.find(eos_str)
    if pos != -1:
        core = full_sequence[:pos].rstrip("\r\n ")
        clean_sequence = core + "\n" + eos_str
    else:
        clean_sequence = full_sequence.rstrip()
    return clean_sequence

def generate_sequences_for_cell_line(
    cell_line: str,
    target_count: int,
    min_length: int,
    max_length: int,
    prefix_generator: ConditionPrefixGenerator,
    model: AutoModelForCausalLM,
    tokenizer,
    cond_vector: List[float],
    prompt_ids: torch.LongTensor,
    args,
    logits_processor: LogitsProcessorList,
    device: str,
    base_seed: Optional[int] = None,
) -> List[Dict]:
    """
    Generates exactly target_count unique sequences for a given cell line
    with lengths in the range [min_length, max_length].
    """
    valid_sequences = []
    seen_sequences = set()
    duplicates_count = 0
    length_rejected_count = 0

    pbar = tqdm(total=target_count, desc=f"Generating for {cell_line}")
    
    attempts = 0
    max_attempts = target_count * 20  # infinite loop protection
    batch_counter = 0

    while len(valid_sequences) < target_count and attempts < max_attempts:
        batch_seed = base_seed + batch_counter if base_seed is not None else None

        # Figure out how much more needed
        remaining = target_count - len(valid_sequences)
        # Generate with a buffer (batch size up to gen_batch_size)
        batch_size = min(args.gen_batch_size, remaining * 2)  
        
        # Creating a batch of identical conditions
        cond_mat = torch.tensor([cond_vector] * batch_size, dtype=torch.float32, device=device)
        
        # Generate
        out = generate_batch(
            prefix_generator=prefix_generator,
            model=model,
            tokenizer=tokenizer,
            cond_batch=cond_mat,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.gen_batch_size,
            logits_processor=logits_processor,
            device=device,
            batch_seed=batch_seed,
        )
        
        # Filter by length
        for dec in out:
            if len(valid_sequences) >= target_count:
                break
                
            full_sequence = dec["full"]
            clean_seq = clean_sequence(full_sequence, tokenizer)
            
            # Count amino acids
            aa_count = count_amino_acids(clean_seq)

            if min_length <= aa_count <= max_length:
                if clean_seq in seen_sequences:
                    duplicates_count += 1
                    continue 
                seen_sequences.add(clean_seq)
                
                valid_sequences.append({
                    "sequence": clean_seq,
                    "cell_line": cell_line,
                    "aa_length": aa_count,
                    "params": {
                        "prompt_text": args.prompt_text,
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_k": args.top_k,
                        "top_p": args.top_p,
                        "repetition_penalty": args.repetition_penalty,
                        "no_repeat_ngram_size": args.no_repeat_ngram_size,
                        "base_seed": base_seed,
                        "batch_seed": batch_seed,
                    }
                })
                pbar.update(1)

            else: 
                length_rejected_count += 1
                continue
        
        attempts += len(out)
        batch_counter += 1
    
    pbar.close()
    
    if duplicates_count > 0:
        print(f"  Duplicates skipped for {cell_line}: {duplicates_count}")

    if length_rejected_count > 0:
        print(f"  Length_rejected for {cell_line}: {length_rejected_count}")

    if len(valid_sequences) < target_count:
        print(f"Warning: Could only generate {len(valid_sequences)}/{target_count} valid sequences for {cell_line}")
    
    return valid_sequences[:target_count]  # return exactly target_count

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Soft-prompt generation for ProtGPT2")
    # Model/ckpt
    p.add_argument("--model_name_or_path", type=str, default="nferruz/ProtGPT2",
                   help="HF model name or local path")
    p.add_argument("--ckpt_path", type=str, default="models/cpp_soft_prompt_model_active_learning.pt",
                   help="Path to trained soft-prompt checkpoint (.pt)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)

    # Conditioning
    p.add_argument("--cell_lines", type=str, nargs="*", default=None,
                   help="List of cell line names to generate for. If None, uses all from checkpoint.")
    p.add_argument("--sequences_per_cell_line", type=int, default=100,
                   help="Number of sequences to generate per cell line")
    p.add_argument("--min_aa_length", type=int, default=7,
                   help="Minimum length in amino acids")
    p.add_argument("--max_aa_length", type=int, default=22,
                   help="Maximum length in amino acids")
    p.add_argument("--gen_batch_size", type=int, default=8,
                   help="Batch size over conditions (for speed)")

    # Prompt/sampling
    p.add_argument("--prompt_text", type=str, default="", help="Optional textual prompt to append after <|endoftext|>")
    p.add_argument("--add_bos_eos", action="store_true", default=True, help="Prepend <|endoftext|> to prompt")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=1.0)

    # Repetition control (optional)
    p.add_argument("--repetition_penalty", type=float, default=1.0, help=">=1.0; 1.0 disables")
    p.add_argument("--no_repeat_ngram_size", type=int, default=0, help="0 disables")

    # Output
    p.add_argument("--out_dir", type=str, default="runs_gen")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--save_csv", action="store_true", default=True)
    p.add_argument("--save_jsonl", action="store_true", default=False)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    run_name = args.run_name or datetime.now().strftime("gen_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Loading tokenizer/model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    embed_dim = model.config.hidden_size
    print(f"Loading soft-prompt checkpoint: {args.ckpt_path}")
    prefix_generator, cell_line_mapping, cond_dim, prefix_len = load_prefix_generator(args.ckpt_path, embed_dim, args.device)
    print(f"Loaded soft-prompt: cond_dim={cond_dim}, prefix_len={prefix_len}, "
          f"num_cell_lines={len(cell_line_mapping) if cell_line_mapping else 'unknown'}")

    # choose cell lines
    if args.cell_lines and len(args.cell_lines) > 0:
        target_cell_lines = args.cell_lines
    else:
        if not cell_line_mapping:
            raise ValueError("cell_line_mapping not found in checkpoint")
        target_cell_lines = sorted(list(cell_line_mapping.keys()))
    
    print(f"Target cell lines ({len(target_cell_lines)}): {target_cell_lines}")
    print(f"Sequences per cell line: {args.sequences_per_cell_line}")
    print(f"AA length range: [{args.min_aa_length}, {args.max_aa_length}]")

    # Build prompt tokens once
    prompt_ids = prepare_prompt_ids(tokenizer, args.prompt_text, add_bos_eos=args.add_bos_eos, device=args.device)
    logits_processor = build_logits_processor(args)

    # Helper to make condition vectors
    def make_cond_vec_for_line(cl_name: str):
        if not cell_line_mapping or cl_name not in cell_line_mapping:
            raise KeyError(f"Cell line '{cl_name}' not found in mapping.")
        v = [0.0] * cond_dim
        v[0] = 1.0  # is_cpp = 1
        idx = cell_line_mapping[cl_name]
        v[1 + idx] = 1.0
        return v

    all_results = []

    for idx, cell_line in enumerate(target_cell_lines):
        print(f"\n{'='*80}")
        print(f"Processing cell line: {cell_line}")
        print(f"{'='*80}")

        
        cell_line_base_seed = args.seed + idx * 100000 if args.seed is not None else None

        print(f"Using base_seed: {cell_line_base_seed} (global_seed={args.seed}, cell_line_idx={idx})")

        cond_vector = make_cond_vec_for_line(cell_line)
        
        sequences = generate_sequences_for_cell_line(
            cell_line=cell_line,
            target_count=args.sequences_per_cell_line,
            min_length=args.min_aa_length,
            max_length=args.max_aa_length,
            prefix_generator=prefix_generator,
            model=model,
            tokenizer=tokenizer,
            cond_vector=cond_vector,
            prompt_ids=prompt_ids,
            args=args,
            logits_processor=logits_processor,
            device=args.device,
            base_seed=cell_line_base_seed
        )
        
        all_results.extend(sequences)
        print(f"Generated {len(sequences)} valid sequences for {cell_line}")

    print(f"\n{'='*80}")
    print(f"Total generated: {len(all_results)} sequences")
    print(f"Saving to {run_dir}")

    # Save params
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Save JSONL
    if args.save_jsonl:
        out_jsonl = os.path.join(run_dir, "raw.jsonl")
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved JSONL: {out_jsonl}")

    # Save CSV
    if args.save_csv:
        df = pd.DataFrame([
            {
                "sequence": r["sequence"],
                "cell_line": r["cell_line"],
                "aa_length": r["aa_length"]
            } for r in all_results
        ])
        out_csv = os.path.join(run_dir, "generated_sequences.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved CSV: {out_csv}")
        
        # Statistics by length
        print(f"\nLength statistics:")
        print(df.groupby('cell_line')['aa_length'].describe())

    print("\nDone.")


if __name__ == "__main__":
    main()