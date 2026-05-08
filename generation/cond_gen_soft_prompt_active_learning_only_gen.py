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

import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np

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


# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Soft-prompt generation for ProtGPT2")
    # Model/ckpt
    p.add_argument("--model_name_or_path", type=str, default="nferruz/ProtGPT2",
                   help="HF model name or local path")
    p.add_argument("--ckpt_path", type=str, default="cpp_soft_prompt_model_active_learning.pt",
                   help="Path to trained soft-prompt checkpoint (.pt)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)

    # Conditioning
    p.add_argument("--cell_line", type=str, nargs="*", default=None,
                   help="One or multiple cell line names to generate for. If omitted, use --total_random.")
    p.add_argument("--total_random", type=int, default=0,
                   help="If >0 and --cell_line not set, generate this many sequences across random cell lines (one per sequence).")
    p.add_argument("--num_per_cell_line", type=int, default=1,
                   help="How many sequences to sample per provided cell line.")
    p.add_argument("--gen_batch_size", type=int, default=32,
                   help="Batch size over conditions (for speed)")

    # Prompt/sampling
    p.add_argument("--prompt_text", type=str, default="", help="Optional textual prompt to append after <|endoftext|>")
    p.add_argument("--add_bos_eos", action="store_true", default=True, help="Prepend <|endoftext|> to prompt")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--num_return_sequences", type=int, default=1,
                   help="Number of samples per condition in one forward pass (used with --cell_line).")

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
    if args.cell_line and len(args.cell_line) > 0:
        target_cell_lines = args.cell_line
        per_cell_line = args.num_return_sequences  # we'll use num_return_sequences per condition
        total_expected = len(target_cell_lines) * per_cell_line
        mode = "by_list"
    else:
        if not cell_line_mapping:
            raise ValueError("cell_line_mapping not found in checkpoint, please provide --cell_line explicitly.")
        if args.total_random <= 0:
            raise ValueError("Either provide --cell_line list or set --total_random > 0.")
        all_lines = sorted(list(cell_line_mapping.keys()))
        target_cell_lines = random.choices(all_lines, k=args.total_random)
        per_cell_line = 1
        total_expected = args.total_random
        mode = "random"

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

    # Prepare iterations
    results = []
    if mode == "by_list":
        # Batch over provided lines; each condition can request N samples via num_return_sequences
        all_lines = target_cell_lines
        batch_idx = 0
        for i in tqdm(range(0, len(all_lines), args.gen_batch_size), desc="Generating (by_list)"):
            current_batch_seed = args.seed + batch_idx  
            batch_lines = all_lines[i:i + args.gen_batch_size]
            cond_mat = torch.tensor([make_cond_vec_for_line(cl) for cl in batch_lines], dtype=torch.float32, device=args.device)
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
                num_return_sequences=per_cell_line,
                logits_processor=logits_processor,
                device=args.device,
                batch_seed=current_batch_seed
            )
            # out length = len(batch_lines) * per_cell_line
            # Expand metadata accordingly
            expanded_lines = []
            for cl in batch_lines:
                expanded_lines.extend([cl] * per_cell_line)
            assert len(expanded_lines) == len(out)
            for dec, cl in zip(out, expanded_lines):
                full_sequence = dec["full"]
                eos_str = tokenizer.eos_token or "<|endoftext|>"

                pos = full_sequence.find(eos_str)
                if pos != -1:
                    core = full_sequence[:pos].rstrip("\r\n ")
                    clean_sequence = core + "\n" + eos_str
                else:
                    clean_sequence = full_sequence.rstrip()

                results.append({
                    "sequence": clean_sequence,
                    "cell_line": cl,
                    "params": {
                        "prompt_text": args.prompt_text,
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_k": args.top_k,
                        "top_p": args.top_p,
                        "repetition_penalty": args.repetition_penalty,
                        "no_repeat_ngram_size": args.no_repeat_ngram_size,
                        "num_return_sequences": per_cell_line,
                        "seed": current_batch_seed
                    }
                })

                batch_idx += 1
    else:
        # Random mode: each requested sequence uses one randomly chosen line (already sampled in target_cell_lines)
        # We'll batch them too for speed (1 sample per condition)
        batch_idx = 0
        for i in tqdm(range(0, len(target_cell_lines), args.gen_batch_size), desc="Generating (random)"):
            current_batch_seed = args.seed + batch_idx  
            batch_lines = target_cell_lines[i:i + args.gen_batch_size]
            cond_mat = torch.tensor([make_cond_vec_for_line(cl) for cl in batch_lines], dtype=torch.float32, device=args.device)
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
                num_return_sequences=1,
                logits_processor=logits_processor,
                device=args.device,
                batch_seed=current_batch_seed
            )
            for dec, cl in zip(out, batch_lines):
                full_sequence = dec["full"]
                eos_str = tokenizer.eos_token or "<|endoftext|>"

                pos = full_sequence.find(eos_str)
                if pos != -1:
                    core = full_sequence[:pos].rstrip("\r\n ")
                    clean_sequence = core + "\n" + eos_str
                else:
                    clean_sequence = full_sequence.rstrip()

                results.append({
                    "sequence": clean_sequence,
                    "cell_line": cl,
                    "params": {
                        "prompt_text": args.prompt_text,
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_k": args.top_k,
                        "top_p": args.top_p,
                        "repetition_penalty": args.repetition_penalty,
                        "no_repeat_ngram_size": args.no_repeat_ngram_size,
                        "num_return_sequences": 1,
                        "seed": current_batch_seed
                    }
                })

                batch_idx += 1

    print(f"Generated {len(results)} sequences (expected ~{total_expected}). Saving to {run_dir}")

    # Save params snapshot
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Save JSONL
    if args.save_jsonl:
        out_jsonl = os.path.join(run_dir, "raw.jsonl")
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved JSONL: {out_jsonl}")

    # Save CSV
    if args.save_csv:
        df = pd.DataFrame([{"sequence": r["sequence"], "cell_line": r["cell_line"]} for r in results])
        out_csv = os.path.join(run_dir, "generated_sequences.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved CSV: {out_csv}")

    print("Done.")


if __name__ == "__main__":
    main()