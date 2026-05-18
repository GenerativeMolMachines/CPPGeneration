#!/usr/bin/env python3
"""
Generates exactly 100 CPP sequences for each cell line
"""


import argparse
import subprocess
import pandas as pd
from pathlib import Path
import json

from typing import Optional
import random
import numpy as np
import torch
import os

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

def generate_cpp_dataset(
    target_cpp_per_line=100,
    initial_multiplier=2.5,  # Generate 2.5x more sequences as a buffer
    max_iterations=3,
    gen_script="cond_gen_soft_prompt_active_learning_only_gen_100.py",
    eval_script="validation/internal/run_pipeline.py",
    seed = 42,
    **gen_params
):
    """
    Iteratively generate sequences until reaching 100 CPP per cell line.
    """

    set_seed(seed)

    all_cpp_sequences = []
    iteration = 1
    sequences_per_line = int(target_cpp_per_line * initial_multiplier)
    
    while iteration <= max_iterations:
        print("\n" + "="*80)
        print(f"ITERATION {iteration}/{max_iterations}")
        print(f"Generating {sequences_per_line} sequences per cell line")
        print("="*80)
        
        run_name = f"cpp_gen_iter_{iteration}"
        
        # 1. Generation
        print(f"\n[{iteration}.1] Generating sequences...")

        iteration_seed = seed + (iteration - 1) * 100000000 if seed is not None else None

        cmd_gen = [
            "python3", gen_script,
            "--sequences_per_cell_line", str(sequences_per_line),
            "--run_name", run_name,
            "--min_aa_length", str(gen_params.get('min_aa_length', 7)),
            "--max_aa_length", str(gen_params.get('max_aa_length', 22)),
            "--temperature", str(gen_params.get('temperature', 1.0)),
            "--repetition_penalty", str(gen_params.get('repetition_penalty', 1.15)),
            "--gen_batch_size", str(gen_params.get('gen_batch_size', 16)),
        ]

        if iteration_seed is not None:
            cmd_gen.extend(["--seed", str(iteration_seed)])
        
        print(f"Using seed for iteration {iteration}: {iteration_seed}")

        subprocess.run(cmd_gen, check=True)
        
        csv_path = f"runs_gen/{run_name}/generated_sequences.csv"
        
        # 2. Classification
        print(f"\n[{iteration}.2] Running CPP classification...")
        cmd_eval = [
            "python3", eval_script,
            "--model_version_name", run_name,
            "--generated_sequences_file", csv_path,
            "--input_type", "soft_prompt_csv",
            "--protgpt2_model_path", gen_params.get('ckpt_path', 'cpp_soft_prompt_model_active_learning.pt'),
            "--calculate_efficiency", "false",
            "--calculate_similarity", "false"
        ]
        subprocess.run(cmd_eval, check=True)
        
        # 3. CPP filtering
        print(f"\n[{iteration}.3] Filtering CPP sequences...")
        metrics_csv = f"metrics_{run_name}/sequences_with_metrics.csv"
        df = pd.read_csv(metrics_csv)
        
        cpp_df = df[df['prediction_cpp'] == 1].copy()
        all_cpp_sequences.append(cpp_df)
        
        # 4. Sufficiency check
        combined_cpp = pd.concat(all_cpp_sequences, ignore_index=True)
        combined_cpp = combined_cpp.drop_duplicates(subset=['sequence', 'cell_line'])
        cpp_counts = combined_cpp.groupby('cell_line').size()
        
        print(f"\n[{iteration}.4] CPP counts per cell line:")
        insufficient_lines = []
        for cell_line, count in cpp_counts.items():
            status = "✓" if count >= target_cpp_per_line else "✗"
            print(f"  {status} {cell_line:30s}: {count:4d} / {target_cpp_per_line}")
            if count < target_cpp_per_line:
                insufficient_lines.append((cell_line, count))
        
        # Check whether all lines have enough CPP
        if not insufficient_lines:
            print("\n✓ All cell lines have sufficient CPP sequences!")
            break
        
        if iteration == max_iterations:
            print(f"\n⚠ Max iterations reached. Some cell lines still insufficient:")
            for line, count in insufficient_lines:
                print(f"  - {line}: {count}/{target_cpp_per_line}")
            break
        
        # Adaptive adjustment for the next iteration
        min_acceptance = min(count / sequences_per_line for _, count in insufficient_lines)
        sequences_per_line = int((target_cpp_per_line / min_acceptance) * 1.2)
        iteration += 1
    
    # 5. Final selection of exactly 100 per line
    print("\n" + "="*80)
    print("FINAL STEP: Selecting exactly 100 CPP per cell line")
    print("="*80)
    
    final_cpp_list = []
    for cell_line in combined_cpp['cell_line'].unique():
        line_cpp = combined_cpp[combined_cpp['cell_line'] == cell_line]
        
        if len(line_cpp) >= target_cpp_per_line:
            # Take the first 100
            selected = line_cpp.head(target_cpp_per_line)
            final_cpp_list.append(selected)
            print(f"✓ {cell_line:30s}: selected {len(selected)} CPP")
        else:
            # Take all available
            final_cpp_list.append(line_cpp)
            print(f"⚠ {cell_line:30s}: only {len(line_cpp)} CPP available")
    
    final_df = pd.concat(final_cpp_list, ignore_index=True)
    
    # 6. Save the result
    output_path = "final_100_cpp_per_line.csv"
    final_df.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total CPP sequences: {len(final_df)}")
    print(f"Cell lines: {final_df['cell_line'].nunique()}")
    print(f"\nSaved to: {output_path}")
    
    # Statistics
    print(f"\nDistribution:")
    print(final_df.groupby('cell_line').size().to_string())
    
    # Save metadata
    metadata = {
        'target_cpp_per_line': target_cpp_per_line,
        'iterations': iteration,
        'total_sequences': len(final_df),
        'generation_params': gen_params,
        'per_line_counts': final_df.groupby('cell_line').size().to_dict(),
        'seed': seed
    }
    
    with open('final_100_cpp_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return final_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate exactly 100 CPP sequences per cell line"
    )
    parser.add_argument("--target_cpp_per_line", type=int, default=100)
    parser.add_argument("--initial_multiplier", type=float, default=2.5,
                       help="Initial overgeneration multiplier")
    parser.add_argument("--max_iterations", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--min_aa_length", type=int, default=7)
    parser.add_argument("--max_aa_length", type=int, default=22)
    parser.add_argument("--gen_batch_size", type=int, default=16)
    parser.add_argument("--ckpt_path", type=str, 
                       default="models/cpp_soft_prompt_model_active_learning.pt")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    generate_cpp_dataset(
        target_cpp_per_line=args.target_cpp_per_line,
        initial_multiplier=args.initial_multiplier,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        min_aa_length=args.min_aa_length,
        max_aa_length=args.max_aa_length,
        gen_batch_size=args.gen_batch_size,
        ckpt_path=args.ckpt_path,
        seed=args.seed,
    )