#!/bin/sh

# Set visible CUDA devices
export CUDA_VISIBLE_DEVICES=3

echo "=============================================="
echo "Generating 100 CPP sequences per cell line"
echo "=============================================="

LOG_FILE="log_run_pipeline_fast.txt"

python3 cond_gen_soft_prompt_active_learning_only_gen_100_fast_cpp_only.py \
    --target_cpp_per_line 100 \
    --initial_multiplier 2.5 \
    --max_iterations 3 \
    --temperature 1.0 \
    --repetition_penalty 1.1 \
    --min_aa_length 7 \
    --max_aa_length 22 \
    --gen_batch_size 32 \
    --ckpt_path "cpp_soft_prompt_model_active_learning.pt" \
    --seed 42 \
    > "${LOG_FILE}" 

echo ""
echo "=============================================="
echo "Generation complete!"
echo "Results saved to: final_100_cpp_per_line.csv"
echo "=============================================="