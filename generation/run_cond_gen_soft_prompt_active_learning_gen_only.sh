#!/bin/sh

export CUDA_VISIBLE_DEVICES=3

echo "Starting generation loop..."

for rep_penalty in 1.0 1.1 1.15 1.2 1.25 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0
do
    echo ""
    echo "######################################################################"
    echo "### Starting generation for repetition_penalty = $rep_penalty"
    echo "######################################################################"

    rep_filename_part=$(echo "$rep_penalty" | tr '.' '_')

    base_name="generated_cpp_sequences_multi_line_active_learning_3500_ends_no_overlap_seed_temp_1"
    run_name="${base_name}_rep_${rep_filename_part}"
    final_csv_name="${run_name}.csv"

    python3 cond_gen_soft_prompt_active_learning_only_gen.py \
	--seed 42 \
        --total_random 3500 \
	--temperature 1.0 \
        --repetition_penalty "$rep_penalty" \
        --run_name "$run_name" \
	--gen_batch_size 32

    original_csv_path="runs_gen/${run_name}/generated_sequences.csv"
    destination_csv_path="runs_gen/${run_name}/${final_csv_name}"

    if [ -f "$original_csv_path" ]; then
        echo "Renaming output file to: ${final_csv_name}"
        mv "$original_csv_path" "$destination_csv_path"
    else
        echo "Warning: Output file $original_csv_path not found. Skipping rename."
    fi

    echo "### Finished generation for repetition_penalty = $rep_penalty"
done

echo ""
echo "######################################################################"
echo "All generation runs are complete."
echo "######################################################################"