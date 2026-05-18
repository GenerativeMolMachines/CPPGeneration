#!/usr/bin/bash

# Set visible CUDA devices
export CUDA_VISIBLE_DEVICES=3

# --- Define the parameters for the current run ---
MODEL_VERSION_BASE="soft_prompt_model_v2_"
PROMPT_MODEL_ARTIFACTS_PATH="models/cpp_soft_prompt_model_active_learning.pt" 
RUNS_DIR="results"

PATTERN="generated_cpp_sequences_multi_line_active_learning_3500_ends_no_overlap_seed_temp_1_rep_*.csv"

# --- Form path to metrics folder and logs ---

# Create a folder for the metrics if it doesn't already exist.
METRICS_BASE_DIR="metrics_${MODEL_VERSION_BASE}"
mkdir -p "${METRICS_BASE_DIR}"

# --- Finding files ---
mapfile -t CSV_FILES < <(find "${RUNS_DIR}" -type f -name "${PATTERN}" | sort)

if [ "${#CSV_FILES[@]}" -eq 0 ]; then
  echo "No CSV files found under ${RUNS_DIR} matching pattern '${PATTERN}'."
  exit 1
fi

echo "Found ${#CSV_FILES[@]} CSV files to evaluate."
echo

# --- Switch for raw_efficiency calculation --- #
CALCULATE_EFFICIENCY="false" 
CALCULATE_SIMILARITY="false"

# --- Run the Python script ---

for CSV_PATH in "${CSV_FILES[@]}"; do
  BASENAME="$(basename "${CSV_PATH}")"
  NAME_NOEXT="${BASENAME%.csv}"

  CLEAN_SUFFIX="${NAME_NOEXT#generated_cpp_sequences_multi_line_}"

  MODEL_VERSION_NAME="${MODEL_VERSION_BASE}_${NAME_NOEXT}"

  METRICS_OUTPUT_DIR="${METRICS_BASE_DIR}"
  LOG_FILE="${METRICS_OUTPUT_DIR}/log_run_pipeline.txt"
  mkdir -p "${METRICS_OUTPUT_DIR}"

  echo "Evaluating: ${CSV_PATH}"
  echo "  -> model_version_name: ${MODEL_VERSION_NAME}"
  echo "  -> metrics dir:        ${METRICS_OUTPUT_DIR}"

  python3 run_pipeline.py \
    --model_version_name "${MODEL_VERSION_NAME}" \
    --protgpt2_model_path "${PROMPT_MODEL_ARTIFACTS_PATH}" \
    --generated_sequences_file "${CSV_PATH}" \
    --input_type "soft_prompt_csv" \
    --calculate_efficiency "${CALCULATE_EFFICIENCY}" \
    --calculate_similarity "${CALCULATE_SIMILARITY}" \
    > "${LOG_FILE}" 2>&1

  echo "Done. Log: ${LOG_FILE}"
  echo
done

echo "Pipeline finished. Check logs in ${LOG_FILE}"