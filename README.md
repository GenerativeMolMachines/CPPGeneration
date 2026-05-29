# CPPGeneration

## Project Overview

`CPPGeneration` is a peptide generation and active-learning pipeline for cell-penetrating peptides (CPPs). The repository includes preprocessing, classifier-driven active sampling, soft-prompt modeling with ProtGPT2, conditional peptide generation, and validation utilities.

## Repository Structure

- `data/`
  - `raw/` : original peptide datasets.
  - `processed/` : cleaned and formatted datasets used for model training and generation.
  - `active_sampling/` : input CSV files used by the active sampling pipeline.
- `preprocessing/`
  - `sequences_for_conditional_generator_preprocessing.ipynb` : notebook for dataset cleaning, filtering, balancing, and formatting for ProtGPT2.
  - `sequence_preprocessing_utils.py` : helper functions for peptide standardization, validation, and sequence feature extraction.
  - `vizualize_sequence_validity.py` : helper visualizations for sequence validity and preprocessing checks.
- `training/`
  - `train_generator.py` : soft-prompt training pipeline for ProtGPT2-based conditional generation.
  - `utility_scripts.py` : training helper utilities.
  - `train_classifier.ipynb` : notebook for building classifier models and exploring training metrics.
- `generation/`
  - `cond_gen_soft_prompt_active_learning_only_gen.py` : generation script for producing CPP sequences using a trained soft prompt.
  - `cond_gen_soft_prompt_active_learning_only_gen_100.py` : extended generation variant for larger sequence batches.
  - `cond_gen_soft_prompt_active_learning_only_gen_100_cpp_only.py` : generation script focused on per-cell-line CPP output.
  - `run_pipeline_arg_soft_prompt_active_learning_100_cpp_only.sh` : example shell script for batch generation.
- `active_sampling/`
  - `with_active_sampling.py` : active learning pipeline using classifier uncertainty, diverse sampling, and budgeted data selection.
- `validation/` and `results/`
  - `validation/internal/` : validation and metrics scripts.
  - `results/` : generated dataset outputs, metrics, and visualizations.
- `models/`
  - trained model artifacts such as `cpp_soft_prompt_model_active_learning.pt` and classifier artifacts.
- `config/`
  - model and active learning parameters stored in JSON files.
- `environment.yml`
  - conda environment specification with all dependencies and versions.

## Main Pipeline Components

### 1. Preprocessing

The preprocessing pipeline formats raw peptide sequences into a cleaned and tokenizable dataset for ProtGPT2.

Primary entrypoints:
- `preprocessing/sequences_for_conditional_generator_preprocessing.ipynb`
- `preprocessing/sequence_preprocessing_utils.py`

Key preprocessing steps:
- Load peptide CSV data.
- Standardize and validate sequences.
- Remove excluded or duplicate peptides.
- Filter CPP sequences (`is_cpp == 1`).
- Group rare cell lines into `other`.
- Optionally oversample minority cell-line groups.
- Format sequences with ProtGPT2 markers and line breaks for `generation_train_dataset_formatted.csv`.

### 2. Active Sampling and Classifier Training

The active sampling pipeline is designed to select valuable peptide candidates for labeling and model refinement.

Primary entrypoint:
- `active_sampling/with_active_sampling.py`

Capabilities include:
- feature extraction and data splitting
- Random Forest training
- calibrated probability estimation
- core-set selection, hard-negative mining, and margin sampling
- budgeted batch selection for active learning

Configuration is controlled by:
- `config/active_learning_config.json`
- `config/rf_model_config.json`

### 3. Soft-Prompt Training

The soft-prompt model is trained using ProtGPT2 as a frozen base model and a trainable prefix generator.

Primary entrypoint:
- `training/train_generator.py`

Highlights:
- Loads `nferruz/ProtGPT2` by default.
- Converts cell-line conditions into a binary condition vector.
- Builds a small MLP to produce prefix embeddings.
- Trains only the prefix generator while freezing the base model.
- Uses `data/processed/generation_train_dataset_formatted.csv` as the training dataset.

### 4. Conditional Sequence Generation

Generated CPP sequences are produced by combining the trained soft prompt with ProtGPT2 and conditioning on cell lines.

**Generation Scripts:**

1. **`cond_gen_soft_prompt_active_learning_only_gen.py`** 
   - Base variant for generating sequences for specified or random cell lines.
   - Arguments: `--cell_line` (list), `--num_per_cell_line`, `--gen_batch_size`
   - Supports single or multiple sequences per cell line.
   - Used for generating `generated_cpp_sequences_multi_line_active_learning_3500_ends_no_overlap_seed_temp_1_rep_1_1.csv`

2. **`cond_gen_soft_prompt_active_learning_only_gen_100.py`**
   - Extended generation for high-volume sequence production for specific cell lines (default 100 per cell line).
   - Arguments: `--cell_lines` (list), `--sequences_per_cell_line` (default 100)
   - Optimized for larger batches with length filtering (`--min_aa_length`, `--max_aa_length`).

3. **`cond_gen_soft_prompt_active_learning_only_gen_100_cpp_only.py`** (active-learning variant)
   - Wrapper for generating only valid CPP sequences after classifier validation, without duplicates.
   - Arguments: `--target_cpp_per_line` (default 100), `--initial_multiplier`, `--max_iterations`
   - Filters output and re-generates to reach target CPP counts.
   - Shell wrapper: `run_pipeline_arg_soft_prompt_active_learning_100_cpp_only.sh`
   - Used for generating `final_100_cpp_per_line_seed_1_1.csv`

**Common Generation Parameters:**
- `--ckpt_path`: Path to trained soft-prompt checkpoint (`.pt`)
- `--temperature` (default 1.0): Sampling temperature for diversity
- `--top_k` (default 50): Top-k sampling threshold
- `--top_p` (default 1.0): Nucleus (top-p) sampling
- `--repetition_penalty` (default 1.1): Penalty for repeating n-grams
- `--max_new_tokens` (default 50): Maximum sequence length per generation
- `--gen_batch_size`: Batch size for generation
- `--prompt_text`: Optional starting amino acids
- `--save_csv` (default True), `--save_jsonl`: Output formats
- `--out_dir`: Output directory for generated sequences and metadata

**Outputs:**
- CSV file with sequence, cell line, and metadata (e.g., `results/generated_sequences.csv`)
- Optional JSONL file with additional generation details
- Sequence filtering includes length validation and optional CPP classification

### 5. Validation and Results

Validation scripts and notebooks help analyze generated sequences and classifier performance.
- `validation/internal/run_pipeline.py`
- `results/visualization.ipynb`
- `results/*.csv`

## Configuration

Project parameters are stored in JSON config files:
- `config/params_generation.json`
- `config/active_learning_config.json`
- `config/rf_model_config.json`

These files control model paths, generation settings, active learning budgets, classifier params, and training hyperparameters.

## Requirements

This repository uses a conda environment specified in `environment.yml`. Key components include:
- Python 3.12.11
- Deep learning: `torch`, `torchvision`, `torchaudio` (with CUDA 12.4)
- NLP: `transformers`, `huggingface-hub`, `tokenizers`, `accelerate`
- Data science: `pandas`, `numpy`, `scipy`, `scikit-learn`, `joblib`
- Cheminformatics: `rdkit`, `biopython`
- Visualization: `matplotlib`, `seaborn`, `plotly`
- Utilities: `tqdm`, `pyyaml`, `requests`, `regex`, `pandarallel`, `pyarrow`
- Hyperparameter tuning: `optuna`

### Setup

Create the conda environment using:
```bash
conda env create -f environment.yml
conda activate cppgen_env
```

## Quick Start

1. Set up the environment:
   ```bash
   conda env create -f environment.yml
   conda activate cppgen_env
   ```

### Training

2. Prepare data:
   - Place raw peptide data in `data/raw/`
   - Use the preprocessing notebook to build `data/processed/generation_train_dataset_formatted.csv`

3. Train the soft prompt:
   - Run `python training/train_generator.py`

### Generation

4. Generate sequences:
   - Run `python generation/cond_gen_soft_prompt_active_learning_only_gen_100_cpp_only.py --ckpt_path models/cpp_soft_prompt_model_active_learning.pt`
   - Generated sequences appear in `results/`.

5. Review outputs:
   - Use the `validation/internal/run_pipeline.py` script and `results/visualization.ipynb` notebook for analysis


## Notes

- The repository is built around ProtGPT2 and custom soft-prompt conditioning with cell-line metadata.
- The active sampling module is designed for iterative dataset expansion and classifier-guided selection.
- Training and generation currently assume a CUDA-enabled environment if available.
