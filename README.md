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

Primary entrypoints:
- `generation/cond_gen_soft_prompt_active_learning_only_gen.py`
- `generation/run_pipeline_arg_soft_prompt_active_learning_100_cpp_only.sh`

Generation options include:
- conditioned cell-line generation
- random generation across cell lines
- sampling temperature, `top_k`, and `top_p`
- prompt text injection
- saving outputs to CSV/JSONL

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

This repository depends on Python packages used across preprocessing, modeling, and validation. Key dependencies include:
- `pandas`
- `numpy`
- `torch`
- `transformers`
- `scikit-learn`
- `tqdm`
- `joblib`
- `rdkit`
- `torchvision` / `matplotlib` / `seaborn` (for notebooks and visualization)

## Quick Start

1. Prepare data:
   - Place raw peptide data in `data/raw/`
   - Use the preprocessing notebook or utility scripts to build `data/processed/generation_train_dataset_formatted.csv`

2. Train the soft prompt:
   - Run `python training/train_generator.py`

3. Generate sequences:
   - Run `python generation/cond_gen_soft_prompt_active_learning_only_gen.py --ckpt_path models/cpp_soft_prompt_model_active_learning.pt`

4. Review outputs:
   - Generated sequences and metrics appear in `results/` and `data/processed/`.

## Notes

- The repository is built around ProtGPT2 and custom soft-prompt conditioning with cell-line metadata.
- The active sampling module is designed for iterative dataset expansion and classifier-guided selection.
- Training and generation currently assume a CUDA-enabled environment if available.
