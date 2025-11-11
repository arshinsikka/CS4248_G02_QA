# Margin-Triggered Reranking for Extractive Question Answering on SQuAD v1.1

**CS4248 Group G02** - National University of Singapore

## Overview

This project studies extractive question answering on SQuAD v1.1 using a RoBERTa-base model with a margin-triggered reranking layer. We fine-tune RoBERTa to achieve a strong baseline (84.28 EM, 90.93 F1), then introduce a lightweight bi-encoder reranker that only operates when the baseline model's confidence margin between top candidates is small. Our final system improves to **84.40 EM and 91.04 F1** while modifying only 55 out of 10,570 predictions (0.5%).

### Key Contributions

- **Margin-triggered reranking**: Only rerank when baseline score margin is small (threshold-based)
- **Top-2 candidate focus**: Reranking operates on only the top two spans for robustness
- **Bi-encoder reranker**: Uses sentence-transformers for efficient semantic similarity scoring
- **Comprehensive analysis**: Detailed candidate distribution and margin analysis tools

## Model Download

**Fine-tuned RoBERTa-base model (4.5GB):** [Download from Google Drive](https://drive.google.com/file/d/1XNCI0GWPADil13jA2u0uug43mSDTsnA6/view?usp=sharing)

**Automatic Download:** The `run_complete_workflow.py` script will automatically download and extract the model if it's not found. No manual download needed when using the workflow script!

**Manual Download:** If you prefer to download manually, extract the zip file to `models/roberta_base_d2e5_wd01_ep2_acc2/`. This is the baseline model used for all experiments.

## Setup

### Step 1: Install Python

This project requires **Python 3.8 or higher**. If you don't have Python installed:

**macOS:**
```bash
# Using Homebrew (recommended)
brew install python3

# Or download from https://www.python.org/downloads/
```

Verify installation:
```bash
python3 --version  # Should show Python 3.8 or higher
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

**For GPU support (CUDA 12.1):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Project Structure

```
CS4248_G02_QA/
├── data/                          # SQuAD v1.1 dataset
│   ├── train-v1.1.json           # 87,599 training questions
│   └── dev-v1.1.json             # 10,570 development questions
├── models/                        # Fine-tuned models (download from Drive)
├── predictions/                   # Prediction files (many experimental files)
├── results/                      # Evaluation results (many experimental files)
└── src/                          # Source code
    ├── run_complete_workflow.py   # ⚡ Automated complete workflow script
    ├── fine_tune_roberta.py      # Fine-tune RoBERTa on SQuAD
    ├── evaluate.py                # Generate single best predictions
    ├── evaluate_k_candidates.py   # Generate top-k candidate predictions
    ├── evaluate-v2.0.py          # Official SQuAD evaluator
    ├── rerank_squad_candidates.py # Global reranking (initial implementation)
    ├── rerank_squad_candidates_threshold.py  # FINAL: Margin-triggered reranking
    ├── search_rerank_hyperparams.py  # Hyperparameter grid search
    ├── compare_rerank_stats.py   # Compare reranked vs original predictions
    ├── calculate_candidate_stats.py  # Score difference statistics
    ├── count_gold_candidate_positions.py  # Gold answer position analysis
    └── evaluate_k_position_stats.py  # Position-based evaluation metrics
```

**Note:** The `predictions/` and `results/` folders contain many files generated during experimentation and hyperparameter search. Only a few key files are needed to reproduce the main results.

## Source Code Files

### Complete Workflow

#### `run_complete_workflow.py` ⚡ **RECOMMENDED**
Automated script that runs the complete pipeline: candidate generation → reranking → evaluation. Uses best default parameters from hyperparameter search.

**Usage:**
```bash
# Use best defaults (top_k=2, alpha=0.5, min_gap=0.05)
python src/run_complete_workflow.py

# Custom parameters
python src/run_complete_workflow.py --top_k 3 --alpha 0.6 --min_gap 0.1

# Skip candidate generation (use existing file)
python src/run_complete_workflow.py --skip_candidates --nbest_file predictions/candidates/predictions_with_2_acc2.json
```

**Key features:**
- **Automatic model download**: Downloads the fine-tuned model from Google Drive if not found (4.5GB)
- Automatically generates top-k candidates
- Applies margin-triggered reranking with configurable parameters
- Evaluates results and displays final metrics
- Creates necessary output directories
- Uses best default parameters from hyperparameter search

### Core Training & Evaluation

#### `fine_tune_roberta.py`
Fine-tunes RoBERTa-base on SQuAD v1.1. Supports configurable epochs, batch size, gradient accumulation, and learning rate.

**Usage:**
```bash
python src/fine_tune_roberta.py \
  --model_name roberta-base \
  --train_path data/train-v1.1.json \
  --dev_path data/dev-v1.1.json \
  --output_dir models/roberta_base_d2e5_wd01_ep2_acc2 \
  --epochs 2 \
  --train_batch 8 \
  --gradient_accumulation 2 \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --max_train -1 \
  --max_eval -1
```

#### `evaluate.py`
Generates single best predictions for each question using the fine-tuned model.

**Usage:**
```bash
python src/evaluate.py \
  --model_path models/roberta_base_d2e5_wd01_ep2_acc2 \
  --dev_file data/dev-v1.1.json \
  --out_file predictions/predictions_baseline.json
```

#### `evaluate_k_candidates.py`
Generates top-k candidate predictions with scores and positions. Used to extract candidate lists for reranking. Output files should be saved to `predictions/candidates/` folder.

**Usage:**
```bash
python src/evaluate_k_candidates.py \
  --model_path models/roberta_base_d2e5_wd01_ep2_acc2 \
  --dev_file data/dev-v1.1.json \
  --out_file predictions/candidates/predictions_with_5_acc2.json \
  --top_k 5
```

#### `evaluate-v2.0.py`
Official SQuAD evaluation script. Computes exact match (EM) and F1 scores.

**Usage:**
```bash
python src/evaluate-v2.0.py \
  data/dev-v1.1.json \
  predictions/predictions_baseline.json \
  --out-file results/results_baseline.json
```

### Reranking Implementation

#### `rerank_squad_candidates.py`
Initial implementation of global reranking (reranks all candidates for all questions). This was our first approach before introducing margin-triggered reranking.

**Usage:**
```bash
python src/rerank_squad_candidates.py \
  --dev_file data/dev-v1.1.json \
  --nbest_file predictions/candidates/predictions_with_5_acc2.json \
  --out_file predictions/reranked/predictions_reranked_global.json \
  --alpha 0.7 \
  --reranker_type bi_encoder \
  --model_name sentence-transformers/all-MiniLM-L6-v2
```

#### `rerank_squad_candidates_threshold.py` ⭐ **FINAL METHOD**
Margin-triggered reranking: only reranks when the score margin between top-2 candidates is below a threshold. This is our final, best-performing approach. Output files should be saved to `predictions/reranked/` folder.

**Usage:**
```bash
python src/rerank_squad_candidates_threshold.py \
  --dev_file data/dev-v1.1.json \
  --nbest_file predictions/candidates/predictions_with_2_acc2.json \
  --out_file predictions/reranked/predictions_top2_alpha05_gap005_bienc.json \
  --alpha 0.5 \
  --min_gap 0.05 \
  --normalize minmax \
  --candidate_text_mode answer_is \
  --cap_topk 2 \
  --reranker_type bi_encoder \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --batch_size 32
```

**Key parameters:**
- `--alpha`: Weight for baseline score vs. reranker score (0.5 = equal weight)
- `--min_gap`: Margin threshold - only rerank if `score1 - score2 < min_gap`
- `--reranker_type`: `bi_encoder` or `cross_encoder`

### Analysis & Hyperparameter Search

#### `search_rerank_hyperparams.py`
Grid search over hyperparameters (alpha, min_gap, top-k) with real-time progress updates and CSV output. Can search over multiple top-k candidate files simultaneously.

**Usage:**
```bash
# Search over single top-k file
python src/search_rerank_hyperparams.py \
  --dev_file data/dev-v1.1.json \
  --nbest top2=predictions/candidates/predictions_with_2_acc2.json \
  --alpha_start 0.3 --alpha_end 0.6 --alpha_step 0.1 \
  --mingap_start 0.05 --mingap_end 0.25 --mingap_step 0.05 \
  --reranker_type bi_encoder \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --results_csv results/hparam_search.csv \
  --results_table results/hparam_search_table.txt

# Search over multiple top-k files simultaneously
python src/search_rerank_hyperparams.py \
  --dev_file data/dev-v1.1.json \
  --nbest top2=predictions/candidates/predictions_with_2_acc2.json \
  --nbest top3=predictions/candidates/predictions_with_3_acc2.json \
  --nbest top5=predictions/candidates/predictions_with_5_acc2.json \
  --alpha_start 0.2 --alpha_end 0.8 --alpha_step 0.1 \
  --mingap_start 0.15 --mingap_end 0.35 --mingap_step 0.05 \
  --reranker_type bi_encoder \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --results_csv results/hparam_search_all.csv
```

**Note:** You can specify multiple `--nbest` arguments to search over different top-k candidate files (top-2, top-3, top-5, etc.) in a single run. The script will evaluate all combinations of hyperparameters for each candidate file.

#### `compare_rerank_stats.py`
Compares reranked predictions against original baseline. Reports how many predictions changed and whether changes improved or hurt accuracy.

**Usage:**
```bash
python src/compare_rerank_stats.py \
  --dev_file data/dev-v1.1.json \
  --nbest_file predictions/candidates/predictions_with_2_acc2.json \
  --reranked_file predictions/reranked/predictions_top2_alpha05_gap005_bienc.json \
  --out_file results/rerank_change_stats.json
```

#### `calculate_candidate_stats.py`
Computes statistics on score differences between candidates (mean, median, percentiles). Analyzes differences when gold is at rank 1 vs. rank 2.

**Usage:**
```bash
python src/calculate_candidate_stats.py \
  --dev_file data/dev-v1.1.json \
  --nbest_file predictions/candidates/predictions_with_5_acc2.json \
  --out_file results/candidate_score_stats.json
```

#### `count_gold_candidate_positions.py`
Counts how often the gold answer appears at each rank (1, 2, 3, 4, 5, or not in top-5).

**Usage:**
```bash
python src/count_gold_candidate_positions.py \
  --dev_file data/dev-v1.1.json \
  --nbest_file predictions/candidates/predictions_with_5_acc2.json \
  --max_k 5 \
  --out_file results/gold_position_counts.json
```

#### `evaluate_k_position_stats.py`
Evaluates performance metrics broken down by candidate position (e.g., what if we always picked rank 2?).

**Usage:**
```bash
python src/evaluate_k_position_stats.py \
  --dev_file data/dev-v1.1.json \
  --nbest_file predictions/candidates/predictions_with_5_acc2.json \
  --out_file results/results_k_position_stats.json
```

## Complete Workflow

### Quick Start: Automated Workflow Script ⚡

The easiest way to run the complete pipeline is using the automated workflow script:

```bash
# Use best default parameters (top_k=2, alpha=0.5, min_gap=0.05)
python src/run_complete_workflow.py

# Custom parameters
python src/run_complete_workflow.py --top_k 3 --alpha 0.6 --min_gap 0.1

# Skip candidate generation (use existing file)
python src/run_complete_workflow.py --skip_candidates --nbest_file predictions/candidates/predictions_with_2_acc2.json

# Use cross-encoder instead of bi-encoder
python src/run_complete_workflow.py --reranker_type cross_encoder
```

**Default parameters (best from hyperparameter search):**
- `top_k`: 2
- `alpha`: 0.5
- `min_gap`: 0.05
- `reranker_type`: bi_encoder

The script automatically:
1. **Downloads the model** (if not found) from Google Drive (4.5GB)
2. Generates top-k candidate predictions
3. Applies margin-triggered reranking
4. Evaluates results and displays metrics

**Note:** On first run, the script will download the model (~4.5GB). This may take several minutes depending on your internet connection. The model is saved to `models/roberta_base_d2e5_wd01_ep2_acc2/` and won't be downloaded again on subsequent runs.

See `python src/run_complete_workflow.py --help` for all options.

### Manual Step-by-Step Workflow

If you prefer to run each step manually (or if you want more control):

### Step 1: Download Model (Optional)
The automated workflow script will download the model automatically. If you prefer to download manually:

Download the fine-tuned model from [Google Drive](https://drive.google.com/file/d/1XNCI0GWPADil13jA2u0uug43mSDTsnA6/view?usp=sharing) and extract to `models/roberta_base_d2e5_wd01_ep2_acc2/`.

### Step 2: Generate Top-K Candidates
```bash
python src/evaluate_k_candidates.py \
  --model_path models/roberta_base_d2e5_wd01_ep2_acc2 \
  --dev_file data/dev-v1.1.json \
  --out_file predictions/candidates/predictions_with_2_acc2.json \
  --top_k 2
```

### Step 3: Apply Margin-Triggered Reranking
```bash
python src/rerank_squad_candidates_threshold.py \
  --dev_file data/dev-v1.1.json \
  --nbest_file predictions/candidates/predictions_with_2_acc2.json \
  --out_file predictions/reranked/predictions_top2_alpha05_gap005_bienc.json \
  --alpha 0.5 \
  --min_gap 0.05 \
  --normalize minmax \
  --candidate_text_mode answer_is \
  --cap_topk 2 \
  --reranker_type bi_encoder \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --batch_size 32
```

### Step 4: Evaluate Results
```bash
python src/evaluate-v2.0.py \
  data/dev-v1.1.json \
  predictions/reranked/predictions_top2_alpha05_gap005_bienc.json \
  --out-file results/results_top2_alpha05_gap005_bienc.json
```

## Results Summary

| System | EM | F1 | Changed |
|--------|----|----|---------|
| Baseline (RoBERTa, 2 epochs, GA=2) | 84.28 | 90.93 | 0 |
| Margin-triggered rerank (top-2, α=0.5, gap<0.05) | 84.40 | 91.04 | 55 |

The reranker improves EM by 0.12 and F1 by 0.11 while modifying only 55 predictions (0.5% of the dataset). Among the 55 changes:
- 14 examples become correct (orig incorrect → new correct)
- 1 example becomes incorrect (orig correct → new incorrect)
- 40 examples remain incorrect (both incorrect)

## Key Findings

1. **95% coverage**: Gold answer appears in top-5 candidates for 95.1% of questions
2. **Margin correlation**: Small margins (score1 - score2) correlate with misranked questions
3. **Top-2 is optimal**: Reranking top-2 candidates outperforms top-3 or top-5
4. **Conservative reranking**: Margin-triggered approach is more robust than global reranking

## External Code & Libraries

- **Hugging Face Transformers**: Model fine-tuning and inference (`transformers` library)
- **Sentence-Transformers**: Bi-encoder reranking (`sentence-transformers` library)
- **Cross-Encoder models**: Cross-encoder reranking experiments (`sentence-transformers` with cross-encoder models)
- **Official SQuAD evaluator**: `evaluate-v2.0.py` (from SQuAD official repository)

All reranking logic, margin-triggered decision rules, hyperparameter search, and analysis tools are our own contributions.

## File Progression

Our implementation evolved as follows:

1. **Baseline**: `evaluate.py` - Single best prediction
2. **Candidate extraction**: `evaluate_k_candidates.py` - Top-k candidates with scores
3. **Global reranking**: `rerank_squad_candidates.py` - Rerank all candidates (initial approach)
4. **Margin-triggered reranking**: `rerank_squad_candidates_threshold.py` - **Final method** (only rerank when margin is small)

The analysis scripts (`calculate_candidate_stats.py`, `count_gold_candidate_positions.py`, etc.) were developed to understand candidate distributions and guide the reranking design.

## Notes on Experimental Files

The `predictions/` and `results/` folders contain many files generated during:
- Hyperparameter grid searches (multiple alpha/gap combinations)
- Different reranker types (bi-encoder vs. cross-encoder)
- Different top-k settings (top-2, top-3, top-5)
- Comparison experiments

Only a few key files are needed to reproduce the main results. The experimental files are retained for completeness and analysis.

## Requirements

See `requirements.txt` for full dependency list. Key dependencies:
- `torch` (PyTorch)
- `transformers` (Hugging Face)
- `sentence-transformers`
- `numpy`
- `datasets`


