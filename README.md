## CS4248 G02 — Extractive Question Answering (SQuAD v1.1)

**[⬇️ Download best model checkpoint (acc2)](https://www.dropbox.com/scl/fi/2j2k8z8u7eh9z7qijtnq2/roberta_base_d2e5_wd01_ep2_acc2.zip?rlkey=aigdofz0snfq98hkqdm4o9y48&dl=1)**

This repo trains and evaluates an extractive QA model using RoBERTa on SQuAD v1.1. The typical flow is:

1) Fine-tune RoBERTa using `src/fine_tune_roberta.py` → saves a model folder under `models/`
2) Run inference with `src/evaluate.py` → writes `predictions/predictions_*.json`
3) Score with the official SQuAD script `src/evaluate-v2.0.py` → writes `results/results_*.json`

Folders used:
- `data/`: SQuAD data (`train-v1.1.json`, `dev-v1.1.json`)
- `models/`: fine-tuned models (each run in its own subfolder)
- `predictions/`: model predictions JSONs
- `results/`: EM/F1 metrics from the SQuAD scorer

At the moment, the best checkpoint is `models/roberta_base_d2e5_wd01_ep2_acc2` ("acc2"). Later, we will not commit model weights (too large); the README will include a download link for the best model checkpoint.

## Environment

Create a venv and install deps (CPU or GPU):
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
# CPU-only minimal:
pip install transformers datasets evaluate
# GPU (CUDA 12.1) + extras:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets evaluate accelerate
```

## Fine-tuning

Script: `src/fine_tune_roberta.py`

Key arguments:
- `--model_name` (default: `roberta-base`): base model to fine-tune (e.g., `distilroberta-base`).
- `--train_path`/`--dev_path`: paths to SQuAD files (default to `../data/...`, pass explicit `./data/...` if needed).
- `--output_dir`: where the trained model will be saved (a folder under `models/`).
- `--epochs`, `--train_batch`, `--eval_batch`, `--learning_rate`, `--weight_decay`.
- `--gradient_accumulation`: simulate larger batch with limited VRAM.
- `--max_train`, `--max_eval`: set `-1` to use full datasets; otherwise cap for quick runs (e.g., 2000/500).

Examples:
```bash
# Quick subset sanity check (CPU or GPU)
python src/fine_tune_roberta.py \
  --model_name roberta-base \
  --output_dir ./models/roberta_base_subset \
  --epochs 2 --train_batch 8 --eval_batch 8 \
  --max_train 2000 --max_eval 500 \
  --train_path ./data/train-v1.1.json --dev_path ./data/dev-v1.1.json

# Full-data training (GPU recommended)
CUDA_VISIBLE_DEVICES=0 python src/fine_tune_roberta.py \
  --model_name roberta-base \
  --output_dir ./models/roberta_base_full \
  --epochs 3 --train_batch 16 --eval_batch 16 \
  --gradient_accumulation 1 \
  --max_train -1 --max_eval -1 \
  --train_path ./data/train-v1.1.json --dev_path ./data/dev-v1.1.json
```

Notes:
- Mixed precision is enabled automatically on supported GPUs (bf16 on A100/H100, fp16 as fallback).
- If you see CUDA OOM, reduce `--train_batch` and increase `--gradient_accumulation`.

## Inference (Generate predictions)

Script: `src/evaluate.py`

This script runs the fine-tuned model on the full SQuAD dev set and writes a predictions JSON mapping `qid -> answer`.

Usage (defaults to `models/roberta_base_full` and writes `predictions/predictions.json` if you pass `--out_file`):
```bash
CUDA_VISIBLE_DEVICES=0 python src/evaluate.py \
  --model_path ./models/roberta_base_full \
  --dev_file ./data/dev-v1.1.json \
  --out_file ./predictions/predictions_roberta_base_full.json
```

If you omit `--out_file`, it writes to `predictions.json` in the repo root; you can move it afterwards:
```bash
mkdir -p predictions
mv predictions.json predictions/predictions.json
```

## Scoring (EM/F1)

Script: `src/evaluate-v2.0.py` (official SQuAD evaluator; works for v1.1 predictions)

Example:
```bash
python src/evaluate-v2.0.py \
  data/dev-v1.1.json \
  predictions/predictions_roberta_base_full.json \
  --out-file results/roberta_base_full.json
```

The result file contains `exact`, `f1`, and counts. You can keep multiple results files for comparison.

## Best Model (current)

- Current best: `models/roberta_base_d2e5_wd01_ep2_acc2` ("acc2"). Use it with `--model_path` in `src/evaluate.py` and score as shown above.
- Large model folders are not committed. We will host the best checkpoint externally and put the link here.

## External Model Download

We do not commit model weights to the repo due to size. A download link (Dropbox/Drive) for the best model will be added here:


**Model Download Links**

| Model Name                    | Download Link                                                                                       | Status (Usage)      |
|-------------------------------|-----------------------------------------------------------------------------------------------------|---------------------|
| Best ("acc2")                 | [Google Drive](https://drive.google.com/file/d/1XNCI0GWPADil13jA2u0uug43mSDTsnA6/view?usp=sharing) | **Recommended**     |
| Old (not recommended)         | [Google Drive (old model)](https://drive.google.com/file/d/1tv1_8kLF8VFFhWBLA4v7UYb3A8380RBJ/view?usp=sharing) | **DO NOT USE**      |

**Note:**  
- The "acc2" checkpoint is the official best checkpoint and should be used for all experiments.
- The "old model" is retained for reference only and should **not be used in experiments or submissions**.


