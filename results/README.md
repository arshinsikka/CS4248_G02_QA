# Results Folder Organization

## Main Results (Essential Files)

These files are needed to reproduce the main results:

- `results_acc2_best_original.json` - Baseline RoBERTa results (EM=84.28, F1=90.93)
- `best_results_top2_alpha05_gap005_bienc.json` - **Best bi-encoder reranked results** (EM=84.40, F1=91.04)
- `results_top2_alpha05_gap005_cross.json` - Best cross-encoder reranked results
- `results_top2_alpha05_gap02_bienc.json` - Alternative bi-encoder configuration
- `results_top2_alpha05_gap02_cross.json` - Alternative cross-encoder configuration

## Analysis Files

- `candidate_score_stats.json` - Score difference statistics between candidates (mean, median, percentiles)
- `gold_position_counts.json` - Distribution of gold answer positions (rank 1-5)
- `results_k_position_stats.json` - Position-based evaluation metrics

## Hyperparameter Search Results

- `hparam_search.csv` - Main hyperparameter grid search results (all configurations, CSV format)
- `hparam_search_02_biencoder.txt` - Bi-encoder search results (text format)
- `hyperparameter_search.txt` - Cross-encoder search results (text format)

## Comparison Statistics

- `rerank_change_stats.json` - Comparison of reranked vs. original predictions (change breakdown)

## Experimental Folder

The `experimental/` subfolder contains intermediate results from various experiments:
- Different epoch configurations (1 epoch, 3 epochs)
- Different gradient accumulation settings (GA=4)
- Early reranking attempts (global reranking, different thresholds)
- Cross-encoder experiments that didn't improve results
- Intermediate hyperparameter search outputs

These are kept for completeness but **not needed to reproduce the main results**.

