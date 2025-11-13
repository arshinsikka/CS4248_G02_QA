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

### Comprehensive Search (Top 2, 3, 5 Candidates)

- `hparam_search_top2_3_5_biencoder.csv` - Bi-encoder hyperparameter search results for top 2, 3, and 5 candidates (CSV format)
- `hparam_search_top2_3_5_biencoder.txt` - Bi-encoder search results summary table (text format)
- `hparam_search_top2_3_5_biencoder_output.txt` - Full bi-encoder search output with live progress logs
- `hparam_search_top2_3_5_cross.csv` - Cross-encoder hyperparameter search results for top 2, 3, and 5 candidates (CSV format)
- `hparam_search_top2_3_5_cross.txt` - Cross-encoder search results summary table (text format)
- `hparam_search_top2_3_5_cross_output.txt` - Full cross-encoder search output with live progress logs

These files contain grid search results across alpha (0.3-0.6) and min_gap (0.05-0.25) parameters for all three candidate set sizes.

## Comparison Statistics

- `rerank_change_stats.json` - Comparison of reranked vs. original predictions (change breakdown)

## Experimental Folder

The `experimental/` subfolder contains intermediate results from various experiments:
- Different epoch configurations (1 epoch, 3 epochs)
- Different gradient accumulation settings (GA=4)
- Early reranking attempts (global reranking, different thresholds)
- Cross-encoder experiments that didn't improve results

These are kept for completeness but **not needed to reproduce the main results**.

