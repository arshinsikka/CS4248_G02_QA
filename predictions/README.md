# Predictions Folder Organization

## Essential Files

### Baseline Predictions
- `predictions_acc2.json` - Single best predictions from baseline RoBERTa model

### Candidate Files (Before Reranking)
Located in `candidates/` subfolder:
- `predictions_with_2_acc2.json` - Top-2 candidates per question (used for final reranking)
- `predictions_with_5_acc2.json` - Top-5 candidates per question (for analysis)

These files contain multiple candidates per question with scores and positions.

### Reranked Predictions (After Reranking)
Located in `reranked/` subfolder:
- `predictions_top2_alpha05_gap005_bienc.json` - **Best bi-encoder reranked predictions** (final result, EM=84.40, F1=91.04)
- `predictions_top2_alpha05_gap005_cross.json` - Best cross-encoder reranked predictions
- `predictions_top2_alpha05_gap02_bienc.json` - Alternative bi-encoder configuration
- `predictions_top2_alpha05_gap02_cross.json` - Alternative cross-encoder configuration

These files contain single best answer per question (same format as baseline).

## Folder Structure

- `candidates/` - Top-k candidate files (multiple answers per question)
- `reranked/` - Final reranked predictions (single answer per question)
- `experimental/` - Intermediate experimental predictions (not needed for reproduction)

## File Format

**Candidate files** (`predictions_with_*.json`):
```json
{
  "question_id": [
    {"text": "answer1", "score": 0.95, "start": 10, "end": 20},
    {"text": "answer2", "score": 0.80, "start": 30, "end": 40}
  ]
}
```

**Reranked files** (`predictions_*.json`):
```json
{
  "question_id": "answer text"
}
```

