# Architecture

## Modules

- `config.py`: YAML loading plus CLI overrides.
- `labels.py`: BIO labels and mappings.
- `data.py`: essay ingestion, annotation loading, fold assignment, and entity alignment.
- `dataset.py`: tokenizer-driven dataset for token classification.
- `modeling.py`: Transformer backbone plus token classification head.
- `metrics.py`: overlap scoring for span matching.
- `postprocess.py`: threshold-based span extraction.
- `pipeline.py`: prediction aggregation and export formatting.
- `train.py`: train, evaluate, and predict orchestration.
- `cli.py`: command-line entrypoints.

## Data flow

1. Load config.
2. Read essay text files and annotation CSVs.
3. Convert per-essay annotations into BIO token labels.
4. Build datasets and dataloaders with tokenizer-to-word alignment.
5. Train or load the model.
6. Convert logits back to word-level labels.
7. Apply post-processing thresholds.
8. Score on validation data or export prediction files.

## Design choices

- Paths are config-driven instead of hard-coded for a specific runtime.
- The package keeps scoring and post-processing logic explicit rather than hiding it in notebook cells.
- Tests rely on synthetic fixtures so CI does not depend on external datasets.
