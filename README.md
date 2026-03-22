# Essay Discourse Labeler

An open-source NLP project for essay discourse segmentation and labeling. The
repository is structured as a professional Python package with
configuration-driven training, evaluation, and inference workflows.

## Why this repository exists

The original project was notebook-first and tightly coupled to a one-off
runtime. This version makes the same core ideas reusable:

- token classification with Transformer backbones
- BIO discourse labels
- fold-based training
- overlap-based span scoring
- thresholded post-processing
- single-model or multi-checkpoint inference

## Project structure

```text
.
├── configs/                # YAML experiment configs
├── data/                   # Local data root (ignored by git)
├── docs/                   # Additional documentation
├── scripts/                # Convenience wrappers
├── src/essay_labeler/      # Package source
├── tests/                  # Unit and smoke tests
└── artifacts/              # Checkpoints and predictions (ignored by git)
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Data setup

This repository does not publish third-party datasets or model artifacts.

1. Prepare essay text files and annotation CSVs in your local workspace.
2. Place the essay text files under:
   - `data/essay-labeler/train/`
   - `data/essay-labeler/test/`
3. Place the annotation CSV under `data/essay-labeler/train.csv`, or point the
   config at an externally stored annotation file.
4. Update the paths in [`configs/base.yaml`](/Users/wangmuyao/Documents/Projects/nlp/configs/base.yaml).

## Training

```bash
essay-labeler train --config configs/base.yaml
```

This performs training and writes checkpoints plus metrics under
`artifacts/`.

## Evaluation

```bash
essay-labeler evaluate \
  --config configs/base.yaml \
  --checkpoint artifacts/checkpoints/fold_0/model.pt
```

This loads a checkpoint, runs validation inference, and reports span-level F1
using the configured overlap metric.

## Prediction

```bash
essay-labeler predict \
  --config configs/base.yaml \
  --checkpoint artifacts/checkpoints/fold_0/model.pt \
  --input-dir data/essay-labeler/test \
  --output artifacts/predictions/predictions.csv
```

You can also pass multiple `--checkpoint` values to average logits across folds.

## Configuration

The main config controls:

- data paths
- model backbone and tokenizer
- max sequence lengths
- training hyperparameters
- validation split settings
- post-processing thresholds
- artifact output locations

CLI flags can override selected config values with `--set key=value`.

## Results

The current default configuration uses a Transformer backbone plus
threshold-based post-processing for essay discourse labeling. The repository is
organized to make model behavior, preprocessing, and evaluation explicit rather
than hiding them in notebooks.

## Architecture notes

See [`docs/architecture.md`]
for the module breakdown and data flow.
