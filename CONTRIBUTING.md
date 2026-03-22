# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Workflow

1. Create a feature branch.
2. Keep changes scoped and documented.
3. Run `pytest` before opening a pull request.
4. Include config or CLI changes in the README when relevant.

## Data and artifacts

Do not commit proprietary datasets, model checkpoints, or large generated artifacts.
Use the documented `data/` and `artifacts/` directories locally.
