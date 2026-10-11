# NLP-task1

This repository is a cleaned, refactored reimplementation of a lightweight text-classification assignment inspired by the original project architecture. The goal is to compare several document representation methods on a shared split and validate the final metrics.

The project is intentionally structured to be easy to understand and quick to run on a compact sample dataset while preserving the same overall logic flow:

- data split and validation
- feature extraction with bag-of-words and embedding-style vectors
- model selection on a held-out validation set
- final evaluation on a test set
- result verification and report generation

## Project layout

- `experiment.py` — train and evaluate the classification pipeline
- `validate_results.py` — recompute metrics and verify output consistency
- `report.py` — generate a simple PDF summary from the saved results
- `run_all.py` — execute the end-to-end workflow
- `sample_data/news_sample.csv` — compact demo dataset for quick execution
- `results/` — generated outputs from experiments

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run_all.py
```

## Notes

- The sample dataset is intentionally small so the project can be tested quickly.
- The code still follows the same high-level workflow as the original assignment: split -> train -> validate -> evaluate -> report.
- The BERT branch uses a lightweight fallback if the full pretrained model cannot be downloaded in the current environment.
