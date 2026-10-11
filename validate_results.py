import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def validate_json_result(result_path: Path, predictions_path: Path | None = None):
    payload = load_json(result_path)
    test_metrics = payload.get('test', {})
    if not isinstance(test_metrics, dict):
        raise ValueError(f'{result_path.name}: missing valid "test" metrics section.')

    accuracy = float(test_metrics.get('accuracy', np.nan))
    macro_f1 = float(test_metrics.get('macro_f1', np.nan))
    if not np.isfinite(accuracy) or not np.isfinite(macro_f1):
        raise ValueError(f'{result_path.name}: invalid metrics -> {test_metrics}')

    if predictions_path is not None and predictions_path.exists():
        frame = pd.read_csv(predictions_path)
        required = {'true', 'predicted'}
        if required.issubset(frame.columns):
            y_true = frame['true'].astype(str).tolist()
            y_pred = frame['predicted'].astype(str).tolist()
            labels = sorted(set(y_true) | set(y_pred))
            recomputed_accuracy = float(accuracy_score(y_true, y_pred))
            recomputed_macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0))

            if not np.isclose(recomputed_accuracy, accuracy, atol=1e-8):
                raise ValueError(f'{result_path.name}: accuracy mismatch {accuracy} vs {recomputed_accuracy}')
            if not np.isclose(recomputed_macro_f1, macro_f1, atol=1e-8):
                raise ValueError(f'{result_path.name}: macro_f1 mismatch {macro_f1} vs {recomputed_macro_f1}')

    return {
        'method': payload.get('method', result_path.stem),
        'accuracy': accuracy,
        'macro_f1': macro_f1,
    }


def main():
    parser = argparse.ArgumentParser(description='Validate saved metrics against prediction CSV outputs.')
    parser.add_argument('--results-dir', type=Path, default=Path('results'))
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f'Results directory not found: {results_dir}')

    records = []
    for json_path in sorted(results_dir.glob('*.json')):
        if json_path.name in {'split.json', 'dataset.json'}:
            continue
        predictions_path = results_dir / f'{json_path.stem}_predictions.csv'
        records.append(validate_json_result(json_path, predictions_path))

    if not records:
        print(f'No experiment result files were found under {results_dir}.')
        return

    for item in records:
        print(f"{item['method']}: accuracy={item['accuracy']:.6f}, macro_f1={item['macro_f1']:.6f}")


if __name__ == '__main__':
    main()
