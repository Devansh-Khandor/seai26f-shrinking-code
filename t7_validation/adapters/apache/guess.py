#!/usr/bin/env python3
"""Apache T7 Guess adapter for the unchanged common prudence checker.

Uses only human_t7/apache/{remember,guess}.py for calculations; never
uses the checker or reference oracle to construct predictions or intervals.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'human_t7' / 'apache'))

from remember import GOAL, adjustable_settings, load_configurations  # noqa: E402
from guess import (  # noqa: E402
    estimate_pair_difference,
    evaluate_predictions,
    one_setting_pairs,
    predict_performance,
)


def answers(csv_path):
    rows = load_configurations(csv_path)
    settings = adjustable_settings(rows)

    # The COMMON evaluation protocol, not T7's original random split:
    # every 5th file row is held out; its Performance- cannot train kNN.
    train = [r for r in rows if r['row_number'] % 5 != 0]
    test = [r for r in rows if r['row_number'] % 5 == 0]
    if not train or not test:
        raise ValueError('Need training and held-out configurations')

    predictions = {
        str(row['row_number']): predict_performance(row, train, settings)['prediction']
        for row in test
    }
    mae = statistics.mean(
        abs(predictions[str(row['row_number'])] - row[GOAL])
        for row in test
    )

    # APACHE-04 asks about historically observed valid, one-setting pairs.
    # Choose by *input settings only*: outcomes are never used to select a pair.
    # Use the first ten deterministic eligible pairs to avoid cherry-picking
    # favorable realized differences or uncertainty coverage.
    candidates = one_setting_pairs(rows, settings)
    if len(candidates) < 10:
        raise ValueError('Need at least ten one-setting observed pairs')
    reported_pairs = []
    for a, b, setting in candidates[:10]:
        result = estimate_pair_difference(a, b, train, settings)
        estimate_a = result['estimate_a']
        estimate_b = result['estimate_b']

        # Conservative, descriptive bounds derived ONLY from the neighbors'
        # observed ranges, never from either pair's realized difference.
        # This is not a calibrated frequentist prediction interval.
        low = estimate_b['neighbor_min'] - estimate_a['neighbor_max']
        high = estimate_b['neighbor_max'] - estimate_a['neighbor_min']
        reported_pairs.append({
            'a': a['row_number'],
            'b': b['row_number'],
            'setting': setting,
            'difference': result['predicted_difference'],
            'low': low,
            'high': high,
        })

    return {'predictions': predictions, 'mae': mae}, {'pairs': reported_pairs}


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: guess.py PATH_TO_APACHE_CSV')
    prediction, pairs = answers(Path(sys.argv[1]))
    print('== APACHE-03 ==')
    print('Five-nearest-neighbor performance predictions')
    print(f"Predictions for {len(prediction['predictions'])} held-out configurations")
    print(f"Held-out MAE: {prediction['mae']:.4f}")
    print('ANSWER APACHE-03 ' + json.dumps(prediction, sort_keys=True))
    print('== APACHE-04 ==')
    print('Observed pairs that differ in exactly one setting')
    print(f"Pairs reported: {len(pairs['pairs'])}")
    print('Uncertainty ranges derive from historical neighbor extrema')
    print('ANSWER APACHE-04 ' + json.dumps(pairs, sort_keys=True))


if __name__ == '__main__':
    main()
