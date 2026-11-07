"""Run the Cars Remember and Guess milestones from the repository root."""

import argparse
from pathlib import Path
from .data import Table
from .ranking import rank_rows
from .stats import summarize
from .prediction import KNNRegressor
from .evaluation import evaluate, train_test_split


def main(argv=None):
    parser = argparse.ArgumentParser(description='Cars / Remember and Guess')
    parser.add_argument('--domain', required=True)
    parser.add_argument('--skill', required=True)
    args = parser.parse_args(argv)
    if args.domain != 'cars' or args.skill not in ('remember', 'guess'):
        parser.error('Unsupported domain or skill; only cars / remember and cars / guess are implemented')
    path = Path(__file__).resolve().parents[1] / 'data/raw/auto93.csv'
    table = Table.load(path)
    if args.skill == 'guess':
        guess(table)
        return
    print('Cars / Remember — CAR-01 and CAR-02')
    print(f'Records: {len(table.rows)}')
    for role, label in [('x', 'X'), ('y', 'Y'), ('ignored', 'Ignored')]:
        print(f'{label}: ' + ', '.join(c.name for c in table.role(role)))
    print('\nCAR-01: recorded goal summaries (best/worst follow header directions)')
    for c in table.role('y'):
        s = summarize(table, c)
        best, worst = ((s.minimum, s.maximum) if c.ideal == 0
                       else (s.maximum, s.minimum))
        print(f'{c.name}: n={s.count}, min={s.minimum:g}, max={s.maximum:g}, '
              f'mean={s.mean:.6f}, median={s.median:g}, best={best:g}, worst={worst:g}')
    ranked = rank_rows(table)
    print('\nCAR-02: top five historical outcomes; equal goal weights; lower d2h is better')
    print(f'Excluded for missing goals: {len(table.rows) - len(ranked)}')
    columns = table.role('x') + table.role('y')
    print('row ' + ' '.join(c.name for c in columns) + ' d2h')
    for row, score in ranked[:5]:
        values = [f'{row.cells[c.index]:g}' if c.numeric else str(row.cells[c.index])
                  for c in columns]
        print(f'{row.number} ' + ' '.join(values) + f' {score:.6f}')
    print('\nHistorical ranking, not prediction for unseen vehicles. '
          'Acc+ is maximized as recorded; its units and physical meaning are uncertain. '
          'Scores imply neither a lower purchase price nor faster physical acceleration.')



def guess(table):
    training, test = train_test_split(table)
    model = KNNRegressor.fit(training)
    print('Cars / Guess — CAR-03 and CAR-04')
    print(f'Records: {len(table.rows)}; seed=42; train={len(training.rows)}; '
          f'test={len(test.rows)}; k={model.k}; p={model.distance.p}')
    print('X normalization fitted on training records only; no held-out tuning.')
    columns = training.role('x')
    seen = {tuple(r.cells[c.index] for c in columns) for r in training.rows}
    overlap = sum(tuple(r.cells[c.index] for c in columns) in seen for r in test.rows)
    print(f'Test rows sharing an X combination with training: {overlap}/{len(test.rows)}')
    print('This row split does not test exclusively unseen configurations.')
    for requirement, target in [('CAR-03', 'Mpg+'), ('CAR-04', 'Lbs-')]:
        result = evaluate(model, test, target)
        column = model.target_column(target)
        print(f'\n{requirement}: {target}; training={result.training_count}; '
              f'observed test targets={result.test_count}')
        print('row actual predicted abs_error neighbor_min neighbor_max neighbor_stddev n')
        for row, prediction in result.predictions[:5]:
            actual = row.cells[column.index]
            print(f'{row.number} {actual:g} {prediction.value:.6f} '
                  f'{abs(actual - prediction.value):.6f} {prediction.minimum:g} '
                  f'{prediction.maximum:g} {prediction.stddev:.6f} {len(prediction.neighbors)}')
        print(f'MAE={result.mae:.6f}; training_mean={result.baseline:.6f}; '
              f'baseline_MAE={result.baseline_mae:.6f}')
    print('\nPredictions use held-out historical records, not a live vehicle evaluation. '
          'Neighbor ranges and population standard deviations describe dispersion; '
          'they are not calibrated prediction intervals. Single-split MAEs do not '
          'establish statistical significance or an experimental treatment result.')


if __name__ == '__main__':
    main()
