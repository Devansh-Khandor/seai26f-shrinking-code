"""Run the Cars Remember and Guess milestones from the repository root."""

import argparse
from itertools import combinations
from pathlib import Path
from .data import Table
from .ranking import rank_rows
from .stats import NumericSummary, summarize
from .certification import compare
from .prediction import KNNRegressor
from .evaluation import evaluate, train_test_split


def main(argv=None):
    parser = argparse.ArgumentParser(description='Cars / Remember, Guess and Certify')
    parser.add_argument('--domain', required=True)
    parser.add_argument('--skill', required=True)
    args = parser.parse_args(argv)
    if args.domain != 'cars' or args.skill not in ('remember', 'guess', 'certify'):
        parser.error('Unsupported domain or skill; only cars / remember, guess, and certify are implemented')
    path = Path(__file__).resolve().parents[1] / 'data/raw/auto93.csv'
    table = Table.load(path)
    if args.skill == 'guess':
        guess(table)
        return
    if args.skill == 'certify':
        certify(table)
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



def cylinder_groups(table):
    columns = {c.name: c for c in table.columns}
    groups = {}
    for row in table.rows:
        cylinder = row.cells[columns['Clndrs'].index]
        value = row.cells[columns['Mpg+'].index]
        if cylinder is not None:
            groups.setdefault(cylinder, [])
            if value is not None:
                groups[cylinder].append(value)
    return groups


def model_year_groups(table):
    model = next(c for c in table.columns if c.name == 'Model')
    groups = {'70-76': [], '77-82': []}
    # Score once on the full reference table, never normalize each group separately.
    for row, score in rank_rows(table):
        year = row.cells[model.index]
        if year is not None:
            if 70 <= year <= 76:
                groups['70-76'].append(score)
            elif 77 <= year <= 82:
                groups['77-82'].append(score)
    return groups


def print_comparison(label, xs, ys):
    result = compare(xs, ys)
    if result.same is None:
        print(f'{label}: insufficient data (n={result.left.count},{result.right.count}; '
              'need at least 2 observed values per group)')
        return
    print(f'{label}: mean_gap(left-right)={result.left.mean - result.right.mean:.6f}; '
          f'cohen={result.cohen:.6f}; D={result.ks_d:.6f}; KS={result.ks:.6f}; '
          f'cliffs={result.cliffs:.6f}; same={result.same}; '
          f'exceeded={",".join(result.exceeded) or "none"}')


def certify(table):
    print('Cars / Certify — CAR-05 and CAR-06')
    print(f'Records: {len(table.rows)}; thresholds: Cohen<=0.35, cliffs<=0.195, KS<=1.36')
    print('same=True: not distinguished by this rule, not proven equivalent. '
          'same=False: distinguished by at least one screen, not automatically significant.')
    print('KS is scaled D; effect-size screens alone do not establish significance.')
    print('\nCAR-05: Mpg+ by observed cylinder count')
    cylinders = cylinder_groups(table)
    for key, values in sorted(cylinders.items()):
        summary = NumericSummary.from_values(values)
        print(f'Clndrs={key:g}: n={summary.count}; mean={summary.mean}; median={summary.median}')
    for a, b in combinations(sorted(cylinders), 2):
        print_comparison(f'{a:g} vs {b:g} cylinders', cylinders[a], cylinders[b])
    print('\nCAR-06: fixed model-year codes; d2h uses full-table goal bounds; lower is better')
    groups = model_year_groups(table)
    for key, values in groups.items():
        summary = NumericSummary.from_values(values)
        print(f'Model={key}: n={summary.count}; mean={summary.mean}; median={summary.median}')
    print_comparison('70-76 vs 77-82', groups['70-76'], groups['77-82'])
    print('\nAll six cylinder pairs are reported without multiplicity adjustment. '
          'Small groups and tied/coarse MPG values limit asymptotic KS interpretation. '
          'These are historical associations, not causal effects or guarantees for inventory decisions. '
          'No failure to distinguish proves equivalence; Acc+ physical meaning remains uncertain.')


if __name__ == '__main__':
    main()
