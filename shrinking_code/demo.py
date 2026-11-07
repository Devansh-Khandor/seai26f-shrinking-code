"""Run the Cars Remember milestone from the repository root."""

import argparse
from pathlib import Path
from .data import Table
from .ranking import rank_rows
from .stats import summarize


def main(argv=None):
    parser = argparse.ArgumentParser(description='Step 1: Cars / Remember only')
    parser.add_argument('--domain', required=True)
    parser.add_argument('--skill', required=True)
    args = parser.parse_args(argv)
    if (args.domain, args.skill) != ('cars', 'remember'):
        parser.error('Unsupported domain or skill; only cars / remember is implemented')
    path = Path(__file__).resolve().parents[1] / 'data/raw/auto93.csv'
    table = Table.load(path)
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


if __name__ == '__main__':
    main()
