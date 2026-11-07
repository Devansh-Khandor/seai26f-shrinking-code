"""NASA Batch A domain policies and presentation; algorithms stay shared."""

from collections import Counter
from itertools import combinations
from .data import Row
from .stats import NumericSummary, summarize
from .ranking import rank_rows
from .prediction import KNNRegressor
from .evaluation import train_test_split, evaluate
from .certification import compare
from .diagnosis import outcome_groups, compare_groups
from .proposals import ProposalStream

TARGETS = ('Effort-', 'Defects-')


def audit(table):
    print(f'Records={len(table.rows)}; columns={len(table.columns)}')
    for role in ('x', 'y', 'ignored'):
        print(f'{role}: ' + ', '.join(f'{c.name} ({"numeric" if c.numeric else "categorical"})' for c in table.role(role)))
    print('Missing cells: ' + str(sum(v is None for r in table.rows for v in r.cells)))
    print('Constants: ' + ', '.join(c.name for c in table.columns if len({r.cells[c.index] for r in table.rows}) == 1))
    counts = Counter(tuple(r.cells[c.index] for c in table.role('x')) for r in table.rows)
    print(f'X combinations={len(counts)}; repeated groups={sum(n>1 for n in counts.values())}; '
          f'rows in repeated groups={sum(n for n in counts.values() if n>1)}')


def remember(table):
    print('NASA / Remember — NASA-01 and NASA-02')
    audit(table)
    for c in table.role('y'):
        s = summarize(table, c)
        best, worst = (s.minimum,s.maximum) if c.ideal == 0 else (s.maximum,s.minimum)
        print(f'{c.name}: n={s.count} range={s.minimum}..{s.maximum} mean={s.mean:.9f} '
              f'median={s.median} best={best} worst={worst}')
    for row, score in rank_rows(table)[:5]:
        print(f'row={row.number} goals={tuple(row.cells[c.index] for c in table.role("y"))} d2h={score:.9f}')
    print('Historical equal-weight ranking: Kloc+ maximized, Effort-/Defects- minimized. '
          'Low d2h does not establish operational superiority; size is a goal, not a prediction input.')


def guess(table):
    training, test = train_test_split(table)
    model = KNNRegressor.fit(training)
    seen = {tuple(r.cells[c.index] for c in training.role('x')) for r in training.rows}
    overlap = sum(tuple(r.cells[c.index] for c in test.role('x')) in seen for r in test.rows)
    print('NASA / Guess — NASA-03 and NASA-04')
    print(f'seed=42 training={len(training.rows)} test={len(test.rows)} k=5; training-only normalization; '
          f'test rows sharing training X={overlap}/{len(test.rows)}')
    for target in TARGETS:
        e = evaluate(model, test, target)
        c = model.target_column(target)
        for row, p in e.predictions[:5]:
            print(f'{target} row={row.number} actual={row.cells[c.index]} predicted={p.value:.9f} '
                  f'neighbor_range={p.minimum}..{p.maximum} stddev={p.stddev:.9f} '
                  f'neighbors={tuple(r.number for r in p.neighbors)}')
        print(f'{target}: scored_test={e.test_count} MAE={e.mae:.9f} '
              f'training_mean={e.baseline:.9f} baseline_MAE={e.baseline_mae:.9f}')
    print('Dispersion is descriptive, not a calibrated interval. Single historical split; '
          'shared X combinations do not demonstrate generalization to exclusively novel proposals.')


def rating_groups(table, rating, target):
    columns = {c.name:c for c in table.columns}
    group, goal = columns[rating], columns[target]
    groups = {}
    for row in table.rows:
        key, value = row.cells[group.index], row.cells[goal.index]
        if key is not None:
            groups.setdefault(key, [])
            if value is not None:
                groups[key].append(value)
    return groups


def certify(table):
    print('NASA / Certify — NASA-05 and NASA-06')
    for rating, target in [('pcap','Effort-'), ('cplx','Defects-')]:
        groups = rating_groups(table, rating, target)
        print(f'{target} by {rating}: recorded rating categories, no numeric subtraction')
        for key, values in sorted(groups.items()):
            s = NumericSummary.from_values(values)
            print(f'  {key}: n={s.count} mean={s.mean} median={s.median}' +
                  ('; caution: fewer than five observed outcomes' if s.count<5 else ''))
        for a,b in combinations(sorted(groups),2):
            c = compare(groups[a],groups[b])
            print(f'  {a} vs {b}: same={c.same} cohen={c.cohen} cliffs={c.cliffs} KS={c.ks} '
                  f'exceeded={c.exceeded}' + ('; insufficient: need >=2 per group' if c.same is None else ''))
    print('Course same is a descriptive screen; False need not mean statistical significance, '
          'True does not prove equivalence. Small groups, tied ratings, confounding, and '
          'unadjusted multiple comparisons limit planning conclusions.')


def flow(table):
    training, incoming = train_test_split(table)
    stream = ProposalStream(training, TARGETS)
    print('NASA / Flow — NASA-07 and NASA-08')
    print(f'Initial completed={len(training.rows)}; simulated proposals={len(incoming.rows)}; seed=42; k=5')
    print('Synthetic row-ID order: assess masked proposal, then reveal completion, then assess next proposal. '
          'Risk: prediction strictly above current observed nearest-rank Q75; not a calibrated risk probability.')
    alerts = Counter()
    for row in incoming.rows:
        query = Row(row.number, tuple(row.cells[c.index] if c.role=='x' else None for c in table.columns))
        assessment = stream.assess(query)
        for risk in assessment.risks:
            alerts[risk.target] += risk.high
            print(f'proposal={row.number} reference_n={assessment.reference_count} {risk.target} '
                  f'prediction={risk.prediction.value:.9f} Q75={risk.threshold} high={risk.high} '
                  f'neighbor_stddev={risk.prediction.stddev:.9f}')
        # Only now are the completed record's outcomes handed to the shared state.
        stream.complete(row)
        print(f'completion={row.number} accepted; historical_n={len(stream.inventory.table.rows)}')
    print(f'Alerts={dict(alerts)}; completed_added={stream.inventory.accepted_count}')
    for c in table.role('y'):
        s = stream.inventory.summaries[c.index]
        print(f'Final {c.name}: n={s.count} mean={s.mean:.9f} median={s.median} range={s.minimum}..{s.maximum}')
    if stream.inventory.table != table:
        raise RuntimeError('Final historical inventory differs from the full table')
    print('Full-batch inventory match: PASS. No actual proposal outcomes enter arrival predictions; '
          'later assessments use earlier completions only. This simulated ordering is not real chronology.')


def blame(table):
    print('NASA / Blame — NASA-09 and NASA-10')
    for target in TARGETS:
        g = outcome_groups(table,target,higher_better=False)
        print(f'{target}: poor_n={len(g.poor.rows)} favorable_n={len(g.favorable.rows)} '
              f'poor_mean={g.poor_outcome.mean:.9f} favorable_mean={g.favorable_outcome.mean:.9f}')
        for f in compare_groups(g.poor,g.favorable):
            print(f'  {f.column.name}: {"numeric Cliff" if f.column.numeric else "categorical TV"}={f.separation}; '
                  f'observed_n={f.poor.count}/{f.favorable.count}')
            for e in f.categories:
                print(f'    {e.value}: poor/favorable={e.poor_count}/{e.favorable_count}; '
                      f'proportions={e.poor_proportion:.6f}/{e.favorable_proportion:.6f}')
            if f.column.numeric:
                print(f'    means={f.poor.mean}/{f.favorable.mean}; same={f.certification.same}')
            for note in f.limitations:
                print('    '+note)
    print('Disjoint floor(n/4) extremes; ties by original row ID; middle omitted. '
          'Observed associations, not causes. Rating order is not a numeric distance; '
          'constant categories are retained. Category scores are descriptive, not significance tests.')
