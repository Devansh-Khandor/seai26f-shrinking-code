"""Run the completed Cars milestones from the repository root."""

import argparse
from itertools import combinations
from pathlib import Path
from .data import Table
from .ranking import rank_rows
from .stats import NumericSummary, summarize
from .certification import compare
from .streaming import Inventory
from .diagnosis import outcome_groups, compare_groups
from .explanation import Explainer
from .alternatives import Alternatives, target_status
from .selection import HistoricalOracle, select_candidates
from .prediction import KNNRegressor
from .evaluation import evaluate, train_test_split


def main(argv=None):
    parser = argparse.ArgumentParser(description='Cars / Remember, Guess, Certify, Flow, Blame, Justify, Choose and Fix')
    parser.add_argument('--domain', required=True)
    parser.add_argument('--skill', required=True)
    args = parser.parse_args(argv)
    if args.domain != 'cars' or args.skill not in ('remember', 'guess', 'certify', 'flow', 'blame', 'justify', 'choose', 'fix'):
        parser.error('Unsupported domain or skill; only cars / remember, guess, certify, flow, blame, justify, choose, and fix are implemented')
    path = Path(__file__).resolve().parents[1] / 'data/raw/auto93.csv'
    table = Table.load(path)
    if args.skill == 'fix':
        fix(table)
        return
    if args.skill == 'choose':
        choose(table)
        return
    if args.skill == 'guess':
        guess(table)
        return
    if args.skill == 'certify':
        certify(table)
        return
    if args.skill == 'flow':
        flow(table)
        return
    if args.skill == 'blame':
        blame(table)
        return
    if args.skill == 'justify':
        justify(table)
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



def flow(table):
    initial, incoming = train_test_split(table)
    inventory = Inventory(initial)
    print('Cars / Flow — CAR-07 and CAR-08')
    print(f'Initial inventory: {len(initial.rows)}; incoming records: {len(incoming.rows)}')
    print('Synthetic arrival order: seed-42 partition, incoming records in original row order; '
          'not a real chronology. Separate from the frozen Guess evaluation.')
    print('Each arrival is compared with current inventory using common bounds including '
          'that arrival, before acceptance. Observed outcomes, not predictions.')
    decisions = []
    for row in incoming.rows:
        decisions.append(inventory.accept(row.cells, number=row.number))
    print('CAR-07 sample decisions (first three and last three arrivals):')
    for i, decision in enumerate(decisions, 1):
        if i <= 3 or i > len(decisions) - 3:
            score = 'unranked: incomplete goals' if decision.score is None else f'{decision.score:.6f}'
            print(f'arrival={i} row={decision.row.number} prior_n={decision.previous_count} '
                  f'rank={decision.rank} top_five={decision.top_five} d2h={score}')
    print(f'CAR-08 accepted arrivals: {inventory.accepted_count}; '
          f'final inventory: {len(inventory.table.rows)}')
    for c in inventory.table.role('y'):
        s = inventory.summaries[c.index]
        print(f'{c.name}: n={s.count}, min={s.minimum:g}, max={s.maximum:g}, '
              f'mean={s.mean:.6f}, median={s.median:g}')
    for c in inventory.table.columns:
        if not c.numeric:
            s = inventory.summaries[c.index]
            print(f'{c.name}: n={s.count}, frequencies={s.frequencies}, mode={s.mode}')
    print('Final top-five row / d2h:')
    for row, score in inventory.shortlist:
        print(f'{row.number} {score:.6f}')
    # Only after the stream finishes may the full dataset be used for verification.
    matches = (inventory.table == table
               and inventory.summaries == tuple(summarize(table, c) for c in table.columns)
               and inventory.shortlist == tuple(rank_rows(table)[:5]))
    print('Full-batch Remember verification: ' + ('PASS' if matches else 'FAIL'))
    if not matches:
        raise RuntimeError('Stream final state does not match full-batch results')
    print('Exact summaries and rankings recomputed per arrival; no constant-time or '
          'real-time deployment claim. Missing goals are accepted but unranked; '
          'future records never enter arrival assessments.')



def blame(table):
    print('Cars / Blame — CAR-09 and CAR-10')
    print(f'Records: {len(table.rows)}; quartiles are floor(n/4) at each end of '
          '(outcome, original row ID) ascending; middle records excluded.')
    print('Ranks are separate by type: numeric absolute Cliff delta; categorical total '
          'variation (TV). Descending score, ties by original column order.')
    for requirement, outcome, label in [('CAR-09', 'Mpg+', 'Mpg+'), ('CAR-10', None, 'd2h')]:
        groups = outcome_groups(table, outcome)
        print(f'\n{requirement}: {label}; eligible={groups.eligible_count}')
        for name, summary in [('poor', groups.poor_outcome), ('favorable', groups.favorable_outcome)]:
            print(f'{name}: n={summary.count}, mean={summary.mean:.6f}, median={summary.median:g}, '
                  f'range={summary.minimum:.6f}..{summary.maximum:.6f}')
        for finding in compare_groups(groups.poor, groups.favorable):
            a, b = finding.poor, finding.favorable
            score = 'insufficient' if finding.separation is None else f'{finding.separation:.6f}'
            print(f'{finding.column.name} ({"numeric" if finding.column.numeric else "categorical"}): '
                  f'separation={score}, observed n={a.count}/{b.count} (poor/favorable)')
            if finding.column.numeric:
                print(f'  mean={a.mean}/{b.mean}; median={a.median}/{b.median}; '
                      f'{finding.direction}; same={finding.certification.same}')
            else:
                for e in finding.categories:
                    print(f'  category={e.value}: counts={e.poor_count}/{e.favorable_count}; '
                          f'proportions={e.poor_proportion:.3f}/{e.favorable_proportion:.3f}; '
                          f'poor-minus-favorable={e.difference:+.3f}')
            for note in finding.limitations:
                print('  ' + note)
    print('\nHistorical group associations, not causal conclusions, individual causes, or unseen-car '
          'predictions. Coarse tied MPG makes quartile boundaries depend on row IDs. '
          'Feature scores are descriptive, not significance tests; same=False can reflect an '
          'effect-size screen alone. Missing values and small counts limit evidence. '
          'd2h retains Step 1 full-table bounds and uncertain Acc+ meaning.')



def print_score_explanation(explanation):
    print(f'row={explanation.row.number}, rank={explanation.rank}, d2h={explanation.score:.9f}')
    for goal in explanation.goals:
        print(f'  {goal.name}: raw={goal.raw:g}, {goal.direction}, '
              f'bounds={goal.minimum:g}..{goal.maximum:g}, normalized={goal.normalized:.9f}, '
              f'ideal={goal.ideal}, squared={goal.squared:.9f}')
    average = sum(g.squared for g in explanation.goals) / len(explanation.goals)
    print(f'  mean squared contribution={average:.9f}; sqrt={average ** 0.5:.9f}')


def justify(table):
    explainer = Explainer(table)
    print('Cars / Justify — CAR-11 and CAR-12')
    print('Historical observed outcomes; all goal and X bounds use this inventory. '
          'Similarity in X is separate from desirability in Y.')
    if not explainer.ranked:
        print('No complete-goal historical rows to recommend.')
        return
    selected = explainer.explain(explainer.ranked[0][0].number)
    print('\nCAR-11: best-ranked historical vehicle')
    print('Recorded fields: ' + ', '.join(f'{c.name}={selected.row.cells[c.index]}' for c in table.columns))
    print_score_explanation(selected)
    print('Three nearest X neighbors when available; ties rounded to 12 decimals, then row ID:')
    neighbors = explainer.neighbors(selected.row.number)
    if not neighbors:
        print('No comparable X neighbors available.')
    for neighbor in neighbors:
        other = neighbor.explanation
        pair = explainer.compare(selected.row.number, other.row.number)
        values = ', '.join(f'{g.name}={g.raw}' for g in other.goals)
        print(f'  row={other.row.number}, X-distance={neighbor.distance:.6f}, '
              f'd2h={other.score}, {values}')
        print(f'    selected wins: {pair.favors_a}; neighbor wins: {pair.favors_b}; '
              f'equal: {pair.equal}; unknown: {pair.unknown}')
    print('\nCAR-12: first tradeoff pair in nested historical ranking order')
    pair = explainer.tradeoff_pair()
    if pair is None:
        print('No complete-goal pair with a strict score difference and wins on both sides.')
    else:
        print_score_explanation(pair.a)
        print_score_explanation(pair.b)
        print(f'Row {pair.a.row.number} wins: {pair.favors_a}; '
              f'row {pair.b.row.number} wins: {pair.favors_b}; equal: {pair.equal}')
        print(f'Row {pair.winner} ranks higher because its average squared distance from '
              'the goal ideals is smaller; every goal has equal weight.')
    if len(explainer.ranked) >= 4:
        groups = outcome_groups(table)
        context = next((f for f in compare_groups(groups.poor, groups.favorable)
                        if f.column.numeric and f.separation is not None), None)
        if context:
            print(f'\nPopulation context only (Blame): {context.column.name}, '
                  f'absolute Cliff delta={context.separation:.6f}; {context.direction}. '
                  'This is not a cause or an individual feature attribution.')
    print('A customer prioritizing a different goal could prefer another car; no weights '
          'were changed here. Observed outcomes were not predicted. No claims about purchase '
          'price, mechanical reliability, or faster physical acceleration; Acc+ meaning is '
          'uncertain. Similarity with missing X can use fewer fields; incomplete goals remain '
          'unranked. This explains a historical score, not a causal mechanism or held-out prediction.')


def choose(table):
    training, hidden = train_test_split(table)
    budget = 40
    oracle = HistoricalOracle(hidden, budget)
    print('Cars / Choose — CAR-13 and CAR-14')
    print('Offline inspection simulation using held-out historical records; seed=42.')
    print(f'Historical training: {len(training.rows)}; candidate pool: {len(oracle.visible.rows)}')
    print(f'Policy: greedy lowest predicted d2h, ties by original row ID; k=5; budget={budget}')
    print('Fixed initial-training goal bounds; no clipping; constants normalize to 0.5:')
    for c in training.role('y'):
        summary = summarize(training, c)
        print(f'  {c.name}: {summary.minimum:g}..{summary.maximum:g}')
    result = select_candidates(training, oracle.visible, oracle.inspect, budget)
    print('CAR-13: predictions recorded before inspection; actual goals shown only after inspection.')
    for cost, decision in enumerate(result.inspections, 1):
        print(f'inspection={cost} row={decision.row.number} '
              f'predicted_goals={decision.predicted_goals} predicted_d2h={decision.predicted_score:.9f} '
              f'actual_goals={tuple(decision.row.cells[c.index] for c in training.role("y"))} '
              f'actual_d2h={decision.observed_score:.9f}')
    print(f'Inspected IDs: {oracle.inspected}')
    print(f'Actual inspection cost: {len(oracle.inspected)}; '
          f'per-goal prediction computations: {result.prediction_count}')
    print('CAR-14: checkpoints from the same run (lower best-so-far d2h is better):')
    for cost in (10, 20, 40):
        state = result.checkpoint(cost)
        print(f'budget={cost} best_row={state.best_row.number} best_d2h={state.best_score:.9f}')
    final = result.inspections[-1]
    print(f'Final recommended inspected car: row={final.best_row.number}, d2h={final.best_score:.9f}')
    print('Recorded X and revealed Y: ' + ', '.join(
        f'{c.name}={final.best_row.cells[c.index]}' for c in training.columns if c.role != 'ignored'))
    print('Heuristic best among inspected candidates; no global optimality or inspection-efficiency '
          'claim. No random baseline or Race comparison. Acc+ is maximized as recorded; '
          'physical meaning remains uncertain.')


def print_alternative(result, table):
    pair = result.comparison
    print(f'row={pair.b.row.number} source_d2h={pair.a.score:.9f} '
          f'alternative_d2h={pair.b.score:.9f} improvement={result.improvement:.9f} '
          f'X_distance={result.distance:.9f} changed_X={len(result.changed)}')
    print('  X: ' + ', '.join(f'{c.name}={pair.b.row.cells[c.index]}' for c in table.role('x')))
    changes = ', '.join(f'{c.name}: {c.before} -> {c.after}' for c in result.changed)
    print(f'  changes: {changes or "none"}; unknown_X={result.unknown_x}')
    print(f'  improved={pair.favors_b}; worsened={pair.favors_a}; tied={pair.equal}; '
          f'Pareto_dominates={result.pareto_dominates}')
    for a, b in zip(pair.a.goals, pair.b.goals):
        print(f'  {a.name}: {a.raw:g} -> {b.raw:g}; raw_difference={b.raw-a.raw:+g}')
    print(f'  order: {result.rationale}')


def fix(table):
    search = Alternatives(table)
    source = search.worst()
    print('Cars / Fix — CAR-15 and CAR-16')
    print('Historically observed alternatives; full-table common bounds; equal-weight d2h.')
    if source is None:
        print('No complete-goal source available.')
        return
    print(f'Source: worst complete record, ties by smallest row ID; row={source.number}')
    print('Source fields: ' + ', '.join(f'{c.name}={source.cells[c.index]}'
                                      for c in table.columns if c.role != 'ignored'))
    print('CAR-15: up to five nearest strictly better overall historical alternatives')
    results = search.better(source.number)
    if not results:
        print('No qualifying better alternative.')
    for result in results:
        print_alternative(result, table)
    targets = {c.name: search.explainer.summaries[c.index].median for c in table.role('y')}
    print('CAR-16: targets are full historical medians, an initial policy, not customer thresholds:')
    for c in table.role('y'):
        print(f'  {c.name} {"<=" if c.ideal == 0 else ">="} {targets[c.name]:g}')
    status = target_status(source, table.columns, targets)
    print(f'Source targets: {status.count}/{len(targets)}; satisfied={status.satisfied}; failed={status.failed}')
    result = search.meet_more(source.number, targets)
    if result is None:
        print('No qualifying alternative with more satisfied targets and fully comparable X.')
    else:
        print(f'Recommended alternative: row={result.comparison.b.row.number}; '
              f'target count={result.source_targets.count} -> {result.alternative_targets.count}; '
              f'satisfied={result.alternative_targets.satisfied}; failed={result.alternative_targets.failed}')
        print_alternative(result, table)
    print('Differences describe recorded vehicles, not feasible mechanical modifications, costs, '
          'or causal improvements. Lower d2h does not imply every goal improves. '
          'Missing X differences are unknown; CAR-16 excludes such pairs. '
          'Acc+ follows its header direction; physical interpretation remains uncertain.')


if __name__ == '__main__':
    main()
