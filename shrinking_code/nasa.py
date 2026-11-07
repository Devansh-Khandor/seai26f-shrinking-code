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


# NASA adjustment policy v1: fixed before evaluating hypothetical outcomes.
ADJUSTMENT_VERSION = 'NASA-adjustments-v1'
ADJUSTABLE = ('docu', 'tool', 'site', 'sced')
MIN_REDUCTION = .05


def proposal_context(table):
    from .selection import HistoricalOracle
    training, hidden = train_test_split(table)
    visible = HistoricalOracle(hidden, 0).visible
    stream = ProposalStream(training, TARGETS)
    assessments = tuple(stream.assess(r) for r in visible.rows)
    return training, visible, assessments


def risk_pair(assessments):
    pairs = [(a,b) for a,b in combinations(sorted(assessments,key=lambda a:a.number),2)
             if tuple(r.high for r in a.risks)!=tuple(r.high for r in b.risks)]
    for a,b in pairs:
        diffs=[x.prediction.value-y.prediction.value for x,y in zip(a.risks,b.risks)]
        if any(d<0 for d in diffs) and any(d>0 for d in diffs):
            return a,b,True
    return (*pairs[0],False) if pairs else None


def justify(table):
    from .explanation import explain_prediction
    training, visible, assessments = proposal_context(table)
    model = KNNRegressor.fit(training)
    query = visible.rows[0]
    e = explain_prediction(model,query,'Effort-')
    p = e.prediction
    print('NASA / Justify — NASA-11 and NASA-12')
    print(f'NASA-11 proposal={query.number} X={tuple((c.name,query.cells[c.index]) for c in table.role("x"))}')
    for row,distance in zip(p.neighbors,e.neighbor_distances):
        c=model.target_column('Effort-')
        differences=tuple((x.name,query.cells[x.index],row.cells[x.index]) for x in table.role('x') if query.cells[x.index]!=row.cells[x.index])
        print(f'neighbor={row.number} X_distance={distance:.9f} observed_effort={row.cells[c.index]} X_differences={differences}')
    print(f'Unweighted neighbor mean={p.value:.9f}; range={p.minimum}..{p.maximum}; stddev={p.stddev:.9f}')
    pair=risk_pair(assessments)
    if pair is None:
        print('NASA-12: no proposals with different alert assessments.')
    else:
        a,b,tradeoff=pair
        print(f'NASA-12 proposals={a.number},{b.number}; strict predicted tradeoff={tradeoff}')
        if not tradeoff: print('No different-risk strict tradeoff pair; first different-alert pair shown.')
        for assessment in (a,b):
            for r in assessment.risks:
                print(f'row={assessment.number} {r.target} predicted={r.prediction.value:.9f} Q75={r.threshold} '
                      f'margin={r.prediction.value-r.threshold:+.9f} high={r.high}')
        rows={r.number:r for r in visible.rows}
        print('Observed X differences: '+str(tuple((c.name,rows[a.number].cells[c.index],rows[b.number].cells[c.index])
              for c in table.role('x') if rows[a.number].cells[c.index]!=rows[b.number].cells[c.index])))
    print('No held-out outcomes used. X differences are observed associations, not feature attributions '
          'or causal effects. Neighbor dispersion is not a calibrated prediction interval.')


def choose(table):
    from .explanation import Explainer
    from .data import Table
    from .selection import HistoricalOracle,select_candidates
    e=Explainer(table)
    print('NASA / Choose — NASA-13 and NASA-14')
    print('NASA-13: observed full-history favorable combinations; Kloc maximized only by header policy.')
    for row,score in e.ranked[:5]:
        print(f'row={row.number} d2h={score:.9f} goals={tuple(row.cells[c.index] for c in table.role("y"))} '
              f'X={tuple((c.name,row.cells[c.index]) for c in table.role("x"))}')
    for a,b in combinations([r for r,_ in e.ranked[:5]],2):
        pair=e.compare(a.number,b.number)
        print(f'rows={a.number},{b.number} favors_first={pair.favors_a} favors_second={pair.favors_b} equal={pair.equal}')
    top=Table(table.columns,tuple(r for r,_ in e.ranked[:5]))
    for c in table.role('x'):
        if not c.numeric and len(summarize(table,c).frequencies)>1:
            for value,count in summarize(top,c).frequencies.items():
                if count>=3: print(f'Common among top five: {c.name}={value} n={count}/5')
    training, hidden=train_test_split(table)
    oracle=HistoricalOracle(hidden,10)
    result=select_candidates(training,oracle.visible,oracle.inspect,10)
    print(f'NASA-14 training={len(training.rows)} candidates={len(hidden.rows)} inspections={len(oracle.inspected)} predictions={result.prediction_count}')
    for d in result.inspections:
        print(f'inspected={d.row.number} predicted_d2h={d.predicted_score:.9f} actual_d2h={d.observed_score:.9f}')
    for budget in (3,5,10):
        d=result.checkpoint(budget)
        print(f'budget={budget} best_row={d.best_row.number} best_d2h={d.best_score:.9f}')
    print('Historical combinations are not a success recipe. Offline inspection simulation; outcomes '
          'are not instantly obtainable in real projects. Fixed training scoring bounds, no global-optimum claim.')


def fix(table):
    from .hypotheticals import search_changes
    training,visible,assessments=proposal_context(table)
    rows={r.number:r for r in visible.rows}
    print('NASA / Fix — NASA-15 and NASA-16')
    print(f'Policy={ADJUSTMENT_VERSION} allowlist={ADJUSTABLE}; minimum reduction={MIN_REDUCTION:.0%}; '
          'training-observed categorical levels only; all suggestions hypothetical, feasibility unverified.')
    print('Training levels: '+str({name: sorted({r.cells[next(c.index for c in training.columns if c.name==name)] for r in training.rows}) for name in ADJUSTABLE}))
    for target,max_changes,requirement in [('Effort-',1,'NASA-15'),('Defects-',2,'NASA-16')]:
        high=[a for a in assessments if next(r for r in a.risks if r.target==target).high]
        if not high:
            print(f'{requirement}: no high predicted {target} proposal; no suggestion.'); continue
        source=min(high,key=lambda a:a.number)
        result=search_changes(training,rows[source.number],target,ADJUSTABLE,TARGETS,max_changes,MIN_REDUCTION)
        print(f'{requirement} source={source.number} variants_evaluated={result.evaluations} '
              f'original={tuple((n,p.value) for n,p in result.original)}')
        for name,p in result.original:
            print(f'Original {name}: range={p.minimum}..{p.maximum}; stddev={p.stddev:.9f}')
        if not result.candidates:
            print('No solution meeting the improvement target within the allowed search.'); continue
        for v in result.candidates[:5]:
            print(f'Hypothetical changes={v.changes} improvement={v.improvement:.9f} '
                  f'historically_observed_X={v.historically_observed}')
            old=dict(result.original)
            for name,p in v.predictions:
                print(f'  {name}: {old[name].value:.9f} -> {p.value:.9f}; delta={p.value-old[name].value:+.9f}; '
                      f'neighbor_range={p.minimum}..{p.maximum} stddev={p.stddev:.9f}')
    print('Predicted differences are not causal reductions. Only category levels are observed; '
          'synthetic combinations may be infeasible. docu/site are constant, so cannot vary here. '
          'Personnel capabilities, identifiers, and other properties remain fixed.')


def spot(table):
    from .anomalies import input_isolation,outcome_residuals
    print('NASA / Spot — NASA-17 and NASA-18')
    for r in input_isolation(table):
        if r.flagged:
            print(f'NASA-17 row={r.row.number} isolation={r.score:.9f} neighbors={r.neighbors}')
    training,test=train_test_split(table)
    for target in TARGETS:
        for r in outcome_residuals(training,test,target):
            if r.flagged:
                print(f'NASA-18 {target} row={r.row.number} actual={r.actual} predicted={r.prediction.value:.9f} '
                      f'signed={r.signed:+.9f} absolute={r.score:.9f} range={r.prediction.minimum}..{r.prediction.maximum} '
                      f'stddev={r.prediction.stddev:.9f}')
    print('Fixed top-five review lists; not confirmed errors or calibrated probabilities. '
          'X-only historical isolation and separate held-out target residuals; k=5, seed=42.')


def race(table):
    from .baselines import prediction_race,selection_race
    training,test=train_test_split(table)
    p=prediction_race(training,test,'Effort-')
    print('NASA / Race — NASA-19 and NASA-20')
    print(f'NASA-19 kNN_MAE={p.evaluation.mae:.9f} mean_MAE={p.evaluation.baseline_mae:.9f} '
          f'percent_reduction={p.percentage_improvement:.9f} wins/losses/ties={p.wins}/{p.losses}/{p.ties} '
          f'same={p.comparison.same} exceeded={p.comparison.exceeded}')
    print('Paired errors (row, kNN, baseline, kNN-minus-baseline): '+str(tuple((r.number,r.knn_error,r.mean_error,r.difference) for r in p.pairs)))
    r=selection_race(table,budgets=(3,5,10))
    for c in r.checkpoints:
        s=c.random_summary
        wins=sum(t.inspections[c.budget-1].best_score<=c.model_score for t in r.random_trials)
        print(f'NASA-20 budget={c.budget} model={c.model_score:.9f} random_mean={s.mean:.9f} median={s.median:.9f} '
              f'min={s.minimum:.9f} max={s.maximum:.9f} match_or_beat={wins}/30 '
              f'inspections_per_run={c.budget} model_predictions={c.model_predictions} random_predictions=0')
    for t in r.random_trials:
        print(f'seed={t.seed} inspected={tuple(d.row.number for d in t.inspections)} '
              f'checkpoints={tuple(t.inspections[b-1].best_score for b in (3,5,10))}')
    print('One deterministic model trajectory, 30 random seeds 0..29 on one fixed dataset. '
          'Course same ignores pairing and a 19-record test set limits inference; no formal superiority '
          'or equivalence claim. No same test for one model trajectory versus random trials. '
          'These are customer-method comparisons, not independent development treatment repetitions.')
