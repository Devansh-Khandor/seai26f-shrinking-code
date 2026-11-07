"""Apache Batch A policies and presentation, backed by shared algorithms."""

from itertools import combinations
from math import ceil
from .data import Row, Table
from .stats import NumericSummary, SymbolicSummary, summarize
from .ranking import rank_rows
from .prediction import KNNRegressor
from .evaluation import train_test_split, evaluate
from .certification import compare
from .streaming import Inventory
from .explanation import Explainer, explain_prediction
from .diagnosis import outcome_groups, compare_groups, categorical_pairs

TARGET = 'Performance-'


def varying(table):
    return tuple(c for c in table.role('x') if len(summarize(table,c).frequencies)>1)


def masked(row, table):
    return Row(row.number,tuple(row.cells[c.index] if c.role=='x' else None for c in table.columns))


def show_findings(findings, limit=5):
    for f in findings[:limit]:
        print(f'{f.column.name}: categorical TV={f.separation:.9f}')
        for c in f.categories:
            print(f'  {c.value}: poor/favorable={c.poor_count}/{c.favorable_count}; '
                  f'proportions={c.poor_proportion:.6f}/{c.favorable_proportion:.6f}')


def remember(table):
    print('Apache / Remember — APACHE-01 and APACHE-02')
    print(f'Records={len(table.rows)} columns={len(table.columns)}; X=categorical setting codes; Y={TARGET} minimize; ignored=none')
    print(f'Missing={sum(v is None for r in table.rows for v in r.cells)}; '
          f'unique_X={len({tuple(r.cells[c.index] for c in table.role("x")) for r in table.rows})}')
    for c in table.role('x'): print(f'{c.name}: {summarize(table,c).frequencies}')
    goal=table.role('y')[0]; s=summarize(table,goal)
    print(f'APACHE-01 n={s.count} range={s.minimum}..{s.maximum} mean={s.mean:.9f} median={s.median}; best={s.minimum} worst={s.maximum}')
    print('Observed performance frequencies: '+str(dict(sorted(SymbolicSummary.from_values(r.cells[goal.index] for r in table.rows).frequencies.items()))))
    ranked=rank_rows(table)
    for row,score in ranked[:5]:
        print(f'APACHE-02 row={row.number} performance={row.cells[goal.index]} d2h={score:.9f} '
              f'X={tuple((c.name,row.cells[c.index]) for c in table.role("x"))}')
    q=len(ranked)//4
    favorable=Table(table.columns,tuple(r for r,_ in ranked[:q]))
    rest=Table(table.columns,tuple(r for r,_ in ranked[q:]))
    variable={c.name for c in varying(table)}
    print(f'Distinguishing settings: best quartile n={q} vs rest n={len(rest.rows)}; poor label below means rest.')
    show_findings(tuple(f for f in compare_groups(rest,favorable) if f.column.name in variable))
    print('Single-goal d2h equals normalized recorded Performance-. Units are unknown; '
          'best observed does not mean best possible. base is constant and not a distinguishing option.')


def one_setting_pair(table):
    columns=varying(table)
    for a,b in combinations(sorted(table.rows,key=lambda r:r.number),2):
        differences=tuple(c.name for c in table.role('x') if a.cells[c.index]!=b.cells[c.index])
        if len(differences)==1 and differences[0] in {c.name for c in columns}:
            return a,b,differences[0]
    return None


def pair_estimate(table):
    pair=one_setting_pair(table)
    if pair is None: return None
    a,b,setting=pair
    reference=Table(table.columns,tuple(r for r in table.rows if r.number not in (a.number,b.number)))
    if not reference.rows: return None
    model=KNNRegressor.fit(reference)
    left=explain_prediction(model,masked(a,table),TARGET)
    right=explain_prediction(model,masked(b,table),TARGET)
    return a,b,setting,left.prediction,right.prediction,len(reference.rows)


def guess(table):
    print('Apache / Guess — APACHE-03 and APACHE-04')
    training,test=train_test_split(table); model=KNNRegressor.fit(training)
    e=evaluate(model,test,TARGET); c=model.target_column(TARGET)
    print(f'APACHE-03 seed=42 training={len(training.rows)} test={len(test.rows)} k=5; training-only X fit')
    for row,p in e.predictions[:5]:
        print(f'row={row.number} actual={row.cells[c.index]} predicted={p.value:.9f} error={abs(row.cells[c.index]-p.value):.9f} '
              f'range={p.minimum}..{p.maximum} stddev={p.stddev:.9f}')
    print(f'MAE={e.mae:.9f} training_mean={e.baseline:.9f} baseline_MAE={e.baseline_mae:.9f}; '
          f'observed_lower_MAE={e.mae<e.baseline_mae}')
    pair=pair_estimate(table)
    if pair is None:
        print('APACHE-04: no eligible observed one-setting pair with a remaining reference.'); return
    a,b,setting,p,q,n=pair
    print(f'APACHE-04 first observed pair by IDs: {a.number},{b.number}; only changed setting={setting}; '
          f'reference_n={n}, excludes both queries; k=5')
    for row,prediction in ((a,p),(b,q)):
        print(f'row={row.number} predicted={prediction.value:.9f} range={prediction.minimum}..{prediction.maximum} '
              f'stddev={prediction.stddev:.9f} neighbors={tuple(r.number for r in prediction.neighbors)}')
    print(f'Predicted difference(second-first)={q.value-p.value:+.9f}; '
          f'neighbor-range difference envelope={q.minimum-p.maximum}..{q.maximum-p.minimum}')
    print(f'Retrospective actual difference(second-first)={b.cells[c.index]-a.cells[c.index]:+g}')
    print('Both configurations are observed historical records. Difference envelope is descriptive, '
          'not a calibrated interval; shared neighbors invalidate naive independent-error assumptions. '
          'No causal toggle effect or validated accuracy for single-setting differences is claimed.')


def setting_groups(table, setting):
    c=next(c for c in table.role('x') if c.name==setting)
    goal=next(c for c in table.role('y') if c.name==TARGET)
    return {value:[r.cells[goal.index] for r in table.rows if r.cells[c.index]==value and r.cells[goal.index] is not None]
            for value in sorted({r.cells[c.index] for r in table.rows if r.cells[c.index] is not None})}


def certify(table):
    print('Apache / Certify — APACHE-05 and APACHE-06')
    print('Operational coding assumption: 1=enabled, 0=disabled; CSV supplies codes, not a setting dictionary.')
    for setting in ('keepAlive','accessLog'):
        groups=setting_groups(table,setting)
        for value,values in groups.items():
            s=NumericSummary.from_values(values)
            print(f'{setting}={value}: n={s.count} mean={s.mean} median={s.median}' + ('; small group caution' if s.count<5 else ''))
        c=compare(groups.get('1',[]),groups.get('0',[]))
        print(f'{setting} enabled vs disabled: same={c.same} cohen={c.cohen} cliffs={c.cliffs} KS={c.ks} exceeded={c.exceeded}')
        if c.same is None: print('Insufficient observations: need at least two per group.')
        else:
            print(f'Observed lower-mean code: {"1" if c.left.mean<c.right.mean else "0" if c.left.mean>c.right.mean else "tie"}')
    print('Unmatched historical groups: other settings vary; no repeated benchmarks establish noise or '
          'consistent per-configuration benefit. same=False is not automatic significance; same=True '
          'does not prove equivalence. Associations, not causal effects.')


def flow(table):
    print('Apache / Flow — APACHE-07 and APACHE-08')
    initial,incoming=train_test_split(table); inventory=Inventory(initial)
    goal=next(c for c in table.role('y') if c.name==TARGET)
    print(f'Initial benchmarks={len(initial.rows)} incoming={len(incoming.rows)}; seed=42, original-row-ID arrival order.')
    print('Frozen APACHE-07 concerns measured results, not proposal predictions. Prior-only nearest-rank Q75; alert if measured value > threshold.')
    alerts=0
    for row in incoming.rows:
        values=sorted(r.cells[goal.index] for r in inventory.table.rows if r.cells[goal.index] is not None)
        threshold=values[ceil(.75*len(values))-1] if values else None
        observed=row.cells[goal.index]
        high=observed>threshold if observed is not None and threshold is not None else None
        alerts+= high is True
        print(f'arrival={row.number} prior_n={len(inventory.table.rows)} measured={observed} prior_Q75={threshold} high={high}')
        inventory.accept(row.cells,number=row.number)
    print(f'APACHE-08 accepted={inventory.accepted_count} alerts={alerts} final_n={len(inventory.table.rows)}')
    print(f'Final performance summary: {inventory.summaries[goal.index]}')
    print('Final shortlist: '+str(tuple((r.number,s) for r,s in inventory.shortlist)))
    if inventory.table!=table or inventory.shortlist!=tuple(rank_rows(table)[:5]):
        raise RuntimeError('Final inventory/ranking differs from batch result')
    print('Full-batch verification: PASS. Exact summaries and ranks recomputed after each completion; '
          'no constant-time claim. Unbenchmarked proposals require separate X-only prediction, not this measured-result assessment.')


def blame(table):
    print('Apache / Blame — APACHE-09 and APACHE-10')
    groups=outcome_groups(table,TARGET,higher_better=False)
    variable={c.name for c in varying(table)}
    findings=tuple(f for f in compare_groups(groups.poor,groups.favorable) if f.column.name in variable)
    print(f'Poor/favorable quartiles: {len(groups.poor.rows)}/{len(groups.favorable.rows)}; '
          f'means={groups.poor_outcome.mean}/{groups.favorable_outcome.mean}; constants excluded from feature rankings.')
    show_findings(findings)
    e=Explainer(table); source=min(e.ranked,key=lambda item:(-item[1],item[0].number))[0]
    print(f'APACHE-09 poor source={source.number} performance={source.cells[next(c.index for c in table.role("y") if c.name==TARGET)]}; source settings={tuple((c.name,source.cells[c.index]) for c in table.role("x"))}')
    for neighbor in e.neighbors(source.number,5):
        row=neighbor.explanation.row
        differences=tuple(c.name for c in table.role('x') if source.cells[c.index]!=row.cells[c.index])
        print(f'Comparable row={row.number} X_distance={neighbor.distance:.9f} '
              f'goals={tuple(g.raw for g in neighbor.explanation.goals)} changed_settings={differences}')
    projected=categorical_pairs(table)
    poor_ids={r.number for r in groups.poor.rows}; favorable_ids={r.number for r in groups.favorable.rows}
    pair_findings=compare_groups(Table(projected.columns,tuple(r for r in projected.rows if r.number in poor_ids)),
                                 Table(projected.columns,tuple(r for r in projected.rows if r.number in favorable_ids)))
    evidence=sorted(((c.difference,f.column.name,c) for f in pair_findings for c in f.categories
                     if c.poor_count>=5 and c.difference>0),key=lambda item:(-item[0],item[1],item[2].value))
    print('APACHE-10 pairwise joint settings: poor support >=5/48, ranked by poor-minus-favorable proportion, top five.')
    for difference,name,c in evidence[:5]:
        print(f'{name} values={c.value}: poor/favorable={c.poor_count}/{c.favorable_count}; '
              f'proportions={c.poor_proportion:.6f}/{c.favorable_proportion:.6f}; difference={difference:.6f}')
    print('Fixed floor(n/4) extremes, ties by original ID. Overlapping pair counts are not independent; '
          'no interaction significance or causal explanation of an individual configuration is established.')
