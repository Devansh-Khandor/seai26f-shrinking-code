"""Bounded categorical what-if search; predictions, never causal interventions."""

from dataclasses import dataclass
from itertools import combinations, product
from math import isfinite
from .data import Row
from .prediction import KNNRegressor


@dataclass(frozen=True)
class Variant:
    query: Row
    changes: tuple
    predictions: tuple
    improvement: float
    historically_observed: bool


@dataclass(frozen=True)
class WhatIf:
    original: tuple
    candidates: tuple
    evaluations: int


def search_changes(training, query, target, fields, other_targets=(), max_changes=1,
                   min_fraction=.05, k=5, cap=1000):
    """Return improving variants at the first successful change count, sorted by
    improvement descending then (field name, value). Predict only masked X.
    """
    if isinstance(max_changes,bool) or not isinstance(max_changes,int) or max_changes not in (1,2):
        raise ValueError('Search supports one or two changed fields')
    if not isfinite(min_fraction) or not 0 < min_fraction <= 1:
        raise ValueError('Minimum fractional improvement must be in (0,1]')
    if isinstance(cap,bool) or not isinstance(cap,int) or cap<1:
        raise ValueError('Positive enumeration cap required')
    if len(query.cells)!=len(training.columns) or any(query.cells[c.index] is not None for c in training.columns if c.role!='x'):
        raise ValueError('Query must contain X only')
    columns={c.name:c for c in training.role('x')}
    fields=tuple(fields)
    if not fields or len(set(fields))!=len(fields) or any(f not in columns or columns[f].numeric for f in fields):
        raise ValueError('Allowlist must contain unique categorical X fields')
    model=KNNRegressor.fit(training,k=k)
    if model.target_column(target).ideal!=0:
        raise ValueError('Improvement target must be minimized')
    targets=tuple(dict.fromkeys((target,)+tuple(other_targets)))
    predict=lambda row: tuple((name,model.predict(row,name)) for name in targets)
    original=predict(query)
    baseline=original[0][1].value
    required=max(abs(baseline)*min_fraction,1e-9)
    levels={f:sorted({r.cells[columns[f].index] for r in training.rows if r.cells[columns[f].index] is not None}
                     - {query.cells[columns[f].index]}) for f in fields}
    plans=[]
    for size in range(1,max_changes+1):
        layer=[]
        for names in combinations(sorted(fields),size):
            for values in product(*(levels[name] for name in names)):
                layer.append(tuple(zip(names,values)))
                if sum(map(len,plans))+len(layer)>cap:
                    raise ValueError('Search space exceeds enumeration cap')
        plans.append(layer)
    x=training.role('x')
    observed={tuple(r.cells[c.index] for c in x) for r in training.rows}
    evaluations=0
    for layer in plans:
        winners=[]
        for changes in layer:
            cells=list(query.cells)
            for name,value in changes: cells[columns[name].index]=value
            variant=Row(query.number,tuple(cells))
            predictions=predict(variant); evaluations+=1
            improvement=baseline-predictions[0][1].value
            if improvement>=required:
                winners.append(Variant(variant,changes,predictions,improvement,
                                       tuple(variant.cells[c.index] for c in x) in observed))
        if winners:
            return WhatIf(original,tuple(sorted(winners,key=lambda v:(-v.improvement,v.changes))),evaluations)
    return WhatIf(original,(),evaluations)
