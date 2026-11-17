"""A known-good solution to all 60 requests, used only to test the semantic checker.

Standard library only, and deliberately written apart from harness/oracle.py
(different aggregates, distances and tests where the requests allow it), so
passing shows the checks accept legitimate methods rather than one recipe.

    python reference_solution.py <csv> <skill 1..10>
"""

import json
import random
import sys
from math import sqrt

ORD = {"vl": 0, "l": 1, "n": 2, "h": 3, "vh": 4, "xh": 5}


def cell(s):
    s = s.strip()
    if s in ("", "?"):
        return None
    try:
        return float(s)
    except ValueError:
        return s


def load(path):
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    names = [s.strip() for s in lines[0].split(",")]
    rows = [[cell(c) for c in ln.split(",")] for ln in lines[1:]]
    return names, rows


class T:
    def __init__(self, path):
        self.names, self.rows = load(path)
        self.n = len(self.rows)
        self.goals = [c for c in self.names if c[-1] in "+-"]
        self.inputs = [c for c in self.names if c[-1] not in "+-X"]
        self.features = [c for c in self.names if c[-1] not in "+-"]
        self.domain = "cars" if "Mpg+" in self.names else "nasa" if "Effort-" in self.names else "apache"

    def v(self, r, c):
        return self.rows[r - 1][self.names.index(c)]

    def col(self, c, rows=None):
        rows = range(1, self.n + 1) if rows is None else rows
        return [self.v(r, c) for r in rows if self.v(r, c) is not None]

    def num(self, c):
        return all(isinstance(x, float) for x in self.col(c))


def mean(xs):
    return sum(xs) / len(xs)


def median(xs):
    s = sorted(xs)
    return (s[(len(s) - 1) // 2] + s[len(s) // 2]) / 2


def q(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


def sd(xs):
    m = mean(xs)
    return sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))


def numeric(x):
    return x if isinstance(x, float) else ORD.get(str(x).strip().lower(), None) if x is not None else None


def held(n):
    return [r for r in range(1, n + 1) if r % 5 == 0]


def k_arrivals(n):
    return int(0.2 * n + 0.5)


def budget(n):
    return min(30, n // 5)


def score(t, rows=None):
    """Rank-sum aggregate (lower = better): deliberately not distance-to-heaven."""
    rows = list(range(1, t.n + 1)) if rows is None else list(rows)
    total = {r: 0.0 for r in rows}
    for g in t.goals:
        sign = -1 if g.endswith("+") else 1
        order = sorted(rows, key=lambda r: sign * t.v(r, g))
        for i, r in enumerate(order):
            total[r] += i
    return total


def zscore(t, rows=None):
    """Z-score aggregate (lower = better), for places a rank sum is too coarse."""
    rows = list(range(1, t.n + 1)) if rows is None else list(rows)
    out = {r: 0.0 for r in rows}
    for g in t.goals:
        vs = t.col(g, rows)
        m, s = mean(vs), sd(vs) or 1.0
        sign = -1 if g.endswith("+") else 1
        for r in rows:
            out[r] += sign * (t.v(r, g) - m) / s
    return out


def dist_fn(t, cols, rows=None):
    rows = list(range(1, t.n + 1)) if rows is None else rows
    span = {}
    for c in cols:
        vs = [numeric(t.v(r, c)) for r in rows]
        vs = [x for x in vs if x is not None]
        span[c] = (min(vs), max(vs)) if vs else (0, 1)

    def d(a, b):
        tot = 0.0
        for c in cols:
            x, y = numeric(t.v(a, c)), numeric(t.v(b, c))
            if x is None or y is None:
                tot += 1
                continue
            lo, hi = span[c]
            tot += (abs(x - y) / (hi - lo) if hi > lo else 0) ** 2
        return sqrt(tot / len(cols))
    return d


def knn(t, target, train, query, k=5):
    d = dist_fn(t, t.inputs, train)
    near = sorted(train, key=lambda r: d(query, r))[:k]
    return mean([t.v(r, target) for r in near])


def ans(rid, obj):
    print(f"== {rid} ==")
    print("reference answer")
    print("value 1")
    print("value 2")
    print(f"ANSWER {rid} {json.dumps(obj)}")


def cliffs(xs, ys):
    gt = sum(x > y for x in xs for y in ys)
    lt = sum(x < y for x in xs for y in ys)
    return (gt - lt) / (len(xs) * len(ys))


def mw_verdict(xs, ys):
    return "different" if abs(cliffs(xs, ys)) >= 0.33 else "same"


IDS = {
    "cars": ["CAR-%02d" % i for i in range(1, 21)],
    "nasa": ["NASA-%02d" % i for i in range(1, 21)],
    "apache": ["APACHE-%02d" % i for i in range(1, 21)],
}


# ---- the kinds ----------------------------------------------------------------

def summary(t):
    out = {}
    for g in t.goals:
        vs = t.col(g)
        best, worst = (max(vs), min(vs)) if g.endswith("+") else (min(vs), max(vs))
        out[g] = {"mean": mean(vs), "median": median(vs), "min": min(vs), "max": max(vs), "best": best, "worst": worst}
    return out


def top(t):
    s = zscore(t)
    return {"top": sorted(s, key=s.get)[:10]}


def top_settings(t):
    g = t.goals[0]
    order = sorted(range(1, t.n + 1), key=lambda r: t.v(r, g))
    tops = order[:max(5, t.n // 15)][:20]
    rest = [r for r in range(1, t.n + 1) if r not in tops]
    sh = lambda c, rs: sum(t.v(r, c) == 1.0 for r in rs) / len(rs)
    items = sorted(({"setting": c, "top_share": sh(c, tops), "rest_share": sh(c, rest)} for c in t.inputs),
                   key=lambda e: -abs(e["top_share"] - e["rest_share"]))
    return {"top": tops, "settings": items[:3]}


def predict(t, target):
    train = [r for r in range(1, t.n + 1) if r % 5]
    preds = {r: knn(t, target, train, r) for r in held(t.n)}
    mae = mean([abs(p - t.v(r, target)) for r, p in preds.items()])
    return {"predictions": {str(r): p for r, p in preds.items()}, "mae": mae}


def predict_interval(t, target):
    train = [r for r in range(1, t.n + 1) if r % 5]
    resid = [abs(knn(t, target, [x for x in train if x != r], r) - t.v(r, target)) for r in train]
    w = q(resid, 0.8)
    out = {}
    for r in held(t.n):
        e = knn(t, target, train, r)
        out[str(r)] = {"estimate": e, "low": e - w, "high": e + w}
    return {"predictions": out}


def pair_diff(t, target):
    effects = {}
    pairs = []
    for a in range(1, t.n + 1):
        for b in range(a + 1, t.n + 1):
            diff = [c for c in t.inputs if t.v(a, c) != t.v(b, c)]
            if len(diff) == 1:
                c = diff[0]
                sign = 1 if t.v(b, c) > t.v(a, c) else -1
                effects.setdefault(c, []).append(sign * (t.v(b, target) - t.v(a, target)))
                pairs.append((a, b, c, sign))
    out = []
    for a, b, c, sign in pairs[:8]:
        e = effects[c]
        out.append({"a": a, "b": b, "setting": c, "difference": sign * median(e),
                    "low": sign * q(e, 0.1) if sign > 0 else sign * q(e, 0.9),
                    "high": sign * q(e, 0.9) if sign > 0 else sign * q(e, 0.1)})
    return {"pairs": out}


def groups(t, group, outcome):
    vals = sorted(set(t.col(group)), key=str)
    mem = {v: [t.v(r, outcome) for r in range(1, t.n + 1) if t.v(r, group) == v] for v in vals}
    key = lambda v: str(int(v)) if isinstance(v, float) else v.strip()
    out = {key(v): {"n": len(ys), "median": median(ys), "mean": mean(ys)} for v, ys in mem.items()}
    comps = []
    for i, a in enumerate(vals):
        for b in vals[i + 1:]:
            comps.append({"a": key(a), "b": key(b), "verdict": mw_verdict(mem[a], mem[b])})
    return {"groups": out, "comparisons": comps}


def year_groups(t):
    s = zscore(t)
    a = [s[r] for r in s if t.v(r, "Model") <= 75]
    b = [s[r] for r in s if t.v(r, "Model") >= 76]
    return {"groups": [{"from": 70, "to": 75, "n": len(a)}, {"from": 76, "to": 82, "n": len(b)}],
            "verdict": mw_verdict(a, b)}


def stream_desirable(t):
    k = k_arrivals(t.n)
    out = []
    for r in range(t.n - k + 1, t.n + 1):
        seen = list(range(1, r + 1))
        s = zscore(t, seen)
        out.append({"row": r, "desirable": s[r] <= q(list(s.values()), 0.25)})
    return {"assessments": out}


def stream_alert_pred(t):
    k = k_arrivals(t.n)
    out = []
    for r in range(t.n - k + 1, t.n + 1):
        before = list(range(1, r))
        pe, pd = knn(t, "Effort-", before, r), knn(t, "Defects-", before, r)
        alert = pe > q(t.col("Effort-", before), 0.75) or pd > q(t.col("Defects-", before), 0.75)
        out.append({"row": r, "predicted_effort": pe, "predicted_defects": pd, "alert": alert})
    return {"assessments": out}


def stream_alert_poor(t, target):
    k = k_arrivals(t.n)
    out = []
    for r in range(t.n - k + 1, t.n + 1):
        before = t.col(target, range(1, r))
        out.append({"row": r, "alert": t.v(r, target) > q(before, 0.75)})
    return {"assessments": out}


def stream_summary(t):
    k = k_arrivals(t.n)
    cps = []
    for m in (0, k // 2, k):
        seen = list(range(1, t.n - k + m + 1))
        summ = {g: {"n": len(t.col(g, seen)), "mean": mean(t.col(g, seen)), "min": min(t.col(g, seen)),
                    "max": max(t.col(g, seen))} for g in t.goals}
        s = zscore(t, seen)
        cps.append({"after": m, "summary": summ, "shortlist": sorted(s, key=s.get)[:5]})
    return {"checkpoints": cps}


def _bad_score(t, outcome):
    if outcome:
        sign = -1 if outcome.endswith("+") else 1
        return {r: sign * t.v(r, outcome) for r in range(1, t.n + 1)}
    return zscore(t)


def blame_rank(t, outcome):
    s = _bad_score(t, outcome)
    order = sorted(s, key=s.get, reverse=True)
    poor = order[:max(5, t.n // 5)]
    rest = [r for r in range(1, t.n + 1) if r not in poor]
    assoc = {}
    for f in t.features:
        xs = [numeric(t.v(r, f)) for r in poor]
        ys = [numeric(t.v(r, f)) for r in rest]
        xs, ys = [x for x in xs if x is not None], [y for y in ys if y is not None]
        assoc[f] = abs(cliffs(xs, ys)) if xs and ys else 0
    return {"poor": poor, "ranking": sorted(assoc, key=lambda f: -assoc[f])[:5]}


def _typical(t, f, rs):
    vs = t.col(f, rs)
    if t.num(f):
        return mean(vs)
    return max(sorted(set(vs)), key=vs.count).strip()


def blame_contrast(t, outcome):
    s = _bad_score(t, outcome)
    order = sorted(s, key=s.get, reverse=True)
    bad, good = order[:t.n // 4], order[-(t.n // 4):]
    assoc = {}
    for f in t.features:
        xs = [numeric(t.v(r, f)) for r in bad]
        ys = [numeric(t.v(r, f)) for r in good]
        xs, ys = [x for x in xs if x is not None], [y for y in ys if y is not None]
        assoc[f] = abs(cliffs(xs, ys)) if xs and ys else 0
    feats = sorted(assoc, key=lambda f: -assoc[f])[:4]
    return {"bad": bad, "good": good,
            "differences": [{"feature": f, "bad": _typical(t, f, bad), "good": _typical(t, f, good)} for f in feats]}


def blame_combos(t, outcome):
    s = _bad_score(t, outcome)
    order = sorted(s, key=s.get, reverse=True)
    bad, good = order[:t.n // 4], order[-(t.n // 4):]
    best = None
    for i, a in enumerate(t.inputs):
        for b in t.inputs[i + 1:]:
            for va in (0.0, 1.0):
                for vb in (0.0, 1.0):
                    m = lambda r: t.v(r, a) == va and t.v(r, b) == vb
                    sb, sg = sum(map(m, bad)) / len(bad), sum(map(m, good)) / len(good)
                    if best is None or sb - sg > best[0]:
                        best = (sb - sg, {"settings": {a: int(va), b: int(vb)}, "bad_share": sb, "good_share": sg})
    return {"bad": bad, "good": good, "combinations": [best[1]]}


def justify_neighbors(t):
    s = zscore(t)
    rec = min(s, key=s.get)
    d = dist_fn(t, t.inputs)
    near = sorted((r for r in range(1, t.n + 1) if r != rec), key=lambda r: d(rec, r))[:4]
    cols = t.goals + t.inputs[:2]
    return {"recommended": rec, "neighbors": [{"row": r, "values": {c: t.v(r, c) for c in cols}} for r in near]}


def justify_estimate(t):
    p = held(t.n)[0]
    train = [r for r in range(1, t.n + 1) if r % 5]
    d = dist_fn(t, t.inputs, train)
    near = sorted(train, key=lambda r: d(p, r))[:3]
    return {"project": p, "estimate": mean([t.v(r, "Effort-") for r in near]),
            "neighbors": [{"row": r, "values": {"Effort-": t.v(r, "Effort-"), "Kloc+": t.v(r, "Kloc+")}} for r in near]}


def justify_pair(t):
    s = zscore(t)
    order = sorted(s, key=s.get)
    a = order[0]
    for b in order[1:]:
        better = [g for g in t.goals if (t.v(b, g) > t.v(a, g)) == g.endswith("+") and t.v(b, g) != t.v(a, g)]
        if s[b] != s[a] and (better or len(t.goals) == 1):
            break
    diffs = [{"feature": c, "a": t.v(a, c), "b": t.v(b, c)} for c in t.names if t.v(a, c) != t.v(b, c)][:4]
    return {"a": a, "b": b, "preferred": a, "scores": {str(a): -s[a], str(b): -s[b]}, "differences": diffs,
            "tradeoffs": [g for g in t.goals if t.v(b, g) != t.v(a, g) and (t.v(b, g) > t.v(a, g)) == g.endswith("+")]}


def _active(t, seed=1):
    """Label a few random rows, then repeatedly label the unlabelled row nearest the best labelled one."""
    rng = random.Random(seed)
    b = budget(t.n)
    order = rng.sample(range(1, t.n + 1), 4)
    d = dist_fn(t, t.inputs)
    while len(order) < b:
        s = zscore(t, order)
        best = min(order, key=s.get)
        rest = [r for r in range(1, t.n + 1) if r not in order]
        order.append(min(rest, key=lambda r: d(best, r)))
    return order


def choose_best(t):
    ev = _active(t)
    s = zscore(t, ev)
    return {"evaluated": ev, "best": min(ev, key=s.get)}


def choose_rule(t):
    s = zscore(t)
    best = None
    for f in t.inputs:
        for v in sorted(set(t.col(f)), key=str):
            rs = [r for r in range(1, t.n + 1) if t.v(r, f) == v]
            if len(rs) >= 5:
                m = median([s[r] for r in rs])
                if best is None or m < best[0]:
                    best = (m, f, v, len(rs))
    _, f, v, n = best
    return {"rule": {f: v.strip() if isinstance(v, str) else [v, v]}, "matching": n}


def choose_order(t):
    return {"order": _active(t)}


def fix_alternatives(t):
    s = zscore(t)
    poor = max(s, key=s.get)
    d = dist_fn(t, t.inputs)
    near = sorted((r for r in range(1, t.n + 1) if r != poor), key=lambda r: d(poor, r))[:8]
    alts = [r for r in near if s[r] < s[poor]][:3]
    out = []
    for a in alts:
        ch = {c: [t.v(poor, c), t.v(a, c)] for c in t.inputs if t.v(poor, c) != t.v(a, c)}
        if ch:
            out.append({"row": a, "changes": ch})
    return {"row": poor, "alternatives": out}


def fix_observed(t, target):
    order = sorted(range(1, t.n + 1), key=lambda r: -t.v(r, target))
    for r in order:
        cands = [a for a in range(1, t.n + 1)
                 if 1 <= sum(t.v(r, c) != t.v(a, c) for c in t.inputs) <= 2 and t.v(a, target) < t.v(r, target)]
        if cands:
            a = min(cands, key=lambda x: t.v(x, target))
            ch = {c: [t.v(r, c), t.v(a, c)] for c in t.inputs if t.v(r, c) != t.v(a, c)}
            return {"row": r, "alternative": a, "changes": ch,
                    "performance": {"before": t.v(r, target), "after": t.v(a, target)}}


def _predict_row(t, target, cells, train):
    """kNN prediction for a hypothetical row given as {column: value}."""
    cols = t.inputs
    span = {}
    for c in cols:
        vs = [numeric(t.v(r, c)) for r in train]
        vs = [x for x in vs if x is not None]
        span[c] = (min(vs), max(vs)) if vs else (0, 1)

    def d(r):
        tot = 0.0
        for c in cols:
            x, y = numeric(cells[c]), numeric(t.v(r, c))
            if x is None or y is None:
                tot += 1
                continue
            lo, hi = span[c]
            tot += (abs(x - y) / (hi - lo) if hi > lo else 0) ** 2
        return tot
    near = sorted(train, key=d)[:5]
    return mean([t.v(r, target) for r in near])


def fix_predicted(t, what, max_changes=None, tradeoff=False):
    target = {"effort": "Effort-", "defects": "Defects-"}[what]
    allowed = [c for c in t.inputs if c != "Kloc+"]
    order = sorted(range(1, t.n + 1), key=lambda r: -t.v(r, target))
    for p in order:
        train = [r for r in range(1, t.n + 1) if r != p]
        cells = {c: t.v(p, c) for c in t.inputs}
        before = _predict_row(t, target, cells, train)
        best = None
        for c in allowed:
            for v in sorted(set(t.col(c)), key=str):
                if v == cells[c]:
                    continue
                new = dict(cells)
                new[c] = v
                after = _predict_row(t, target, new, train)
                if after < before and (best is None or after < best[0]):
                    best = (after, c, v, new)
        if best:
            after, c, v, new = best
            out = {"project": p, "changes": {c: [cells[c].strip() if isinstance(cells[c], str) else cells[c],
                                                 v.strip() if isinstance(v, str) else v]},
                   f"predicted_{what}": {"before": before, "after": after}}
            if tradeoff:
                out["predicted_effort"] = {"before": _predict_row(t, "Effort-", cells, train),
                                           "after": _predict_row(t, "Effort-", new, train)}
            return out


def fix_targets(t):
    th = {g: median(t.col(g)) for g in t.goals}
    meets = lambda x: sum((t.v(x, g) >= th[g]) if g.endswith("+") else (t.v(x, g) <= th[g]) for g in t.goals)
    s = zscore(t)
    r = max(s, key=s.get)
    diff = lambda x: [c for c in t.inputs if t.v(r, c) != t.v(x, c)]
    cands = [x for x in range(1, t.n + 1) if x != r and meets(x) > meets(r)]
    a = min(cands, key=lambda x: (len(diff(x)), x))
    return {"row": r, "targets": th, "alternative": a, "changed": diff(a)}


def fix_target(t, target):
    r = max(range(1, t.n + 1), key=lambda x: t.v(x, target))
    tv = median(t.col(target))
    gap = lambda x: abs(t.v(x, target) - tv)
    diff = lambda x: [c for c in t.inputs if t.v(r, c) != t.v(x, c)]
    cands = [x for x in range(1, t.n + 1) if gap(x) < gap(r)]
    a = min(cands, key=lambda x: (len(diff(x)), gap(x)))
    return {"row": r, "target": tv, "alternative": a, "changed": diff(a)}


def spot_x(t):
    d = dist_fn(t, t.inputs)
    sc = {}
    for r in range(1, t.n + 1):
        ds = sorted(d(r, o) for o in range(1, t.n + 1) if o != r)
        sc[r] = mean(ds[:3])
    fl = sorted(sc, key=lambda r: -sc[r])[:max(1, t.n // 25)]
    return {"flagged": fl, "scores": {str(r): sc[r] for r in fl}}


def spot_y(t, cols):
    d = dist_fn(t, t.inputs)
    items = []
    for c in cols:
        s = sd(t.col(c)) or 1
        for r in range(1, t.n + 1):
            near = sorted((o for o in range(1, t.n + 1) if o != r), key=lambda o: d(r, o))[:4]
            exp = mean([t.v(o, c) for o in near])
            items.append((abs(t.v(r, c) - exp) / s, r, c, exp))
    items.sort(reverse=True)
    return {"flagged": [{"row": r, "column": c, "actual": t.v(r, c), "expected": e}
                        for _, r, c, e in items[:max(1, t.n // 25)]]}


def race_select(t):
    b, reps = budget(t.n), 20
    s = zscore(t)
    ours, rnd = [], []
    for i in range(reps):
        ev = _active(t, seed=i + 1)
        se = zscore(t, ev)
        ours.append(min(ev, key=se.get))
        pick = random.Random(100 + i).sample(range(1, t.n + 1), b)
        sp = zscore(t, pick)
        rnd.append(min(pick, key=sp.get))
    xo, xr = [s[r] for r in ours], [s[r] for r in rnd]
    v = "no clear difference" if abs(cliffs(xo, xr)) < 0.33 else ("ours better" if median(xo) < median(xr) else "random better")
    return {"budget": b, "repeats": reps, "ours": ours, "random": rnd, "verdict": v}


def race_predict(t, target):
    train = [r for r in range(1, t.n + 1) if r % 5]
    preds = {r: knn(t, target, train, r) for r in held(t.n)}
    m = mean(t.col(target, train))
    em = [abs(p - t.v(r, target)) for r, p in preds.items()]
    eb = [abs(m - t.v(r, target)) for r in preds]
    v = "no clear difference" if abs(cliffs(em, eb)) < 0.33 else ("better" if median(em) < median(eb) else "worse")
    return {"predictions": {str(r): p for r, p in preds.items()}, "baseline": m, "mae_model": mean(em),
            "mae_baseline": mean(eb), "verdict": v}


SOLVERS = {
    "cars": [summary, top, lambda t: predict(t, "Mpg+"), lambda t: predict_interval(t, "Lbs-"),
             lambda t: groups(t, "Clndrs", "Mpg+"), year_groups, stream_desirable, stream_summary,
             lambda t: blame_rank(t, "Mpg+"), lambda t: blame_contrast(t, None), justify_neighbors, justify_pair,
             choose_best, choose_order, fix_alternatives, fix_targets, spot_x, lambda t: spot_y(t, ["Mpg+", "Lbs-"]),
             race_select, lambda t: race_predict(t, "Mpg+")],
    "nasa": [summary, top, lambda t: predict(t, "Effort-"), lambda t: predict_interval(t, "Defects-"),
             lambda t: groups(t, "pcap", "Effort-"), lambda t: groups(t, "cplx", "Defects-"), stream_alert_pred,
             stream_summary, lambda t: blame_rank(t, "Effort-"), lambda t: blame_contrast(t, "Defects-"),
             justify_estimate, justify_pair, choose_rule, choose_order, lambda t: fix_predicted(t, "effort"),
             lambda t: fix_predicted(t, "defects", 3, True), spot_x, lambda t: spot_y(t, ["Effort-", "Defects-"]),
             lambda t: race_predict(t, "Effort-"), race_select],
    "apache": [summary, top_settings, lambda t: predict(t, "Performance-"), lambda t: pair_diff(t, "Performance-"),
               lambda t: groups(t, "keepAlive", "Performance-"), lambda t: groups(t, "accessLog", "Performance-"),
               lambda t: stream_alert_poor(t, "Performance-"), stream_summary, lambda t: blame_rank(t, "Performance-"),
               lambda t: blame_combos(t, "Performance-"), justify_neighbors, justify_pair, choose_best, choose_order,
               lambda t: fix_observed(t, "Performance-"), lambda t: fix_target(t, "Performance-"),
               lambda t: spot_y(t, ["Performance-"]), spot_x, race_select, lambda t: race_predict(t, "Performance-")],
}


def main(path, skill):
    t = T(path)
    ids = IDS[t.domain]
    for i in (2 * skill - 2, 2 * skill - 1):
        ans(ids[i], SOLVERS[t.domain][i](t))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
