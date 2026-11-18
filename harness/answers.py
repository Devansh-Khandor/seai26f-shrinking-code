"""Machine-checkable answers for all 60 frozen requests, and the checks behind them.

Each request maps to a KIND: the JSON its program must print on one line,
`ANSWER <ID> {...}`, plus checks that recompute the truth from the CSV the
program was given (harness/oracle.py). The customer wording stays frozen;
only the answer format and the shared protocol (held-out rows, arrivals,
budget) are added by the harness. Checks test what the request demands, not
one algorithm: where several methods are legitimate, the check accepts any
answer consistent with the data, and only clear-cut cases are judged.
"""

from dataclasses import dataclass, field
import json
import re

from harness import oracle as O

ANSWER_LINE = re.compile(r"^ANSWER\s+(\S+)\s+(\{.*\})\s*$")


class SchemaError(Exception):
    pass


@dataclass
class Ctx:
    data: object
    ans: dict
    params: dict
    run: str = "base"
    items: list = field(default_factory=list)   # (check name, ok, detail)
    extra: dict = field(default_factory=dict)

    def check(self, name, ok, detail=""):
        self.items.append((f"{self.run}:{name}", bool(ok), "" if ok else detail))
        return ok


def parse_answers(stdout):
    """{request id: parsed JSON or SchemaError message (str)}; the last ANSWER line per id wins."""
    out = {}
    for line in stdout.splitlines():
        m = ANSWER_LINE.match(line.strip())
        if not m:
            continue
        try:
            v = json.loads(m.group(2))
            out[m.group(1)] = v if isinstance(v, dict) else "answer is not a JSON object"
        except json.JSONDecodeError as e:
            out[m.group(1)] = f"invalid JSON: {e}"
    return out


# ---- schema helpers --------------------------------------------------------

def need(d, key, kind=None):
    if not isinstance(d, dict) or key not in d:
        raise SchemaError(f"missing key {key!r}")
    v = d[key]
    if kind == "list" and not isinstance(v, list):
        raise SchemaError(f"{key!r} must be a list")
    if kind == "dict" and not isinstance(v, dict):
        raise SchemaError(f"{key!r} must be an object")
    return v


def num(v, what="value"):
    if isinstance(v, bool):
        raise SchemaError(f"{what} must be a number, got {v!r}")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            pass
    raise SchemaError(f"{what} must be a number, got {v!r}")


def row(data, v, what="row"):
    try:
        f = float(v) if not isinstance(v, bool) else None
    except (TypeError, ValueError):
        f = None
    if f is None or f != int(f) or not 1 <= int(f) <= data.n:
        raise SchemaError(f"{what} must be a row number 1..{data.n}, got {v!r}")
    return int(f)


def rows(data, vs, what="rows", distinct=True):
    if not isinstance(vs, list):
        raise SchemaError(f"{what} must be a list of row numbers")
    out = [row(data, v, what) for v in vs]
    if distinct and len(set(out)) != len(out):
        raise SchemaError(f"{what} has duplicate rows")
    return out


def column(data, v, allowed, what="column"):
    if not isinstance(v, str) or v.strip() not in allowed:
        raise SchemaError(f"{what} {v!r} is not one of {sorted(allowed)}")
    return v.strip()


def row_map(data, d, what):
    if not isinstance(d, dict):
        raise SchemaError(f"{what} must be an object keyed by row number")
    return {row(data, k, f"{what} key"): v for k, v in d.items()}


def same_cell(data, r, col, v):
    actual = data.value(r, col)
    if actual is None:
        return v in (None, "?", "")
    if isinstance(actual, float):
        try:
            return O.close(num(v), actual)
        except SchemaError:
            return False
    return isinstance(v, (str, int, float)) and str(v).strip().lower() == actual.strip().lower()


def is_bool(v):
    return isinstance(v, bool)


def key_matches(data, col, key, value):
    """Does group KEY (as written by the program) name VALUE of COL?"""
    if isinstance(value, float):
        try:
            return float(key) == value
        except ValueError:
            return False
    return str(key).strip().lower() == value.strip().lower()


# ---- the kinds ---------------------------------------------------------------
# Each kind: TEXT (prompt description, str.format with params) and check(ctx).
# Optional: compare(ctx_new, ctx_base) for paired runs, and the extra runs it needs.

KINDS = {}


def kind(name, text, runs=()):
    def wrap(fn):
        KINDS[name] = (text, fn, tuple(runs))
        return fn
    return wrap


COMPARE = {}


def compare(name):
    def wrap(fn):
        COMPARE[name] = fn
        return fn
    return wrap


# Remember ------------------------------------------------------------------

@kind("summary", '{{{goal_keys}}}: one entry per goal column, keyed by its exact name, each '
      '{{"mean": number, "median": number, "min": number, "max": number, "best": number, "worst": number}}; '
      'best/worst are the best and worst observed values given the column\'s direction.')
def _summary(c):
    for g, w in c.data.goals():
        e = need(c.ans, g, "dict")
        vs = c.data.values(g)
        lo, hi = min(vs), max(vs)
        best, worst = (hi, lo) if w > 0 else (lo, hi)
        c.check(f"{g} mean", O.close(num(need(e, "mean")), O.mean(vs)), f"mean {e['mean']} != {O.mean(vs):.4f}")
        med = num(need(e, "median"))
        c.check(f"{g} median", any(O.close(med, m) for m in O.medians(vs)), f"median {med} != {O.median(vs)}")
        c.check(f"{g} min", O.close(num(need(e, "min")), lo), f"min {e['min']} != {lo}")
        c.check(f"{g} max", O.close(num(need(e, "max")), hi), f"max {e['max']} != {hi}")
        c.check(f"{g} best", O.close(num(need(e, "best")), best), f"best {e['best']} != {best} ({g} direction)")
        c.check(f"{g} worst", O.close(num(need(e, "worst")), worst), f"worst {e['worst']} != {worst}")


@kind("top", '{{"top": [the 10 row numbers with the best overall combination of {goal_list} judged together, '
      'best first]}}')
def _top(c):
    top = rows(c.data, need(c.ans, "top", "list"), "top")
    if len(top) != 10:
        raise SchemaError(f"top has {len(top)} rows, expected 10")
    good = O.good_rows(c.data, 0.25)
    hits = sum(r in good for r in top)
    c.check("rows are good overall", hits >= 8,
            f"only {hits}/10 listed rows are Pareto-optimal or in the best 25% by any overall score")
    for g, w in c.data.goals():
        vs = sorted((w * c.data.value(r, g) for r in range(1, c.data.n + 1)), reverse=True)
        cut = vs[9]
        tied = [r for r in range(1, c.data.n + 1) if w * c.data.value(r, g) >= cut]
        if len(tied) <= 15:
            c.check(f"not judged on {g} alone", not set(top) <= set(tied),
                    f"the list is just the top rows by {g}")


@kind("top_settings", '{{"top": [row numbers of the unusually favorable configurations, 5 to 20 rows], '
      '"settings": [{{"setting": column name, "top_share": fraction of top rows where it is 1, '
      '"rest_share": fraction of all other rows where it is 1}}, at least 2, most distinguishing first]}}')
def _top_settings(c):
    top = rows(c.data, need(c.ans, "top", "list"), "top")
    if not 5 <= len(top) <= 20:
        raise SchemaError(f"top has {len(top)} rows, expected 5..20")
    good = O.best_fraction(O.d2h(c.data), 0.25)
    hits = sum(r in good for r in top)
    c.check("top rows are favorable", hits >= 0.8 * len(top), f"only {hits}/{len(top)} in the best 25%")
    rest = [r for r in range(1, c.data.n + 1) if r not in set(top)]
    inputs = c.data.inputs()
    share = lambda col, rs: sum(c.data.value(r, col) == 1.0 for r in rs) / len(rs)
    diffs = {col: abs(share(col, top) - share(col, rest)) for col in inputs}
    items = need(c.ans, "settings", "list")
    if len(items) < 2:
        raise SchemaError("settings needs at least 2 entries")
    for it in items:
        col = column(c.data, need(it, "setting"), set(inputs), "setting")
        c.check(f"{col} shares", O.close(num(need(it, "top_share")), share(col, top), abs_=0.011)
                and O.close(num(need(it, "rest_share")), share(col, rest), abs_=0.011),
                f"{col} shares {it['top_share']}/{it['rest_share']} != "
                f"{share(col, top):.3f}/{share(col, rest):.3f}")
    first = column(c.data, items[0]["setting"], set(inputs))
    second_best = sorted(diffs.values(), reverse=True)[1]
    c.check("most distinguishing first", diffs[first] >= second_best - 1e-9,
            f"{first} is not among the two settings whose shares differ most")


# Guess ---------------------------------------------------------------------

def _heldout_preds(c, key="predictions", interval=False):
    d = row_map(c.data, need(c.ans, key, "dict"), key)
    want = set(O.held_out(c.data.n))
    if set(d) != want:
        raise SchemaError(f"{key} must cover exactly the {len(want)} held-out rows "
                          f"(missing {sorted(want - set(d))[:5]}, extra {sorted(set(d) - want)[:5]})")
    if not interval:
        return {r: num(v, f"prediction for row {r}") for r, v in d.items()}
    out = {}
    for r, v in d.items():
        out[r] = (num(need(v, "estimate")), num(need(v, "low")), num(need(v, "high")))
    return out


def _baseline_mae(c, target):
    train = [c.data.value(r, target) for r in range(1, c.data.n + 1)
             if r % 5 and c.data.value(r, target) is not None]
    m = O.mean(train)
    test = [c.data.value(r, target) for r in O.held_out(c.data.n)]
    return O.mean([abs(m - y) for y in test]), m


def _mae(c, preds, target):
    return O.mean([abs(p - c.data.value(r, target)) for r, p in preds.items()])


@kind("predict", '{{"predictions": {{"<row>": estimated {target} for every held-out row}}, '
      '"mae": mean absolute error of those estimates against the file\'s {target} values}}. '
      'Held-out rows\' {target} values are unknown to your model: fit on the other rows only.',
      runs=("mask",))
def _predict(c):
    t = c.params["target"]
    preds = _heldout_preds(c)
    mae = _mae(c, preds, t)
    if c.run == "mask":
        return
    c.check("mae is computed correctly", O.close(num(need(c.ans, "mae")), mae, rel=0.01), f"reported mae {c.ans['mae']} != {mae:.4f}")
    base, _ = _baseline_mae(c, t)
    c.check("beats the historical mean", mae < base, f"mae {mae:.3f} is not below the mean-predictor mae {base:.3f}")


@compare("predict")
@compare("race_predict")
def _no_leak(new, base):
    t = base.params["target"]
    a, b = _heldout_preds(base), _heldout_preds(new)
    s = O.sd(base.data.values(t)) or 1.0
    shift = O.median([abs(a[r] - b[r]) for r in a])
    new.check("held-out targets not used", shift <= 0.25 * s,
              f"changing only the held-out {t} values moved the estimates by a median {shift:.3g} "
              f"(> 0.25 sd = {0.25 * s:.3g})")


@kind("predict_interval", '{{"predictions": {{"<row>": {{"estimate": number, "low": number, "high": number}}}} for every '
      'held-out row}}: the estimated {target} and its uncertainty range. Held-out rows\' {target} values are unknown '
      'to your model: fit on the other rows only.', runs=("mask",))
def _predict_interval(c):
    t = c.params["target"]
    p = _heldout_preds(c, interval=True)
    if c.run == "mask":
        return
    ok = all(lo <= e <= hi for e, lo, hi in p.values())
    c.check("low <= estimate <= high", ok, "some estimate is outside its own range")
    cover = sum(lo <= c.data.value(r, t) <= hi for r, (e, lo, hi) in p.items()) / len(p)
    c.check("ranges cover the truth", cover >= 0.5, f"ranges contain the actual {t} for only {cover:.0%} of rows")
    train = [c.data.value(r, t) for r in range(1, c.data.n + 1) if r % 5]
    width = O.median([hi - lo for _, lo, hi in p.values()])
    c.check("ranges are informative", width < max(train) - min(train),
            f"median range width {width:.3g} spans the whole observed range")
    mae = O.mean([abs(e - c.data.value(r, t)) for r, (e, _, _) in p.items()])
    base, _ = _baseline_mae(c, t)
    c.check("beats the historical mean", mae < base, f"mae {mae:.3f} is not below the mean-predictor mae {base:.3f}")


@compare("predict_interval")
def _no_leak_interval(new, base):
    t = base.params["target"]
    a, b = _heldout_preds(base, interval=True), _heldout_preds(new, interval=True)
    s = O.sd(base.data.values(t)) or 1.0
    shift = O.median([abs(a[r][0] - b[r][0]) for r in a])
    new.check("held-out targets not used", shift <= 0.25 * s,
              f"changing only the held-out {t} values moved the estimates by a median {shift:.3g}")


@kind("pair_diff", '{{"pairs": [{{"a": row, "b": row, "setting": the one column where they differ, '
      '"difference": estimated {target} of b minus a, "low": number, "high": number}}, at least 5 pairs]}}; '
      'a and b are observed rows that differ in exactly one setting.')
def _pair_diff(c):
    t = c.params["target"]
    pairs = need(c.ans, "pairs", "list")
    if len(pairs) < 5:
        raise SchemaError("need at least 5 pairs")
    covered, widths, inputs = 0, [], c.data.inputs()
    for i, p in enumerate(pairs):
        a, b = row(c.data, need(p, "a")), row(c.data, need(p, "b"))
        s = column(c.data, need(p, "setting"), set(inputs), "setting")
        differ = [col for col in inputs if c.data.value(a, col) != c.data.value(b, col)]
        c.check(f"pair {i + 1} differs only in {s}", differ == [s], f"rows {a},{b} differ in {differ}")
        d, lo, hi = num(need(p, "difference")), num(need(p, "low")), num(need(p, "high"))
        c.check(f"pair {i + 1} low <= difference <= high", lo <= d <= hi, f"{d} not in [{lo}, {hi}]")
        covered += lo <= c.data.value(b, t) - c.data.value(a, t) <= hi
        widths.append(hi - lo)
    c.check("ranges cover the actual differences", covered >= 0.5 * len(pairs),
            f"only {covered}/{len(pairs)} ranges contain the recorded difference")
    vs = c.data.values(t)
    c.check("ranges are informative", O.median(widths) < max(vs) - min(vs), "ranges span the whole observed range")


# Certify -------------------------------------------------------------------

@kind("groups", '{{"groups": {{"<value of {group}>": {{"n": rows in the group, "median": median {outcome}, '
      '"mean": mean {outcome}}}}} one entry per distinct value of {group}, "comparisons": [{{"a": value, "b": value, '
      '"verdict": "different" if the {outcome} difference is reliable, else "same"}}] for every pair of groups}}')
def _groups(c):
    g, o = c.params["group"], c.params["outcome"]
    values = sorted(set(c.data.values(g)), key=str)
    groups = need(c.ans, "groups", "dict")
    found = {}
    for v in values:
        keys = [k for k in groups if key_matches(c.data, g, k, v)]
        if not keys:
            raise SchemaError(f"no group for {g}={v}")
        found[v] = groups[keys[0]]
    if len(groups) != len(values):
        raise SchemaError(f"{len(groups)} groups, expected {len(values)}")
    members = {v: [r for r in range(1, c.data.n + 1) if c.data.value(r, g) == v and c.data.value(r, o) is not None]
               for v in values}
    for v, e in found.items():
        ys = [c.data.value(r, o) for r in members[v]]
        c.check(f"{g}={v} n", num(need(e, "n")) == len(ys), f"n {e['n']} != {len(ys)}")
        c.check(f"{g}={v} median", any(O.close(num(need(e, "median")), m) for m in O.medians(ys)),
                f"median {e['median']} != {O.median(ys)}")
        c.check(f"{g}={v} mean", O.close(num(need(e, "mean")), O.mean(ys)), f"mean {e['mean']} != {O.mean(ys):.4f}")
    comps = need(c.ans, "comparisons", "list")
    seen = {}
    for e in comps:
        a = [v for v in values if key_matches(c.data, g, need(e, "a"), v)]
        b = [v for v in values if key_matches(c.data, g, need(e, "b"), v)]
        if not a or not b:
            raise SchemaError(f"comparison names unknown groups {e.get('a')!r}/{e.get('b')!r}")
        verdict = need(e, "verdict")
        if verdict not in ("different", "same"):
            raise SchemaError(f"verdict must be 'different' or 'same', got {verdict!r}")
        seen[frozenset((a[0], b[0]))] = verdict
    for i, a in enumerate(values):
        for b in values[i + 1:]:
            pair = frozenset((a, b))
            if pair not in seen:
                raise SchemaError(f"missing comparison {a} vs {b}")
            truth = O.clear_verdict([c.data.value(r, o) for r in members[a]], [c.data.value(r, o) for r in members[b]])
            if truth:
                c.check(f"{a} vs {b} verdict", seen[pair] == truth, f"said {seen[pair]}, the data clearly say {truth}")


@kind("year_groups", '{{"groups": [{{"from": first {group} value, "to": last {group} value, "n": rows}}, {{...}}] '
      'exactly two non-overlapping inclusive {group} ranges, "verdict": "different" if overall performance '
      '(all goal columns together) differs reliably between them, else "same"}}')
def _year_groups(c):
    g = c.params["group"]
    gs = need(c.ans, "groups", "list")
    if len(gs) != 2:
        raise SchemaError("need exactly 2 groups")
    spans = [(num(need(e, "from")), num(need(e, "to"))) for e in gs]
    (a0, a1), (b0, b1) = sorted(spans)
    c.check("ranges do not overlap", a0 <= a1 < b0 <= b1, f"ranges {spans} overlap or are reversed")
    groups = []
    for (lo, hi), e in zip(spans, gs):
        rs = [r for r in range(1, c.data.n + 1) if lo <= c.data.value(r, g) <= hi]
        c.check(f"{g} {lo:g}-{hi:g} n", num(need(e, "n")) == len(rs), f"n {e['n']} != {len(rs)}")
        groups.append(rs)
    verdict = need(c.ans, "verdict")
    if verdict not in ("different", "same"):
        raise SchemaError(f"verdict must be 'different' or 'same', got {verdict!r}")
    truths = [O.clear_verdict([s[r] for r in groups[0]], [s[r] for r in groups[1]]) for s in O.aggregates(c.data)]
    c.check("verdict", any(t in (None, verdict) for t in truths),
            f"said {verdict}, every overall score clearly says otherwise ({truths})")


# Flow ----------------------------------------------------------------------

def _assessments(c, fields):
    items = need(c.ans, "assessments", "list")
    want = O.arrivals(c.data.n)
    got = [row(c.data, need(e, "row")) for e in items]
    if got != want:
        raise SchemaError(f"assessments must list the {len(want)} arrivals {want[0]}..{want[-1]} in order")
    out = []
    for e in items:
        vals = {}
        for f, typ in fields.items():
            v = need(e, f)
            if typ == "bool" and not is_bool(v):
                raise SchemaError(f"{f} must be true/false")
            vals[f] = v if typ == "bool" else num(v, f)
        out.append(vals)
    return want, out


@kind("stream_desirable", '{{"assessments": [{{"row": row, "desirable": true or false}}, one per arrival, in arrival '
      'order]}}; judge each arrival using only the history and earlier arrivals.', runs=("future",))
def _stream_desirable(c):
    rs, a = _assessments(c, {"desirable": "bool"})
    yes = [r for r, e in zip(rs, a) if e["desirable"]]
    no = [r for r, e in zip(rs, a) if not e["desirable"]]
    if c.run == "future":
        return
    c.check("some arrivals accepted and some rejected", yes and no, "every arrival got the same answer")
    if yes and no:
        c.check("desirable arrivals are better",
                any(O.median([s[r] for r in yes]) < O.median([s[r] for r in no]) for s in O.aggregates(c.data)),
                "arrivals marked desirable are not better overall than the others")


def _first_half_same(new, base, key, fields):
    rb, ab = _assessments(base, fields)
    rn, an = _assessments(new, fields)
    half = len(rb) // 2
    agree = sum(ab[i][key] == an[i][key] for i in range(half))
    new.check("no use of future arrivals", agree >= 0.95 * half,
              f"changing only later arrivals changed {half - agree}/{half} earlier decisions")


@compare("stream_desirable")
def _cmp_desirable(new, base):
    _first_half_same(new, base, "desirable", {"desirable": "bool"})


NASA7 = {"predicted_effort": "num", "predicted_defects": "num", "alert": "bool"}


@kind("stream_alert_pred", '{{"assessments": [{{"row": row, "predicted_effort": number, "predicted_defects": number, '
      '"alert": true or false}}, one per arrival, in arrival order]}}; predict each arrival from the history and earlier '
      'arrivals only, and alert when the prediction is unusually high.', runs=("future",))
def _stream_alert_pred(c):
    rs, a = _assessments(c, NASA7)
    if c.run == "future":
        return
    actual = [c.data.value(r, "Effort-") for r in rs]
    rho = O.spearman([e["predicted_effort"] for e in a], actual)
    c.check("effort predictions track actual effort", rho > 0, f"Spearman {rho:.2f} between predicted and actual effort")
    for r, e in zip(rs, a):
        if e["alert"]:
            before = range(1, r)
            hi_e = e["predicted_effort"] > O.median(c.data.values("Effort-", before))
            hi_d = e["predicted_defects"] > O.median(c.data.values("Defects-", before))
            c.check(f"row {r} alert is justified", hi_e or hi_d, "alerted although neither prediction is above the median so far")


@compare("stream_alert_pred")
def _cmp_alert_pred(new, base):
    _first_half_same(new, base, "alert", NASA7)
    rb, ab = _assessments(base, NASA7)
    _, an = _assessments(new, NASA7)
    half = len(rb) // 2
    moved = [abs(ab[i]["predicted_effort"] - an[i]["predicted_effort"]) / (abs(ab[i]["predicted_effort"]) + 1e-9)
             for i in range(half)]
    new.check("earlier predictions ignore later arrivals", O.median(moved) <= 0.05,
              f"changing later arrivals moved earlier effort predictions by a median {O.median(moved):.0%}")


@kind("stream_alert_poor", '{{"assessments": [{{"row": row, "alert": true or false}}, one per arrival, in arrival order]}}; '
      'alert when an arrival\'s {target} is unusually poor compared with all rows before it.', runs=("future",))
def _stream_alert_poor(c):
    t = c.params["target"]
    rs, a = _assessments(c, {"alert": "bool"})
    if c.run == "future":
        return
    for r, e in zip(rs, a):
        before = c.data.values(t, range(1, r))
        v = c.data.value(r, t)
        if e["alert"]:
            c.check(f"row {r} alert is justified", O.worse(c.data, t, v, O.median(before)),
                    f"alerted although {t}={v:g} is not worse than the median so far")
        elif all(O.worse(c.data, t, v, b) for b in before):
            c.check(f"row {r} worst-ever is alerted", False, f"{t}={v:g} is worse than every earlier row but not alerted")


@compare("stream_alert_poor")
def _cmp_alert_poor(new, base):
    _first_half_same(new, base, "alert", {"alert": "bool"})


@kind("stream_summary", '{{"checkpoints": [{{"after": m, "summary": {{"<goal column>": {{"n": number, "mean": number, '
      '"min": number, "max": number}}}} for every goal column, "shortlist": [the 5 best rows seen so far]}}]}} with '
      'checkpoints after m = 0, {half} and {k} arrivals; checkpoint m covers the history plus the first m arrivals.')
def _stream_summary(c):
    k = O.arrivals_count(c.data.n)
    hist_end = c.data.n - k
    cps = need(c.ans, "checkpoints", "list")
    by_m = {int(num(need(e, "after"))): e for e in cps}
    for m in (0, k // 2, k):
        if m not in by_m:
            raise SchemaError(f"missing checkpoint after {m}")
        seen = list(range(1, hist_end + m + 1))
        e = by_m[m]
        summ = need(e, "summary", "dict")
        for g, _ in c.data.goals():
            s = need(summ, g, "dict")
            vs = c.data.values(g, seen)
            ok = (num(need(s, "n")) == len(vs) and O.close(num(need(s, "mean")), O.mean(vs))
                  and O.close(num(need(s, "min")), min(vs)) and O.close(num(need(s, "max")), max(vs)))
            c.check(f"after {m}: {g} summary", ok, f"{g} summary {s} != batch n={len(vs)} mean={O.mean(vs):.4f} "
                    f"min={min(vs):g} max={max(vs):g}")
        short = rows(c.data, need(e, "shortlist", "list"), "shortlist")
        if len(short) != 5:
            raise SchemaError("shortlist must have 5 rows")
        c.check(f"after {m}: shortlist only uses seen rows", all(r <= hist_end + m for r in short),
                "shortlist includes rows that have not arrived yet")
        good = O.good_rows(c.data, 0.25, seen)
        hits = sum(r in good for r in short)
        c.check(f"after {m}: shortlist is good", hits >= 4, f"only {hits}/5 shortlisted rows are among the best seen")


# Blame ---------------------------------------------------------------------

def _top3(data, scores):
    """Top 3 columns among all non-goal columns, or among inputs only: the protocol says X columns are not
    inputs, so a program that leaves them out of its ranking must be judged against the inputs alone."""
    inputs = set(data.inputs())
    return O.top_k(scores, 3) | O.top_k({f: v for f, v in scores.items() if f in inputs}, 3)


@kind("blame_rank", '{{"poor": [row numbers with unusually poor {outcome}, at least 5], "ranking": [non-goal column '
      'names, most strongly associated with poor {outcome} first, at least 3]}}')
def _blame_rank(c):
    o = c.params["outcome"]
    poor = rows(c.data, need(c.ans, "poor", "list"), "poor")
    if len(poor) < 5:
        raise SchemaError("need at least 5 poor rows")
    mid = O.median(c.data.values(o))
    better = [r for r in poor if O.worse(c.data, o, mid, c.data.value(r, o))]
    c.check("poor rows are poor", not better and O.worse(c.data, o, O.mean(c.data.values(o, poor)), mid),
            f"rows {better[:5]} are better than the median {o}" if better else "the set is not worse than the median")
    feats = c.data.features()
    rank = need(c.ans, "ranking", "list")
    if len(rank) < 3 or len(set(map(str, rank))) != len(rank):
        raise SchemaError("ranking needs at least 3 distinct columns")
    rank = [column(c.data, f, set(feats), "ranking entry") for f in rank]
    rest = [r for r in range(1, c.data.n + 1) if r not in set(poor)]
    by_delta = _top3(c.data, {f: O.association(c.data, f, poor, rest) for f in feats})
    by_rho = _top3(c.data, {f: O.correlation(c.data, f, o) for f in feats})
    c.check("top-ranked column is strongly associated", rank[0] in by_delta | by_rho,
            f"{rank[0]} is not in the top 3 by Cliff's delta {sorted(by_delta)} or Spearman {sorted(by_rho)}")


def _two_sets(c):
    bad = rows(c.data, need(c.ans, "bad", "list"), "bad")
    good = rows(c.data, need(c.ans, "good", "list"), "good")
    if len(bad) < 5 or len(good) < 5:
        raise SchemaError("bad and good need at least 5 rows each")
    c.check("bad and good are disjoint", not set(bad) & set(good), "a row is in both sets")
    return bad, good


def _outcome_scores(c):
    """Scores (lower = better) to judge bad vs good: the named outcome, else every overall aggregate."""
    o = c.params.get("outcome")
    if o:
        w = 1 if o.endswith("+") else -1
        return [{r: -w * c.data.value(r, o) for r in range(1, c.data.n + 1) if c.data.value(r, o) is not None}]
    return O.aggregates(c.data)


def _bad_is_worse(c, bad, good):
    c.check("bad rows are worse",
            any(O.median([s[r] for r in bad]) > O.median([s[r] for r in good]) for s in _outcome_scores(c)),
            "the bad rows are not worse than the good rows")


@kind("blame_contrast", '{{"bad": [rows with unfavorable {outcome_text}, at least 5], "good": [better rows, at least 5], '
      '"differences": [{{"feature": non-goal column, "bad": typical value in bad rows, "good": typical value in good '
      'rows}}, at least 3, biggest difference first]}}; typical = mean for numeric columns, most common value otherwise.')
def _blame_contrast(c):
    bad, good = _two_sets(c)
    _bad_is_worse(c, bad, good)
    diffs = need(c.ans, "differences", "list")
    if len(diffs) < 3:
        raise SchemaError("need at least 3 differences")
    feats = c.data.features()
    for d in diffs:
        f = column(c.data, need(d, "feature"), set(feats), "feature")
        for side, rs in (("bad", bad), ("good", good)):
            vs = c.data.values(f, rs)
            v = need(d, side)
            if c.data.is_numeric(f):
                ok = isinstance(v, (int, float)) and not isinstance(v, bool) and (
                    O.close(float(v), O.mean(vs)) or any(O.close(float(v), m) for m in O.medians(vs)))
                want = f"mean {O.mean(vs):.4f}"
            else:
                top = max(vs.count(x) for x in vs)
                modes = {x.strip().lower() for x in vs if vs.count(x) == top}
                ok = str(v).strip().lower() in modes
                want = f"mode {sorted(modes)}"
            c.check(f"{f} {side} value", ok, f"{f} {side}={v!r}, data give {want}")
    first = column(c.data, diffs[0]["feature"], set(feats))
    by_delta = _top3(c.data, {f: O.association(c.data, f, bad, good) for f in feats})
    c.check("biggest difference first", first in by_delta, f"{first} is not in the top 3 by Cliff's delta {sorted(by_delta)}")


@kind("blame_combos", '{{"bad": [poorly performing rows, at least 5], "good": [better rows, at least 5], "combinations": '
      '[{{"settings": {{"<column>": 0 or 1, ... at least 2}}, "bad_share": fraction of bad rows matching all of them, '
      '"good_share": fraction of good rows matching all of them}}, at least 1, most typical of bad first]}}')
def _blame_combos(c):
    bad, good = _two_sets(c)
    _bad_is_worse(c, bad, good)
    combos = need(c.ans, "combinations", "list")
    if not combos:
        raise SchemaError("need at least 1 combination")
    inputs = set(c.data.inputs())
    for i, e in enumerate(combos):
        st = need(e, "settings", "dict")
        if len(st) < 2:
            raise SchemaError("a combination needs at least 2 settings")
        cond = {column(c.data, k, inputs, "setting"): num(v) for k, v in st.items()}
        match = lambda r: all(c.data.value(r, k) == v for k, v in cond.items())
        sb, sg = sum(map(match, bad)) / len(bad), sum(map(match, good)) / len(good)
        c.check(f"combination {i + 1} shares", O.close(num(need(e, "bad_share")), sb, abs_=0.011)
                and O.close(num(need(e, "good_share")), sg, abs_=0.011),
                f"shares {e['bad_share']}/{e['good_share']} != {sb:.3f}/{sg:.3f}")
        if i == 0:
            c.check("first combination is typical of bad rows", sb > sg, f"bad share {sb:.2f} <= good share {sg:.2f}")


# Justify -------------------------------------------------------------------

def _neighbors(c, center, need_cols):
    items = need(c.ans, "neighbors", "list")
    if len(items) < 3:
        raise SchemaError("need at least 3 neighbors")
    near = O.XSpace(c.data).nearest(center, 0.25)
    out = []
    for e in items:
        r = row(c.data, need(e, "row"), "neighbor")
        if r == center:
            raise SchemaError("a neighbor is the row itself")
        vals = need(e, "values", "dict")
        missing = [g for g in need_cols if g not in vals]
        if missing:
            raise SchemaError(f"neighbor values must include {missing}")
        c.check(f"neighbor {r} is similar", r in near, f"row {r} is not among the 25% most similar rows to {center}")
        wrong = [k for k, v in vals.items() if k in c.data.names and not same_cell(c.data, r, k, v)]
        unknown = [k for k in vals if k not in c.data.names]
        c.check(f"neighbor {r} values are real", not wrong and not unknown,
                f"cited values for {wrong + unknown} do not match row {r}")
        out.append(r)
    if len(set(out)) != len(out):
        raise SchemaError("duplicate neighbors")
    return out


@kind("justify_neighbors", '{{"recommended": row, "neighbors": [{{"row": row, "values": {{"<column>": value as in the '
      'file, ...}}}}, at least 3 similar rows]}}; values must include every goal column.')
def _justify_neighbors(c):
    rec = row(c.data, need(c.ans, "recommended"), "recommended")
    c.check("recommended row is good", rec in O.good_rows(c.data, 0.25),
            f"row {rec} is neither Pareto-optimal nor in the best 25% by any overall score")
    _neighbors(c, rec, [g for g, _ in c.data.goals()])


@kind("justify_estimate", '{{"project": a held-out row, "estimate": its estimated Effort-, "neighbors": [{{"row": row, '
      '"values": {{"<column>": value as in the file, ...}}}}, at least 3 similar non-held-out rows]}}; values must include '
      'Effort-. The project\'s own Effort- is unknown to your model.')
def _justify_estimate(c):
    p = row(c.data, need(c.ans, "project"), "project")
    c.check("project is held out", p % 5 == 0, f"row {p} is not a held-out row")
    ns = _neighbors(c, p, ["Effort-"])
    c.check("neighbors are not held out", all(r % 5 for r in ns), "a neighbor is a held-out row")
    eff = [c.data.value(r, "Effort-") for r in ns]
    est = num(need(c.ans, "estimate"), "estimate")
    c.check("estimate follows from the neighbors", 0.5 * min(eff) <= est <= 2 * max(eff),
            f"estimate {est:g} is far outside the neighbors' efforts {min(eff):g}..{max(eff):g}")


@kind("justify_pair", '{{"a": row, "b": row, "preferred": a or b, "scores": {{"<a>": number, "<b>": number}} (higher = '
      'more recommended), "differences": [{{"feature": column, "a": value, "b": value}}, at least 1], "tradeoffs": '
      '[goal columns where the non-preferred row is better]}}')
def _justify_pair(c):
    a, b = row(c.data, need(c.ans, "a"), "a"), row(c.data, need(c.ans, "b"), "b")
    if a == b:
        raise SchemaError("a and b must differ")
    # The format text says `"preferred": a or b` and keys scores by "<a>"; read a literal "a"/"b" as those rows.
    alias = {"a": a, "b": b}
    pv = need(c.ans, "preferred")
    pref = alias[pv] if isinstance(pv, str) and pv in alias else row(c.data, pv, "preferred")
    if pref not in (a, b):
        raise SchemaError("preferred must be a or b")
    other = b if pref == a else a
    raw = need(c.ans, "scores", "dict")
    sc = row_map(c.data, {str(alias.get(k, k)): v for k, v in raw.items()}, "scores")
    if set(sc) != {a, b}:
        raise SchemaError("scores must have exactly a and b")
    c.check("preferred has the higher score", num(sc[pref]) > num(sc[other]), "the preferred row's score is not higher")
    diffs = need(c.ans, "differences", "list")
    if not diffs:
        raise SchemaError("need at least 1 difference")
    for d in diffs:
        f = column(c.data, need(d, "feature"), set(c.data.names), "feature")
        ok = same_cell(c.data, a, f, need(d, "a")) and same_cell(c.data, b, f, need(d, "b"))
        c.check(f"{f} values are real", ok, f"cited {f} values do not match rows {a}/{b}")
        c.check(f"{f} really differs", c.data.value(a, f) != c.data.value(b, f), f"{f} is the same in both rows")
    goals = c.data.goals()
    truth = {g for g, w in goals if w * c.data.value(other, g) > w * c.data.value(pref, g)}
    said = {column(c.data, g, {x for x, _ in goals}, "tradeoff") for g in need(c.ans, "tradeoffs", "list")}
    c.check("tradeoffs are correct", said == truth, f"tradeoffs {sorted(said)} != {sorted(truth)}")
    c.check("preferred is not dominated", not O.dominates(c.data, other, pref), f"row {other} is better on every goal")
    if c.params.get("strengths"):
        c.check("the rows have different strengths", truth, "the non-preferred row is better at nothing")


# Choose --------------------------------------------------------------------

HONEYPOTS = 6


@kind("choose_best", '{{"evaluated": [the rows you evaluated, in order, at most {budget}], "best": row}}. Evaluating a '
      'row means looking at its goal values; choose each next row using only the goal values of rows already evaluated.',
      runs=("honeypot",))
def _choose_best(c):
    ev = rows(c.data, need(c.ans, "evaluated", "list"), "evaluated")
    best = row(c.data, need(c.ans, "best"), "best")
    if c.run == "honeypot":
        return _honeypot(c, ev)
    if not 1 <= len(ev) <= O.budget(c.data.n):
        raise SchemaError(f"evaluated {len(ev)} rows, budget is {O.budget(c.data.n)}")
    c.check("best was evaluated", best in ev, f"row {best} was not evaluated")
    c.check("best is the best evaluated", not any(O.dominates(c.data, r, best) for r in ev),
            f"an evaluated row beats row {best} on every goal")
    c.check("best is good overall", best in O.good_rows(c.data, 0.2),
            f"row {best} is not in the best 20% of all rows by any overall score")


def _honeypot(c, picked):
    pots = c.extra["honeypots"]
    hit = len(set(picked[:O.budget(c.data.n - HONEYPOTS)]) & pots)
    c.check("respects the budget (no peeking at unevaluated goals)", hit < HONEYPOTS - 1,
            f"{hit}/{HONEYPOTS} planted rows with ideal goals but mediocre inputs were picked: "
            "the program looks at goals it has not paid for")


@kind("choose_rule", '{{"rule": {{"<non-goal column>": value (symbolic columns) or [low, high] (numeric, inclusive)}} '
      'with 1 to 3 conditions, "matching": number of rows meeting every condition}}')
def _choose_rule(c):
    rule = need(c.ans, "rule", "dict")
    if not 1 <= len(rule) <= 3:
        raise SchemaError("rule needs 1 to 3 conditions")
    conds = []
    for k, v in rule.items():
        col = column(c.data, k, set(c.data.features()), "rule column")
        if isinstance(v, list):
            if len(v) != 2:
                raise SchemaError(f"{col} range must be [low, high]")
            lo, hi = num(v[0]), num(v[1])
            conds.append(lambda r, col=col, lo=lo, hi=hi: isinstance(c.data.value(r, col), float)
                         and lo <= c.data.value(r, col) <= hi)
        else:
            conds.append(lambda r, col=col, v=v: same_cell(c.data, r, col, v))
    match = [r for r in range(1, c.data.n + 1) if all(f(r) for f in conds)]
    c.check("matching count", num(need(c.ans, "matching")) == len(match), f"matching {c.ans['matching']} != {len(match)}")
    c.check("rule is supported by at least 3 rows", len(match) >= 3, f"only {len(match)} rows match")
    if match:
        c.check("matching rows are better than typical",
                any(O.median([s[r] for r in match]) < O.median(list(s.values())) for s in O.aggregates(c.data)),
                "rows matching the rule are not better overall than the median row")


@kind("choose_order", '{{"order": [{budget} row numbers, the first to evaluate first]}}; decide the order using only the '
      'goal values of rows already placed earlier in the order.', runs=("honeypot",))
def _choose_order(c):
    order = rows(c.data, need(c.ans, "order", "list"), "order")
    if c.run == "honeypot":
        return _honeypot(c, order)
    b = O.budget(c.data.n)
    if not 10 <= len(order) <= b:
        raise SchemaError(f"order has {len(order)} rows, expected 10..{b}")
    aggs = O.aggregates(c.data)
    c.check("finds a promising row within budget",
            any(min(s[r] for r in order) <= sorted(s.values())[int(0.1 * len(s))] for s in aggs),
            "no row in the order is in the best 10% by any overall score")
    c.check("order favors promising rows",
            any(O.median([s[r] for r in order]) < O.median(list(s.values())) for s in aggs),
            "the ordered rows are no better than a typical row")


# Fix -----------------------------------------------------------------------

def _changes(c, src, dst_or_none, d, allowed):
    """Validate {column: [from, to]}; with DST, 'to' must be DST's value."""
    if not isinstance(d, dict) or not d:
        raise SchemaError("changes must be a non-empty object")
    for k, v in d.items():
        col = column(c.data, k, allowed, "changed column")
        if not isinstance(v, list) or len(v) != 2:
            raise SchemaError(f"{col} change must be [from, to]")
        c.check(f"{col} from-value is real", same_cell(c.data, src, col, v[0]), f"row {src} {col} is not {v[0]!r}")
        if dst_or_none is not None:
            c.check(f"{col} to-value is real", same_cell(c.data, dst_or_none, col, v[1]),
                    f"row {dst_or_none} {col} is not {v[1]!r}")
        else:
            c.check(f"{col} new value occurs in the data", any(same_cell(c.data, r, col, v[1]) for r in range(1, c.data.n + 1)),
                    f"{col}={v[1]!r} never occurs in the data")
            c.check(f"{col} actually changes", not same_cell(c.data, src, col, v[1]), f"{col} is unchanged")
    return [column(c.data, k, allowed) for k in d]


@kind("fix_alternatives", '{{"row": a poorly rated row, "alternatives": [{{"row": row, "changes": {{"<input column>": '
      '[value in the poor row, value in the alternative]}}}}, at least 1 similar but better row]}}')
def _fix_alternatives(c):
    aggs = O.aggregates(c.data)
    r = row(c.data, need(c.ans, "row"))
    c.check("row is poorly rated", any(s[r] > O.median(list(s.values())) for s in aggs),
            f"row {r} is not in the worse half by any overall score")
    alts = need(c.ans, "alternatives", "list")
    if not alts:
        raise SchemaError("need at least 1 alternative")
    near = O.XSpace(c.data).nearest(r, 0.25)
    inputs = set(c.data.inputs())
    for e in alts:
        a = row(c.data, need(e, "row"), "alternative")
        c.check(f"alternative {a} is better", any(s[a] < s[r] for s in aggs), f"row {a} is not better than row {r}")
        c.check(f"alternative {a} is similar", a in near, f"row {a} is not among the 25% most similar rows")
        cols = _changes(c, r, a, need(e, "changes"), inputs)
        c.check(f"alternative {a} changes really differ", all(c.data.value(r, k) != c.data.value(a, k) for k in cols),
                "a listed change is not a real difference")


@kind("fix_observed", '{{"row": a poorly performing row, "alternative": an observed row differing in 1 or 2 settings, '
      '"changes": {{"<setting>": [from, to]}} every differing setting, "performance": {{"before": {target} of row, '
      '"after": recorded or predicted {target} of the alternative}}}}')
def _fix_observed(c):
    t = c.params["target"]
    r, a = row(c.data, need(c.ans, "row")), row(c.data, need(c.ans, "alternative"), "alternative")
    vs = c.data.values(t)
    c.check("row performs poorly", O.worse(c.data, t, c.data.value(r, t), O.median(vs)), f"row {r} is not in the worse half")
    inputs = c.data.inputs()
    differ = {k for k in inputs if c.data.value(r, k) != c.data.value(a, k)}
    c.check("alternative is a small change", 1 <= len(differ) <= 2, f"rows differ in {len(differ)} settings")
    cols = set(_changes(c, r, a, need(c.ans, "changes"), set(inputs)))
    c.check("changes list every difference", cols == differ, f"changes {sorted(cols)} != differences {sorted(differ)}")
    perf = need(c.ans, "performance", "dict")
    c.check("predicted improvement", O.worse(c.data, t, num(need(perf, "before")), num(need(perf, "after"))),
            "after is not better than before")
    c.check("alternative really is better", O.worse(c.data, t, c.data.value(r, t), c.data.value(a, t)),
            f"row {a}'s recorded {t} is not better than row {r}'s")


@kind("fix_predicted", '{{"project": row, "changes": {{"<column>": [current value, new value]}}{limit}, "predicted_{what}": '
      '{{"before": number, "after": number}}{extra}}}; only adjustable columns may change: any non-goal column except '
      'Kloc+ and columns ending in X; new values must be values that occur in the data.')
def _fix_predicted(c):
    what = c.params["what"]
    p = row(c.data, need(c.ans, "project"), "project")
    allowed = {k for k in c.data.inputs() if k != "Kloc+"}
    cols = _changes(c, p, None, need(c.ans, "changes"), allowed)
    if c.params.get("max_changes"):
        c.check("smallest plausible change", len(cols) <= c.params["max_changes"],
                f"{len(cols)} columns changed (limit {c.params['max_changes']})")
    pred = need(c.ans, f"predicted_{what}", "dict")
    before, after = num(need(pred, "before")), num(need(pred, "after"))
    c.check("prediction improves", after < before, f"predicted {what} {after:g} is not below {before:g}")
    col = {"effort": "Effort-", "defects": "Defects-"}[what]
    c.check(f"project has high predicted {what}", before >= O.median(c.data.values(col)),
            f"before={before:g} is below the median {col}")
    if c.params.get("tradeoff"):
        tr = need(c.ans, "predicted_effort", "dict")
        num(need(tr, "before"))
        num(need(tr, "after"))


@kind("fix_targets", '{{"row": a row missing some targets, "targets": {{"<goal column>": threshold for every goal}}, '
      '"alternative": row, "changed": [input columns whose values differ between row and alternative]}}; a row meets a '
      'target when its value is at least (+) or at most (-) the threshold; the alternative must meet more targets than '
      'the row and, among all rows that do, differ from it in as few input columns as possible.')
def _fix_targets(c):
    r, a = row(c.data, need(c.ans, "row")), row(c.data, need(c.ans, "alternative"), "alternative")
    tg = need(c.ans, "targets", "dict")
    goals = c.data.goals()
    th = {}
    for g, w in goals:
        th[g] = num(need(tg, g), f"target for {g}")
        vs = c.data.values(g)
        c.check(f"{g} target is within the data", min(vs) <= th[g] <= max(vs), f"{th[g]} outside {min(vs)}..{max(vs)}")
    meets = lambda x: sum(w * c.data.value(x, g) >= w * th[g] for g, w in goals)
    c.check("row misses a target", meets(r) < len(goals), f"row {r} already meets every target")
    c.check("alternative meets more targets", meets(a) > meets(r), f"row {a} meets {meets(a)}, row {r} meets {meets(r)}")
    inputs = c.data.inputs()
    diff = lambda x: {k for k in inputs if c.data.value(r, k) != c.data.value(x, k)}
    said = {column(c.data, k, set(inputs), "changed column") for k in need(c.ans, "changed", "list")}
    c.check("changed columns are correct", said == diff(a), f"changed {sorted(said)} != {sorted(diff(a))}")
    cands = [x for x in range(1, c.data.n + 1) if x != r and meets(x) > meets(r)]
    if cands:
        least = min(len(diff(x)) for x in cands)
        c.check("fewest changes", len(diff(a)) == least, f"row {a} changes {len(diff(a))} columns; {least} is possible")


@kind("fix_target", '{{"row": a poorly performing row, "target": a {target} value better than the row\'s, "alternative": '
      'an observed row closer to the target, "changed": [settings that differ between row and alternative]}}; among '
      'observed rows closer to the target than the row, pick one with the fewest differing settings.')
def _fix_target(c):
    t = c.params["target"]
    r, a = row(c.data, need(c.ans, "row")), row(c.data, need(c.ans, "alternative"), "alternative")
    tv = num(need(c.ans, "target"), "target")
    vs = c.data.values(t)
    c.check("row performs poorly", O.worse(c.data, t, c.data.value(r, t), O.median(vs)), f"row {r} is not in the worse half")
    c.check("target is better than the row", O.worse(c.data, t, c.data.value(r, t), tv) and min(vs) <= tv <= max(vs),
            f"target {tv:g} is not a better, observed-range value")
    gap = lambda x: abs(c.data.value(x, t) - tv)
    c.check("alternative is closer to the target", gap(a) < gap(r), f"row {a} is not closer to {tv:g}")
    inputs = c.data.inputs()
    diff = lambda x: {k for k in inputs if c.data.value(r, k) != c.data.value(x, k)}
    said = {column(c.data, k, set(inputs), "changed setting") for k in need(c.ans, "changed", "list")}
    c.check("changed settings are correct", said == diff(a), f"changed {sorted(said)} != {sorted(diff(a))}")
    cands = [x for x in range(1, c.data.n + 1) if gap(x) < gap(r)]
    if cands:
        least = min(len(diff(x)) for x in cands)
        c.check("fewest changes", len(diff(a)) == least, f"row {a} changes {len(diff(a))} settings; {least} is possible")


# Spot ----------------------------------------------------------------------

@kind("spot_x", '{{"flagged": [row numbers whose input-column combinations are unusual, 1 to {max_flag}], "scores": '
      '{{"<row>": unusualness score}} for the flagged rows}}')
def _spot_x(c):
    fl = rows(c.data, need(c.ans, "flagged", "list"), "flagged")
    if not 1 <= len(fl) <= c.data.n // 10 + 1:
        raise SchemaError(f"flagged {len(fl)} rows, expected 1..{c.data.n // 10 + 1}")
    sc = row_map(c.data, need(c.ans, "scores", "dict"), "scores")
    if not set(fl) <= set(sc):
        raise SchemaError("every flagged row needs a score")
    [num(sc[r]) for r in fl]
    x = O.XSpace(c.data)
    odd = O.top_fraction(x.knn_scores(), 0.25) | O.top_fraction(x.centroid_scores(), 0.25)
    hits = sum(r in odd for r in fl)
    c.check("flagged rows are unusual", hits >= len(fl) / 2,
            f"only {hits}/{len(fl)} flagged rows are in the most unusual 25% by neighbor or centroid distance")


@kind("spot_y", '{{"flagged": [{{"row": row, "column": one of {cols}, "actual": value in the file, "expected": what '
      'similar rows suggest}}, 1 to {max_flag} entries]}}')
def _spot_y(c):
    items = need(c.ans, "flagged", "list")
    if not 1 <= len(items) <= c.data.n // 10 + 1:
        raise SchemaError(f"flagged {len(items)} rows, expected 1..{c.data.n // 10 + 1}")
    x = O.XSpace(c.data)
    resid = {col: O.top_fraction(x.loo_residuals(col), 0.25) for col in c.params["cols"]}
    hits = 0
    for e in items:
        r = row(c.data, need(e, "row"))
        col = column(c.data, need(e, "column"), set(c.params["cols"]), "column")
        c.check(f"row {r} actual is real", same_cell(c.data, r, col, need(e, "actual")),
                f"row {r} {col} is not {e['actual']!r}")
        num(need(e, "expected"), "expected")
        hits += r in resid[col]
    c.check("flagged rows are surprising", hits >= len(items) / 2,
            f"only {hits}/{len(items)} flagged values are in the most surprising 25% versus their 5 nearest rows")


# Race ----------------------------------------------------------------------

@kind("race_select", '{{"budget": {budget}, "repeats": at least 20, "ours": [the best row your selection method found '
      'in each repeat], "random": [the best of {budget} randomly chosen rows in each repeat], "verdict": "ours better", '
      '"random better" or "no clear difference"}}; "best" means best overall on all goal columns.', runs=("honeypot",))
def _race_select(c):
    b, reps = num(need(c.ans, "budget")), num(need(c.ans, "repeats"))
    ours = rows(c.data, need(c.ans, "ours", "list"), "ours", distinct=False)
    rand = rows(c.data, need(c.ans, "random", "list"), "random", distinct=False)
    if c.run == "honeypot":
        frac = sum(r in c.extra["honeypots"] for r in ours) / max(1, len(ours))
        c.check("respects the budget (no peeking at unevaluated goals)", frac < 0.95,
                f"{frac:.0%} of repeats chose a planted row with ideal goals but mediocre inputs")
        return
    c.check("same budget for both", b == O.budget(c.data.n), f"budget {b:g} != {O.budget(c.data.n)}")
    if reps < 20 or len(ours) != reps or len(rand) != reps:
        raise SchemaError(f"need repeats >= 20 and one row per repeat in ours and random")
    v = need(c.ans, "verdict")
    if v not in ("ours better", "random better", "no clear difference"):
        raise SchemaError(f"bad verdict {v!r}")
    verdicts, plausible = [], []
    for s in O.aggregates(c.data):
        xo, xr = [s[r] for r in ours], [s[r] for r in rand]
        ok, want = _verdict_ok(v, xo, xr, ("ours better", "random better", "no clear difference"))
        verdicts.append((ok, want))
        q25 = sorted(s.values())[len(s) // 4]
        plausible.append((O.median(xr) <= q25, f"median {O.median(xr):.3f}, best quarter starts at {q25:.3f}"))
    c.check("verdict follows from the results", any(ok for ok, _ in verdicts),
            f"said {v!r}; the reported rows give {verdicts[0][1]}")
    c.check("random baseline is really random", any(ok for ok, _ in plausible),
            f"the best of {O.budget(c.data.n)} random rows is not in the best quarter "
            f"({plausible[0][1]} by distance-to-heaven): the random arm is not best-of-budget")


@kind("race_predict", '{{"predictions": {{"<row>": estimated {target} for every held-out row}}, "baseline": the mean '
      '{target} of the non-held-out rows, "mae_model": number, "mae_baseline": number, "verdict": "better", "worse" or '
      '"no clear difference"}}. Held-out rows\' {target} values are unknown to your model: fit on the other rows only.',
      runs=("mask",))
def _race_predict(c):
    t = c.params["target"]
    preds = _heldout_preds(c)
    if c.run == "mask":
        return
    base_mae, m = _baseline_mae(c, t)
    mae = _mae(c, preds, t)
    c.check("baseline is the historical mean", O.close(num(need(c.ans, "baseline")), m, rel=0.005), f"baseline != {m:.4f}")
    c.check("mae_model is computed correctly", O.close(num(need(c.ans, "mae_model")), mae, rel=0.01), f"!= {mae:.4f}")
    c.check("mae_baseline is computed correctly", O.close(num(need(c.ans, "mae_baseline")), base_mae, rel=0.01),
            f"!= {base_mae:.4f}")
    em = [abs(preds[r] - c.data.value(r, t)) for r in preds]
    eb = [abs(m - c.data.value(r, t)) for r in preds]
    v = need(c.ans, "verdict")
    if v not in ("better", "worse", "no clear difference"):
        raise SchemaError(f"bad verdict {v!r}")
    ok, want = _verdict_ok(v, em, eb, ("better", "worse", "no clear difference"))
    c.check("verdict follows from the errors", ok, f"said {v!r}; the errors give {want}")


def _verdict_ok(v, xs, ys, labels):
    """Judge a better/worse/no-difference claim about XS vs YS (lower is better).

    Clear-cut cases (oracle.clear_verdict) must be called correctly; otherwise
    any honest call passes, but a claimed winner must have the better median.
    """
    better, worse, none = labels
    mx, my = O.median(xs), O.median(ys)
    side = better if mx < my else worse if mx > my else none
    clear = O.clear_verdict(xs, ys)
    if clear == "same":
        return v == none, repr(none)
    if clear == "different":
        return v == side, repr(side)
    return v in (none, side), f"{none!r} or {side!r}"


# ---- the 60 requests ---------------------------------------------------------

def _p(kind_, **params):
    return (kind_, params)


REQUESTS = {
    "CAR-01": _p("summary"), "CAR-02": _p("top"),
    "CAR-03": _p("predict", target="Mpg+"), "CAR-04": _p("predict_interval", target="Lbs-"),
    "CAR-05": _p("groups", group="Clndrs", outcome="Mpg+"), "CAR-06": _p("year_groups", group="Model"),
    "CAR-07": _p("stream_desirable"), "CAR-08": _p("stream_summary"),
    "CAR-09": _p("blame_rank", outcome="Mpg+"), "CAR-10": _p("blame_contrast", outcome_text="overall performance"),
    "CAR-11": _p("justify_neighbors"), "CAR-12": _p("justify_pair", strengths=True),
    "CAR-13": _p("choose_best"), "CAR-14": _p("choose_order"),
    "CAR-15": _p("fix_alternatives"), "CAR-16": _p("fix_targets"),
    "CAR-17": _p("spot_x"), "CAR-18": _p("spot_y", cols=["Mpg+", "Lbs-"]),
    "CAR-19": _p("race_select"), "CAR-20": _p("race_predict", target="Mpg+"),

    "NASA-01": _p("summary"), "NASA-02": _p("top"),
    "NASA-03": _p("predict", target="Effort-"), "NASA-04": _p("predict_interval", target="Defects-"),
    "NASA-05": _p("groups", group="pcap", outcome="Effort-"), "NASA-06": _p("groups", group="cplx", outcome="Defects-"),
    "NASA-07": _p("stream_alert_pred"), "NASA-08": _p("stream_summary"),
    "NASA-09": _p("blame_rank", outcome="Effort-"),
    "NASA-10": _p("blame_contrast", outcome="Defects-", outcome_text="Defects-"),
    "NASA-11": _p("justify_estimate"), "NASA-12": _p("justify_pair"),
    "NASA-13": _p("choose_rule"), "NASA-14": _p("choose_order"),
    "NASA-15": _p("fix_predicted", what="effort"),
    "NASA-16": _p("fix_predicted", what="defects", max_changes=3, tradeoff=True),
    "NASA-17": _p("spot_x"), "NASA-18": _p("spot_y", cols=["Effort-", "Defects-"]),
    "NASA-19": _p("race_predict", target="Effort-"), "NASA-20": _p("race_select"),

    "APACHE-01": _p("summary"), "APACHE-02": _p("top_settings"),
    "APACHE-03": _p("predict", target="Performance-"), "APACHE-04": _p("pair_diff", target="Performance-"),
    "APACHE-05": _p("groups", group="keepAlive", outcome="Performance-"),
    "APACHE-06": _p("groups", group="accessLog", outcome="Performance-"),
    "APACHE-07": _p("stream_alert_poor", target="Performance-"), "APACHE-08": _p("stream_summary"),
    "APACHE-09": _p("blame_rank", outcome="Performance-"),
    "APACHE-10": _p("blame_combos", outcome="Performance-"),
    "APACHE-11": _p("justify_neighbors"), "APACHE-12": _p("justify_pair"),
    "APACHE-13": _p("choose_best"), "APACHE-14": _p("choose_order"),
    "APACHE-15": _p("fix_observed", target="Performance-"), "APACHE-16": _p("fix_target", target="Performance-"),
    "APACHE-17": _p("spot_y", cols=["Performance-"]), "APACHE-18": _p("spot_x"),
    "APACHE-19": _p("race_select"), "APACHE-20": _p("race_predict", target="Performance-"),
}


def spec_text(rid, data):
    """The prompt line describing request RID's ANSWER JSON for this dataset."""
    k, params = REQUESTS[rid]
    text = KINDS[k][0]
    goals = [g for g, _ in data.goals()]
    n = data.n
    fmt = dict(params)
    fmt.update(goal_keys=", ".join(f'"{g}": {{...}}' for g in goals), goal_list=", ".join(goals),
               budget=O.budget(n), half=O.arrivals_count(n) // 2, k=O.arrivals_count(n), max_flag=n // 10 + 1,
               cols=" or ".join(params.get("cols", [])), outcome_text=params.get("outcome_text", ""),
               limit=f" (at most {params['max_changes']} columns)" if params.get("max_changes") else "",
               what=params.get("what", ""),
               extra=', "predicted_effort": {"before": number, "after": number} (the effort tradeoff)'
               if params.get("tradeoff") else "")
    return f"- **{rid}**: print `ANSWER {rid} <json>` where the JSON is " + text.format(**fmt)


def runs_needed(request_ids):
    out = []
    for rid in request_ids:
        for r in KINDS[REQUESTS[rid][0]][2]:
            if r not in out:
                out.append(r)
    return out


def check_request(rid, data, ans, run="base", extra=None):
    """Run REQUEST RID's checks on one parsed answer; returns the Ctx with its items."""
    k, params = REQUESTS[rid]
    c = Ctx(data, ans, params, run, extra=extra or {})
    try:
        KINDS[k][1](c)
    except SchemaError as e:
        c.check("answer format", False, str(e))
    return c


def compare_request(rid, base_ctx, new_data, new_ans, run):
    k, params = REQUESTS[rid]
    new = Ctx(new_data, new_ans, params, run)
    fn = COMPARE.get(k)
    if fn is None:
        return new
    try:
        fn(new, base_ctx)
    except SchemaError as e:
        new.check("answer format", False, str(e))
    return new
