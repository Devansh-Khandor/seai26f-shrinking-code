"""Independent reference calculations for the semantic checks. Never runs generated code.

Everything here is computed from the CSV a program was given, with the
standard library only, so a check never trusts the program it is checking.
Rows are numbered 1..n in file order (the header is not a row).
"""

from dataclasses import dataclass
from math import sqrt
from pathlib import Path

ORDINAL = {"vl": 0, "l": 1, "n": 2, "h": 3, "vh": 4, "xh": 5}


def _cell(s):
    s = s.strip()
    if s == "?" or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return s


@dataclass
class Data:
    names: list          # header names, in order
    raw: list            # rows of original cell strings (for writing perturbed copies)
    rows: list           # rows of parsed cells: float, str, or None

    @property
    def n(self):
        return len(self.rows)

    def col(self, name):
        return self.names.index(name)

    def goals(self):
        """[(name, +1 maximize | -1 minimize)]"""
        return [(c, 1 if c.endswith("+") else -1) for c in self.names if c[-1] in "+-"]

    def inputs(self):
        """Columns a model may use: not goals, not ending in X."""
        return [c for c in self.names if c[-1] not in "+-X"]

    def features(self):
        """Every non-goal column (inputs plus X columns)."""
        return [c for c in self.names if c[-1] not in "+-"]

    def value(self, row, name):
        return self.rows[row - 1][self.col(name)]

    def values(self, name, rows=None):
        i = self.col(name)
        rows = range(1, self.n + 1) if rows is None else rows
        return [self.rows[r - 1][i] for r in rows if self.rows[r - 1][i] is not None]

    def is_numeric(self, name):
        return all(isinstance(v, float) for v in self.values(name))

    def text(self):
        return "\n".join(",".join(r) for r in [self.names] + self.raw) + "\n"

    def with_raw(self, raw):
        return Data(list(self.names), raw, [[_cell(c) for c in r] for r in raw])


def load(path):
    return parse(Path(path).read_text(encoding="utf-8-sig"))


def parse(text):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    names = [s.strip() for s in lines[0].split(",")]
    raw = [ln.split(",") for ln in lines[1:]]
    return Data(names, raw, [[_cell(c) for c in r] for r in raw])


# ---- protocol shared with the prompt -------------------------------------

def held_out(n):
    return [r for r in range(1, n + 1) if r % 5 == 0]


def arrivals_count(n):
    return int(0.2 * n + 0.5)


def arrivals(n):
    k = arrivals_count(n)
    return list(range(n - k + 1, n + 1))


def budget(n):
    return min(30, n // 5)


# ---- statistics ----------------------------------------------------------

def mean(xs):
    return sum(xs) / len(xs)


def medians(xs):
    """Every common median convention: middle-average, lower, upper."""
    s = sorted(xs)
    n = len(s)
    lo, hi = s[(n - 1) // 2], s[n // 2]
    return {(lo + hi) / 2, lo, hi}


def median(xs):
    s = sorted(xs)
    n = len(s)
    return (s[(n - 1) // 2] + s[n // 2]) / 2


def sd(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def close(a, b, rel=1e-3, abs_=0.006):
    return abs(a - b) <= max(abs_, rel * abs(b))


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(xs, ys):
    if len(xs) < 3:
        return 0.0
    rx, ry = ranks(xs), ranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def cliffs_delta(xs, ys):
    gt = sum(1 for x in xs for y in ys if x > y)
    lt = sum(1 for x in xs for y in ys if x < y)
    return (gt - lt) / (len(xs) * len(ys))


def ks_rejects(xs, ys, a=1.36):
    """True when the two-sample KS statistic exceeds the 95% critical value."""
    n, m = len(xs), len(ys)
    pts = sorted(set(xs) | set(ys))
    d = max(abs(sum(x <= v for x in xs) / n - sum(y <= v for y in ys) / m) for v in pts)
    return d > a * sqrt((n + m) / (n * m))


def clear_verdict(xs, ys):
    """'different' / 'same' when the evidence is unambiguous, else None (either answer accepted).

    Large effect (|Cliff's delta| >= 0.474) and KS rejection -> different.
    Negligible effect (|delta| < 0.147) and no KS rejection -> same.
    Groups under 5 rows are never clear.
    """
    if len(xs) < 5 or len(ys) < 5:
        return None
    d, ks = abs(cliffs_delta(xs, ys)), ks_rejects(xs, ys)
    if d >= 0.474 and ks:
        return "different"
    if d < 0.147 and not ks:
        return "same"
    return None


def numeric(v):
    """Numbers stay; NASA-style ratings become ordinals; other symbols None."""
    if isinstance(v, float):
        return v
    if isinstance(v, str):
        return ORDINAL.get(v.strip().lower())
    return None


# ---- goals: distance to heaven, dominance --------------------------------

def d2h(data, rows=None):
    """{row: distance to heaven} using min-max bounds over ROWS (default: all rows)."""
    rows = list(range(1, data.n + 1)) if rows is None else list(rows)
    goals = data.goals()
    bounds = {}
    for g, _ in goals:
        vs = data.values(g, rows)
        bounds[g] = (min(vs), max(vs))
    out = {}
    for r in rows:
        total, ok = 0.0, True
        for g, w in goals:
            v = data.value(r, g)
            if v is None:
                ok = False
                break
            lo, hi = bounds[g]
            x = (v - lo) / (hi - lo) if hi > lo else 0.0
            total += (x - (1 if w > 0 else 0)) ** 2
        if ok:
            out[r] = sqrt(total / len(goals))
    return out


def zsum(data, rows=None):
    """{row: sum of goal z-scores, oriented so lower = better}."""
    rows = list(range(1, data.n + 1)) if rows is None else list(rows)
    out = {r: 0.0 for r in rows}
    for g, w in data.goals():
        vs = data.values(g, rows)
        m, s = mean(vs), sd(vs) or 1.0
        for r in rows:
            out[r] += -w * (data.value(r, g) - m) / s
    return out


def ranksum(data, rows=None):
    """{row: sum of per-goal ranks, best = 0, ties averaged}."""
    rows = list(range(1, data.n + 1)) if rows is None else list(rows)
    out = {r: 0.0 for r in rows}
    for g, w in data.goals():
        rk = ranks([-w * data.value(r, g) for r in rows])
        for r, k in zip(rows, rk):
            out[r] += k
    return out


def aggregates(data, rows=None):
    """Three standard ways to score 'overall' (lower = better). The requests never fix one,
    so an overall-quality claim passes if it holds under any of them."""
    return [d2h(data, rows), zsum(data, rows), ranksum(data, rows)]


def good_rows(data, frac, rows=None):
    """Rows in the best FRAC under any aggregate, plus the Pareto front."""
    out = pareto(data, rows)
    for s in aggregates(data, rows):
        out |= best_fraction(s, frac)
    return out


def dominates(data, a, b):
    better = False
    for g, w in data.goals():
        va, vb = data.value(a, g), data.value(b, g)
        if va is None or vb is None:
            return False
        if w * va < w * vb:
            return False
        if w * va > w * vb:
            better = True
    return better


def pareto(data, rows=None):
    rows = list(range(1, data.n + 1)) if rows is None else list(rows)
    return {r for r in rows if not any(dominates(data, o, r) for o in rows if o != r)}


def best_fraction(scores, frac):
    """Rows whose score is within the best FRAC (lowest scores), ties included."""
    s = sorted(scores.values())
    cut = s[max(0, int(len(s) * frac) - 1)]
    return {r for r, v in scores.items() if v <= cut}


def worse(data, name, a, b):
    """Is value A worse than value B for goal NAME?"""
    return a < b if name.endswith("+") else a > b


# ---- distances over inputs -------------------------------------------------

class XSpace:
    """Min-max numeric gaps, 0/1 symbol mismatch, missing = 1; Euclidean over input columns."""

    def __init__(self, data, columns=None):
        self.data = data
        self.cols = [data.col(c) for c in (columns or data.inputs())]
        self.bounds = {}
        for i in self.cols:
            vs = [r[i] for r in data.rows if isinstance(r[i], float)]
            if vs and len(vs) == len([r for r in data.rows if r[i] is not None]):
                self.bounds[i] = (min(vs), max(vs))

    def gap(self, i, a, b):
        if a is None or b is None:
            return 1.0
        if i in self.bounds:
            lo, hi = self.bounds[i]
            return abs(a - b) / (hi - lo) if hi > lo else 0.0
        return 0.0 if a == b else 1.0

    def dist(self, r1, r2):
        a, b = self.data.rows[r1 - 1], self.data.rows[r2 - 1]
        return sqrt(sum(self.gap(i, a[i], b[i]) ** 2 for i in self.cols) / len(self.cols))

    def nearest(self, r, frac):
        """Rows (other than R) within the nearest FRAC of all other rows, ties included."""
        others = [o for o in range(1, self.data.n + 1) if o != r]
        ds = {o: self.dist(r, o) for o in others}
        cut = sorted(ds.values())[max(0, int(len(others) * frac) - 1)]
        return {o for o, d in ds.items() if d <= cut + 1e-12}

    def knn_scores(self, k=5):
        out = {}
        for r in range(1, self.data.n + 1):
            ds = sorted(self.dist(r, o) for o in range(1, self.data.n + 1) if o != r)
            out[r] = mean(ds[:k])
        return out

    def centroid_scores(self):
        """Distance from each row to the column-wise middle (mean / mode)."""
        mids = {}
        for i in self.cols:
            vs = [row[i] for row in self.data.rows if row[i] is not None]
            if i in self.bounds:
                mids[i] = mean(vs)
            else:
                mids[i] = max(set(vs), key=vs.count) if vs else None
        out = {}
        for r in range(1, self.data.n + 1):
            row = self.data.rows[r - 1]
            out[r] = sqrt(sum(self.gap(i, row[i], mids[i]) ** 2 for i in self.cols) / len(self.cols))
        return out

    def loo_residuals(self, name, k=5):
        """|actual - mean of the K nearest other rows' NAME| / sd(NAME), per row."""
        j = self.data.col(name)
        s = sd(self.data.values(name)) or 1.0
        out = {}
        for r in range(1, self.data.n + 1):
            v = self.data.rows[r - 1][j]
            if v is None:
                continue
            others = sorted((self.dist(r, o), o) for o in range(1, self.data.n + 1)
                            if o != r and self.data.rows[o - 1][j] is not None)
            exp = mean([self.data.rows[o - 1][j] for _, o in others[:k]])
            out[r] = abs(v - exp) / s
        return out


def top_fraction(scores, frac):
    """Rows whose score is within the top FRAC (highest scores), ties included."""
    s = sorted(scores.values(), reverse=True)
    cut = s[max(0, int(len(s) * frac) - 1)]
    return {r for r, v in scores.items() if v >= cut}


# ---- association measures for blame ---------------------------------------

def association(data, feature, outcome_rows, other_rows):
    """|Cliff's delta| of FEATURE (numeric or ordinal) between two row sets, or None."""
    xs = [numeric(data.value(r, feature)) for r in outcome_rows]
    ys = [numeric(data.value(r, feature)) for r in other_rows]
    xs, ys = [x for x in xs if x is not None], [y for y in ys if y is not None]
    if not xs or not ys:
        return None
    return abs(cliffs_delta(xs, ys))


def correlation(data, feature, outcome):
    pairs = [(numeric(data.value(r, feature)), data.value(r, outcome)) for r in range(1, data.n + 1)]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    return abs(spearman([a for a, _ in pairs], [b for _, b in pairs]))


def top_k(scores, k):
    """Names in the top K of SCORES (dict name -> score or None), ties included."""
    vals = sorted((v for v in scores.values() if v is not None), reverse=True)
    if not vals:
        return set()
    cut = vals[min(k, len(vals)) - 1]
    return {n for n, v in scores.items() if v is not None and v >= cut - 1e-12}


# ---- the professor's same(), used for verdict consistency ------------------

def same(xs, ys):
    """timm's stats.same with eps = 0.35 * sd of the first sample (see harness/vendor/stats.py)."""
    from harness.vendor_stats import same as _same, spread_of
    return _same(xs, ys, 0.35 * spread_of(xs))
