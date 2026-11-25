"""Turn runs.csv into histogram data, `same`/`bestRanks` tables and pass rates. Never calls an LLM.

    python -m harness.analyze --model qwen2.5-coder:14b

Statistics are the professor's same() and bestRanks() (harness/vendor_stats.py,
copied unchanged): same() is a median-difference shortcut, then Cliff's delta
and KS; eps = 0.35 * sd of the better-median sample, as bestRanks uses.

Samples (fixed 2026-11-10, before any data existed):
  primary   trial_total: per COMPLETE trial, total new_loc over skills 2-10.
            A trial is complete when every skill 2-10 has a numeric new_loc
            (rule revised 2026-11-10 after code review, still before any data:
            counting blank steps as 0 would let a failing arm look like reuse).
  secondary pooled:      every non-blank new_loc for skills 2-10
"""

import argparse
from collections import Counter
import csv
from pathlib import Path
from statistics import median

from harness import vendor_stats
from harness.results import REQ_KEY, latest_rows, load_rows
from harness.workspace import slug

HARNESS_DIR = Path(__file__).resolve().parent
RESULTS = HARNESS_DIR / "results" / "v2" / "runs.csv"
PAIRS = [("T2", "T1"), ("T4", "T3"), ("T6", "T5")]
FOUNDATIONS = [("plain Python", ("T1", "T2")), ("scikit-learn", ("T3", "T4")), ("ezr", ("T5", "T6"))]
LADDER_TAIL = range(2, 11)
SEM_STATUSES = ("pass", "fail", "unverified", "parse_failure", "execution_failure", "checker_error")


def same(xs, ys):
    """The professor's same(), eps = 0.35 * sd of the sample with the lower median."""
    if not xs or not ys:
        raise ValueError("same() needs two non-empty samples")
    first = xs if median(xs) <= median(ys) else ys
    return vendor_stats.same(list(map(float, xs)), list(map(float, ys)),
                             0.35 * vendor_stats.spread_of(list(map(float, first))))


def best_ranks(samples):
    """Treatments tied for best (lowest median) by the professor's bestRanks; empty samples skipped."""
    d = {k: list(map(float, v)) for k, v in samples.items() if v}
    return list(vendor_stats.bestRanks(d)) if d else []


def for_model(rows, model):
    return [r for r in rows if r["model"] == model]


def _select(rows, domain, treatment):
    return [r for r in rows if r["domain"] == domain and r["treatment"] == treatment]


def _num(x):
    return None if x in ("", None) else float(x)


def _by_trial(rows, domain, treatment):
    trials = {}
    for r in _select(rows, domain, treatment):
        if int(r["skill_index"]) in LADDER_TAIL:
            trials.setdefault(int(r["trial"]), {})[int(r["skill_index"])] = _num(r["new_loc"])
        else:
            trials.setdefault(int(r["trial"]), {})
    return trials


def _complete(steps):
    return all(steps.get(s) is not None for s in LADDER_TAIL)


def trial_totals(rows, domain, treatment):
    trials = _by_trial(rows, domain, treatment)
    return [sum(trials[k].values()) for k in sorted(trials) if _complete(trials[k])]


def incomplete_trials(rows, domain, treatment):
    return sum(1 for steps in _by_trial(rows, domain, treatment).values() if not _complete(steps))


def pooled(rows, domain, treatment):
    return [_num(r["new_loc"]) for r in _select(rows, domain, treatment)
            if int(r["skill_index"]) in LADDER_TAIL and _num(r["new_loc"]) is not None]


SAMPLES = {"trial_total": trial_totals, "pooled": pooled}


def histogram(rows):
    groups = {}
    for r in rows:
        v = _num(r["pct_new"])
        if v is not None:
            groups.setdefault((r["domain"], r["treatment"], int(r["skill_index"])), []).append(v)
    return {k: median(v) for k, v in groups.items()}


def pass_rates(rows, domain, treatment):
    # unresolved backend errors are the machine's problem, not a step outcome
    sel = [r for r in _select(rows, domain, treatment) if r["status"] != "backend_error"]
    return _rates(sel)


def _rates(sel):
    n = len(sel)
    count = lambda f: sum(1 for r in sel if f(r))
    sem = lambda r: r.get("semantic") or ""
    verified = count(lambda r: sem(r) in ("pass", "fail"))
    return {"n": n,
            "runs_rate": count(lambda r: r["runs"] == "True") / n if n else 0.0,
            "contract_rate": count(lambda r: r["contract"] == "True") / n if n else 0.0,
            "semantic_rate": count(lambda r: sem(r) == "pass") / n if n else 0.0,
            "semantic_verified_rate": count(lambda r: sem(r) == "pass") / verified if verified else 0.0,
            "verified": verified,
            "unverified": count(lambda r: sem(r) == "unverified"),
            "semantic_fail": count(lambda r: sem(r) == "fail"),
            "contract_not_semantic": count(lambda r: r["contract"] == "True" and sem(r) != "pass"),
            "checker_error": count(lambda r: sem(r) == "checker_error"),
            "overflow": count(lambda r: r["status"] == "context_overflow"),
            "parse_failed": count(lambda r: r["status"] == "parse_failed"),
            "timeout": count(lambda r: r["status"] == "timeout"),
            "output_truncated": count(lambda r: r["status"] == "output_truncated"),
            "crashed": count(lambda r: r["status"] == "ok" and r["runs"] != "True")}


def trial_semantic(rows, domain, treatment):
    """Skills (of 10) passing the semantic check, per trial that has all 10 steps recorded."""
    trials = {}
    sel = _select(rows, domain, treatment)
    if not any(r.get("semantic") for r in sel):
        return []          # v1 results: no semantic check was run
    for r in sel:
        trials.setdefault(int(r["trial"]), {})[int(r["skill_index"])] = r.get("semantic") == "pass"
    return [sum(v.values()) for k, v in sorted(trials.items()) if len(v) == 10]


def failure_counts(rows, domain, treatment):
    return Counter(r.get("failure_category") or "none (passed)" for r in _select(rows, domain, treatment)
                   if r["status"] != "backend_error")


def sizing(rows):
    out = {}
    for r in rows:
        v = _num(r.get("prompt_tokens")) or _num(r.get("prompt_tokens_est"))
        if v is not None:
            k = (r["domain"], r["treatment"], int(r["skill_index"]))
            out[k] = max(out.get(k, 0), int(v))
    return out


def verdict(reuse, base):
    if same(reuse, base):
        return "no certified difference"
    return "reuse lower (certified)" if median(reuse) < median(base) else "reuse higher (certified)"


def _requirement_table(req_rows, model):
    rows = [r for r in latest_rows_by(req_rows, REQ_KEY) if r["model"] == model]
    if not rows:
        return None
    domains = sorted({r["domain"] for r in rows})
    treatments = sorted({r["treatment"] for r in rows})
    md = [f"# Requirement-level semantic results — model `{model}`", "",
          "One row per request per step (two per skill). Rate = pass / all attempts; "
          "unverified and failures stay in the denominator.", "",
          "| domain | treatment | requests | " + " | ".join(SEM_STATUSES) + " | pass rate |",
          "|---|---|---|" + "---|" * len(SEM_STATUSES) + "---|"]
    for d in domains:
        for t in treatments:
            sel = [r for r in rows if r["domain"] == d and r["treatment"] == t]
            if not sel:
                continue
            c = Counter(r["status"] for r in sel)
            md.append(f"| {d} | {t} | {len(sel)} | " + " | ".join(str(c.get(s, 0)) for s in SEM_STATUSES)
                      + f" | {c.get('pass', 0) / len(sel):.0%} |")
    md += ["", "## Pass rate per request", "", "| request | " + " | ".join(treatments) + " |",
           "|---|" + "---|" * len(treatments)]
    for rid in sorted({r["request_id"] for r in rows}, key=lambda x: (x.split("-")[0], int(x.split("-")[1]))):
        cells = []
        for t in treatments:
            sel = [r for r in rows if r["request_id"] == rid and r["treatment"] == t]
            cells.append(f"{sum(r['status'] == 'pass' for r in sel)}/{len(sel)}" if sel else "—")
        md.append(f"| {rid} | " + " | ".join(cells) + " |")
    return md


def latest_rows_by(rows, key):
    latest = {}
    for r in rows:
        latest[tuple(str(r[k]) for k in key)] = r
    return list(latest.values())


def _fmt(xs):
    return ", ".join(f"{x:g}" for x in xs)


def _same_section(rows, domain, sample, label):
    lines = [f"### {label}: `{sample}`", "",
             "| pair | reuse values | base values | reuse median | base median | n | excluded trials | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    n_seen = 0
    for reuse_t, base_t in PAIRS:
        if not _select(rows, domain, reuse_t) and not _select(rows, domain, base_t):
            continue
        a, b = SAMPLES[sample](rows, domain, reuse_t), SAMPLES[sample](rows, domain, base_t)
        excl = f"{incomplete_trials(rows, domain, reuse_t)}/{incomplete_trials(rows, domain, base_t)}"
        if not a or not b:
            lines.append(f"| {reuse_t} vs {base_t} | {_fmt(a)} | {_fmt(b)} | — | — | {len(a)}/{len(b)} "
                         f"| {excl} | not enough data |")
            continue
        n_seen = max(n_seen, len(a), len(b))
        lines.append(f"| {reuse_t} vs {base_t} | {_fmt(a)} | {_fmt(b)} | {median(a):g} | {median(b):g} "
                     f"| {len(a)}/{len(b)} | {excl} | {verdict(a, b)} |")
    return lines, n_seen


def write_reports(rows, model, out_dir, req_rows=()):
    rows = for_model(latest_rows(rows), model)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    domains = sorted({r["domain"] for r in rows})
    treatments = sorted({r["treatment"] for r in rows})
    written = []

    hist = histogram(rows)
    p = out_dir / "histogram.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["domain", "treatment", "skill_index", "median_pct_new"])
        for (d, t, s), v in sorted(hist.items()):
            w.writerow([d, t, s, v])
    written.append(p)

    md = [f"# Median % new code per step — model `{model}`", ""]
    for d in domains:
        md += [f"## {d}", "", "| treatment | " + " | ".join(str(s) for s in range(1, 11)) + " |",
               "|---|" + "---|" * 10]
        for t in treatments:
            md.append(f"| {t} | " + " | ".join(f"{hist[(d, t, s)]:g}" if (d, t, s) in hist else "—"
                                              for s in range(1, 11)) + " |")
        md.append("")
    p = out_dir / "histogram.md"
    p.write_text("\n".join(md), encoding="utf-8")
    written.append(p)

    md, n_max = [f"# `same` tables — model `{model}`", "",
                 "`same` = the professor's same() from stats.py (median-difference shortcut, Cliff's delta, KS), "
                 "called unchanged with eps = 0.35 sd.", ""], 0
    for d in domains:
        md += [f"## {d}", ""]
        for sample, label in (("trial_total", "primary"), ("pooled", "secondary")):
            lines, n = _same_section(rows, d, sample, label)
            if sample == "trial_total":
                n_max = max(n_max, n)
            md += lines + [""]
    md += ["## bestRanks", "",
           "Treatments tied for best by the professor's bestRanks(). LOC: lowest total new LOC over skills 2-10 "
           "(complete trials only). Correctness: most skills passing the semantic check per trial "
           "(trials with all 10 steps recorded).", "",
           "| domain | LOC: tied for best | n (complete/attempted trials) | correctness: tied for best |",
           "|---|---|---|---|"]
    for d in domains:
        loc = {t: trial_totals(rows, d, t) for t in treatments}
        cor = {t: [-x for x in trial_semantic(rows, d, t)] for t in treatments}
        counts = ", ".join(f"{t} {len(loc[t])}/{len(_by_trial(rows, d, t))}" for t in treatments)
        md.append(f"| {d} | {', '.join(best_ranks(loc)) or 'not enough data'} | {counts} | "
                  f"{', '.join(best_ranks(cor)) or 'no semantic data'} |")
    md.append("")
    md.append(f"N={n_max} trials: `same` on small samples has little power; "
              "a 'no certified difference' is not evidence of equivalence. "
              "Excluded trials = trials missing a numeric new_loc for some skill 2-10 "
              "(overflow, parse failure, cut-off reply, or not yet run); see pass_rates.md.")
    p = out_dir / "same_tables.md"
    p.write_text("\n".join(md) + "\n", encoding="utf-8")
    written.append(p)

    md = [f"# Pass rates — model `{model}`", "",
          "Per skill attempt (both requests must pass for semantic). Every attempt stays in the denominator; "
          "`semantic (verified)` divides by attempts the checker could judge (pass + fail) instead. "
          "`contract only` = passed the original format contract but not the semantic check.", "",
          "| domain | treatment | steps | runs | contract | semantic | semantic (verified) | contract only "
          "| unverified | checker errors | crashed | parse_failed | overflow | truncated | timeout |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def line(label_d, label_t, pr):
        return (f"| {label_d} | {label_t} | {pr['n']} | {pr['runs_rate']:.0%} | {pr['contract_rate']:.0%} "
                f"| {pr['semantic_rate']:.0%} | {pr['semantic_verified_rate']:.0%} of {pr['verified']} "
                f"| {pr['contract_not_semantic']} | {pr['unverified']} | {pr['checker_error']} | {pr['crashed']} "
                f"| {pr['parse_failed']} | {pr['overflow']} | {pr['output_truncated']} | {pr['timeout']} |")
    for d in domains:
        for t in treatments:
            pr = pass_rates(rows, d, t)
            if pr["n"]:
                md.append(line(d, t, pr))
    md += ["", "## By foundation (all domains)", "", md[4], md[5]]
    for name, ts in FOUNDATIONS:
        sel = [r for r in rows if r["treatment"] in ts and r["status"] != "backend_error"]
        if sel:
            md.append(line("all", f"{name} ({'+'.join(ts)})", _rates(sel)))
    md += ["", "## Failure categories", "",
           "`system:*` and `evaluator:*` are the machine's or the checker's problem, not the model's.", "",
           "| domain | treatment | categories |", "|---|---|---|"]
    for d in domains:
        for t in treatments:
            fc = failure_counts(rows, d, t)
            if fc:
                md.append(f"| {d} | {t} | " + ", ".join(f"{k} {v}" for k, v in sorted(fc.items())) + " |")
    p = out_dir / "pass_rates.md"
    p.write_text("\n".join(md) + "\n", encoding="utf-8")
    written.append(p)

    req = _requirement_table(list(req_rows), model)
    if req:
        p = out_dir / "requirements.md"
        p.write_text("\n".join(req) + "\n", encoding="utf-8")
        written.append(p)

    size = sizing(rows)
    md = [f"# Max prompt tokens per step — model `{model}`", "",
          "Reported tokens where the backend gives them, else the chars/4 estimate.", ""]
    for d in domains:
        md += [f"## {d}", "", "| treatment | " + " | ".join(str(s) for s in range(1, 11)) + " |",
               "|---|" + "---|" * 10]
        for t in treatments:
            md.append(f"| {t} | " + " | ".join(str(size.get((d, t, s), "—")) for s in range(1, 11)) + " |")
        md.append("")
    p = out_dir / "sizing.md"
    p.write_text("\n".join(md), encoding="utf-8")
    written.append(p)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m harness.analyze")
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    out = a.out or Path(a.results).parent / slug(a.model)
    req = load_rows(Path(a.results).parent / "requirements.csv")
    for p in write_reports(load_rows(a.results), a.model, out, req):
        print(p)


if __name__ == "__main__":
    main()
