"""Turn runs.csv into histogram data and `same` tables. Never calls an LLM.

    python -m harness.analyze --model qwen2.5-coder:14b

Samples (fixed 2026-11-10, before any data existed):
  primary   trial_total: per COMPLETE trial, total new_loc over skills 2-10.
            A trial is complete when every skill 2-10 has a numeric new_loc
            (rule revised 2026-11-10 after code review, still before any data:
            counting blank steps as 0 would let a failing arm look like reuse).
  secondary pooled:      every non-blank new_loc for skills 2-10
"""

import argparse
import csv
import os
from pathlib import Path
from statistics import median
import subprocess

from harness.checks import lua_executable
from harness.prompts import VENDOR_DIR
from harness.results import latest_rows, load_rows
from harness.workspace import slug

HARNESS_DIR = Path(__file__).resolve().parent
SAME_LUA = HARNESS_DIR / "same.lua"
PAIRS = [("T2", "T1"), ("T4", "T3"), ("T6", "T5")]
LADDER_TAIL = range(2, 11)


def same(xs, ys):
    if not xs or not ys:
        raise ValueError("same() needs two non-empty samples")
    exe = lua_executable()
    if exe is None:
        raise RuntimeError("Lua required for same")
    data = " ".join(map(repr, map(float, xs))) + "\n" + " ".join(map(repr, map(float, ys))) + "\n"
    env = {**os.environ, "LUA_PATH": f"{VENDOR_DIR}/?.lua;;"}
    p = subprocess.run([exe, str(SAME_LUA)], input=data, capture_output=True, text=True, env=env, check=True)
    return p.stdout.strip() == "true"


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
    n = len(sel)
    count = lambda f: sum(1 for r in sel if f(r))
    return {"n": n,
            "runs_rate": count(lambda r: r["runs"] == "True") / n if n else 0.0,
            "contract_rate": count(lambda r: r["contract"] == "True") / n if n else 0.0,
            "overflow": count(lambda r: r["status"] == "context_overflow"),
            "parse_failed": count(lambda r: r["status"] == "parse_failed"),
            "timeout": count(lambda r: r["status"] == "timeout"),
            "output_truncated": count(lambda r: r["status"] == "output_truncated")}


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


def write_reports(rows, model, out_dir):
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
                 "`same` = the course's ezr.lua Cohen + Cliff's delta + KS test, called unchanged.", ""], 0
    for d in domains:
        md += [f"## {d}", ""]
        for sample, label in (("trial_total", "primary"), ("pooled", "secondary")):
            lines, n = _same_section(rows, d, sample, label)
            if sample == "trial_total":
                n_max = max(n_max, n)
            md += lines + [""]
    md.append(f"N={n_max} trials: `same` on small samples has little power; "
              "a 'no certified difference' is not evidence of equivalence. "
              "Excluded trials = trials missing a numeric new_loc for some skill 2-10 "
              "(overflow, parse failure, cut-off reply, or not yet run); see pass_rates.md.")
    p = out_dir / "same_tables.md"
    p.write_text("\n".join(md) + "\n", encoding="utf-8")
    written.append(p)

    md = [f"# Pass rates — model `{model}`", "",
          "| domain | treatment | steps | runs | contract | overflow | parse_failed | output_truncated | timeout |",
          "|---|---|---|---|---|---|---|---|---|"]
    for d in domains:
        for t in treatments:
            pr = pass_rates(rows, d, t)
            if pr["n"]:
                md.append(f"| {d} | {t} | {pr['n']} | {pr['runs_rate']:.0%} | {pr['contract_rate']:.0%} "
                          f"| {pr['overflow']} | {pr['parse_failed']} | {pr['output_truncated']} | {pr['timeout']} |")
    p = out_dir / "pass_rates.md"
    p.write_text("\n".join(md) + "\n", encoding="utf-8")
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
    ap.add_argument("--results", type=Path, default=HARNESS_DIR / "results" / "runs.csv")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    out = a.out or HARNESS_DIR / "results" / slug(a.model)
    for p in write_reports(load_rows(a.results), a.model, out):
        print(p)


if __name__ == "__main__":
    main()
