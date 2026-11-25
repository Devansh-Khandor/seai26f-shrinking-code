"""Phase 6A: consolidate the frozen internal metrics and the post-hoc external metrics (offline).

Reads only frozen or already-generated files and writes to harness/results/consolidated/:
  treatment_internal.csv   one row per domain x treatment (T1-T6), from the frozen runs.csv / requirements.csv
  reuse_pair_comparison.csv  matched reuse pairs (T2/T1, T4/T3, T6/T5) per domain, with same() diagnostics
  treatment_external.csv   one row per measured external request x domain x treatment, from the phase 1-5H CSVs
  t7_comparison.csv        T7 (n = 1 per domain) beside T1-T6, with LOC and external coverage
  statistical_validity.md  what same() and bestRanks() can and cannot say at these sample sizes
  consolidated_report.md   publication tables and slide-ready findings

No LLM call, no generated program and no T7 adapter is run. Metric algorithms are not re-implemented: internal
samples come from harness.analyze, external values from the earlier phase CSVs, same()/bestRanks() from the
vendored stats.py. Run: python -m harness.consolidate
"""

import argparse
import csv
import hashlib
import io
import json
from math import sqrt
from pathlib import Path
import subprocess
import sys

from harness import analyze as AN
from harness import loc as L
from harness import oracle as O
from harness import vendor_stats as VS

HARNESS_DIR = Path(__file__).resolve().parent
REPO = HARNESS_DIR.parent
RUNS = HARNESS_DIR / "results" / "v2" / "runs.csv"
REQS = HARNESS_DIR / "results" / "v2" / "requirements.csv"
EXT = HARNESS_DIR / "results" / "v2-external" / "gpt-oss-20b"
OUT = HARNESS_DIR / "results" / "consolidated"
FROZEN = HARNESS_DIR / "results" / "v2"
MODEL = "gpt-oss-20b"
DOMAINS = ("cars", "nasa", "apache")
TREATMENTS = ("T1", "T2", "T3", "T4", "T5", "T6")
PAIRS = AN.PAIRS                                  # (reuse, base)
FAMILY = {"T1": "plain Python", "T2": "plain Python", "T3": "scikit-learn", "T4": "scikit-learn",
          "T5": "ezr", "T6": "ezr"}
REUSE = {"T1": False, "T2": True, "T3": False, "T4": True, "T5": False, "T6": True}
TAIL = AN.LADDER_TAIL                             # skills 2..10, the frozen LOC rule

# ---- internal ---------------------------------------------------------------------


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _sel(rows, d, t):
    return [r for r in rows if r["model"] == MODEL and r["domain"] == d and r["treatment"] == t]


def _loc(r):
    return None if r["new_loc"] in ("", None) else float(r["new_loc"])


def loc_trend(runs, d, t):
    """Spearman of new LOC against skill index over every measured step 2-10 (H1 win condition 1)."""
    pts = [(int(r["skill_index"]), _loc(r)) for r in _sel(runs, d, t)
           if int(r["skill_index"]) in TAIL and _loc(r) is not None]
    return (O.spearman([a for a, _ in pts], [b for _, b in pts]) if len(pts) >= 3 else ""), len(pts)


def internal_row(runs, reqs, d, t):
    steps = _sel(runs, d, t)
    req = _sel(reqs, d, t)
    totals = AN.trial_totals(runs, d, t)
    tail = [r for r in steps if int(r["skill_index"]) in TAIL]
    measured = [r for r in tail if _loc(r) is not None]
    loc_all = sum(_loc(r) for r in measured)
    loc_crash = sum(_loc(r) for r in measured if r["runs"] != "True")
    loc_unsem = sum(_loc(r) for r in measured if r["semantic"] != "pass")
    fails = lambda prefix: sum(any(f.startswith(prefix) for f in x["failed_checks"].split(" | ")) for x in req)
    rho, npts = loc_trend(runs, d, t)
    sem_trial = AN.trial_semantic(runs, d, t)
    return {
        "domain": d, "treatment": t, "family": FAMILY[t], "reuse": REUSE[t],
        "trials_attempted": len({r["trial"] for r in steps}),
        "complete_loc_trajectories": len(totals),
        "loc_total_2_10_values": ";".join(f"{v:g}" for v in totals),
        "loc_total_2_10_median": O.median(totals) if totals else "",
        "loc_total_2_10_min": min(totals) if totals else "", "loc_total_2_10_max": max(totals) if totals else "",
        "loc_steps_measured": len(measured), "loc_step_median": O.median([_loc(r) for r in measured]) if measured else "",
        "loc_share_in_crashed_steps": loc_crash / loc_all if loc_all else "",
        "loc_share_in_semantic_failures": loc_unsem / loc_all if loc_all else "",
        "loc_trend_spearman": rho, "loc_trend_points": npts,
        "steps": len(steps), "steps_generated": sum(r["status"] == "ok" for r in steps),
        "steps_parse_failed": sum(r["status"] == "parse_failed" for r in steps),
        "steps_executable": sum(r["runs"] == "True" for r in steps),
        "steps_crashed": sum(r["status"] == "ok" and r["runs"] != "True" for r in steps),
        "steps_semantic_pass": sum(r["semantic"] == "pass" for r in steps),
        "step_semantic_rate": sum(r["semantic"] == "pass" for r in steps) / len(steps),
        "skills_passing_per_trial": ";".join(str(x) for x in sem_trial),
        "requests": len(req), "requests_pass": sum(x["status"] == "pass" for x in req),
        "request_pass_rate": sum(x["status"] == "pass" for x in req) / len(req),
        "requests_execution_failure": sum(x["status"] == "execution_failure" for x in req),
        "requests_parse_failure": sum(x["status"] == "parse_failure" for x in req),
        "requests_schema_failure": sum(any(f.startswith(("base:answer format", "base:answer present"))
                                           for f in x["failed_checks"].split(" | ")) for x in req),
        "leakage_failures": fails("mask:held-out targets not used"),
        "budget_peeking_failures": fails("honeypot:respects the budget"),
    }


# ---- the professor's same(), with the reason for its answer ----------------------------


def same_diagnostics(xs, ys):
    """harness.analyze.same (eps = 0.35 sd of the lower-median sample) plus which test decided."""
    xs, ys = sorted(map(float, xs)), sorted(map(float, ys))
    n, m = len(xs), len(ys)
    first = xs if O.median(xs) <= O.median(ys) else ys
    eps = 0.35 * VS.spread_of(list(first))
    mdiff = abs(xs[n // 2] - ys[m // 2])
    gt = sum(1 for a in xs for b in ys if a > b)
    lt = sum(1 for a in xs for b in ys if a < b)
    cliffs = (gt - lt) / (n * m)          # only |cliffs| is used, as in stats.py
    pts = sorted(set(xs) | set(ys))
    ks = max(abs(sum(a <= v for a in xs) / n - sum(b <= v for b in ys) / m) for v in pts)
    crit = VS.the.stats.conf * sqrt((n + m) / (n * m))
    verdict = AN.same(xs, ys)
    if mdiff <= eps:
        route = "medians within eps -> same"
    elif abs(cliffs) > VS.the.stats.cliffs:
        route = "Cliff's delta above 0.195 -> different"
    else:
        route = "KS decides"
    return {"n_reuse": n, "n_base": m, "eps": eps, "median_gap": mdiff, "abs_cliffs": abs(cliffs),
            "ks_stat": ks, "ks_critical": crit, "ks_can_reject": crit < 1.0, "ks_rejects": ks > crit,
            "same": verdict, "route": route}


def _pair_row(runs, reqs, ext_delivery, d, reuse, base, scope):
    if scope == "trial_total":
        xs, ys = AN.trial_totals(runs, d, reuse), AN.trial_totals(runs, d, base)
    else:
        xs, ys = AN.pooled(runs, d, reuse), AN.pooled(runs, d, base)
    ir, ib = internal_row(runs, reqs, d, reuse), internal_row(runs, reqs, d, base)
    row = {"domain": d, "reuse": reuse, "base": base, "family": FAMILY[reuse], "loc_sample": scope,
           "reuse_loc_median": O.median(xs) if xs else "", "base_loc_median": O.median(ys) if ys else "",
           "reuse_loc_lower": bool(xs and ys and O.median(xs) < O.median(ys)),
           "reuse_complete": ir["complete_loc_trajectories"], "base_complete": ib["complete_loc_trajectories"],
           "reuse_step_semantic_rate": ir["step_semantic_rate"], "base_step_semantic_rate": ib["step_semantic_rate"],
           "reuse_semantic_lower": ir["step_semantic_rate"] < ib["step_semantic_rate"],
           "reuse_request_pass_rate": ir["request_pass_rate"], "base_request_pass_rate": ib["request_pass_rate"],
           "reuse_external_delivery": ext_delivery.get((d, reuse), ""), "base_external_delivery":
           ext_delivery.get((d, base), "")}
    if xs and ys:
        diag = same_diagnostics(xs, ys)
        row.update(diag)
        row["frozen_verdict"] = AN.verdict(xs, ys)
        row["certification_informative"] = diag["ks_can_reject"] and min(len(xs), len(ys)) >= 4
    return row


# ---- external ---------------------------------------------------------------------------

SKILLS = [
    # (skill, summary file, metric filter, request kinds measured, customer question, outcome, reference type,
    #  success column, valid column, quality column, quality label)
    ("Remember", "remember_summary.csv", None, "recommended shortlist (-02)",
     "Which records are our most desirable options?", "shortlist precision against the declared-goal truth set",
     "declared goal (Pareto front / best Performance- quartile); -01 is fidelity only",
     "successful", "integrity_valid", "median_precision", "median shortlist precision"),
    ("Guess", "prediction_summary.csv", None, "prediction (-03); intervals (-04) not measured",
     "How accurate are our estimates for new records?", "MAE on held-out rows vs the historical-mean baseline",
     "observed outcome", "beats_baseline_n", "primary_n", "median_mae_ratio", "median MAE ratio (model/baseline)"),
    ("Certify", "certify_summary.csv", None, "group comparisons (-05, -06)",
     "Are observed group differences reliable?", "agreement of verdicts with the frozen oracle",
     "statistical oracle (clear_verdict)", "oracle_consistent", "integrity_valid", "median_agreement_rate",
     "median oracle agreement on decidable pairs"),
    ("Flow", "flow_summary.csv", None, "streaming decisions/predictions (-07), checkpoints (-08)",
     "Do streaming assessments stay useful as records arrive?",
     "alert/decision precision-recall, streaming MAE ratio, checkpoint shortlist quality",
     "observed outcome (NASA-07 effort) / oracle 'best seen' set / frozen median threshold",
     "successful", "integrity_valid", None, "see flow_report.md"),
    ("Blame", "blame_summary.csv", None, "ranking (-09), contrast (-10), combinations (APACHE-10)",
     "Which characteristics are associated with poor results?",
     "top-3 overlap with the program-independent association ranking",
     "statistical oracle (|Spearman|, |Cliff's delta|); association, not causation",
     "successful", "integrity_valid", "median_rho_top3_overlap", "median rho top-3 overlap"),
    ("Justify", "justify_summary.csv", None, "explanations (-11, -12)",
     "Are explanations truthful about the data?", "share of explanatory claims that are true",
     "observed values and declared goals", "successful", "integrity_valid", "median_precision",
     "median claim precision"),
    ("Choose", "selection_summary.csv", None, "ordered selection (-14); best-row choice (-13) not measured",
     "Does guided selection find good options within the inspection budget?",
     "normalized regret of the best row inspected within the budget",
     "declared goal (d2h)", "beats_random_n", "primary_n", "median_normalized_regret",
     "median normalized regret (0 = best)"),
    ("Fix", "fix_summary.csv", None, "observed alternatives (Cars/Apache -15, -16); NASA not measurable",
     "Which observed changes would improve a weak record?", "improvement of the proposed observed alternative",
     "observed outcome / declared goal", "successful", "integrity_valid", "median_improvement_captured",
     "median share of the attainable improvement captured"),
    ("Race", "race_summary.csv", "selection", "guided vs random selection (-19/-20 selection request)",
     "Does guided search beat random sampling?", "best d2h found by guided vs random selection",
     "declared goal (d2h)", "beats_n", "primary_n", "median_advantage", "median advantage over random"),
    ("Race", "race_summary.csv", "prediction", "model vs historical mean (race_predict request)",
     "Does the model beat the historical mean?", "MAE ratio on held-out rows", "observed outcome",
     "beats_n", "primary_n", "median_mae_ratio", "median MAE ratio (model/baseline)"),
]
SPOT_NOTE = ("not measurable: the datasets carry no anomaly labels, so verified anomaly precision cannot be "
             "computed; the frozen checker's internal rules are the only evidence")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def external_rows(ext_dir=EXT):
    out = []
    for skill, fname, metric, measured, question, outcome, ref, succ, valid, qcol, qlabel in SKILLS:
        for r in read(Path(ext_dir) / fname):
            if metric is not None and r.get("metric") != metric:
                continue
            if r["scope"] != "domain_treatment":
                continue
            if r.get("measurable") == "False" or r.get("measurable") == "false":
                continue
            att = int(r["attempted"])
            ok = int(r[succ])
            va = int(r[valid])
            fidelity = r.get("kind") == "summary"          # Remember -01: describes the data, not a business outcome
            ql, qc = ("summary fidelity (not an external metric)", "median_summary_fidelity") if fidelity else (qlabel,
                                                                                                             qcol)
            out.append({"skill": skill, "request_id": r["request_id"], "domain": r["domain"],
                        "counts_as_external": not fidelity,
                        # Race prediction repeats the Guess predictions in most trials, so it is not counted twice
                        "in_delivery_total": not fidelity and not (skill == "Race" and metric == "prediction"),
                        "treatment": r["treatment"], "customer_question": question, "outcome_measured": outcome,
                        "reference_type": ref, "requests_measured": measured,
                        "attempted": att, "integrity_valid": va, "successful": ok,
                        "delivery_rate": ok / att if att else "", "conditional_success_rate": ok / va if va else "",
                        "survivorship_gap": (ok / va - ok / att) if va and att else "",
                        "quality_label": ql, "quality_value": r.get(qc, "") if qc else "",
                        "source": fname})
    return out


def external_delivery(rows):
    """{(domain, treatment): successes / attempts} over every measured external request (counts, not a score)."""
    acc = {}
    for r in rows:
        if not r["in_delivery_total"]:
            continue
        k = (r["domain"], r["treatment"])
        a, s, v = acc.get(k, (0, 0, 0))
        acc[k] = (a + r["attempted"], s + r["successful"], v + r["integrity_valid"])
    return {k: s / a for k, (a, s, v) in acc.items()}, acc


# ---- T7 -------------------------------------------------------------------------------------

T7_EXTERNAL = {
    "Guess": "evaluated: common-protocol adapter, all three domains (CAR-03, NASA-03, APACHE-03)",
    "Race (prediction)": "evaluated, but the same predictions as Guess in every domain; counted once",
    "Remember": "adapters exist for NASA and Apache only; not evaluated under the common external protocol",
    "Certify": "adapters exist for NASA and Apache (different decision rules from the oracle); not evaluated",
    "Flow": "adapters exist for NASA and Apache (same arrival rule); not evaluated",
    "Blame": "adapters exist for NASA and Apache; not evaluated",
    "Justify": "adapters exist for NASA and Apache; not evaluated",
    "Choose": "adapters exist for NASA and Apache; the honeypot gate is structurally vacuous for T7; not evaluated",
    "Fix": "adapters exist for NASA and Apache; not evaluated",
    "Race (selection)": "adapters exist for NASA and Apache; not evaluated",
    "Spot": "not measurable for any treatment (no anomaly labels)",
}


def _git_show(rev, path):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{path}"], capture_output=True,
                          check=True).stdout.decode("utf-8", "replace")


def _git_files(rev, prefix):
    out = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", rev, prefix], capture_output=True,
                         check=True).stdout.decode()
    return [p for p in out.split() if p.endswith(".py")]


def t7_loc(domain):
    """Recompute standalone T7 LOC per step from the t7-<domain>-stepN tags with harness.loc.measure_step."""
    prev, steps = {}, []
    for i in range(1, 11):
        rev = f"t7-{domain}-step{i}"
        cur = {p: _git_show(rev, p) for p in _git_files(rev, f"human_t7/{domain}/")}
        steps.append(L.measure_step(prev, cur).new_loc)
        prev = cur
    return steps


def adapter_loc(domain):
    files = sorted((REPO / "t7_validation" / "adapters" / domain).glob("*.py"))
    return sum(len(L.code_lines(f.read_text(encoding="utf-8"), ".py")) for f in files), len(files)


def t7_rows(internal, ext_rows):
    t7 = {r["domain"]: r for r in read(EXT / "t7_prediction.csv") if r["kind"] == "predict"}
    race = {r["domain"]: r for r in read(EXT / "t7_prediction.csv") if r["kind"] == "race_predict"}
    cars = read(REPO / "results" / "human_t7_cars_loc.csv")
    out = []
    for d in DOMAINS:
        if d == "cars":
            steps = [int(r["added_impl_loc"]) for r in cars]
            base = int(cars[0]["base_impl_loc"])
            loc_src = "results/human_t7_cars_loc.csv (experiments/measure_loc.py convention)"
            foundation = "human-directed scripts on the LLM-assisted shared shrinking_code/ library"
        else:
            steps = t7_loc(d)
            loc_src = f"recomputed from tags t7-{d}-step1..10 with harness.loc.measure_step"
            base = 0
            foundation = f"standalone human-directed modules (human_t7/{d})"
        a_loc, a_files = adapter_loc(d)
        g = t7[d]
        llm = [r for r in ext_rows if r["skill"] == "Guess" and r["domain"] == d]
        out.append({
            "domain": d, "treatment": "T7", "trials": 1, "independent_trials": 0, "foundation": foundation,
            "library_baseline_loc": base, "loc_steps_1_10": sum(steps), "loc_steps_2_10": sum(steps[1:]),
            "loc_per_step": ";".join(map(str, steps)), "loc_source": loc_src,
            "adapter_loc": a_loc, "adapter_files": a_files,
            "guess_mae_ratio": float(g["mae_ratio"]), "guess_beats_baseline": g["beats_baseline"],
            "race_prediction_identical": abs(float(race[d]["mae_ratio"]) - float(g["mae_ratio"])) < 1e-12,
            "llm_guess_mae_ratio_medians": ";".join(f"{r['treatment']}:{_fmt(r['quality_value'])}" for r in llm),
            "llm_guess_valid_of_attempted": ";".join(f"{r['treatment']}:{r['integrity_valid']}/{r['attempted']}"
                                                     for r in llm),
            "llm_loc_2_10_medians": ";".join(f"{r['treatment']}:{_fmt(r['loc_total_2_10_median'])}"
                                             for r in internal if r["domain"] == d),
            "external_coverage": "Guess only (Race prediction duplicates it)",
        })
    return out


# ---- output ---------------------------------------------------------------------------------


def _fmt(v, spec=".3f"):
    x = _f(v)
    if x is None:
        return "—" if v in ("", None) else str(v)
    return f"{x:g}" if abs(x) >= 10 else format(x, spec)


def to_csv(rows, fields=None):
    fields = fields or list(rows[0])
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else (f"{r[k]:.10g}" if isinstance(r.get(k), float) else r.get(k)))
                    for k in fields})
    return buf.getvalue()


def best_ranks_table(runs):
    out = []
    for d in DOMAINS:
        samples = {t: AN.trial_totals(runs, d, t) for t in TREATMENTS}
        sem = {t: [-x for x in AN.trial_semantic(runs, d, t)] for t in TREATMENTS}
        out.append({"domain": d, "loc_best": AN.best_ranks(samples), "loc_n": {t: len(v) for t, v in samples.items()},
                    "semantic_best": AN.best_ranks(sem), "semantic_n": {t: len(v) for t, v in sem.items()}})
    return out


def cross_domain_exploratory(runs):
    """Exploratory: same() on trial totals pooled over the three domains (unit = one complete trial)."""
    out = []
    for reuse, base in PAIRS:
        xs = [v for d in DOMAINS for v in AN.trial_totals(runs, d, reuse)]
        ys = [v for d in DOMAINS for v in AN.trial_totals(runs, d, base)]
        out.append(dict(same_diagnostics(xs, ys), reuse=reuse, base=base, reuse_median=O.median(xs),
                        base_median=O.median(ys)))
    return out


def guess_race_duplicates():
    g = {(r["domain"], r["treatment"], r["trial"]): r for r in read(EXT / "prediction_mae.csv") if r["primary"] == "true"}
    r_ = {(r["domain"], r["treatment"], r["trial"]): r for r in read(EXT / "race_prediction.csv")
          if r["primary"] == "true"}
    both = sorted(set(g) & set(r_))
    same_ = [k for k in both if abs(float(g[k]["model_mae"]) - float(r_[k]["model_mae"])) < 1e-9]
    return len(both), len(same_)


def statistical_validity(pairs, br, cross, dup):
    tt = [p for p in pairs if p["loc_sample"] == "trial_total"]
    lines = [
        "# Statistical validity of the treatment comparisons (phase 6A)", "",
        "Tools: the professor's `same()` and `bestRanks()` from `stats.py` (vendored unchanged in "
        "`harness/vendor_stats.py`; `the.stats` = cliffs 0.195, conf 1.36, eps 0.35), called through "
        "`harness/analyze.py` (eps = 0.35 × sd of the lower-median sample).", "",
        "## How same() decides", "",
        "1. If the two medians differ by at most eps, the samples are *same*.",
        "2. Otherwise, if |Cliff's delta| > 0.195, they are *different*.",
        "3. Otherwise the KS statistic is compared with 1.36 × sqrt((n+m)/(nm)).", "",
        "**KS cannot reject at these sizes.** The statistic is at most 1, and the critical value is above 1 whenever "
        "both samples have 3 or fewer values (n = m = 3: 1.11; 2 vs 3: 1.24; 1 vs 3: 1.57). So with n ≤ 3 every "
        "'different' comes from step 2 alone. With 3 vs 3 values, |delta| > 0.195 means only two of the nine pairs "
        "need to be out of order. With a single value whose median is the lower one, its sd is 0, so eps = 0 and any "
        "median gap moves on to the Cliff step, where one value below all three others gives |delta| = 1.", "",
        "## Matched pairs, trial totals of new LOC over skills 2-10 (the primary frozen sample)", "",
        "| domain | pair | n reuse/base | reuse median | base median | eps | median gap | abs Cliff | KS / critical | "
        "decided by | verdict | informative? |", "|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|"]
    for p in tt:
        lines.append(f"| {p['domain']} | {p['reuse']} vs {p['base']} | {p['n_reuse']}/{p['n_base']} | "
                     f"{_fmt(p['reuse_loc_median'])} | {_fmt(p['base_loc_median'])} | {p['eps']:.1f} | "
                     f"{p['median_gap']:.1f} | {p['abs_cliffs']:.2f} | {p['ks_stat']:.2f} / {p['ks_critical']:.2f} | "
                     f"{p['route']} | {p['frozen_verdict']} | {'yes' if p['certification_informative'] else 'no'} |")
    cert = [p for p in tt if p["frozen_verdict"].startswith("reuse lower")]
    lines += ["", f"The frozen tables report {len(cert)} of 9 trial-total pairs as 'reuse lower (certified)'. All of "
              "them are decided by the Cliff step with at most 3 values per side, and several rest on one complete "
              "reuse trial (n = 1). They show that every complete reuse trial was smaller than most baseline trials, "
              "not a statistically certified effect.", "",
              "## Pooled per-step LOC (secondary frozen sample)", "",
              "Pooling the 9 steps of each trial gives n ≈ 24-27 per side, where KS can reject. But the steps of one "
              "trial share one development history and differ by skill, so they are not independent development "
              "trials; these verdicts are exploratory at best.", "",
              "| domain | pair | n reuse/base | reuse median | base median | decided by | verdict |",
              "|---|---|---|---:|---:|---|---|"]
    for p in [p for p in pairs if p["loc_sample"] == "pooled"]:
        lines.append(f"| {p['domain']} | {p['reuse']} vs {p['base']} | {p['n_reuse']}/{p['n_base']} | "
                     f"{_fmt(p['reuse_loc_median'])} | {_fmt(p['base_loc_median'])} | {p['route']} | "
                     f"{p['frozen_verdict']} |")
    lines += ["", "## bestRanks()", "",
              "bestRanks sorts treatments by median and keeps those `same()` as the lowest. With 1-3 values per "
              "treatment, 'tied for best' inherits every limitation above, and treatments with fewer complete trials "
              "(survivors) enter with fewer values. Correctness ranks use skills passing per trial (negated so lower "
              "is better).", "",
              "| domain | LOC: tied for best | complete trials | correctness: tied for best |", "|---|---|---|---|"]
    for b in br:
        lines.append(f"| {b['domain']} | {', '.join(b['loc_best'])} | "
                     f"{', '.join(f'{t} {n}' for t, n in b['loc_n'].items())} | {', '.join(b['semantic_best'])} |")
    lines += ["", "## Exploratory cross-domain comparison", "",
              "Unit = one complete trial (domain × trial), pooled over the three domains, so n ≤ 9 per side and KS can "
              "reject. Domains differ in size and difficulty, the pairs are balanced only by design, and incomplete "
              "trials drop out, so this is exploratory and not a pre-registered test.", "",
              "| pair | n reuse/base | reuse median | base median | abs Cliff | KS / critical | decided by | same? |",
              "|---|---|---:|---:|---:|---|---|---|"]
    for c in cross:
        lines.append(f"| {c['reuse']} vs {c['base']} | {c['n_reuse']}/{c['n_base']} | {c['reuse_median']:g} | "
                     f"{c['base_median']:g} | {c['abs_cliffs']:.2f} | {c['ks_stat']:.2f} / {c['ks_critical']:.2f} | "
                     f"{c['route']} | {c['same']} |")
    lines += ["", "## What must stay descriptive", "",
              "- Every external metric: at most 3 attempts per cell, often 0-2 valid; no same() is applied.",
              "- T7: one implementation per domain (n = 1), so same() and bestRanks() are not applied to T7.",
              f"- Guess and Race predictions: identical in {dup[1]} of {dup[0]} trials where both are valid, so they "
              "are not independent evidence.",
              "- Skill requests within a trial are not independent development trials; neither are held-out rows, "
              "arrivals or checkpoints.", ""]
    return "\n".join(lines)


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(runs_path=RUNS, reqs_path=REQS, ext_dir=EXT):
    runs, reqs = read(runs_path), read(reqs_path)
    internal = [internal_row(runs, reqs, d, t) for d in DOMAINS for t in TREATMENTS]
    ext = external_rows(ext_dir)
    deliv, counts = external_delivery(ext)
    for r in internal:
        a, s, v = counts[(r["domain"], r["treatment"])]
        r.update(external_attempts=a, external_valid=v, external_successes=s, external_delivery=s / a,
                 external_conditional=s / v if v else "")
    pairs = [_pair_row(runs, reqs, deliv, d, ru, ba, sc) for sc in ("trial_total", "pooled") for d in DOMAINS
             for ru, ba in PAIRS]
    t7 = t7_rows(internal, ext)
    br = best_ranks_table(runs)
    cross = cross_domain_exploratory(runs)
    dup = guess_race_duplicates()
    return {"internal": internal, "external": ext, "pairs": pairs, "t7": t7, "best_ranks": br, "cross": cross,
            "dup": dup, "runs": runs, "reqs": reqs}


def write(res, out_dir=OUT, report=None):
    out = Path(out_dir)
    if out.resolve() == FROZEN.resolve() or FROZEN.resolve() in out.resolve().parents:
        raise ValueError("refusing to write into the frozen results")
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "treatment_internal.csv": to_csv(res["internal"]),
        "treatment_external.csv": to_csv(res["external"]),
        "reuse_pair_comparison.csv": to_csv(res["pairs"], list(dict.fromkeys(k for p in res["pairs"] for k in p))),
        "t7_comparison.csv": to_csv(res["t7"]),
        "statistical_validity.md": statistical_validity(res["pairs"], res["best_ranks"], res["cross"], res["dup"]),
    }
    if report is not None:
        files["consolidated_report.md"] = report(res)
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
    manifest = {"phase": "6A consolidation (post-hoc synthesis; nothing re-run)",
                "inputs": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [RUNS, REQS] + sorted(EXT.glob("*.csv"))},
                "outputs": {n: _sha(t) for n, t in files.items()},
                "llm_calls": 0, "programs_rerun": 0, "t7_adapters_run": 0}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out, files


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args(sys.argv[1:] if argv is None else argv)
    from harness.consolidate_report import report
    out, files = write(build(), a.out, report)
    print(f"wrote {len(files)} files to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
