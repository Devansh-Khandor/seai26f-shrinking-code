"""Append-only results CSV: one row per (model, domain, treatment, trial, skill)."""

import csv
import os
from pathlib import Path

FIELDS = ("model", "backend", "domain", "treatment", "trial", "skill_index", "skill", "status",
          "runs", "contract", "new_loc", "deleted_loc", "test_loc", "total_loc", "pct_new",
          "prompt_tokens", "response_tokens", "prompt_tokens_est", "finish_reason", "output_capped",
          "seconds", "timestamp", "prompt_template_hash", "temperature", "think", "quantization", "model_digest")

# backend_error is deliberately absent: resume retries it.
DONE_STATUSES = {"ok", "parse_failed", "context_overflow", "output_truncated", "timeout"}
KEY = ("model", "domain", "treatment", "trial", "skill_index")


def key_of(row):
    return tuple(str(row[k]) for k in KEY)


def append_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: "" if row.get(k) is None else row.get(k) for k in FIELDS})
        f.flush()
        os.fsync(f.fileno())


def load_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest_rows(rows):
    latest = {}
    for row in rows:
        latest[key_of(row)] = row
    return list(latest.values())


def completed_keys(path):
    return {key_of(r) for r in latest_rows(load_rows(path)) if r["status"] in DONE_STATUSES}
