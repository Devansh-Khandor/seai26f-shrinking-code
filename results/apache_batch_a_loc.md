# APACHE batch a transfer LOC

Baseline: `41e9c47a3919984d5cb0715729054e00752d66ad`. Target: **WORKTREE (provisional)**.

| File | Category | Added | Removed | Total |
|---|---|---:|---:|---:|
| shrinking_code/diagnosis.py | shared_additions | 16 | 0 | 125 |
| shrinking_code/apache.py | apache_adapter | 156 | 0 | 156 |
| shrinking_code/demo.py | shared_dispatch | 6 | 1 | 445 |
| tests/test_apache.py | apache_tests | 148 | 0 | 148 |
| tests/test_remember.py | test_integration | 1 | 1 | 102 |
| README.md | documentation | 28 | 2 | 289 |
| experiments/measure_transfer.py | measurement_infrastructure | 23 | 5 | 112 |

Implementation additions: 178; removals: 1.

Baseline is the committed nasa-complete tag. WORKTREE is provisional; --revision <accepted-commit> reads only committed blobs. Git diffs count nonblank/noncomment Python lines including docstrings; documentation counts nonblank lines. Added/removed counts are separate, not net growth. Shared additions, Apache adapter, dispatch edits, tests, documentation and infrastructure are separated. Hashes identify measured bytes. Generated reports and regression JSON fixtures are excluded. Renames/moves/edits may be counted as additions; these counts do not establish new functionality or independent experimental repetitions. Existing Cars/NASA LOC reports are not rewritten by --domain apache.
