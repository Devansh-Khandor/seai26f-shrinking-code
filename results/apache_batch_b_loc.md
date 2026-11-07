# APACHE batch b transfer LOC

Baseline: `05b6e99399e1de10f128a770d547cf3997cd52d4`. Target: **WORKTREE (provisional)**.

| File | Category | Added | Removed | Total |
|---|---|---:|---:|---:|
| shrinking_code/alternatives.py | shared_additions | 44 | 0 | 150 |
| tests/test_apache_batch_b.py | apache_tests | 127 | 0 | 127 |
| shrinking_code/diagnosis.py | shared_additions | 0 | 0 | 125 |
| shrinking_code/apache.py | apache_adapter | 135 | 0 | 291 |
| shrinking_code/demo.py | shared_dispatch | 2 | 2 | 445 |
| tests/test_apache.py | apache_tests | 1 | 1 | 148 |
| tests/test_remember.py | test_integration | 1 | 1 | 102 |
| README.md | documentation | 35 | 5 | 319 |
| experiments/measure_transfer.py | measurement_infrastructure | 12 | 4 | 120 |

Implementation additions: 181; removals: 2.

Batch B baseline is committed apache-batch-a; cumulative (and explicit --batch a) baseline is committed nasa-complete. Default Apache invocation writes B and cumulative only, preserving Batch A artifacts. WORKTREE is provisional; --revision <accepted-commit> reads only committed blobs. Git diffs count nonblank/noncomment Python lines including docstrings; documentation counts nonblank lines. Added/removed counts are separate, not net growth. Shared additions, Apache adapter, dispatch edits, tests, documentation and infrastructure are separated. Hashes identify measured bytes. Generated reports and regression JSON fixtures are excluded. Renames/moves/edits may be counted as additions; these counts do not establish new functionality or independent experimental repetitions. Existing Cars/NASA LOC reports are not rewritten by --domain apache.
