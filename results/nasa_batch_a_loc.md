# NASA batch a transfer LOC

Baseline: `2d734874a59cacbcc2e91d4ca486546f8ec275aa`. Target: **fd68b8986204150dc7d86bb0a6167653e2e24d7b**.

| File | Category | Added | Removed | Total |
|---|---|---:|---:|---:|
| shrinking_code/explanation.py | shared_explanation_additions | 0 | 0 | 105 |
| shrinking_code/proposals.py | new_shared | 50 | 0 | 50 |
| shrinking_code/nasa.py | nasa_adapter | 124 | 0 | 124 |
| shrinking_code/demo.py | modified_shared | 6 | 1 | 440 |
| tests/test_nasa.py | nasa_tests | 183 | 0 | 183 |
| tests/test_remember.py | test_integration | 1 | 1 | 102 |
| README.md | documentation | 26 | 2 | 235 |
| experiments/measure_transfer.py | measurement_infrastructure | 81 | 0 | 81 |

Implementation additions: 180; removals: 1.

Batch A is finalized from the accepted nasa-batch-a Git checkpoint. Batch B is relative to that checkpoint; cumulative transfer is relative to step10. WORKTREE targets are provisional, never labeled finalized. Use --revision <accepted-commit> to finalize Batch B and cumulative reports after acceptance.

Counts reuse the Cars nonblank/noncomment policy including docstrings. Python uses tokenization; documentation counts nonblank lines. Added/removed are separate. Shared additions in existing files and new generic capabilities are distinguished from NASA adapter and dispatch. Existing-file additions include edits, not necessarily new logic; block moves/renames can count as additions. Endpoint cumulative counts need not equal the sum of batch additions because edits can replace earlier additions.

The explicit scope includes all NASA implementation, tests, README, and transfer measurement source; unchanged shared modules add zero LOC, not zero reuse. Regression JSON fixtures and generated reports are excluded artifacts, not Python LOC. Content hashes identify measured bytes. New untracked scope files are included provisionally. Cars LOC outputs remain unchanged. This is development-history measurement, not an independent treatment experiment.
