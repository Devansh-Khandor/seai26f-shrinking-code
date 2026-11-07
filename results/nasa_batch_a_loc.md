# NASA Batch A transfer LOC

Baseline: `2d734874a59cacbcc2e91d4ca486546f8ec275aa` (step10). Target: **WORKTREE (provisional)**.

| File | Category | Added | Removed | Total |
|---|---|---:|---:|---:|
| shrinking_code/proposals.py | new_shared | 50 | 0 | 50 |
| shrinking_code/nasa.py | nasa_adapter | 124 | 0 | 124 |
| shrinking_code/demo.py | modified_shared | 6 | 1 | 440 |
| tests/test_nasa.py | nasa_tests | 183 | 0 | 183 |
| tests/test_remember.py | test_integration | 1 | 1 | 102 |
| README.md | documentation | 26 | 2 | 235 |
| experiments/measure_transfer.py | measurement_infrastructure | 81 | 0 | 81 |

Counts reuse the Cars nonblank/noncomment policy (including docstrings). Python uses tokenization; documentation counts nonblank text. Added and removed are separate, not net LOC. Modified shared LOC means additions/removals in existing shared dispatch; it is not counted as a new shared algorithm.

Scope is explicitly the Batch A files listed above. Existing unchanged shared modules contribute zero additions, not zero reused code. NASA adapter lines include policies and presentation and count as implementation; no duplicated algorithm is discounted as reuse. Generated results and pre-existing Cars measurement infrastructure are excluded. File hashes in the JSON identify measured bytes.

Default results are provisional working-tree Git diffs against step10, including new untracked files. Finalize after acceptance with `python3 experiments/measure_transfer.py --revision <accepted-commit>`, which reads Git blobs only. Renames within this explicitly named batch scope are treated as deletion/addition; edits/moves are not mechanically distinguishable from new logic. This is one development sequence, not an independent treatment repetition or evidence for H1/H2. Cars LOC artifacts are not changed.
