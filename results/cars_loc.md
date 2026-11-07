# Cars tagged LOC measurements

Regenerate with `python3 experiments/measure_loc.py`. Source: resolved Git tags, not working-tree files.

| Step | Skill | Tag | Commit | Added impl | Removed impl | Total impl | New % | Added tests | Added docs | Source files changed |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Remember | step1 | f5152b81ed58ed5b86a7223466ae6bea8cf3fea6 | 167 | 0 | 167 | 100.000000 | 169 | 39 | 5 |
| 2 | Guess | step2 | 41c12df5ea88992e1bc3811a17499fa7b0ec25b2 | 173 | 4 | 336 | 51.488095 | 175 | 23 | 4 |
| 3 | Certify | step3 | 7fde9ebdc5aafefacd8378265b19256f40ad0e74 | 114 | 4 | 446 | 25.560538 | 105 | 35 | 2 |
| 4 | Flow | step4 | efa31ddc6f9bd9a7755269be47f36114638b6827 | 125 | 4 | 567 | 22.045855 | 158 | 16 | 2 |
| 5 | Blame | step5 | a25e125b6a3573bf673a5a2a0fea42a2309d10ff | 148 | 3 | 712 | 20.786517 | 151 | 18 | 2 |
| 6 | Justify | step6 | fa280e9793a1aedf20cd0ab5faf741273b57382c | 168 | 3 | 877 | 19.156214 | 133 | 17 | 2 |
| 7 | Choose | step7 | 652236654caa61f83f8743d019473164b8f1e3ee | 154 | 3 | 1028 | 14.980545 | 214 | 23 | 2 |
| 8 | Fix | step8 | 52b62c6339bde925a27a0ecface8a949241b64fc | 161 | 3 | 1186 | 13.575042 | 191 | 22 | 2 |
| 9 | Spot | step9 | ad17a8f115c2cf7983c0bdc1541c0597da00e35c | 120 | 3 | 1303 | 9.209517 | 164 | 33 | 2 |
| 10 | Race | step10 | 2d734874a59cacbcc2e91d4ca486546f8ec275aa | 190 | 3 | 1490 | 12.751678 | 163 | 25 | 2 |

## Methodology

Step 1 uses its sole parent; later steps use the preceding tag. Tags are peeled to commits; missing tags, repeated commits, non-ancestor steps, and merge-containing intervals fail. Multiple linear commits are supported by endpoint comparison. CSV records both tag-object and commit IDs plus every comparison base.

Primary LOC is physical nonblank, noncomment Python lines in shrinking_code/. Python tokenization excludes comment-only lines, retains inline-code lines, and counts docstrings/multiline strings as string statements (including nonblank continuation lines). It does not execute source. Blank string lines are excluded. Tests use the same rule. Documentation (root README* and docs/) counts nonblank text lines. experiments/ is separate infrastructure: Python uses token counting, other files use nonblank text. Raw datasets and other unclassified files (including JSON test fixtures/results) are excluded from LOC.

Added LOC counts eligible added lines, never net growth. Deleted LOC is separate. Percentage = 100 × added implementation LOC / total implementation LOC at that tag; an empty denominator is undefined (empty CSV field). Git Myers diffs disable indent heuristics, external diffs, and textconv. Rename detection uses 50% similarity; unchanged renames count zero added/deleted LOC. Crossing category boundaries transfers the file LOC into the new category.

CSV separates additions in new versus existing implementation files, renamed source-file counts, and identical added/deleted line matches (possible moves or repeated text). These matches are not subtracted from the primary metric: identical lines do not prove semantic equivalence or a move. Git diffs cannot perfectly distinguish edits, moves, copies, refactoring, and new logic. Replacements count as deletion plus addition. Cross-file/block moves can still count as additions; endpoint diffs omit churn that was added and removed between tags. Git version/rename heuristics can affect ambiguous matches. Source-files-changed counts implementation change records, including renames and comment-only edits.

The reclassified_impl_loc column exposes any count change in unchanged physical lines caused by lexical context changes, e.g. multiline strings. It reconciles snapshot totals with added/deleted counts; it is not silently attributed to new code. No semantic functionality or effort metric is inferred.

## Growth and descriptive trend

Baseline implementation LOC: 0; final: 1490; net growth: 1490.
Sum of added implementation LOC: 1520; removed: 30; lexical reclassification adjustment: 0.
New implementation LOC sequence: [167, 173, 114, 125, 148, 168, 154, 161, 120, 190].
Descriptive least-squares slope: 1.078788 LOC per step (upward). Increases over the previous milestone occur at steps: 2, 4, 5, 6, 8, 10.
New-code percentage sequence: ['100.000000', '51.488095', '25.560538', '22.045855', '20.786517', '19.156214', '14.980545', '13.575042', '9.209517', '12.751678'].

A declining percentage can result from a growing cumulative denominator even when absolute additions rise. Absolute and percentage trends must not be conflated. This one sequential LLM-assisted development history is descriptive, not an independent experimental repetition or statistical support for H1/H2. No same test is applied to dependent milestones. No seven-treatment or cross-domain conclusion follows.

## Category totals and traceability

| Tag | Base commit | Total tests | Total docs | Added infrastructure | Removed infrastructure | Total infrastructure |
|---|---|---:|---:|---:|---:|---:|
| step1 | e65056e59bb071dfc273b9f42241ffa0c59b10f7 | 169 | 445 | 0 | 0 | 0 |
| step2 | f5152b81ed58ed5b86a7223466ae6bea8cf3fea6 | 342 | 462 | 0 | 0 | 0 |
| step3 | 41c12df5ea88992e1bc3811a17499fa7b0ec25b2 | 445 | 492 | 0 | 0 | 0 |
| step4 | 7fde9ebdc5aafefacd8378265b19256f40ad0e74 | 601 | 503 | 0 | 0 | 0 |
| step5 | efa31ddc6f9bd9a7755269be47f36114638b6827 | 750 | 517 | 0 | 0 | 0 |
| step6 | a25e125b6a3573bf673a5a2a0fea42a2309d10ff | 881 | 530 | 0 | 0 | 0 |
| step7 | fa280e9793a1aedf20cd0ab5faf741273b57382c | 1093 | 549 | 0 | 0 | 0 |
| step8 | 652236654caa61f83f8743d019473164b8f1e3ee | 1282 | 567 | 0 | 0 | 0 |
| step9 | 52b62c6339bde925a27a0ecface8a949241b64fc | 1444 | 596 | 0 | 0 | 0 |
| step10 | ad17a8f115c2cf7983c0bdc1541c0597da00e35c | 1605 | 617 | 0 | 0 | 0 |
