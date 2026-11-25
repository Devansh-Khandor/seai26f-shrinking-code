# Human T7 Cars LOC measurements

Measured from the ten T7 Git commits using the methodology in `experiments/measure_loc.py`.

Implementation LOC includes both `shrinking_code/*.py` and `human_t7/cars/*.py`.

| Step | Skill | Commit | Added impl | Removed impl | Total impl | New % | Added tests | Added docs |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Remember | 91eb95bbf241372098bc0e63399807d90537d663 | 35 | 0 | 2290 | 1.528384 | 0 | 0 |
| 2 | Guess | 36ba9bf00031c19a52f29fed1e2f14f49599cce6 | 54 | 0 | 2344 | 2.303754 | 0 | 0 |
| 3 | Certify | b3b558f5a9c90306e96557fc71bb785f61139a71 | 87 | 0 | 2431 | 3.578774 | 0 | 0 |
| 4 | Flow | 38765b168a1f8317db095f31682ab32c837c7d97 | 49 | 0 | 2480 | 1.975806 | 0 | 0 |
| 5 | Blame | c722fe6321e5af5b58c6c5d750afbeb992bcad48 | 55 | 0 | 2535 | 2.169625 | 0 | 0 |
| 6 | Justify | ff8e07be01be9a9af02e813b77831412d37f2781 | 62 | 0 | 2597 | 2.387370 | 0 | 0 |
| 7 | Choose | ba3c893c0c4041a1d56e591efa7336e9d4c16a81 | 73 | 0 | 2670 | 2.734082 | 0 | 0 |
| 8 | Fix | 0bd2c33e8a9e745721f498bc127b585494bf3a6a | 77 | 0 | 2747 | 2.803058 | 0 | 0 |
| 9 | Spot | 7d9c40b4602cacb7ea3d3211017cfec07d10aa38 | 51 | 0 | 2798 | 1.822731 | 0 | 0 |
| 10 | Race | c996b0e970d5cb672490d059273019b1c58cb160 | 77 | 0 | 2875 | 2.678261 | 0 | 0 |

## Summary

- Baseline implementation LOC: 2255
- Final implementation LOC: 2875
- Total added implementation LOC: 620
- Total removed implementation LOC: 0

## Methodology

This script reuses the official measurement implementation for Python tokenization, snapshots, Git diffs, and LOC calculations. The only category extension is that `human_t7/cars/*.py` is counted as implementation alongside `shrinking_code/*.py`.

The first T7 commit is compared with its parent. Each later commit is compared with the preceding T7 commit. The shared `shrinking_code/` library is therefore included in the baseline and cumulative implementation totals.

Added LOC counts eligible added lines, not net growth. Removed LOC is reported separately. New percentage is added implementation LOC divided by total implementation LOC at that milestone.

These measurements describe one sequential development history; they are not independent experimental repetitions or statistical evidence for H1 or H2.
