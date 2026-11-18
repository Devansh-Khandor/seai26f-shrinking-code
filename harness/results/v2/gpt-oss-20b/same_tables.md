# `same` tables — model `gpt-oss-20b`

`same` = the professor's same() from stats.py (median-difference shortcut, Cliff's delta, KS), called unchanged with eps = 0.35 sd.

## apache

### primary: `trial_total`

| pair | reuse values | base values | reuse median | base median | n | excluded trials | verdict |
|---|---|---|---|---|---|---|---|
| T2 vs T1 | 1284, 1759, 1202 | 1626, 1256, 1343 | 1284 | 1343 | 3/3 | 0/0 | no certified difference |
| T4 vs T3 | 1398, 1565, 1158 | 1424, 1258, 1651 | 1398 | 1424 | 3/3 | 0/0 | no certified difference |
| T6 vs T5 | 1227 | 1687, 1640, 1661 | 1227 | 1661 | 1/3 | 2/0 | reuse lower (certified) |

### secondary: `pooled`

| pair | reuse values | base values | reuse median | base median | n | excluded trials | verdict |
|---|---|---|---|---|---|---|---|
| T2 vs T1 | 175, 126, 162, 131, 148, 74, 160, 145, 163, 298, 132, 108, 181, 188, 207, 177, 227, 241, 165, 91, 85, 124, 176, 118, 165, 129, 149 | 171, 143, 127, 150, 157, 207, 186, 216, 269, 147, 118, 124, 151, 129, 87, 176, 137, 187, 189, 124, 133, 163, 135, 75, 183, 133, 208 | 160 | 150 | 27/27 | 0/0 | no certified difference |
| T4 vs T3 | 170, 127, 116, 131, 141, 130, 157, 169, 257, 147, 162, 130, 162, 195, 163, 211, 156, 239, 125, 90, 117, 158, 135, 100, 166, 115, 152 | 138, 139, 105, 161, 172, 119, 180, 149, 261, 129, 136, 133, 133, 143, 112, 147, 143, 182, 153, 174, 150, 174, 209, 155, 192, 197, 247 | 152 | 150 | 27/27 | 0/0 | no certified difference |
| T6 vs T5 | 136, 84, 140, 114, 103, 227, 124, 159, 151, 143, 121, 181, 130, 113, 114, 223, 119, 100, 117, 136, 144, 129, 161, 119, 202 | 154, 165, 198, 180, 207, 141, 233, 193, 216, 183, 153, 133, 180, 201, 178, 179, 188, 245, 225, 149, 163, 165, 226, 171, 173, 174, 215 | 130 | 180 | 25/27 | 2/0 | reuse lower (certified) |

## cars

### primary: `trial_total`

| pair | reuse values | base values | reuse median | base median | n | excluded trials | verdict |
|---|---|---|---|---|---|---|---|
| T2 vs T1 | 1649, 1448 | 1700, 1762, 1536 | 1548.5 | 1700 | 2/3 | 1/0 | reuse lower (certified) |
| T4 vs T3 | 1544, 1413, 1261 | 1495, 1657 | 1413 | 1576 | 3/2 | 0/1 | reuse lower (certified) |
| T6 vs T5 | 1283, 1257 | 1755, 1591, 1988 | 1270 | 1755 | 2/3 | 1/0 | reuse lower (certified) |

### secondary: `pooled`

| pair | reuse values | base values | reuse median | base median | n | excluded trials | verdict |
|---|---|---|---|---|---|---|---|
| T2 vs T1 | 203, 152, 109, 204, 99, 77, 127, 138, 203, 191, 146, 212, 201, 92, 176, 176, 252, 204, 146, 127, 186, 141, 104, 184, 136, 220 | 186, 208, 186, 178, 159, 119, 256, 163, 245, 227, 178, 157, 245, 198, 124, 223, 174, 236, 171, 175, 116, 184, 154, 89, 229, 161, 257 | 164 | 178 | 26/27 | 1/0 | no certified difference |
| T4 vs T3 | 168, 175, 148, 152, 182, 139, 205, 155, 220, 136, 164, 144, 144, 189, 68, 203, 158, 207, 172, 156, 103, 142, 151, 65, 199, 141, 132 | 115, 174, 145, 167, 157, 109, 240, 180, 208, 158, 186, 134, 183, 229, 107, 219, 181, 260, 106, 174, 136, 157, 165, 89, 167, 160 | 155 | 166 | 27/26 | 0/1 | no certified difference |
| T6 vs T5 | 141, 111, 135, 166, 119, 106, 165, 156, 184, 198, 131, 95, 174, 115, 217, 171, 166, 125, 109, 135, 130, 62, 203, 122, 205 | 239, 180, 181, 178, 184, 136, 227, 196, 234, 179, 137, 128, 205, 196, 94, 238, 165, 249, 184, 207, 170, 237, 222, 240, 264, 168, 296 | 135 | 196 | 25/27 | 1/0 | reuse lower (certified) |

## nasa

### primary: `trial_total`

| pair | reuse values | base values | reuse median | base median | n | excluded trials | verdict |
|---|---|---|---|---|---|---|---|
| T2 vs T1 | 1309, 1665 | 1917, 1602, 2018 | 1487 | 1917 | 2/3 | 1/0 | reuse lower (certified) |
| T4 vs T3 | 1774 | 1773, 1755, 1667 | 1774 | 1755 | 1/3 | 2/0 | no certified difference |
| T6 vs T5 | 1341 | 1507, 1665, 1806 | 1341 | 1665 | 1/3 | 2/0 | reuse lower (certified) |

### secondary: `pooled`

| pair | reuse values | base values | reuse median | base median | n | excluded trials | verdict |
|---|---|---|---|---|---|---|---|
| T2 vs T1 | 155, 178, 197, 190, 176, 184, 226, 202, 121, 121, 131, 152, 186, 103, 241, 119, 135, 142, 115, 172, 163, 304, 142, 249, 189, 189 | 206, 143, 255, 193, 167, 187, 314, 193, 259, 182, 138, 191, 151, 172, 129, 287, 165, 187, 128, 160, 230, 199, 259, 226, 304, 293, 219 | 174 | 193 | 26/27 | 1/0 | reuse lower (certified) |
| T4 vs T3 | 151, 125, 215, 175, 230, 174, 153, 91, 231, 140, 155, 186, 137, 265, 200, 192, 135, 240, 134, 279, 120, 267, 239, 168 | 197, 126, 225, 188, 242, 160, 249, 186, 200, 214, 120, 208, 214, 203, 143, 270, 181, 202, 118, 153, 211, 184, 216, 172, 203, 197, 213 | 174.5 | 200 | 24/27 | 2/0 | no certified difference |
| T6 vs T5 | 99, 82, 122, 138, 133, 162, 125, 171, 115, 109, 156, 192, 182, 142, 194, 100, 151, 99, 118, 131, 196, 104, 186, 137, 125 | 123, 171, 159, 166, 170, 223, 169, 141, 185, 171, 135, 191, 199, 215, 142, 219, 211, 182, 200, 193, 251, 132, 212, 217, 209, 176, 216 | 133 | 185 | 25/27 | 2/0 | reuse lower (certified) |

## bestRanks

Treatments tied for best by the professor's bestRanks(). LOC: lowest total new LOC over skills 2-10 (complete trials only). Correctness: most skills passing the semantic check per trial (trials with all 10 steps recorded).

| domain | LOC: tied for best | n (complete/attempted trials) | correctness: tied for best |
|---|---|---|---|
| apache | T6 | T1 3/3, T2 3/3, T3 3/3, T4 3/3, T5 3/3, T6 1/3 | T5 |
| cars | T6 | T1 3/3, T2 2/3, T3 2/3, T4 3/3, T5 3/3, T6 2/3 | T3 |
| nasa | T6, T2 | T1 3/3, T2 2/3, T3 3/3, T4 1/3, T5 3/3, T6 1/3 | T1, T5 |

N=3 trials: `same` on small samples has little power; a 'no certified difference' is not evidence of equivalence. Excluded trials = trials missing a numeric new_loc for some skill 2-10 (overflow, parse failure, cut-off reply, or not yet run); see pass_rates.md.
