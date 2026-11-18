# Apache T7 — Experimental Results

## Step 1: Remember

Requirements: APACHE-01 and APACHE-02.

### Performance Summary

| Metric | Value |
|---|---:|
| Configurations | 192 |
| Minimum performance | 840 |
| Median performance | 1215 |
| Mean performance | 1427.66 |
| Maximum performance | 2640 |

Lower `Performance-` values are preferable.

### Best Historical Configurations

| CSV row | Performance |
|---|---:|
| 125 | 840 |
| 151 | 840 |
| 73 | 870 |
| 91 | 870 |
| 103 | 870 |

### Favorable Group

Definition: lowest 25% of performance values, including ties.

- Cutoff: 1050
- Favorable configurations: 52
- Remaining configurations: 140

Largest observed setting-frequency differences:

| Setting | Favorable | Remaining |
|---|---:|---:|
| keepAlive=1 | 7.7% | 65.7% |
| hostnameLookups=1 | 86.5% | 36.4% |
| inMemory=1 | 21.2% | 37.9% |

These associations do not establish causation.

### Verification

- 7/7 behavioral unit tests passed.
- The Step 1 demonstration executed successfully.
- Full independent semantic certification is not claimed.

## Incremental LOC

Harness-compatible LOC measurements will be calculated
from the cumulative Git checkpoints.
