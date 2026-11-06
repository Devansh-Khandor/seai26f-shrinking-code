# Dataset inspection: The Shrinking-Code Experiment

Inspected the three local `data/raw/` CSVs. Counts exclude headers. **Observed** statements and ranges come from these files; **interpretations** are explicitly labeled. No data was changed and no models or project skills were implemented.

**Reading conventions:** uppercase initial letters indicate numerical columns; lowercase initials indicate symbolic/categorical columns. A trailing `+` means maximize, `-` means minimize, and `X` means ignore (not an independent variable). Remaining columns are independent variables. These conventions are documented in [Tim Menzies’s data-format description](https://gist.github.com/timm/9d85996f6750f3a828609c4140c0267c). None of these headers uses `!`. Numeric storage is distinguished below from categorical meaning. Ignored columns are still inspected.

**Checks:** CSV row widths, duplicate headers/records, missing markers (`?`, empty/whitespace-only cells, and case-insensitive `NA`, `N/A`, `NaN`, `null`, `none`), surrounding whitespace, numerical ranges, constants, and repeated X combinations. `n` is a categorical value, not a missing marker. All files have consistent row widths, unique headers, and no identical full records. Numerical ranges exclude missing values; all observed numerical values are finite. Absence of a missing marker does not prove completeness of the original measurements.

## 1. `auto93.csv` — Automobile selection

**Exact column names, original order:**

```text
Clndrs,Volume,HpX,Model,origin,Lbs-,Acc+,Mpg+
```

**Observed shape:** 398 records × 8 columns.

**Types and roles:**
- X (4): numerical `Clndrs`, `Volume`, `Model`; categorical `origin` (codes `1`, `2`, `3`; meanings unspecified).
- Y (3), all numerical: `Lbs-`, `Acc+`, `Mpg+`.
- Ignored (1): numerical `HpX`.
- All eight columns contain numeric-form values apart from missing markers; `origin` remains categorical under the naming convention.

**Missing values:** six `?` cells, all in `HpX` (6/398 ≈ 1.51%); data-record numbers 178, 207, 210, 233, 361, 391 (1-based, excluding header). Every other column has zero detected missing values, including all goals.

**Goal meanings — interpretations, not definitions supplied by the CSV:**
- `Lbs-`: likely vehicle weight in pounds; **minimize** by suffix.
- `Acc+`: likely an acceleration-related measure; **maximize** by suffix. Exact definition and units are uncertain. If it measures elapsed acceleration time, maximizing would favor slower acceleration; resolve this before assigning a physical interpretation to “better.”
- `Mpg+`: likely fuel economy in miles per gallon; **maximize** by suffix. Measurement protocol is unspecified.

**Potential quality/requirements issues:** `Mpg+` contains only `{10, 20, 30, 40, 50}`, suggesting rounding or binning, but the transformation is unknown. Do not assume continuous original fuel-economy measurements. `Model` values span 70–82; interpreting them as model-year codes is plausible but unconfirmed. There are only 306 unique X combinations (69 repeated groups; 92 excess records); shared X values do not make complete records duplicates. `HpX` is excluded under the convention, so its missing values do not require imputation for the designated X/Y data. No columns are constant.

**First five records** (header repeated; original cell text preserved):

```csv
Clndrs,Volume,HpX,Model,origin,Lbs-,Acc+,Mpg+
8,304,193,70,1,4732,18.5,10
8,360,215,70,1,4615,14,10
8,307,200,70,1,4376,15,10
8,318,210,70,1,4382,13.5,10
8,429,208,72,1,4633,11,10
```

**Observed numerical ranges** (includes numeric-coded categories and ignored fields):

| Column | Minimum | Maximum |
|---|---:|---:|
| `Clndrs` | 3 | 8 |
| `Volume` | 68 | 455 |
| `HpX` | 46 | 230 |
| `Model` | 70 | 82 |
| `origin` | 1 | 3 |
| `Lbs-` | 1613 | 5140 |
| `Acc+` | 8 | 24.8 |
| `Mpg+` | 10 | 50 |

## 2. `nasa93dem.csv` — Software project estimation

**Exact column names, original order:**

```text
idX,centerX,YearX,prec,flex,resl,team,pmat,rely,data,cplx,ruse,docu,time,stor,pvol,acap,pcap,pcon,apex,plex,ltex,tool,site,sced,Kloc+,Effort-,Defects-,MonthsX
```

**Observed shape:** 93 records × 29 columns.

**Types and roles:**
- X (22), all categorical: `prec`, `flex`, `resl`, `team`, `pmat`, `rely`, `data`, `cplx`, `ruse`, `docu`, `time`, `stor`, `pvol`, `acap`, `pcap`, `pcon`, `apex`, `plex`, `ltex`, `tool`, `site`, `sced`.
- Y (3), all numerical: `Kloc+`, `Effort-`, `Defects-`.
- Ignored (4): `idX`, `centerX`, `YearX`, `MonthsX`. All contain numerical values; the naming convention would classify lowercase `idX` and `centerX` as symbolic before exclusion. Semantically, identifiers and center codes should not be assumed quantitative.
- Storage: 7 numeric-form columns and 22 text-valued columns. After trimming whitespace, the observed text vocabulary is `vl`, `l`, `n`, `h`, `vh`, `xh`. These plausibly encode ordered ratings, but their definitions and numerical spacing are not supplied; retain categorical treatment unless documented.

**Missing values:** zero detected in every column, including zero `?` markers.

**Goal meanings — interpretations, not definitions supplied by the CSV:**
- `Kloc+`: likely software size in thousands of lines of code; **maximize** by suffix. It is a goal here, not an input, despite the estimation context. The project title does not override the file’s direction.
- `Effort-`: likely development effort; **minimize**. Units are uncertain (person-months would need confirmation).
- `Defects-`: likely a defect count; **minimize**. Whether measured or estimated, and the counting period, are uncertain.
- `MonthsX` is ignored, not a time-minimization goal; its name suggests duration in months, but the CSV gives no definition.

**Potential quality/requirements issues:** `idX=29` occurs in records 29 and 30; ID 30 is absent from the otherwise 1–93 sequence. Do not assume ID uniqueness or repair it without provenance. The first three `prec` cells contain eight leading spaces before `h`; trimming would collapse the apparent two raw categories into one. After trimming, eight X columns are constant: `prec=h`, `flex=h`, `resl=h`, `team=vh`, `ruse=n`, `docu=n`, `pcon=n`, `site=n`. There are only 55 unique trimmed X combinations (15 repeated groups; 38 excess records). Goal scales differ substantially; large values are not proven errors. Abbreviated feature meanings, rating scales, units, and measurement provenance need a data dictionary.

**First five records** (header repeated; original cell text preserved):

```csv
idX,centerX,YearX,prec,flex,resl,team,pmat,rely,data,cplx,ruse,docu,time,stor,pvol,acap,pcap,pcon,apex,plex,ltex,tool,site,sced,Kloc+,Effort-,Defects-,MonthsX
1,2,1979,        h,h,h,vh,h,h,l,h,n,n,n,n,l,n,n,n,n,n,h,n,n,l,25.9,117.6,808,15.3
2,2,1979,        h,h,h,vh,h,h,l,h,n,n,n,n,l,n,n,n,n,n,h,n,n,l,24.6,117.6,767,15
3,2,1979,        h,h,h,vh,h,h,l,h,n,n,n,n,l,n,n,n,n,n,h,n,n,l,7.7,31.2,240,10.1
4,2,1979,h,h,h,vh,h,h,l,h,n,n,n,n,l,n,n,n,n,n,h,n,n,l,8.2,36,256,10.4
5,2,1979,h,h,h,vh,h,h,l,h,n,n,n,n,l,n,n,n,n,n,h,n,n,l,9.7,25.2,302,11
```

**Observed numerical ranges** (includes numeric-coded categories and ignored fields):

| Column | Minimum | Maximum |
|---|---:|---:|
| `idX` | 1 | 93 |
| `centerX` | 1 | 6 |
| `YearX` | 1971 | 1987 |
| `Kloc+` | 0.9 | 980 |
| `Effort-` | 8.4 | 8211 |
| `Defects-` | 28 | 50961 |
| `MonthsX` | 4.9 | 96.4 |

## 3. `Apache_AllMeasurements.csv` — Apache server configuration optimization

**Exact column names, original order:**

```text
base,hostnameLookups,keepAlive,enableSendfile,followSymLinks,accessLog,extendedStatus,inMemory,handle,Performance-
```

**Observed shape:** 192 records × 10 columns.

**Types and roles:**
- X (9), all categorical under the naming convention: `base`, `hostnameLookups`, `keepAlive`, `enableSendfile`, `followSymLinks`, `accessLog`, `extendedStatus`, `inMemory`, `handle`.
- Y (1), numerical: `Performance-`.
- Ignored: none.
- All 10 columns are stored as numeric-form values. Eight X columns are binary `{0, 1}`; `base` is always `1`. Interpretation: these appear to encode configuration states, but exact flag semantics and valid combinations are not documented in the CSV.

**Missing values:** zero detected in every column, including zero `?` markers.

**Goal meaning — interpretation:** `Performance-` is a performance measurement to **minimize**, as explicitly encoded by `-`. The metric, units, workload, and whether values aggregate repeated runs are **uncertain**. Do not label it throughput, latency, or runtime without supporting metadata.

**Potential quality/requirements issues:** `base` is constant and cannot distinguish configurations. All 192 X combinations are unique; no repeated configuration measurements are available to assess run-to-run variation. `inMemory` and `handle` each have 128 zeros and 64 ones; the other six varying flags each have 96 zeros and 96 ones. The 192 configurations do not cover all 256 mathematical combinations of eight binary flags; this may reflect configuration constraints rather than missing data. Constraint definitions and performance measurement metadata are needed before generalizing to unseen configurations.

**First five records** (header repeated; original cell text preserved):

```csv
base,hostnameLookups,keepAlive,enableSendfile,followSymLinks,accessLog,extendedStatus,inMemory,handle,Performance-
1,0,0,0,0,0,0,0,0,1200
1,0,1,0,1,0,0,0,0,2100
1,0,1,1,0,0,1,1,0,2310
1,0,0,1,0,1,0,1,0,1260
1,0,0,1,0,1,1,1,0,1140
```

**Observed numerical ranges** (includes numeric-coded categories and ignored fields):

| Column | Minimum | Maximum |
|---|---:|---:|
| `base` | 1 | 1 |
| `hostnameLookups` | 0 | 1 |
| `keepAlive` | 0 | 1 |
| `enableSendfile` | 0 | 1 |
| `followSymLinks` | 0 | 1 |
| `accessLog` | 0 | 1 |
| `extendedStatus` | 0 | 1 |
| `inMemory` | 0 | 1 |
| `handle` | 0 | 1 |
| `Performance-` | 840 | 2640 |

**Dataset comparison**

Counts below use the header conventions for active X/Y roles, not numeric storage.

| Property | Automobile | NASA | Apache |
|---|---|---|---|
| Rows × columns | 398 × 8 | 93 × 29 | 192 × 10 |
| Independent X | 4: 3 numerical, 1 categorical | 22 categorical | 9 categorical |
| Goals Y | 3 numerical | 3 numerical | 1 numerical |
| Maximize | `Acc+`, `Mpg+` | `Kloc+` | None |
| Minimize | `Lbs-` | `Effort-`, `Defects-` | `Performance-` |
| Ignored columns | `HpX` | `idX`, `centerX`, `YearX`, `MonthsX` | None |
| Numeric-form / text-valued columns | 8 / 0 (excluding `?`) | 7 / 22 | 10 / 0 |
| Missing cells | 6, only ignored `HpX` | 0 | 0 |
| Missing goal cells | 0 | 0 | 0 |
| Constant X columns after trimming | 0 | 8 | 1 |
| Unique X combinations after trimming | 306 | 55 | 192 |
| Main unresolved issues | Acceleration definition; coarse MPG values | Repeated ID; whitespace; rating/goal definitions | Performance definition; configuration constraints |

Requirement implications: preserve header-based roles and directions; distinguish numeric-coded categories from quantitative variables; recognize `?`; trim categorical whitespace in any future working copy; and clarify goal definitions before interpreting optimization results. Repeated X combinations are not sufficient grounds to delete observations. This inspection makes no cleaning changes or implementation decisions.
