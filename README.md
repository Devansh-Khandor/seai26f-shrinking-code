# The Shrinking-Code Experiment

CSC 591 (AI & Software Engineering) graduate project.

**Question:** as an AI assistant builds a ladder of ten related capabilities, does reusing a small library of symbolic components reduce the new code each step needs, and does that benefit carry over to new domains?

The [frozen requirements](docs/requirements.md) define 60 requests (two per skill, ten skills, three domains). The [pre-registered hypotheses](docs/hypotheses.md) define the treatments, measurements and win conditions.

## Domains

| Domain | Stakeholder | Data |
|---|---|---|
| Cars | Independent used-car dealership | [auto93.csv](data/raw/auto93.csv) |
| NASA | Software project manager | [nasa93dem.csv](data/raw/nasa93dem.csv) |
| Apache | Web server administrator | [Apache_AllMeasurements.csv](data/raw/Apache_AllMeasurements.csv) |

The ten skills, in order: remember, guess, certify, flow, blame, justify, choose, fix, spot, race.

## Treatments

| # | Background given to the coder | Told to reuse? | Language |
|---|---|---|---|
| T1 / T2 | Plain Python (standard library) | no / yes | Python |
| T3 / T4 | scikit-learn | no / yes | Python |
| T5 / T6 | Course `ezr.lua` library | no / yes | Lua |
| T7 | Human-written | — | — |

T1–T6 are run by an LLM through the [treatment harness](harness/README.md). Each step is one fresh call with no history: the background, the trial's code so far, and the next skill's two requests. Every step's new lines of code are measured and the program is run on the domain CSV.

## Status

- **T1–T6:** complete for all three domains at 3 trials (540 steps, `gpt-oss-20b`, one fixed setup). Running trials 4–10 to reach the pre-registered target of 10.
- **T7 (human):** in progress.
- **Analysis, slides and report:** in progress.

## Repository layout

| Path | Contents |
|---|---|
| `docs/` | Frozen requirements, hypotheses, dataset analysis, [reference implementation notes](docs/reference_implementation.md) |
| `data/raw/` | The three original datasets (never modified) |
| `harness/` | T1–T6 treatment harness ([README](harness/README.md), [run environment](harness/RUN_ENV.md)) |
| `harness/results/` | **Treatment results:** `runs.csv` (one row per step), summary tables per model, zipped raw trial logs |
| `shrinking_code/` | Reference implementation answering all 60 requests with shared components ([notes](docs/reference_implementation.md)) |
| `tests/` | Tests for the reference implementation |
| `experiments/` | Code-growth measurement scripts for the reference implementation |
| `results/` | Code-growth measurements of the reference implementation (development history, **not** treatment results) |
| `deliverables/` | Submitted course deliverables |

## Running

Python 3.10+ for the harness; the reference implementation needs only the standard library.

```sh
# Reference implementation
python -m shrinking_code.demo --domain cars --skill remember   # any domain / skill
python -m unittest discover -s tests -v

# Treatment harness (see harness/README.md for setup, backends and Lua)
python -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
python -m harness --backend <backend> --model <model> --trials 1-3
python -m harness.analyze --model <model>
python -m unittest discover -s harness/tests -t .
```

## Results

Treatment results are in [`harness/results/`](harness/results/):

- `runs.csv`: every step's status, pass/fail checks, new/total lines of code, tokens and timing.
- `<model>/same_tables.md`: T2 vs T1, T4 vs T3, T6 vs T5 with the course `same` test.
- `<model>/histogram.md`: median % new code per step.
- `<model>/pass_rates.md`, `<model>/sizing.md`: run/contract pass rates and prompt sizes.
- `gpt-oss-20b-3trials.zip`: every prompt, reply, code snapshot and program output.

## Disclosure

The reference implementation (`shrinking_code/`, `tests/`, `experiments/`) was developed with LLM assistance. It implementation is one development history, not an experimental treatment, and is not human-written T7 code.

Course references: [project specification](https://github.com/txt/seai26f/blob/main/docs/submit/uproj.md) and [graduate scope](https://github.com/txt/seai26f/blob/main/docs/submit/gproj.md).
