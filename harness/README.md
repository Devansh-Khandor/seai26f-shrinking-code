# Treatment harness (T1–T6)

This runs the six LLM treatments from `uproj.md` ("seven ways up the ladder"). For each domain, treatment, trial and skill, it does the following:

1. Builds a prompt containing the background library, this trial's own code for skills 1..i, the two frozen requests for skill i+1, and the answer format for those two requests.
2. Sends the prompt to the configured LLM in one fresh call with no history.
3. Saves the reply as a code snapshot.
4. Runs the code (tier 1) and checks its output sections (tier 2, the original format contract).
5. Checks whether the program actually does what each request asks (tier 3, the semantic check), including reruns on controlled copies of the data.
6. Measures the new lines of code.
7. Appends one row to `results/v2/runs.csv` and one row per request to `results/v2/requirements.csv`.

T7, the human treatment, is not run here. Its rows can be added to the same CSV with `model=human`.

| # | Background | Told to reuse? | Language |
|---|---|---|---|
| T1 / T2 | plain Python (standard library) | no / yes | Python |
| T3 / T4 | scikit-learn | no / yes | Python |
| T5 / T6 | the course's Python `ezr.py` (`vendor/python/`, unchanged) | no / yes | Python |

## Protocol versions

- **v1** (2026-11-12): the original 3-trial run. T5/T6 used Lua ezr. There was no semantic check. Its results stay as they were in `results/runs.csv`, `results/gpt-oss-20b/` and the zip, and its tags are `t1-cars-r1-step1` and so on. They are exploratory evidence; do not pool them with v2.
- **v2** (2026-11-17): changed after the professor's feedback on the third metric. T5/T6 use Python ezr. Every request has a machine-checked answer, and every step gets a semantic verdict and a failure category. Outputs go to `runs/v2/` and `results/v2/`; tags are `v2-t1-cars-r1-step1` and so on. The customer requests, the skill order, the reuse/no-reuse wording and the LOC rules are unchanged.

## Requirements on the run machine

- Python 3.12+ (the course's `ezr.py` uses 3.12 syntax).
- The pinned packages in the repo-root `requirements.txt` (scikit-learn, numpy and their dependencies). These are only needed to *run* T3/T4 output. The harness itself is standard library only.
- An LLM backend (see below).

Generated Python runs under the same interpreter as the harness, so use a virtual environment and start the harness with its Python:

```sh
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
.venv/bin/python -m pip install -r requirements.txt       # macOS / Linux
.venv/Scripts/python -m harness ...                       # or activate the venv and use python
```

## Backends

Only the backend changes between runs. Everything else stays the same.

```sh
# Local model through Ollama
python -m harness --backend ollama --model qwen2.5-coder:14b --context 32768

# Local model through llama.cpp's llama-server (see "llama.cpp setup" below)
python -m harness --backend llamacpp --model gpt-oss-20b --context 65536 --think medium

# Any OpenAI-compatible API (OpenAI, DeepSeek, LM Studio, vLLM)
python -m harness --backend openai --model gpt-4.1-mini            # uses OPENAI_API_KEY
python -m harness --backend openai --base-url https://api.deepseek.com --model deepseek-chat   # DEEPSEEK_API_KEY
python -m harness --backend openai --base-url http://localhost:1234/v1 --model local-model     # LM Studio

# Claude Code subscription (claude -p), one fresh process per call
python -m harness --backend claude-cli --model sonnet
```

The model names above are examples. Record the exact tag and quantization you actually use in `RUN_ENV.md`.

### llama.cpp setup

The `llamacpp` backend talks to `llama-server` (default `http://localhost:8080`). Compared with `--backend openai` it caps each reply (`max_tokens`) so prompt plus reply fits the context, passes `--think` to the gpt-oss chat template as `reasoning_effort`, never sends an API key, and records the GGUF file, build and server temperature in every row. Before the first call it reads `/props` and refuses to start if the server's per-slot context differs from `--context`.

Start the server in its own terminal and leave it running. Replace the placeholders with where you unpacked the llama.cpp release and saved the GGUF:

```powershell
<llama.cpp dir>\llama-server.exe -m <models dir>\gpt-oss-20b-MXFP4.gguf `
  -c 65536 -np 1 -ngl 99 --jinja --temp 1.0 --top-p 1.0 --port 8080
```

- `-c` must equal the harness `--context`. `-np 1` keeps the whole context in one slot (with `-np N` each slot gets `c/N`).
- `-ngl 99` puts every layer on the GPU. `--temp 1.0 --top-p 1.0` are OpenAI's recommended gpt-oss settings; the value the server uses is what each row records.
- The reasoning trace comes back separately from the answer and is not saved as code, but it does count toward `max_tokens`, so keep one `--think` level for the whole experiment.

`claude-cli` runs each call in an empty temp directory with no tools, no settings files, no MCP servers, no slash commands and no saved session. It can't use `--bare`, because that refuses subscription logins. As a result, a user-level `~/.claude/CLAUDE.md` on the run machine may still load. Run it under an account that doesn't have one.


## Selecting work

| Flag | Default | Example |
|---|---|---|
| `--domains` | `cars,nasa,apache` | `--domains cars` |
| `--treatments` | `T1-T6` | `--treatments T1,T5` |
| `--trials` | `1-3` | `--trials 1-10` |
| `--skills` | `1-10` | `--skills 1-3` |
| `--check-timeout` | `120` seconds | applies to every run of a program, including the semantic reruns |
| `--runs-root` / `--results` | `harness/runs/v2`, `harness/results/v2/runs.csv` | |
| `--dry-run` | off | Writes each skill-1 prompt and prints its estimated size. No LLM calls. |
| `--think` | off | Reasoning effort for thinking models on Ollama or llama.cpp (`low`, `medium`, `high`), e.g. gpt-oss. Reasoning counts toward the reply cap, so use one level for the whole experiment. Every row records it. |

## Safety checks

- The run refuses to start if T5/T6 are selected on Python older than 3.12, or if T3/T4 are selected and scikit-learn won't import. Otherwise a whole night of results would show up as model failures.
- Generated programs run in a throwaway copy of the snapshot, with API keys removed from their environment. Files they write at runtime never count as code; they are listed in `check.txt` and then discarded. Only T5/T6 programs can import `ezr`.
- A program that prints more than 2 MB, or runs past `--check-timeout`, is killed along with its child processes.
- Ollama replies are capped (`num_predict`) so the prompt plus the reply fits the context.
- `runs.csv` refuses to append rows with a different set of columns, so v1 and v2 results can't be mixed in one file.

## Resume

Press Ctrl+C at any time to stop. Rerun the **same command** to continue. Finished steps are skipped. A step whose snapshot directory is missing gets redone. A `backend_error` row (the server was down or retries ran out) stops that trial and is retried on the next run. A crash loses at most the call that was in flight.

## Run procedure

1. `python -m harness --backend llamacpp --model <tag> --context 65536 --think medium --dry-run`. This checks prompt sizes; T5/T6 start about 4k tokens larger (they include `ezr.py`).
2. Smoke test, one ladder per foundation: `python -m harness ... --domains cars --treatments T1,T3,T5 --trials 1 --skills 1-3`. Check that rows show `runs`, `contract` and `semantic`, and that no failure category starts with `system:`. A `system:dependency` row means the machine is missing a package.
3. `python -m harness.analyze --model <tag>`, then open `results/v2/<tag>/pass_rates.md` and `sizing.md`.
4. Full run: `python -m harness ... --trials 1-3`.
5. `python -m harness.analyze --model <tag>`.
6. Optional: `python -m harness.diagnose --backend llamacpp --model <tag>` asks the model why each failed step failed (see [Failure categories](#failure-categories)).
7. `python -m harness.tag --model <tag>`, then `git push origin --tags`. This tags every finished trial (see [Tags](#tags)).

To add trials later, run the same command with the new trial numbers, e.g. `--trials 4-10`, then repeat steps 5–7. Trials that are already tagged are checked, not retagged.

## Outputs

- `runs/v2/<model>/<domain>/<T#>/trial<k>/`:
  - `stepNN/`: the trial's full code after skill NN.
  - `stepNN.prompt.txt` and `stepNN.response.txt`: exactly what was sent and received.
  - `stepNN.check.txt`: the generated program's stdout and stderr.
  - `stepNN.semantic.json`: every semantic check, with what was expected, for both requests and every controlled rerun.
- `results/v2/runs.csv`: one row per step. `status` is one of `ok`, `parse_failed`, `context_overflow`, `output_truncated` (the reply hit the length limit), `backend_error`, `timeout`. `semantic` is the step's semantic verdict, and `failure_category` / `failure_detail` say why it failed. Each row also records the model's quantization, digest and temperature where the backend reports them.
- `results/v2/requirements.csv`: one row per request per step (two per skill), including failed steps: status, checks passed/total, the failed checks, and the path to the evidence file.
- `results/v2/diagnoses.csv` (from `diagnose`): the model's suspected cause for each failed step.
- `results/v2/<model>/` (from `analyze`):
  - `histogram.csv` and `histogram.md`: median % new code per step.
  - `same_tables.md`: T2 vs T1, T4 vs T3, T6 vs T5, plus `bestRanks` over all six treatments for LOC and for correctness.
  - `pass_rates.md`: runs, original contract and semantic pass rates per domain and treatment and per foundation, the count that passed the contract but not the semantic check, and failure categories.
  - `requirements.md`: request-level semantic results.
  - `sizing.md`

## The semantic check (tier 3)

The professor's point was that the format contract (tier 2) passes programs that print the right headings and some numbers while doing the wrong thing. Tier 3 checks the substance.

**Answers.** Each of the 60 requests has an answer format (`harness/answers.py`). After its `== <ID> ==` section the program prints `ANSWER <ID> {json}`. For example, CAR-03 must give a prediction for every held-out row and its MAE. The prompt adds only this format and a shared protocol: held-out rows are rows whose number is divisible by 5, arrivals are the last 20% of rows, and the evaluation budget is min(30, rows/5). The customer wording stays frozen.

**Checks.** `harness/oracle.py` recomputes the truth from the CSV with the standard library and never trusts the program. The checks test what the request demands, not one algorithm:

- Exact facts must be exact: summary statistics, group counts and means, shares, cited row values, which settings differ, the minimal number of changes, and MAE recomputed from the program's own predictions.
- Where several methods are legitimate, any answer consistent with the data passes:
  - "Overall" quality may be judged by distance-to-heaven, z-score sum or rank sum (they disagree on NASA, whose goals conflict).
  - Neighbors must be among the 25% most similar rows.
  - Flagged rows must mostly be in the most unusual 25%.
- Statistical verdicts ("different" / "same", "better" / "no clear difference") are judged only when the data are clear-cut: |Cliff's delta| ≥ 0.474 with a KS rejection, or < 0.147 with none. Otherwise either call passes, but a claimed winner must have the better median.
- Predictions must beat the historical mean. Uncertainty ranges must cover the truth at least half the time and must not span the whole observed range.

**Controlled reruns.** The same program is run again on altered copies of the data:

| Rerun | What changes | What it catches |
|---|---|---|
| `shuffled` (every skill) | rows in a fixed different order; every check repeated | hard-coded answers |
| `mask` (Guess, Race) | held-out target values replaced by absurd ones, one copy per target (other columns, including other goals, stay as they are) | using held-out targets (leakage) |
| `future` (Flow) | later arrivals' goals pushed far beyond the observed range | reading future arrivals |
| `honeypot` (Choose, Race) | six planted rows with ideal goals but middling inputs | looking at goals outside the budget |

**Verdicts.** Each request gets one of: `pass`, `fail`, `unverified` (a rerun timed out, so there isn't enough evidence; never counted as a pass), `parse_failure`, `execution_failure`, or `checker_error` (the checker itself broke). A skill passes only if both of its requests pass. Pass rates use every attempt as the denominator. `pass_rates.md` also shows the rate among verified attempts, labelled as such.

This is *tested* correctness: a finite set of checks, not a proof. `tests/reference_solution.py` is an independent, known-good solution to all 60 requests. It deliberately uses different aggregates, distances and tests than the oracle, and it passes every check. `tests/test_semantic.py` feeds well-formatted wrong answers for every kind of request, plus hard-coded, leaky, peeking and future-reading programs, and each one must fail.

## Failure categories

`failure_category` comes from evidence only (`harness/failures.py`): the status, the exit code, the last traceback line and the semantic result. The part before the colon says whose problem it is:

| Prefix | Meaning | Examples |
|---|---|---|
| `system:` | the machine or harness | `backend`, `context_overflow`, `dependency` (an allowed package is missing), `resource` |
| `generation:` | the reply never became code | `extraction` (no parsable files), `truncated` |
| `code:` | the generated program | `syntax`, `forbidden_import`, `api_misuse` (e.g. a name ezr doesn't have), `runtime`, `missing_entry`, `format`, `no_answer`, `semantic`, `fragile` (crashed on a controlled copy) |
| `runtime:` | ran out of time or flooded output | `timeout`, `output_flood` |
| `evaluator:` | the checker couldn't judge | `checker_error`, `unverified` |

`python -m harness.diagnose` adds a second opinion. It sends each failed step's request, code, errors and failed checks to the same backend and asks whether the cause is a system problem or a code problem. The answers go to `diagnoses.csv`, marked `suspected (LLM)`. They sit beside the evidence-based category and never change a verdict.

## Tags

The class asks for histograms derivable from the repo's tags, so `python -m harness.tag` turns each finished trial into its own small git history, off every branch: an empty start commit, then one commit per step whose tree is exactly that step's snapshot. A failed step repeats the previous snapshot, as the harness does. v2 tags are named `v2-<t#>-<domain>-r<trial>-step<k>`, e.g. `v2-t1-cars-r1-step1`; `--protocol v1` checks the original unprefixed tags.

- Only trials whose ten steps are all finished are tagged. Existing tags are never moved, and a partly tagged trial is an error.
- After tagging, every tagged trial is re-measured from its tags alone with the same counting rules, and any difference from `runs.csv` is an error.
- It never pushes. Run `git push origin --tags` yourself.

`git diff --stat v2-t1-cars-r1-step1..v2-t1-cars-r1-step2` shows a step's change, but it also counts comments and blank lines.

## Measurement rules

- **New code** follows the frozen pre-registration (`docs/hypotheses.md` §5.1–5.2). It counts non-blank, non-comment implementation lines *added* against the previous step's snapshot. Deleted lines and test files are counted separately. `% new = 100 × new / total`.
- **Tier 1** passes when the entry point `skill_NN.py` exits 0 on the domain CSV within the timeout.
- **Tier 2** passes when every request has an `== <ID> ==` section with at least 3 non-empty lines, at least one of them containing a number. It is kept unchanged so it can be compared with v1, but it is only a format check.
- **Tier 3** is the semantic check above.
- **`same` and `bestRanks`** are the professor's functions from `stats.py` (`harness/vendor_stats.py`, copied unchanged): a median-difference shortcut, then Cliff's delta and KS, with eps = 0.35 × sd of the sample with the lower median. The primary sample is the per-trial total new LOC over skills 2–10, counting **complete trials only**. A trial is complete when every one of skills 2–10 has a numeric `new_loc`. Failed steps are never counted as zero. The number of excluded trials is shown in every table row. The pooled per-skill sample is secondary. `bestRanks` ranks all six treatments by LOC (complete trials) and by skills passing the semantic check per trial.

## Tests

```sh
python -m unittest discover -s harness/tests -t . -v
```

The tests use a fake backend, so they make no LLM calls. `test_pipeline.py` runs the whole v2 pipeline end to end with a fake model that writes the reference solution, for plain Python and for Python ezr.
