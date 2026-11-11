# Treatment harness (T1–T6)

This runs the six LLM treatments from `uproj.md` ("seven ways up the ladder"). For each domain, treatment, trial and skill, it does the following:

1. Builds a prompt containing the background library, this trial's own code for skills 1..i, and the two frozen requests for skill i+1.
2. Sends the prompt to the configured LLM in one fresh call with no history.
3. Saves the reply as a code snapshot.
4. Runs the code (tier 1) and checks its output sections (tier 2).
5. Measures the new lines of code.
6. Appends one row to `results/runs.csv`.

T7, the human treatment, is not run here. Its rows can be added to the same CSV with `model=human`.

| # | Background | Told to reuse? | Language |
|---|---|---|---|
| T1 / T2 | plain Python (standard library) | no / yes | Python |
| T3 / T4 | scikit-learn | no / yes | Python |
| T5 / T6 | `ezr.lua` + `ezr-lib.lua` (`vendor/`, unchanged) | no / yes | Lua |

## Requirements on the run machine

- Python 3.10+.
- The pinned packages in the repo-root `requirements.txt` (scikit-learn, numpy and their dependencies). These are only needed to *run* T3/T4 output. The harness itself is standard library only.
- Lua 5.4+ or LuaJIT, on `PATH` or set in `HARNESS_LUA`. This is needed to run T5/T6 output and `same`.
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
python -m harness --backend llamacpp --model gpt-oss-20b --context 32768 --think medium

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
  -c 32768 -np 1 -ngl 99 --jinja --temp 1.0 --top-p 1.0 --port 8080
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
| `--check-timeout` | `120` seconds | |
| `--runs-root` / `--results` | `harness/runs`, `harness/results/runs.csv` | |
| `--dry-run` | off | Writes each skill-1 prompt and prints its estimated size. No LLM calls. |
| `--think` | off | Reasoning effort for thinking models on Ollama or llama.cpp (`low`, `medium`, `high`), e.g. gpt-oss. Reasoning counts toward the reply cap, so use one level for the whole experiment. Every row records it. |

## Safety checks

- The run refuses to start if T5/T6 are selected and no Lua is found, or if T3/T4 are selected and scikit-learn won't import. Otherwise a whole night of results would show up as model failures.
- Generated programs run in a throwaway copy of the snapshot, with API keys removed from their environment. Files they write at runtime never count as code; they are listed in `check.txt` and then discarded.
- A program that prints more than 2 MB, or runs past `--check-timeout`, is killed along with its child processes.
- Ollama replies are capped (`num_predict`) so the prompt plus the reply fits the context.

## Resume

Press Ctrl+C at any time to stop. Rerun the **same command** to continue. Finished steps are skipped. A step whose snapshot directory is missing gets redone. A `backend_error` row (the server was down or retries ran out) stops that trial and is retried on the next run. A crash loses at most the call that was in flight.

## First-run procedure (one domain)

1. `python -m harness --backend ollama --model <tag> --domains cars --dry-run`. This checks the sizes; T5/T6 start about 9k tokens larger.
2. `python -m harness --backend ollama --model <tag> --domains cars --treatments T1,T5 --trials 1`. This is one full ladder each for T1 and T5.
3. `python -m harness.analyze --model <tag>`, then open `results/<tag>/sizing.md`. If T5 at step 10 is near the context limit, raise `--context` if VRAM allows. If it still overflows, stop and decide with the team. Don't trim `ezr.lua`.
4. Full run: `python -m harness --backend ollama --model <tag> --domains cars --trials 1-3`.
5. `python -m harness.analyze --model <tag>`.

## Outputs

- `runs/<model>/<domain>/<T#>/trial<k>/`:
  - `stepNN/`: the trial's full code after skill NN.
  - `stepNN.prompt.txt` and `stepNN.response.txt`: exactly what was sent and received.
  - `stepNN.check.txt`: the generated program's stdout and stderr.
- `results/runs.csv`: one row per step. `status` is one of `ok`, `parse_failed`, `context_overflow`, `output_truncated` (the reply hit the length limit), `backend_error`, `timeout`. Each row also records the model's quantization, digest and temperature where the backend reports them.
- `results/<model>/` (from `analyze`):
  - `histogram.csv` and `histogram.md`: median % new code per step.
  - `same_tables.md`: T2 vs T1, T4 vs T3, T6 vs T5.
  - `pass_rates.md`
  - `sizing.md`

## Measurement rules

- **New code** follows the frozen pre-registration (`docs/hypotheses.md` §5.1–5.2). It counts non-blank, non-comment implementation lines *added* against the previous step's snapshot. Deleted lines and test files are counted separately. `% new = 100 × new / total`.
- **Tier 1** passes when the entry point `skill_NN.py` or `skill_NN.lua` exits 0 on the domain CSV within the timeout.
- **Tier 2** passes when every request has an `== <ID> ==` section with at least 3 non-empty lines, at least one of them containing a number. This is a weak check of correctness. Tier 3 (checking the answers) is not implemented.
- **`same`** calls the course's `ezr.lua` `same` unchanged, through `same.lua`. The primary sample is the per-trial total new LOC over skills 2–10, counting **complete trials only**. A trial is complete when every one of skills 2–10 has a numeric `new_loc`. The number of excluded trials is shown in every table row. The pooled per-skill sample is secondary. Both rules were fixed on 2026-11-10, before any data existed.

## Tests

```sh
python3 -m unittest discover -s harness/tests -t . -v
```

The tests use a fake backend, so they make no LLM calls. Tests that need Lua are skipped if Lua is missing.
