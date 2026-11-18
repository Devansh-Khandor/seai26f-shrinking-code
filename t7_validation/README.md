# T7 common prudence evaluation — Apache Remember pilot

This evaluation layer is separate from historical T7 skill checkpoints.

- `adapters/apache/remember.py`: computes APACHE-01/02 answers using `human_t7/apache/remember.py` only; the adapter does **not** use the oracle.
- `run_apache_remember.py`: executes the adapter and evaluates it with the unchanged `harness.semantic.evaluate` API, saving detailed JSON evidence.
- `tests/test_apache_remember_adapter.py`: verifies the adapter's behavior, including reordered rows.

Run from the repository root after making the treatment-harness branch available as a second worktree:

```sh
git fetch origin treatment-harness
git worktree add --detach ../seai26f-treatment-harness-reference origin/treatment-harness
python3 -m unittest discover -s t7_validation/tests -v
T7_HARNESS_ROOT="$(cd ../seai26f-treatment-harness-reference && pwd)" python3 t7_validation/run_apache_remember.py
```

The runner evaluates the original dataset and the checker-provided shuffled dataset and writes `t7_validation/results/apache_remember.json`.

The adapter reports a ten-row representative shortlist. The complete favorable group of 52 rows, including performance cutoff ties, remains defined in T7's original implementation and is not changed by this output adapter.

Do not alter tagged T7 checkpoints; commit the validation work separately.
