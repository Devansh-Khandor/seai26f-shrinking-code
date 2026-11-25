# Run environment

Fill this in on the machine that runs the treatments, once per model or setup change.

| Field | Value |
|---|---|
| Date | 2026-11-11 |
| Machine / GPU | NVIDIA GeForce RTX 4080, all layers offloaded (`-ngl 99`) |
| OS | Windows 11 (x86_64) |
| Python version | 3.13.7 (venv from `requirements.txt`) |
| scikit-learn version | 1.9.1 (numpy 2.5.4) |
| Lua interpreter + version | Lua 5.4.6 |
| Backend | `llamacpp` (`llama-server`, OpenAI-compatible endpoint) |
| llama.cpp build (if used) | b12242 (commit 3b4d6ab3f) |
| `llama-server` command line (if used) | `llama-server -m gpt-oss-20b-MXFP4.gguf -c 65536 -np 1 -ngl 99 --jinja --temp 1.0 --top-p 1.0 --port 8080` |
| GGUF file + SHA256 (if used) | `gpt-oss-20b-MXFP4.gguf`, `27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901` |
| Model tag | gpt-oss-20b (`--think medium`) |
| Quantization | MXFP4 |
| Context (`--context`) | 65536 |
| Temperature | 1.0 (top-p 1.0, set on the server) |
| Harness commit | 41f74ac |
| `prompt_template_hash` (from runs.csv) | 2d507109b49a |
