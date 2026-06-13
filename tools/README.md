# Tools

Scripts that are not installed as console entry points. Run from a repo with `pip install -e .` on the path.

## Shipped CLIs (`ponyexl3/cli/`)

Installed via `pyproject.toml` as `ponyexl3-generate`, `ponyexl3-compare-layer`, and `ponyexl3-compare-engines`.

## `bench/`

Decode/prefill/kernel throughput sweeps. Examples:

```bash
python tools/bench/benchmark_forward.py MODEL
python tools/bench/benchmark_all_decode.py   # MODEL env var
```

## `dev/`

Diagnostics, speculative-decode A/B tests, parity repair, and GPTQ pilots. Not part of the v0.1 user surface.
