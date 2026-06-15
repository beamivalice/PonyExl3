# Contributing

PonyExl3 targets **macOS on Apple Silicon with Metal**. There is no Linux
support path — inference depends on MLX Metal kernels.

## Setup

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Python **≥ 3.14** is required. This matches the version we develop and test on;
older Python releases are unsupported.

## Tests

```bash
pytest tests/ -q
```

**171 tests** run without any checkpoint on disk (synthetic layers + CPU/MLX
parity). Nine more are skipped unless you set model env vars — see README
[Testing](README.md#testing).

CI runs the same suite on **GitHub-hosted `macos-latest`** (Apple Silicon).
Those runners expose Metal, so MLX kernel tests execute for real — not skipped
the way they would be on Linux.

Optional integration tests (skipped by default):

```bash
export PONYEXL3_MODEL_DIR=/path/to/checkpoint
export PONYEXL3_MODEL_27B=/path/to/27b-exl3
export PONYEXL3_REFERENCE_NPZ=/path/to/reference.npz
pytest tests/ -q
```

## Type annotations

The package ships `ponyexl3/py.typed` (PEP 561). Downstream tools (Pylance,
mypy, pyright) can use our annotations and the `typings/` stubs. Running
pyright locally is optional:

```bash
pyright ponyexl3 tests   # not enforced in CI
```

## Pull requests

- Keep changes focused; match existing style in the file you edit.
- Run `pytest tests/ -q` before opening a PR.
- Do not commit model weights, `.npz` reference bundles, or local paths.

## License

By contributing, you agree that your contributions are licensed under the
Apache-2.0 license in [`LICENSE`](LICENSE).
