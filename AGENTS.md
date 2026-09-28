# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- What this repo is: a Shallow-pi distillation experiment (Pi0.5 LIBERO teacher -> 6-layer student) on a snapshot of the icsl-Jeon/openpi fork. Experiment plan, log and results: `README.md` (top) and `results/phase1/SUMMARY.md`.
- Always `source scripts/phase1/env.sh` first: it supplies the user-local compiler (micromamba env `vla-tools`, the machine has no gcc; Triton/torch.compile and some sdists need one), a PATH without Windows `/mnt/c` binaries, `UV_LINK_MODE=copy` and the LIBERO/MuJoCo EGL settings.
- Two venvs: `.venv` (uv, Python 3.11, model/serving/training) and `examples/libero/.venv` (Python 3.8, LIBERO sim client; build with `scripts/phase1/setup_libero_client.sh`). After any `uv sync` that reinstalls transformers, re-copy `src/openpi/models_pytorch/transformers_replace/*` into `.venv/lib/python3.11/site-packages/transformers/`.
- The torch build must stay a cu128 wheel (see `[tool.uv.sources]` in `pyproject.toml`): the PyPI cu126 wheel has no sm_120 (Blackwell) kernels.
- Every model is compared with the same commands: `scripts/phase1/eval_libero.sh`, `bench_model.py`, `action_agreement.py`; teacher/student layer mapping lives only in `src/openpi/models_pytorch/layer_mapping.py`.
- Checkpoints (`checkpoints/`), datasets and held-out frames (`data/`) and `logs/` are gitignored; commit only raw JSON + markdown under `results/`.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
