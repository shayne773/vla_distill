#!/usr/bin/env bash
# Build the LIBERO simulation-client venv (Python 3.8) used by scripts/phase1/eval_libero.sh.
# Mirrors examples/libero/README.md "Without Docker", with one change: the client only needs torch
# to torch.load() LIBERO init states, so we install the CPU wheels of the same torch version instead
# of the 2 GB cu113 wheels (cu113 kernels cannot run on sm_120 anyway; the client never touches the GPU).
set -euo pipefail
cd "$(dirname "$0")/../.."
VENV=examples/libero/.venv
uv venv --python 3.8 "$VENV"
grep -vE '^(torch|torchvision|torchaudio)==' examples/libero/requirements.txt > "$VENV/requirements.nocuda.txt"
cat >> "$VENV/requirements.nocuda.txt" <<'REQ'
torch==1.11.0+cpu
torchvision==0.12.0+cpu
torchaudio==0.11.0+cpu
REQ
VIRTUAL_ENV=$PWD/$VENV uv pip sync "$VENV/requirements.nocuda.txt" third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy=unsafe-best-match
VIRTUAL_ENV=$PWD/$VENV uv pip install -e packages/openpi-client
VIRTUAL_ENV=$PWD/$VENV uv pip install -e third_party/libero
