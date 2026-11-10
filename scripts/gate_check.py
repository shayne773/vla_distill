"""Fix-is-active gate: fraction of action-expert layer weights (bf16 matrices; every one gets a gradient) with
|w| >= 2^-6 that changed from the student init. Pure-bf16 AdamW at LR <= 2.5e-5 leaves it at exactly 0; with fp32
master weights it rises within a few steps.
usage: python -I scripts/gate_check.py INIT.safetensors CHECKPOINT.safetensors"""

import sys

from safetensors import safe_open

A, B = safe_open(sys.argv[1], "pt"), safe_open(sys.argv[2], "pt")
n = c = 0
for k in A.keys():  # noqa: SIM118 (safe_open handles are not iterable)
    if "gemma_expert.model.layers." in k and "layernorm" not in k:
        a, b = A.get_tensor(k).float(), B.get_tensor(k).float()
        m = a.abs() >= 2.0**-6
        n, c = n + int(m.sum()), c + int((m & (a != b)).sum())
f = c / max(n, 1)
print(f"changed fraction of |w|>=2^-6 expert-layer weights: {f:.4f} ({c}/{n}) -> {'PASS' if f >= 0.25 else 'FAIL'}")
