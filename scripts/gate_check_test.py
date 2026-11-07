import pathlib
import subprocess
import sys

from safetensors.torch import save_file
import torch

GATE = pathlib.Path(__file__).with_name("gate_check.py")
EXPERT = "paligemma_with_expert.gemma_expert.model.layers"


def _init_tensors() -> dict[str, torch.Tensor]:
    # Every element has |w| >= 2^-6, so all of them count; a layernorm and a VLM tensor that the gate must ignore.
    return {
        f"{EXPERT}.0.mlp.down_proj.weight": torch.full((8, 4), 0.05, dtype=torch.bfloat16),
        f"{EXPERT}.1.self_attn.q_proj.weight": torch.full((4, 4), -0.03, dtype=torch.bfloat16),
        f"{EXPERT}.0.input_layernorm.dense.weight": torch.full((4,), 0.5),
        "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.down_proj.weight": torch.full((4, 4), 0.05),
    }


def _gate(tmp_path: pathlib.Path, ckpt: dict[str, torch.Tensor]) -> str:
    save_file(_init_tensors(), tmp_path / "init.safetensors")
    save_file(ckpt, tmp_path / "ckpt.safetensors")
    out = subprocess.run(
        [sys.executable, "-I", str(GATE), str(tmp_path / "init.safetensors"), str(tmp_path / "ckpt.safetensors")],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def test_gate_fails_when_expert_weights_unchanged(tmp_path: pathlib.Path):
    # The pure-bf16 signature: expert layer matrices bit-identical to the init, only ignored tensors moved.
    ckpt = _init_tensors()
    ckpt[f"{EXPERT}.0.input_layernorm.dense.weight"] += 0.1
    ckpt["paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.down_proj.weight"] += 0.1
    assert _gate(tmp_path, ckpt) == "changed fraction of |w|>=2^-6 expert-layer weights: 0.0000 (0/48) -> FAIL"


def test_gate_passes_when_expert_weights_changed(tmp_path: pathlib.Path):
    ckpt = _init_tensors()
    ckpt[f"{EXPERT}.0.mlp.down_proj.weight"] = ckpt[f"{EXPERT}.0.mlp.down_proj.weight"] + 0.01
    assert _gate(tmp_path, ckpt) == "changed fraction of |w|>=2^-6 expert-layer weights: 0.6667 (32/48) -> PASS"
