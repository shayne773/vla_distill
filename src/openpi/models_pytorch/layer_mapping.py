"""Teacher -> student layer mapping for Shallow-pi depth distillation.

Shallow-pi initializes a shallow student by uniformly subsampling the teacher's transformer layers
(TinyBERT-style) in *both* the VLM backbone and the action expert. The upstream fork implements this
inline in `scripts/distill_pytorch.py`, but its prefix check
`k.startswith(expert_prefix, k.startswith(backbone_prefix))` passes a bool as `startswith`'s `start`
argument, so only action-expert layers were remapped and the backbone silently kept teacher layers
0..L_s-1. This module is the single, corrected implementation used by both the distillation script
and `scripts/phase1/make_student_init.py`.
"""

import torch

# State-dict prefixes of the per-layer weights of the two Gemma stacks in PI0Pytorch.
LAYER_PREFIXES = (
    "paligemma_with_expert.paligemma.model.language_model.layers.",
    "paligemma_with_expert.gemma_expert.model.layers.",
)


def uniform_layer_mapping(teacher_depth: int, student_depth: int) -> list[int]:
    """Student layer i <- teacher layer mapping[i]; endpoints kept (18 -> 6 gives [0, 3, 7, 10, 14, 17])."""
    return torch.linspace(0, teacher_depth - 1, steps=student_depth).round().long().tolist()


def map_teacher_layers(
    student_state_dict: dict[str, torch.Tensor],
    teacher_state_dict: dict[str, torch.Tensor],
    teacher_depth: int,
    student_depth: int,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    """Overwrite the per-layer weights of `student_state_dict` with the mapped teacher layers.

    Non-layer weights (vision tower, embeddings, final norms, projections) are left untouched; the
    caller is expected to have copied them from the teacher already (they have identical shapes).
    Returns the updated state dict and the mapping used.
    """
    mapping = uniform_layer_mapping(teacher_depth, student_depth)
    out = dict(student_state_dict)
    n_mapped = 0
    for key in student_state_dict:
        for prefix in LAYER_PREFIXES:
            if not key.startswith(prefix):
                continue
            s_idx, rest = key[len(prefix) :].split(".", 1)
            t_key = f"{prefix}{mapping[int(s_idx)]}.{rest}"
            out[key] = teacher_state_dict[t_key]
            n_mapped += 1
    if n_mapped == 0:
        raise ValueError("No layer weights were mapped; state-dict prefixes do not match PI0Pytorch.")
    return out, mapping
