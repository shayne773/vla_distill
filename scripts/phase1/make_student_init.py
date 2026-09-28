"""Build the untrained Shallow-pi student: teacher weights with transformer layers uniformly subsampled.

This is exactly the initialization `scripts/distill_pytorch.py` starts training from (same helper,
`openpi.models_pytorch.layer_mapping`), saved as a standalone PyTorch checkpoint so the untrained student
can be served and benchmarked like any other model.

    uv run scripts/phase1/make_student_init.py \
        --teacher-dir checkpoints/pi05_libero_pytorch --student-config pi05_libero_l06 \
        --output-dir checkpoints/pi05_libero_l06_init
"""

import dataclasses
import json
import pathlib
import shutil

import safetensors.torch
import torch
import tyro

from openpi.models_pytorch import layer_mapping
from openpi.models_pytorch import pi0_pytorch
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    teacher_dir: str = "checkpoints/pi05_libero_pytorch"
    student_config: str = "pi05_libero_l06"
    output_dir: str = "checkpoints/pi05_libero_l06_init"
    precision: str = "bfloat16"


def main(args: Args) -> None:
    student_cfg = _config.get_config(args.student_config).model
    teacher_cfg = _config.get_config(student_cfg.teacher_config).model
    student_cfg = dataclasses.replace(student_cfg, dtype=args.precision)

    teacher_sd = safetensors.torch.load_file(str(pathlib.Path(args.teacher_dir) / "model.safetensors"))
    student = pi0_pytorch.PI0Pytorch(student_cfg)
    student_sd = student.state_dict()

    # Everything that is not a per-layer weight (vision tower, embeddings, norms, projections) has the same
    # shape in teacher and student and is copied directly; per-layer weights are then remapped.
    init_sd = {k: teacher_sd[k] for k in student_sd if k in teacher_sd}
    missing = sorted(set(student_sd) - set(init_sd))
    init_sd, mapping = layer_mapping.map_teacher_layers(
        init_sd, teacher_sd, teacher_cfg.gemma_depth, student_cfg.gemma_depth
    )
    result = student.load_state_dict(init_sd, strict=False)
    print(f"layer mapping (student <- teacher): {mapping}")
    # embed_tokens is tied to lm_head (safetensors stores it once, as lm_head), so it is filled via lm_head.
    print(f"keys not in teacher file (tied, filled via lm_head): {missing}")
    print(f"load_state_dict: missing={result.missing_keys} unexpected={result.unexpected_keys}")

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    student = student.to(torch.bfloat16 if args.precision == "bfloat16" else torch.float32)
    safetensors.torch.save_model(student, str(out / "model.safetensors"))
    if (pathlib.Path(args.teacher_dir) / "assets").exists():
        shutil.copytree(pathlib.Path(args.teacher_dir) / "assets", out / "assets", dirs_exist_ok=True)
    (out / "init.json").write_text(
        json.dumps(
            {
                "teacher_dir": args.teacher_dir,
                "student_config": args.student_config,
                "teacher_depth": teacher_cfg.gemma_depth,
                "student_depth": student_cfg.gemma_depth,
                "layer_mapping": mapping,
                "precision": args.precision,
                "trained_steps": 0,
            },
            indent=2,
        )
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
