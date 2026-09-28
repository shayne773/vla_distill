"""Offline action agreement: MSE between each model's action chunk and the teacher's on fixed held-out frames.

Frames come from `scripts/phase1/collect_heldout_frames.sh` (teacher rollouts from LIBERO init states 40-49,
which the success-rate evaluation never uses; every 3rd policy query is recorded). Each frame stores its own
noise seed, so every model denoises from exactly the same noise. Actions are compared after the policy's
output transforms (un-normalized 7-D LIBERO actions, full 10-step chunk).

    uv run scripts/phase1/action_agreement.py --model-id teacher_l18 --config pi05_libero \
        --checkpoint-dir checkpoints/pi05_libero_pytorch
    uv run scripts/phase1/action_agreement.py --model-id student_l06_init --config pi05_libero_l06 \
        --checkpoint-dir checkpoints/pi05_libero_l06_init --reference results/phase1/action_agreement/teacher_l18_actions.npz
"""

import dataclasses
import hashlib
import json
import pathlib

import numpy as np
import torch
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    model_id: str
    config: str
    checkpoint_dir: str
    frames: str = "data/phase1/heldout_frames.npz"
    reference: str | None = None  # teacher actions npz; defaults to this run's own actions (zero reference)
    out_dir: str = "results/phase1/action_agreement"
    num_denoise_steps: int = 10
    # Added to every frame's noise seed. 0 for all model comparisons; a nonzero offset on the teacher gives the
    # teacher-vs-teacher sampling-noise floor (how much the teacher's own chunks vary with different noise).
    noise_seed_offset: int = 0


def frames_digest(d) -> str:
    h = hashlib.sha256()
    for k in ("image", "wrist_image", "state", "noise_seed"):
        h.update(np.ascontiguousarray(d[k]).tobytes())
    return h.hexdigest()


def main(args: Args) -> None:
    torch.manual_seed(0)
    d = np.load(args.frames, allow_pickle=False)
    n = len(d["noise_seed"])
    train_config = _config.get_config(args.config)
    horizon, dim = train_config.model.action_horizon, train_config.model.action_dim
    policy = _policy_config.create_trained_policy(
        train_config, args.checkpoint_dir, sample_kwargs={"num_steps": args.num_denoise_steps}
    )
    actions = []
    for i in range(n):
        obs = {
            "observation/image": d["image"][i],
            "observation/wrist_image": d["wrist_image"][i],
            "observation/state": d["state"][i],
            "prompt": str(d["prompt"][i]),
        }
        noise = (
            np.random.default_rng(int(d["noise_seed"][i]) + args.noise_seed_offset)
            .standard_normal((horizon, dim))
            .astype(np.float32)
        )
        actions.append(np.asarray(policy.infer(obs, noise=noise)["actions"], dtype=np.float32))
    actions = np.stack(actions)  # (N, horizon, 7)

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    digest = frames_digest(d)
    np.savez_compressed(out / f"{args.model_id}_actions.npz", actions=actions, frames_sha256=digest)

    ref_path = args.reference or str(out / f"{args.model_id}_actions.npz")
    ref = np.load(ref_path)
    if str(ref["frames_sha256"]) != digest:
        raise ValueError(f"Reference {ref_path} was computed on different frames")
    ref_actions = ref["actions"]
    err = (actions - ref_actions) ** 2  # (N, H, 7)
    per_frame = err.mean(axis=(1, 2))
    first = err[:, 0, :].mean(axis=1)
    rng = np.random.default_rng(0)
    boot = [per_frame[rng.integers(0, n, n)].mean() for _ in range(2000)]
    gripper_agree = float((np.sign(actions[..., 6]) == np.sign(ref_actions[..., 6])).mean())
    result = {
        "model_id": args.model_id,
        "config": args.config,
        "checkpoint_dir": args.checkpoint_dir,
        "reference": ref_path,
        "num_frames": int(n),
        "frames_sha256": digest,
        "num_denoise_steps": args.num_denoise_steps,
        "noise_seed_offset": args.noise_seed_offset,
        "mse_chunk_mean": float(per_frame.mean()),
        "mse_chunk_ci95_bootstrap": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "mse_chunk_median": float(np.median(per_frame)),
        "mse_first_action_mean": float(first.mean()),
        "mse_per_dim": err.mean(axis=(0, 1)).tolist(),
        "gripper_sign_agreement": gripper_agree,
        "per_suite_mse": {s: float(per_frame[d["suite"] == s].mean()) for s in sorted(set(d["suite"].tolist()))},
        "per_frame_mse": [float(x) for x in per_frame],
    }
    (out / f"{args.model_id}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_frame_mse"}, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
