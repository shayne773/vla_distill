"""Inference latency, peak GPU memory and parameter counts for one PyTorch checkpoint.

Latency is the model's action-chunk generation (`PI0Pytorch.sample_actions`: prefix forward + KV cache +
10 Euler denoising steps, torch.compile'd as upstream), batch 1, bfloat16, CUDA-synchronized, on a fixed
observation (first frame of the held-out action-agreement set, or a fixed synthetic frame) and fixed noise.
Warm-up calls are excluded. The end-to-end `Policy.infer` time (input/output transforms, host<->device
copies) is reported separately.

    uv run scripts/phase1/bench_model.py --model-id teacher_l18 --config pi05_libero \
        --checkpoint-dir checkpoints/pi05_libero_pytorch
"""

import dataclasses
import json
import pathlib
import platform
import subprocess
import time

import jax
import numpy as np
import torch
import tyro

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    model_id: str
    config: str
    checkpoint_dir: str
    out_dir: str = "results/phase1/bench"
    frames: str = "data/phase1/heldout_frames.npz"
    num_denoise_steps: int = 10
    warmup: int = 20
    iters: int = 200


def count_params(model: torch.nn.Module) -> dict:
    groups = {
        "vision_encoder": (
            "paligemma_with_expert.paligemma.model.vision_tower.",
            "paligemma_with_expert.paligemma.model.multi_modal_projector.",
        ),
        "vlm_backbone_layers": (
            "paligemma_with_expert.paligemma.model.language_model.layers.",
            "paligemma_with_expert.paligemma.model.language_model.norm.",
        ),
        "vlm_token_embedding": (
            "paligemma_with_expert.paligemma.model.language_model.embed_tokens.",
            "paligemma_with_expert.paligemma.lm_head.",
        ),
        # The action expert's Gemma lm_head (vocab x 1024) is never used by pi0/pi0.5; count it separately.
        "unused_expert_lm_head": ("paligemma_with_expert.gemma_expert.lm_head.",),
        "action_expert": (
            "paligemma_with_expert.gemma_expert.",
            "action_in_proj.",
            "action_out_proj.",
            "time_mlp_in.",
            "time_mlp_out.",
            "state_proj.",
            "action_time_mlp_in.",
            "action_time_mlp_out.",
        ),
    }
    counts = dict.fromkeys(groups, 0)
    other = 0
    seen = set()
    for name, p in model.named_parameters(remove_duplicate=True):
        if p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        for g, prefixes in groups.items():
            if name.startswith(prefixes):
                counts[g] += p.numel()
                break
        else:
            other += p.numel()
    counts["other"] = other
    counts["total"] = sum(counts.values())
    counts["vlm_backbone_total"] = counts["vlm_backbone_layers"] + counts["vlm_token_embedding"]
    counts["total_used_at_inference"] = counts["total"] - counts["unused_expert_lm_head"]
    return counts


def load_obs(frames_path: str) -> dict:
    if pathlib.Path(frames_path).exists():
        d = np.load(frames_path, allow_pickle=False)
        return {
            "observation/image": d["image"][0],
            "observation/wrist_image": d["wrist_image"][0],
            "observation/state": d["state"][0],
            "prompt": str(d["prompt"][0]),
        }
    rng = np.random.default_rng(0)
    return {
        "observation/image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": rng.standard_normal(8).astype(np.float32),
        "prompt": "put the black bowl on the plate",
    }


def env_info() -> dict:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return {
        "gpu": smi,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "kernel": platform.release(),
    }


def main(args: Args) -> None:
    torch.manual_seed(0)
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(
        train_config, args.checkpoint_dir, sample_kwargs={"num_steps": args.num_denoise_steps}
    )
    model = policy._model  # noqa: SLF001
    device = policy._pytorch_device  # noqa: SLF001
    params = count_params(model)
    weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    torch.cuda.synchronize()
    mem_after_load = torch.cuda.memory_allocated()

    obs = load_obs(args.frames)
    noise = (
        np.random.default_rng(1234)
        .standard_normal((train_config.model.action_horizon, train_config.model.action_dim))
        .astype(np.float32)
    )

    # Build the exact model inputs once (same path as Policy.infer) for the model-only timing.
    inputs = policy._input_transform(jax.tree.map(lambda x: x, obs))  # noqa: SLF001
    inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
    observation = _model.Observation.from_dict(inputs)
    noise_t = torch.from_numpy(noise).to(device)[None]

    def model_call():
        with torch.no_grad():
            return model.sample_actions(device, observation, noise=noise_t, num_steps=args.num_denoise_steps)

    t0 = time.time()
    for _ in range(args.warmup):
        model_call()
    torch.cuda.synchronize()
    warmup_s = time.time() - t0

    torch.cuda.reset_peak_memory_stats()
    lat = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t = time.perf_counter()
        model_call()
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1000)
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    e2e = []
    for _ in range(min(args.iters, 100)):
        t = time.perf_counter()
        policy.infer(obs, noise=noise)
        e2e.append((time.perf_counter() - t) * 1000)

    lat, e2e = np.array(lat), np.array(e2e)
    dynamo_stats = {k: dict(v) for k, v in torch._dynamo.utils.counters.items() if k in ("stats", "graph_break")}  # noqa: SLF001
    result = {
        "model_id": args.model_id,
        "config": args.config,
        "checkpoint_dir": args.checkpoint_dir,
        "gemma_depth": train_config.model.gemma_depth,
        "precision": train_config.model.dtype,
        "num_denoise_steps": args.num_denoise_steps,
        "action_horizon": train_config.model.action_horizon,
        "batch_size": 1,
        "warmup_calls": args.warmup,
        "warmup_s_incl_compile": round(warmup_s, 1),
        "timed_calls": int(len(lat)),
        "latency_ms": {
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
            "mean": float(lat.mean()),
            "min": float(lat.min()),
            "max": float(lat.max()),
        },
        "policy_infer_e2e_ms": {
            "p50": float(np.percentile(e2e, 50)),
            "p95": float(np.percentile(e2e, 95)),
            "n": int(len(e2e)),
        },
        "gpu_memory_bytes": {
            "weights": int(weights_bytes),
            "allocated_after_load": int(mem_after_load),
            "peak_allocated_inference": int(peak_alloc),
            "peak_reserved_inference": int(peak_reserved),
        },
        "params": params,
        "torch_compile_counters": dynamo_stats,
        "env": env_info(),
        "latency_samples_ms": [round(x, 3) for x in lat.tolist()],
    }
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.model_id}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "latency_samples_ms"}, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
