"""Policy server for the paired LIBERO evaluation.

Same as `scripts/serve_policy.py` (openpi websocket server + `create_trained_policy`), plus:
  * the number of denoising steps is fixed explicitly (default 10) for every model;
  * if the client sends an integer `noise_seed`, the flow-matching initial noise is drawn from
    `np.random.default_rng(noise_seed)`, so every model gets the same noise for the same
    (suite, task, episode, query) key (common random numbers), and every episode is reproducible.

    uv run scripts/phase1/serve_policy_seeded.py --config pi05_libero --checkpoint-dir checkpoints/pi05_libero_pytorch
"""

import dataclasses
import logging

import numpy as np
import torch
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    config: str
    checkpoint_dir: str
    port: int = 8000
    num_denoise_steps: int = 10
    torch_seed: int = 0


class SeededNoisePolicy:
    def __init__(self, policy, action_horizon: int, action_dim: int):
        self._policy = policy
        self._shape = (action_horizon, action_dim)

    def infer(self, obs: dict) -> dict:
        obs = dict(obs)
        seed = obs.pop("noise_seed", None)
        noise = None
        if seed is not None:
            noise = np.random.default_rng(int(seed)).standard_normal(self._shape).astype(np.float32)
        return self._policy.infer(obs, noise=noise)

    @property
    def metadata(self):
        return self._policy.metadata


def main(args: Args) -> None:
    torch.manual_seed(args.torch_seed)
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(
        train_config, args.checkpoint_dir, sample_kwargs={"num_steps": args.num_denoise_steps}
    )
    model_cfg = train_config.model
    wrapped = SeededNoisePolicy(policy, model_cfg.action_horizon, model_cfg.action_dim)
    # Warm up (torch.compile max-autotune) before accepting clients, so the first real request does not
    # block the event loop long enough to trip websocket keepalive timeouts of other clients.
    dummy = {
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
        "observation/state": np.zeros((8,), np.float32),
        "prompt": "warm up",
    }
    for i in range(3):
        wrapped.infer({**dummy, "noise_seed": i})
    logging.info("Warm-up done")
    metadata = dict(policy.metadata)
    metadata.update(config=args.config, checkpoint_dir=args.checkpoint_dir, num_denoise_steps=args.num_denoise_steps)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped, host="127.0.0.1", port=args.port, metadata=metadata
    )
    logging.info("Serving %s from %s on port %d", args.config, args.checkpoint_dir, args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
