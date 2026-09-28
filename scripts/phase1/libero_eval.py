"""Paired LIBERO evaluation client (runs in the Python 3.8 LIBERO venv, talks to serve_policy_seeded.py).

Rollout logic is `examples/libero/main.py` (same observation preprocessing, 224px resize-with-pad, 10 wait
steps, per-suite max steps, replan every 5 actions); changes are for a fair, reproducible comparison:
  * every (suite, task) is self-contained: global numpy/random RNGs and the env are seeded at task start and
    episodes run in init-state order 0..N-1, so tasks can run in parallel worker processes and every model
    sees exactly the same initial states and env seeds;
  * every policy query carries a deterministic `noise_seed` (suite, task, episode, query index), so all
    models get the same flow-matching noise at the same point of an episode;
  * one JSON file per task with per-episode records (resumable: finished tasks are skipped), then
    `summary.json` with per-suite / overall success rates and Wilson 95% CIs.

    python scripts/phase1/libero_eval.py --model-id teacher_l18 --out-dir results/phase1/libero/teacher_l18
"""

import collections
import dataclasses
import json
import logging
import math
import multiprocessing as mp
import os
import pathlib
import random
import sys
import time
from typing import List, Optional

import numpy as np
import tyro

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import libero_summary

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


@dataclasses.dataclass
class Args:
    model_id: str
    out_dir: str
    host: str = "127.0.0.1"
    port: int = 8000
    suites: List[str] = dataclasses.field(default_factory=lambda: list(SUITES))
    trials_per_task: int = 10
    seed: int = 7
    replan_steps: int = 5
    resize_size: int = 224
    num_steps_wait: int = 10
    workers: int = 4
    video_dir: Optional[str] = None  # e.g. data/libero/videos/<model_id>; off by default
    # Episodes use init states [init_state_offset, init_state_offset + trials_per_task). The success-rate
    # evaluation uses 0-9; held-out frame collection uses 40+ so the two never overlap.
    init_state_offset: int = 0
    # If > 0, save every k-th policy input (images, state, prompt, noise seed) to <out_dir>/frames/.
    record_every: int = 0


def noise_seed(seed: int, suite_idx: int, task_id: int, episode_idx: int, query_idx: int) -> int:
    return seed * 10**8 + suite_idx * 10**7 + task_id * 10**5 + episode_idx * 10**3 + query_idx


def _quat2axisangle(quat):
    # Copied from examples/libero/main.py (robosuite transform_utils).
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def task_file(out_dir: str, suite: str, task_id: int) -> pathlib.Path:
    return pathlib.Path(out_dir) / "episodes" / f"{suite}__task{task_id:02d}.json"


def run_task(unit):
    args, suite, task_id = unit
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(processName)s %(message)s")
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy

    suite_idx = SUITES.index(suite) if suite in SUITES else 9
    np.random.seed(args.seed)
    random.seed(args.seed)
    task_suite = benchmark.get_benchmark_dict()[suite]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=LIBERO_ENV_RESOLUTION, camera_widths=LIBERO_ENV_RESOLUTION
    )
    env.seed(args.seed)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    max_steps = MAX_STEPS[suite]

    episodes = []
    recorded = collections.defaultdict(list)
    for episode_idx in range(args.init_state_offset, args.init_state_offset + args.trials_per_task):
        t_start = time.time()
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx])
        action_plan = collections.deque()
        t, query_idx, done, error = 0, 0, False, None
        infer_ms, frames = [], []
        while t < max_steps + args.num_steps_wait:
            try:
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                wrist = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist, args.resize_size, args.resize_size)
                )
                if args.video_dir:
                    frames.append(img)
                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist,
                        "observation/state": np.concatenate(
                            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                        "prompt": str(task.language),
                        "noise_seed": noise_seed(args.seed, suite_idx, task_id, episode_idx, query_idx),
                    }
                    if args.record_every and query_idx % args.record_every == 0:
                        for k, v in (
                            ("image", img),
                            ("wrist_image", wrist),
                            ("state", element["observation/state"]),
                            ("prompt", element["prompt"]),
                            ("noise_seed", element["noise_seed"]),
                            ("suite", suite),
                            ("task_id", task_id),
                            ("init_state_idx", episode_idx),
                            ("query_idx", query_idx),
                            ("env_step", t - args.num_steps_wait),
                        ):
                            recorded[k].append(v)
                    result = client.infer(element)
                    query_idx += 1
                    infer_ms.append(float(result.get("server_timing", {}).get("infer_ms", float("nan"))))
                    chunk = result["actions"]
                    assert len(chunk) >= args.replan_steps
                    action_plan.extend(chunk[: args.replan_steps])
                action = action_plan.popleft()
                obs, _, done, _ = env.step(action.tolist())
                if done:
                    break
                t += 1
            except Exception as e:  # mirror main.py: an exception ends the episode as a failure
                error = repr(e)
                logging.exception("episode error")
                break
        episodes.append(
            {
                "model_id": args.model_id,
                "suite": suite,
                "task_id": task_id,
                "task": str(task.language),
                "init_state_idx": episode_idx,
                "seed": args.seed,
                "noise_seed_base": noise_seed(args.seed, suite_idx, task_id, episode_idx, 0),
                "success": bool(done),
                "steps": max(0, t - args.num_steps_wait) + (1 if done else 0),
                "max_steps": max_steps,
                "num_queries": query_idx,
                "server_infer_ms_mean": float(np.nanmean(infer_ms)) if infer_ms else None,
                "wall_time_s": round(time.time() - t_start, 2),
                "error": error,
            }
        )
        if args.video_dir and frames:
            import imageio

            pathlib.Path(args.video_dir).mkdir(parents=True, exist_ok=True)
            name = f"{suite}__task{task_id:02d}__ep{episode_idx:02d}_{'success' if done else 'failure'}.mp4"
            imageio.mimwrite(pathlib.Path(args.video_dir) / name, [np.asarray(x) for x in frames], fps=10)
        logging.info("%s task %d ep %d success=%s steps=%d", suite, task_id, episode_idx, done, episodes[-1]["steps"])

    if recorded:
        fpath = pathlib.Path(args.out_dir) / "frames" / f"{suite}__task{task_id:02d}.npz"
        fpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fpath, **{k: np.asarray(v) for k, v in recorded.items()})
    path = task_file(args.out_dir, suite, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"suite": suite, "task_id": task_id, "episodes": episodes}, indent=1))
    os.replace(tmp, path)
    return suite, task_id, sum(e["success"] for e in episodes)


def main(args: Args) -> None:
    from libero.libero import benchmark

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_args.json").write_text(json.dumps(dataclasses.asdict(args), indent=2))
    bench = benchmark.get_benchmark_dict()
    units = []
    # Longest suites first for better load balancing across workers.
    for suite in sorted(args.suites, key=lambda s: -MAX_STEPS[s]):
        units.extend(
            (args, suite, task_id)
            for task_id in range(bench[suite]().n_tasks)
            if not task_file(args.out_dir, suite, task_id).exists()
        )
    logging.info(
        "%d tasks to run (%d already done)", len(units), sum(bench[s]().n_tasks for s in args.suites) - len(units)
    )
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
        for suite, task_id, n_succ in pool.imap_unordered(run_task, units):
            logging.info("done %s task %d: %d/%d", suite, task_id, n_succ, args.trials_per_task)
    summary = libero_summary.summarize_dir(args.out_dir, expected_trials=args.trials_per_task, suites=args.suites)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(processName)s %(message)s")
    main(tyro.cli(Args))
