"""Aggregate per-task LIBERO episode files into success rates with Wilson 95% CIs (stdlib only, py3.8+)."""

import json
import math
import pathlib


def wilson_ci(successes, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion; returns (low, high)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _rate(successes, n):
    lo, hi = wilson_ci(successes, n)
    return {"successes": successes, "n": n, "rate": successes / n if n else float("nan"), "ci95": [lo, hi]}


def load_episodes(out_dir):
    episodes = []
    for f in sorted((pathlib.Path(out_dir) / "episodes").glob("*.json")):
        episodes.extend(json.loads(f.read_text())["episodes"])
    return episodes


def summarize_dir(out_dir, expected_trials=None, suites=None):
    episodes = load_episodes(out_dir)
    by_suite = {}
    for e in episodes:
        by_suite.setdefault(e["suite"], []).append(e)
    summary = {"model_id": episodes[0]["model_id"] if episodes else None, "per_suite": {}, "per_task": {}}
    for suite in suites or sorted(by_suite):
        eps = by_suite.get(suite, [])
        summary["per_suite"][suite] = _rate(sum(e["success"] for e in eps), len(eps))
        summary["per_suite"][suite]["mean_steps"] = sum(e["steps"] for e in eps) / len(eps) if eps else None
        summary["per_suite"][suite]["errors"] = sum(1 for e in eps if e.get("error"))
        tasks = {}
        for e in eps:
            tasks.setdefault(e["task_id"], []).append(e)
        summary["per_task"][suite] = {
            str(t): {"task": v[0]["task"], "successes": sum(x["success"] for x in v), "n": len(v)}
            for t, v in sorted(tasks.items())
        }
    all_eps = [e for s in summary["per_suite"] for e in by_suite.get(s, [])]
    summary["overall"] = _rate(sum(e["success"] for e in all_eps), len(all_eps))
    if expected_trials is not None:
        summary["complete"] = all(
            summary["per_task"][s][t]["n"] == expected_trials for s in summary["per_task"] for t in summary["per_task"][s]
        ) and all(len(summary["per_task"][s]) == 10 for s in summary["per_task"])
    return summary


if __name__ == "__main__":
    import sys

    s = summarize_dir(sys.argv[1], expected_trials=int(sys.argv[2]) if len(sys.argv) > 2 else None)
    (pathlib.Path(sys.argv[1]) / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({k: v for k, v in s.items() if k != "per_task"}, indent=2))
