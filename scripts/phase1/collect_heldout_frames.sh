#!/usr/bin/env bash
# Build the fixed held-out frame set for offline action agreement (run once; reused in phase 3).
# Teacher rollouts from LIBERO init state 40 of every task in the 4 suites (the success-rate evaluation only
# uses init states 0-9); every 3rd policy query is recorded with its noise seed.
# Output: data/phase1/heldout_frames.npz (gitignored; ~tens of MB) + its sha256 in results/phase1/.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT_DIR=data/phase1/heldout_collection scripts/phase1/eval_libero.sh heldout_collection_teacher pi05_libero \
  checkpoints/pi05_libero_pytorch --init-state-offset 40 --trials-per-task 1 --record-every 3
.venv/bin/python - <<'PY'
import glob, hashlib, json, numpy as np, pathlib
files = sorted(glob.glob("data/phase1/heldout_collection/frames/*.npz"))
parts = [dict(np.load(f)) for f in files]
merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
merged["prompt"] = merged["prompt"].astype(str)
merged["suite"] = merged["suite"].astype(str)
np.savez_compressed("data/phase1/heldout_frames.npz", **merged)
h = hashlib.sha256()
for k in ("image", "wrist_image", "state", "noise_seed"):
    h.update(np.ascontiguousarray(merged[k]).tobytes())
info = {"num_frames": int(len(merged["noise_seed"])), "frames_sha256": h.hexdigest(),
        "per_suite": {s: int((merged["suite"] == s).sum()) for s in sorted(set(merged["suite"].tolist()))},
        "source": "teacher_l18 rollouts, init state 40 of each task, every 3rd query"}
pathlib.Path("results/phase1").mkdir(parents=True, exist_ok=True)
pathlib.Path("results/phase1/heldout_frames.json").write_text(json.dumps(info, indent=2))
print(info)
PY
