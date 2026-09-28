#!/usr/bin/env bash
# One command to reproduce the paired LIBERO evaluation for any checkpoint/config:
#   scripts/phase1/eval_libero.sh <model_id> <train_config> <pytorch_checkpoint_dir> [extra libero_eval.py args]
# e.g.
#   scripts/phase1/eval_libero.sh teacher_l18      pi05_libero     checkpoints/pi05_libero_pytorch
#   scripts/phase1/eval_libero.sh student_l06_init pi05_libero_l06 checkpoints/pi05_libero_l06_init
# Fixed for every model: 4 suites x 10 tasks x 10 trials (init states 0-9), seed 7, replan every 5 actions,
# 10 denoising steps, per-query noise seeds. Results: results/phase1/libero/<model_id>/{episodes/*.json,summary.json}.
# Resumable: rerunning skips tasks that already have a result file.
set -euo pipefail
MODEL_ID=$1 CONFIG=$2 CKPT=$3; shift 3
cd "$(dirname "$0")/../.."
source scripts/phase1/env.sh
OUT=${OUT_DIR:-results/phase1/libero/$MODEL_ID}
PORT=${PORT:-8000}
mkdir -p logs "$OUT"
.venv/bin/python scripts/phase1/serve_policy_seeded.py --config "$CONFIG" --checkpoint-dir "$CKPT" --port "$PORT" \
  > "logs/server_$MODEL_ID.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until grep -q "Warm-up done" "logs/server_$MODEL_ID.log"; do
  kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -30 "logs/server_$MODEL_ID.log"; exit 1; }
  sleep 5
done
PYTHONPATH="$PWD/third_party/libero" examples/libero/.venv/bin/python scripts/phase1/libero_eval.py \
  --model-id "$MODEL_ID" --out-dir "$OUT" --port "$PORT" "$@" 2>&1 | tee -a "logs/client_$MODEL_ID.log"
