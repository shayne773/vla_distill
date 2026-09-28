#!/usr/bin/env bash
# Phase-2 readiness check on the laptop: a handful of real Shallow-pi distillation steps (teacher forward +
# student forward/backward + AdamW step + checkpoint save) on the Pi0.5 6-layer student config, on 8 LIBERO
# episodes only. NOT a training run. Everything the phase-2 run changes is a CLI override here:
#   bf16 instead of fp32 (fp32 teacher alone is 13 GB), batch 2 instead of 64, vision tower + token embedding
#   frozen, 8 episodes instead of 1693.
# Usage: scripts/phase1/distill_smoke.sh [config=pi05_libero_l06] [batch=2] [steps=10] [extra args...]
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/phase1/env.sh
CONFIG=${1:-pi05_libero_l06} BATCH=${2:-2} STEPS=${3:-10}; shift $(( $# < 3 ? $# : 3 ))
mkdir -p logs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/distill_pytorch.py "$CONFIG" \
  --exp-name "laptop_smoke_b${BATCH}" --overwrite \
  --batch-size "$BATCH" --num-train-steps "$STEPS" --log-interval 1 --save-interval 100000 --num-workers 2 \
  --pytorch-training-precision bfloat16 \
  --data.lerobot-episodes 0 1 2 3 4 5 6 7 \
  --pytorch-freeze-prefixes paligemma_with_expert.paligemma.model.vision_tower. \
      paligemma_with_expert.paligemma.model.multi_modal_projector. \
      paligemma_with_expert.paligemma.model.language_model.embed_tokens. \
      paligemma_with_expert.paligemma.lm_head. \
  "$@"
