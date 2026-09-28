# vla_distill: distilling π₀.₅ (LIBERO) with Shallow-π

This repository runs a small-budget, controlled experiment: take the official **π₀.₅ LIBERO** checkpoint
(`gs://openpi-assets/checkpoints/pi05_libero`) as the teacher and distill it into a **6-layer student**
(18 → 6 transformer layers in both the PaliGemma VLM backbone and the action expert) using
**Shallow-π** ([arXiv 2601.20262](https://arxiv.org/abs/2601.20262), "Knowledge Distillation for Flow-based VLAs").
The goal is the *comparison*, not the student's absolute score: every model (teacher, untrained student,
distilled student, task-loss-only control) is measured with identical, paired LIBERO rollouts, the same
latency/memory benchmark and the same offline action-agreement metric.

| Phase | Where | What |
|---|---|---|
| 1 (this log's first entry) | laptop, RTX 5080 Laptop 16 GB, free | setup, teacher baseline, untrained 6-layer student, training-step readiness |
| 2 | one rented GPU | short distillation run + no-distillation-loss control |
| 3 | laptop | evaluate every model identically, write the comparison report |

**Lineage and license.** The code base is a snapshot (no history) of
[icsl-Jeon/openpi](https://github.com/icsl-Jeon/openpi) at commit
`a5940ac510c7f5d94918f238cfc3722be1a2c5c8` ("Add distillation code", 2026-01-29), the Shallow-π authors'
fork of [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi). Apache-2.0
([LICENSE](LICENSE)); Gemma weights are subject to [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt). The upstream README
follows unchanged after the experiment log. Submodules `third_party/libero` and `third_party/aloha` are
pinned to the same commits as upstream.

## How to reproduce (laptop)

```bash
git submodule update --init --recursive
source scripts/phase1/env.sh                 # user-local gcc, clean PATH, EGL, LIBERO config (see file)
uv sync                                      # Python 3.11 env; torch 2.7.1+cu128 (see "Deviations")
cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
scripts/phase1/setup_libero_client.sh        # Python 3.8 LIBERO sim-client venv

# teacher: official JAX checkpoint -> PyTorch with openpi's converter
uv run examples/convert_jax_model_to_pytorch.py --config_name pi05_libero \
  --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --output_path checkpoints/pi05_libero_pytorch --precision bfloat16
cp -r ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets checkpoints/pi05_libero_pytorch/
# untrained student: teacher layers [0,3,7,10,14,17] in backbone and action expert
uv run scripts/phase1/make_student_init.py      # -> checkpoints/pi05_libero_l06_init

scripts/phase1/collect_heldout_frames.sh        # once: 457 held-out frames for action agreement
# the same four commands for every model (MODEL_ID CONFIG CHECKPOINT):
scripts/phase1/eval_libero.sh      teacher_l18 pi05_libero checkpoints/pi05_libero_pytorch
uv run scripts/phase1/bench_model.py       --model-id teacher_l18 --config pi05_libero --checkpoint-dir checkpoints/pi05_libero_pytorch
uv run scripts/phase1/action_agreement.py  --model-id teacher_l18 --config pi05_libero --checkpoint-dir checkpoints/pi05_libero_pytorch
uv run scripts/phase1/make_summary.py      --models teacher_l18 student_l06_init   # -> results/phase1/SUMMARY.md
scripts/phase1/distill_smoke.sh pi05_libero_l06 2 10    # distillation-step readiness check (see log)
```

Model identities used in `results/`: `teacher_l18` (config `pi05_libero`), `student_l06_init`
(config `pi05_libero_l06`, layer-mapped teacher weights, 0 training steps), and for phase 2
`student_l06_kd` (config `pi05_libero_l06`: task + teacher-velocity loss) and `student_l06_taskonly`
(config `pi05_libero_l06_taskonly`: identical but `loss_weight_teacher=0`).

## Tests and CI

CI runs only `pre-commit` (uv-lock, ruff, ruff-format; `.github/workflows/pre-commit.yml`). The upstream
`Test` workflow was removed: it required Physical Intelligence's self-hosted `openpi-verylarge` runner,
which this repository does not have, so it could never start. The test suite runs locally instead:

```bash
source scripts/phase1/env.sh
# 1) everything except the one video-decoding test (JAX on CPU so tests don't depend on the GPU)
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest --strict-markers -m "not manual" \
  --deselect src/openpi/training/data_loader_test.py::test_with_real_dataset
# 2) the video-decoding test, with FFmpeg 4-7 shared libraries for torchcodec
#    (one-time: micromamba install -y -n vla-tools -c conda-forge ffmpeg=7)
LD_LIBRARY_PATH=$MAMBA_ROOT_PREFIX/envs/vla-tools/lib JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  .venv/bin/python -m pytest --strict-markers src/openpi/training/data_loader_test.py::test_with_real_dataset
uvx pre-commit@4.6.2 run --all-files      # same checks as CI
```

Prerequisites: the `.venv` from "How to reproduce" (`uv sync` plus the `transformers_replace` copy), the user-local
toolchain from `scripts/phase1/env.sh`, and network access (some tests download small assets from
`gs://openpi-assets` and the LeRobot `aloha_sim` dataset from Hugging Face).
The `LD_LIBRARY_PATH` override is scoped to step 2 only, so the conda libraries never shadow system ones for
the rest of the suite. Without FFmpeg, `test_with_real_dataset` fails with
`Could not load libtorchcodec`. 

Observed on the laptop (JAX on CPU, 16 threads, 2026-09-28), roughly **13-20 minutes** for the suite (59 tests, 3 `manual` deselected):
- Full suite before FFmpeg was installed: 58 passed, 1 failed (`test_with_real_dataset`, libtorchcodec) in 776 s.
- Step 2 (the video test with FFmpeg): passed in 19 s.
- **`scripts/train_test.py::test_train[debug]` (upstream JAX training test) segfaulted in some local runs.** It passed
  in the first full run and in 1 of 2 standalone runs (137 s), and segfaulted (`Fatal Python error: Segmentation fault`
  inside `scripts/train.py` `ptrain_step`) in the other standalone run and in the step-1 run (1,208 s). It also aborted when the conda lib
  directory was on `LD_LIBRARY_PATH` for the whole suite. The cause was not investigated; this PR does not modify
  `scripts/train.py` or `train_test.py`.

## Experiment log

### 2026-09-27 — Phase 1: laptop setup, teacher baseline, untrained 6-layer student, training readiness

**Environment.** Windows 11 + WSL2 Ubuntu 24.04.3 (kernel 6.6.87.2), no sudo; RTX 5080 Laptop GPU 16 GB
(sm_120), NVIDIA driver 616.92 (nvidia-smi 615.71.08, CUDA UMD 13.4); 16 CPU threads, 31 GB RAM.
Python 3.11.16 (`.venv`, uv 0.12.19): torch 2.7.1+cu128, CUDA 12.8, cuDNN 9.7.1, transformers 4.53.2
(+ openpi `transformers_replace`), jax 0.5.3 (CPU only, conversion). LIBERO client: Python 3.8, mujoco
3.2.3, robosuite 1.4.1, torch 1.11.0+cpu. Toolchain: conda-forge gcc 13.4 (micromamba env `vla-tools`).

**What ran** (all raw outputs under `results/phase1/`, rendered table in
[`results/phase1/SUMMARY.md`](results/phase1/SUMMARY.md)):

| | Teacher `teacher_l18` (π₀.₅ LIBERO, 18+18 layers) | Student `student_l06_init` (6+6 layers, no training) |
|---|---|---|
| LIBERO success, overall (400 paired episodes) | **96.0%** [93.6, 97.5] | **0.0%** [0.0, 1.0] |
| spatial / object / goal / libero_10 | 98 / 97 / 96 / 93 | 0 / 0 / 0 / 0 (every episode hits the step limit) |
| Latency per action chunk, p50 / p95 (laptop) | 145.7 / 155.5 ms | 58.6 / 61.0 ms (**2.5× faster**) |
| Peak GPU memory, inference (allocated) | 6.97 GiB | 3.84 GiB |
| Parameters total (used at inference) | 3,617M (3,353M) | 2,012M (1,749M) |
| Transformer layers: VLM backbone / action expert | 1,982M / 430M | 661M / 147M (−67%) |
| Action MSE vs teacher on 457 held-out frames | 0 (reference) | 0.181 [0.167, 0.195] |

Reading the numbers: the teacher reproduces the published π₀.₅ LIBERO result (paper Table 2: 96; openpi:
96.85 at 50 trials/task) on this laptop through the PyTorch path. The teacher's own sampling-noise floor for action MSE
(same frames, different noise seeds) is 0.0086, so the untrained student's actions are about 21× further from the
teacher than the teacher is from itself, and its gripper open/close sign agrees only 71% of the time. **Cutting layers
alone destroys the policy (96% → 0%)**; all of the student's competence has to come from training. That is what
phase 2 measures, with and without the distillation loss. The unchanged SigLIP encoder (415M) and the 527M-parameter
token embedding (a lookup, almost no compute) are why the total only drops 44% while the latency drops 60%.

**Commands** (exact; see "How to reproduce" for setup):
```bash
scripts/phase1/collect_heldout_frames.sh
scripts/phase1/eval_libero.sh teacher_l18      pi05_libero     checkpoints/pi05_libero_pytorch
scripts/phase1/eval_libero.sh student_l06_init pi05_libero_l06 checkpoints/pi05_libero_l06_init
.venv/bin/python scripts/phase1/bench_model.py --model-id teacher_l18      --config pi05_libero     --checkpoint-dir checkpoints/pi05_libero_pytorch
.venv/bin/python scripts/phase1/bench_model.py --model-id student_l06_init --config pi05_libero_l06 --checkpoint-dir checkpoints/pi05_libero_l06_init
.venv/bin/python scripts/phase1/action_agreement.py --model-id teacher_l18 --config pi05_libero --checkpoint-dir checkpoints/pi05_libero_pytorch
.venv/bin/python scripts/phase1/action_agreement.py --model-id teacher_l18_othernoise --config pi05_libero --checkpoint-dir checkpoints/pi05_libero_pytorch \
    --noise-seed-offset 1000000007 --reference results/phase1/action_agreement/teacher_l18_actions.npz
.venv/bin/python scripts/phase1/action_agreement.py --model-id student_l06_init --config pi05_libero_l06 --checkpoint-dir checkpoints/pi05_libero_l06_init \
    --reference results/phase1/action_agreement/teacher_l18_actions.npz
scripts/phase1/distill_smoke.sh pi05_libero_l06 2 10        # run A (fails at AdamW, see below)
FREEZE_EXTRA="paligemma_with_expert.paligemma.model.language_model.layers. paligemma_with_expert.paligemma.model.language_model.norm. paligemma_with_expert.gemma_expert.lm_head." \
  scripts/phase1/distill_smoke.sh pi05_libero_l06 2 10       # run B; run C: same with "8 20"
.venv/bin/python scripts/phase1/make_summary.py --models teacher_l18 student_l06_init
```
Wall-clock on the laptop: teacher eval 57 min, untrained-student eval 1 h 43 min (all episodes run to the step limit), each
benchmark / agreement run 2-4 min, teacher conversion 3 min.

**Evaluation protocol (identical for every model).** openpi's LIBERO example (policy server + sim client),
4 suites × 10 tasks × 10 trials = 400 episodes/model; init states 0-9 of each task, env seed 7, replan every
5 actions of the 10-action chunk, 10 flow-matching denoising steps, 224 px inputs, per-suite step limits
220/280/300/520. The flow-matching noise for query *q* of episode *e* of task *t* is drawn from a fixed seed, so
all models see the same noise at the same point (`scripts/phase1/serve_policy_seeded.py`). Per-episode JSON
(suite, task, init-state index, seed, noise seed, success, steps, errors) is in
`results/phase1/libero/<model>/episodes/`, and `summary.json` has per-suite / overall rates with Wilson 95% CIs. Held-out
frames for action agreement: 457 policy inputs recorded from teacher rollouts on init state 40 (never used by
the success evaluation), every 3rd query, sha256 in `results/phase1/heldout_frames.json`. The frames file itself
(`data/phase1/heldout_frames.npz`, gitignored) is rebuilt by `collect_heldout_frames.sh`.

**Fixes to the Shallow-π fork (needed before paying for phase 2).**
1. *Layer mapping only applied to the action expert.* The fork's init loop calls
   `k.startswith(expert_prefix, k.startswith(backbone_prefix))`; the second argument is `startswith`'s
   *start index* (a bool), so backbone keys never matched and the VLM backbone silently kept teacher layers
   0-5 instead of the uniform subsample [0, 3, 7, 10, 14, 17] the paper describes (TinyBERT-style, both
   components). Fixed in one shared helper, `src/openpi/models_pytorch/layer_mapping.py`, used by both
   `scripts/distill_pytorch.py` and `make_student_init.py`. Measured effect at init: action MSE vs teacher
   0.181 (corrected) vs 0.266 (fork's mapping), gripper-sign agreement 71% vs 49%
   (`results/phase1/action_agreement/student_l06_init_forkmapping.json`).
2. *Teacher and student saw different augmentations.* `DistilledPI0Pytorch.forward` and the teacher's
   `eval_model` each re-ran the random crop/rotate/color-jitter preprocessing, so the teacher-velocity target
   was computed on a different image than the student's input. The teacher now reuses the student's
   preprocessed inputs.
3. *Loss weights were ignored.* `DistilledPi0Config.loss_weight_gt/teacher` existed but the loss was a fixed
   sum. They are now applied, and `loss_weight_teacher=0` skips the teacher forward. That is the phase-2
   control config `pi05_libero_l06_taskonly`.
4. Added `pytorch_freeze_prefixes` (TrainConfig) and `data.lerobot_episodes` (a LeRobot episode subset),
   both used only by the laptop smoke test.

**Deviations from the paper's setup** (the same for every model, so the comparison stays paired):
- *Attention-distillation loss not implemented.* The released `distill_pytorch.py` has only task + teacher-velocity
  loss; the paper's third loss (KL on action→vision-language attention at a middle layer) is missing. Paper
  ablation at L6: task only 93.0, +KD 93.9, +KD+attention 94.6. Phase 2 therefore distills with two of the three
  losses unless someone decides to implement the third one first.
- *Evaluation budget.* 10 trials per task (400 per model) instead of the usual 50 (2000). At p≈0.96 the Wilson
  95% CI half-width is about ±2 points overall and ±4 points per suite. Paired rollouts (identical init states 0-9, env seed 7 and
  per-query noise seeds) plus an exact McNemar test on discordant episodes make the teacher/student
  comparison much tighter than two independent CIs would suggest.
- *Evaluation client changes vs `examples/libero/main.py`.* Rollout logic is unchanged (224 px resize-with-pad,
  10 wait steps, replan every 5 actions, same per-suite max steps). Each task is self-seeded so tasks run in
  4 parallel workers, and every policy query carries a deterministic noise seed.
- *Precision.* All inference is bf16 (openpi's default conversion). The fork's `pi0_libero_l06` config trains
  in fp32; the laptop smoke test uses bf16 (an fp32 teacher alone is 14.5 GB).
- *Latency is laptop latency.* The paper's 25.5 ms (teacher) / 11.3 ms (L6) are on a data-center GPU. Here the
  GPU is a 92 W laptop part shared with the Windows desktop (about 3.3 GB VRAM and some compute are always in use),
  running upstream's `torch.compile(mode="max-autotune")` with eager attention (3 graphs, a few graph breaks
  inside transformers). Compare the teacher/student *ratio*, not absolute milliseconds.
- *Environment.* torch 2.7.1 **cu128** wheel instead of PyPI's cu126 build (no sm_120 kernels; the same torch
  version, only the CUDA build changed). The uv.lock diff is large because uv 0.12 rewrites the lock format.
  There is no system compiler, so gcc 13 / kernel headers / libxcrypt come from a user-local micromamba env
  (`vla-tools`), which evdev (via lerobot→pynput), Triton and the LIBERO client's sdists need. The LIBERO client
  installs the CPU build of torch 1.11 (it only `torch.load`s init states). MuJoCo renders with EGL on Mesa
  llvmpipe (software): the GPU d3d12 Mesa driver hangs creating a MuJoCo context under WSL2 (two attempts)
  and osmesa is not installed. Software rendering gives about 4.5 env steps/s per process, about 18/s with 4 workers.
- *Converter quirk.* `convert_jax_model_to_pytorch.py` copies `assets/` from the checkpoint's parent directory,
  so the norm stats must be copied by hand (see the reproduce block above). Peak host RAM during conversion was 28.8 GB of 31 GB.

**Training readiness (phase 2) — measured on the laptop** (`results/phase1/train_smoke/distill_smoke.json`):
- Real `scripts/distill_pytorch.py` steps on `pi05_libero_l06` (bf16, gradient checkpointing, vision tower +
  token embedding frozen, 8 LIBERO episodes): teacher load, layer-mapped student init, data pipeline with the
  teacher's norm stats, forward + teacher forward + backward all run. Model creation (teacher + student on the GPU) takes
  11.6 GB.
- With the 6 student VLM layers trainable (1.07 B trainable), batch 2 reaches **13.5 GB peak** after backward
  and then fails in the first AdamW step with `RuntimeError: CUDA driver error: device not ready` (WSL2's way
  of reporting GPU OOM; only about 12.9 GB of the 16 GB is usable next to the Windows desktop).
- With the VLM backbone frozen too (147 M trainable, action expert only), the full loop works: 10 steps at
  batch 2 (0.7 s/step, peak 13.3 GB) and 20 steps at batch 8 (**2.3 s/step, 0.29 s/sample, peak 15.0 GB**), plus checkpoint
  save. The saved checkpoint reloads through `create_trained_policy` and produces (10, 7) action chunks, so
  phase-2 outputs drop straight into the phase-3 evaluation commands.

**Phase-2 resource estimate** (full student trainable, batch 64 as in the paper; 1 epoch = 4,273 steps):
- Compute per training sample ≈ teacher forward 4.5 TFLOP (3×256 image tokens + 200 text tokens through
  SigLIP + 18 Gemma-2B layers) + student forward 1.9 + backward 3.8 + checkpoint recompute 1.9 ≈ **12 TFLOP**,
  i.e. ≈ 0.77 PFLOP per batch-64 step (the task-only control skips the teacher: ≈ 7.5 TFLOP/sample).
  Laptop calibration: the frozen-VLM smoke step does ≈ 6.5 TFLOP/sample in 0.29 s ≈ 22 TFLOP/s effective.
- Throughput assumed at ~30-40% bf16 utilization with eager attention: **1× H100 80GB ≈ 2-2.6 s/step**,
  1× A100 80GB ≈ 5-6.5 s/step; the fork's fp32 (TF32) setting would be roughly 1.5-2× slower.
- GPU memory, bf16, full student trainable: teacher 7.5 GB + student weights 4.1 + grads 4.1 +
  AdamW states 8.2 ≈ 24 GB static, plus activations (with gradient checkpointing, estimated 0.3-0.5 GB/sample → 20-30 GB at
  batch 64) ≈ **45-55 GB → use an 80 GB card** (a 48 GB card needs batch 32, which changes the recipe).
  The fork's fp32 setting (teacher 14.5 + student 8.1 + grads 8.1 + AdamW 16.2 ≈ 47 GB static) does not fit
  batch 64 even on 80 GB without more freezing.
- **Recommended short run: 3,000 steps × batch 64 (≈ 0.7 epoch) for each of the two runs**, same seed and data order,
  checkpoints at 1,500 and 3,000 (`--save-interval 1500`), LR schedule shortened so it actually anneals
  (`--lr-schedule.warmup-steps 300 --lr-schedule.decay-steps 3000`, peak 2.5e-5 as in the fork's config),
  bf16 (`--pytorch-training-precision bfloat16`). On 1× H100 80GB: KD run ≈ 1.7-2.2 h, task-only control ≈ 1.1-1.4 h,
  plus ≈ 1 h setup (uv env, 54 GB LeRobot LIBERO download, teacher conversion, which needs ≈ 30 GB host RAM) ≈ **4-5 GPU-hours total**.
  The paper-length run (30 k steps) would be ≈ 17-22 h + 11-14 h for the control, ≈ 8× the cost, so it is not needed for a
  passable effect-of-distillation comparison.
- Why these choices: the question is *distillation vs no distillation at equal compute*, so both runs share
  every setting except `loss_weight_teacher`. 3 k steps is enough for a 6-layer student initialized from teacher layers
  to recover a large part of the gap (it starts from 0% success and 21× the teacher's noise-floor action MSE), while the
  1.5 k checkpoint gives a second point on the curve at no extra cost. Absolute success will likely stay below
  the paper's 95% (they train 30 k steps with the attention loss too).

---

# Upstream README (icsl-Jeon/openpi fork of Physical-Intelligence/openpi, unchanged)

openpi holds open-source models and packages for robotics, published by the [Physical Intelligence team](https://www.physicalintelligence.company/).

Currently, this repo contains three types of models:
- the [π₀ model](https://www.physicalintelligence.company/blog/pi0), a flow-based vision-language-action model (VLA).
- the [π₀-FAST model](https://www.physicalintelligence.company/research/fast), an autoregressive VLA, based on the FAST action tokenizer.
- the [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05), an upgraded version of π₀ with better open-world generalization trained with [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation). Note that, in this repository, we currently only support the flow matching head for both $\pi_{0.5}$ training and inference.

For all models, we provide _base model_ checkpoints, pre-trained on 10k+ hours of robot data, and examples for using them out of the box or fine-tuning them to your own datasets.

This is an experiment: $\pi_0$ was developed for our own robots, which differ from the widely used platforms such as [ALOHA](https://tonyzhaozh.github.io/aloha/) and [DROID](https://droid-dataset.github.io/), and though we are optimistic that researchers and practitioners will be able to run creative new experiments adapting $\pi_0$ to their own platforms, we do not expect every such attempt to be successful. All this is to say: $\pi_0$ may or may not work for you, but you are welcome to try it and see!

## Updates

- [Sept 2025] We released PyTorch support in openpi.
- [Sept 2025] We released pi05, an upgraded version of pi0 with better open-world generalization.
- [Sept 2025]: We have added an [improved idle filter](examples/droid/README_train.md#data-filtering) for DROID training.
- [Jun 2025]: We have added [instructions](examples/droid/README_train.md) for using `openpi` to train VLAs on the full [DROID dataset](https://droid-dataset.github.io/). This is an approximate open-source implementation of the training pipeline used to train pi0-FAST-DROID. 


## Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Please also note that the current training script does not yet support multi-node training.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 8 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090           |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |

The repo has been tested with Ubuntu 22.04, we do not currently support other operating systems.

## Installation

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

**Docker**: As an alternative to uv installation, we provide instructions for installing openpi using Docker. If you encounter issues with your system setup, consider using Docker to simplify installation. See [Docker Setup](docs/docker.md) for more details.




## Model Checkpoints

### Base Models
We provide multiple base VLA model checkpoints. These checkpoints have been pre-trained on 10k+ hours of robot data, and can be used for fine-tuning.

| Model        | Use Case    | Description                                                                                                 | Checkpoint Path                                |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $\pi_0$      | Fine-Tuning | Base [π₀ model](https://www.physicalintelligence.company/blog/pi0) for fine-tuning                | `gs://openpi-assets/checkpoints/pi0_base`      |
| $\pi_0$-FAST | Fine-Tuning | Base autoregressive [π₀-FAST model](https://www.physicalintelligence.company/research/fast) for fine-tuning | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| $\pi_{0.5}$    | Fine-Tuning | Base [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05) for fine-tuning    | `gs://openpi-assets/checkpoints/pi05_base`      |

### Fine-Tuned Models
We also provide "expert" checkpoints for various robot platforms and tasks. These models are fine-tuned from the base models above and intended to run directly on the target robot. These may or may not work on your particular robot. Since these checkpoints were fine-tuned on relatively small datasets collected with more widely available robots, such as ALOHA and the DROID Franka setup, they might not generalize to your particular setup, though we found some of these, especially the DROID checkpoint, to generalize quite broadly in practice.

| Model                    | Use Case    | Description                                                                                                                                                                                              | Checkpoint Path                                       |
| ------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| $\pi_0$-FAST-DROID       | Inference   | $\pi_0$-FAST model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): can perform a wide range of simple table-top manipulation tasks 0-shot in new scenes on the DROID robot platform | `gs://openpi-assets/checkpoints/pi0_fast_droid`       |
| $\pi_0$-DROID            | Fine-Tuning | $\pi_0$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): faster inference than $\pi_0$-FAST-DROID, but may not follow language commands as well                                | `gs://openpi-assets/checkpoints/pi0_droid`            |
| $\pi_0$-ALOHA-towel      | Inference   | $\pi_0$ model fine-tuned on internal [ALOHA](https://tonyzhaozh.github.io/aloha/) data: can fold diverse towels 0-shot on ALOHA robot platforms                                                          | `gs://openpi-assets/checkpoints/pi0_aloha_towel`      |
| $\pi_0$-ALOHA-tupperware | Inference   | $\pi_0$ model fine-tuned on internal [ALOHA](https://tonyzhaozh.github.io/aloha/) data: can unpack food from a tupperware container                                                                                                             | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware` |
| $\pi_0$-ALOHA-pen-uncap  | Inference   | $\pi_0$ model fine-tuned on public [ALOHA](https://dit-policy.github.io/) data: can uncap a pen                                                                                                          | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap`  |
| $\pi_{0.5}$-LIBERO      | Inference   | $\pi_{0.5}$ model fine-tuned for the [LIBERO](https://libero-project.github.io/datasets) benchmark: gets state-of-the-art performance (see [LIBERO README](examples/libero/README.md)) | `gs://openpi-assets/checkpoints/pi05_libero`      |
| $\pi_{0.5}$-DROID      | Inference / Fine-Tuning | $\pi_{0.5}$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/) with [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation): fast inference and good language-following | `gs://openpi-assets/checkpoints/pi05_droid`      |


By default, checkpoints are automatically downloaded from `gs://openpi-assets` and are cached in `~/.cache/openpi` when needed. You can overwrite the download path by setting the `OPENPI_DATA_HOME` environment variable.




## Running Inference for a Pre-Trained Model

Our pre-trained model checkpoints can be run with a few lines of code (here our $\pi_0$-FAST-DROID model):
```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    ...
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]
```
You can also test this out in the [example notebook](examples/inference.ipynb).

We provide detailed step-by-step examples for running inference of our pre-trained checkpoints on [DROID](examples/droid/README.md) and [ALOHA](examples/aloha_real/README.md) robots.

**Remote Inference**: We provide [examples and code](docs/remote_inference.md) for running inference of our models **remotely**: the model can run on a different server and stream actions to the robot via a websocket connection. This makes it easy to use more powerful GPUs off-robot and keep robot and policy environments separate.

**Test inference without a robot**: We provide a [script](examples/simple_client/README.md) for testing inference without a robot. This script will generate a random observation and run inference with the model. See [here](examples/simple_client/README.md) for more details.





## Fine-Tuning Base Models on Your Own Data

We will fine-tune the $\pi_{0.5}$ model on the [LIBERO dataset](https://libero-project.github.io/datasets) as a running example for how to fine-tune a base model on your own data. We will explain three steps:
1. Convert your data to a LeRobot dataset (which we use for training)
2. Defining training configs and running training
3. Spinning up a policy server and running inference

### 1. Convert your data to a LeRobot dataset

We provide a minimal example script for converting LIBERO data to a LeRobot dataset in [`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py). You can easily modify it to convert your own data! You can download the raw LIBERO dataset from [here](https://huggingface.co/datasets/openvla/modified_libero_rlds), and run the script with:

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

**Note:** If you just want to fine-tune on LIBERO, you can skip this step, because our LIBERO fine-tuning configs point to a pre-converted LIBERO dataset. This step is merely an example that you can adapt to your own data.

### 2. Defining training configs and running training

To fine-tune a base model on your own data, you need to define configs for data processing and training. We provide example configs with detailed comments for LIBERO below, which you can modify for your own dataset:

- [`LiberoInputs` and `LiberoOutputs`](src/openpi/policies/libero_policy.py): Defines the data mapping from the LIBERO environment to the model and vice versa. Will be used for both, training and inference.
- [`LeRobotLiberoDataConfig`](src/openpi/training/config.py): Defines how to process raw LIBERO data from LeRobot dataset for training.
- [`TrainConfig`](src/openpi/training/config.py): Defines fine-tuning hyperparameters, data config, and weight loader.

We provide example fine-tuning configs for [π₀](src/openpi/training/config.py), [π₀-FAST](src/openpi/training/config.py), and [π₀.₅](src/openpi/training/config.py) on LIBERO data.

Before we can run training, we need to compute the normalization statistics for the training data. Run the script below with the name of your training config:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

Now we can kick off training with the following command (the `--overwrite` flag is used to overwrite existing checkpoints if you rerun fine-tuning with the same config):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

The command will log training progress to the console and save checkpoints to the `checkpoints` directory. You can also monitor training progress on the Weights & Biases dashboard. For maximally using the GPU memory, set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` before running training -- this enables JAX to use up to 90% of the GPU memory (vs. the default of 75%).

**Note:** We provide functionality for *reloading* normalization statistics for state / action normalization from pre-training. This can be beneficial if you are fine-tuning to a new task on a robot that was part of our pre-training mixture. For more details on how to reload normalization statistics, see the [norm_stats.md](docs/norm_stats.md) file.

### 3. Spinning up a policy server and running inference

Once training is complete, we can run inference by spinning up a policy server and then querying it from a LIBERO evaluation script. Launching a model server is easy (we use the checkpoint for iteration 20,000 for this example, modify as needed):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

This will spin up a server that listens on port 8000 and waits for observations to be sent to it. We can then run an evaluation script (or robot runtime) that queries the server.

For running the LIBERO eval in particular, we provide (and recommend using) a Dockerized workflow that handles both the policy server and the evaluation script together. See the [LIBERO README](examples/libero/README.md) for more details.

If you want to embed a policy server call in your own robot runtime, we have a minimal example of how to do so in the [remote inference docs](docs/remote_inference.md).



### More Examples

We provide more examples for how to fine-tune and run inference with our models on the ALOHA platform in the following READMEs:
- [ALOHA Simulator](examples/aloha_sim)
- [ALOHA Real](examples/aloha_real)
- [UR5](examples/ur5)

## PyTorch Support

openpi now provides PyTorch implementations of π₀ and π₀.₅ models alongside the original JAX versions! The PyTorch implementation has been validated on the LIBERO benchmark (both inference and finetuning). A few features are currently not supported (this may change in the future):

- The π₀-FAST model
- Mixed precision training
- FSDP (fully-sharded data parallelism) training
- LoRA (low-rank adaptation) training
- EMA (exponential moving average) weights during training

### Setup
1. Make sure that you have the latest version of all dependencies installed: `uv sync`

2. Double check that you have transformers 4.53.2 installed: `uv pip show transformers`

3. Apply the transformers library patches:
   ```bash
   cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
   ```

This overwrites several files in the transformers library with necessary model changes: 1) supporting AdaRMS, 2) correctly controlling the precision of activations, and 3) allowing the KV cache to be used without being updated.

**WARNING**: With the default uv link mode (hardlink), this will permanently affect the transformers library in your uv cache, meaning the changes will survive reinstallations of transformers and could even propagate to other projects that use transformers. To fully undo this operation, you must run `uv cache clean transformers`.

### Converting JAX Models to PyTorch

To convert a JAX model checkpoint to PyTorch format:

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint
```

### Running Inference with PyTorch

The PyTorch implementation uses the same API as the JAX version - you only need to change the checkpoint path to point to the converted PyTorch model:

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/converted/pytorch/checkpoint"

# Create a trained policy (automatically detects PyTorch format)
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference (same API as JAX)
action_chunk = policy.infer(example)["actions"]
```

### Policy Server with PyTorch

The policy server works identically with PyTorch models - just point to the converted checkpoint directory:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/path/to/converted/pytorch/checkpoint
```

### Finetuning with PyTorch

To finetune a model in PyTorch:

1. Convert the JAX base model to PyTorch format:
   ```bash
   uv run examples/convert_jax_model_to_pytorch.py \
       --config_name <config name> \
       --checkpoint_dir /path/to/jax/base/model \
       --output_path /path/to/pytorch/base/model
   ```

2. Specify the converted PyTorch model path in your config using `pytorch_weight_path`

3. Launch training using one of these modes:

```bash
# Single GPU training:
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>

# Example:
uv run scripts/train_pytorch.py debug --exp_name pytorch_test
uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume  # Resume from latest checkpoint

# Multi-GPU training (single node):
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Example:
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume

# Multi-Node Training:
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```

### Distillation with PyTorch

```sh
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/distill_pytorch.py pi0_libero_l06 --exp_name pytorch_distill_test
```
### Precision Settings

JAX and PyTorch implementations handle precision as follows:

**JAX:**
1. Inference: most weights and computations in bfloat16, with a few computations in float32 for stability
2. Training: defaults to mixed precision: weights and gradients in float32, (most) activations and computations in bfloat16. You can change to full float32 training by setting `dtype` to float32 in the config.

**PyTorch:**
1. Inference: matches JAX -- most weights and computations in bfloat16, with a few weights converted to float32 for stability
2. Training: supports either full bfloat16 (default) or full float32. You can change it by setting `pytorch_training_precision` in the config. bfloat16 uses less memory but exhibits higher losses compared to float32. Mixed precision is not yet supported.

With torch.compile, inference speed is comparable between JAX and PyTorch.

## Troubleshooting

We will collect common issues and their solutions here. If you encounter an issue, please check here first. If you can't find a solution, please file an issue on the repo (see [here](CONTRIBUTING.md) for guidelines).

| Issue                                     | Resolution                                                                                                                                                                                   |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uv sync` fails with dependency conflicts | Try removing the virtual environment directory (`rm -rf .venv`) and running `uv sync` again. If issues persist, check that you have the latest version of `uv` installed (`uv self update`). |
| Training runs out of GPU memory           | Make sure you set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (or higher) before running training to allow JAX to use more GPU memory. You can also use `--fsdp-devices <n>` where `<n>` is your number of GPUs, to enable [fully-sharded data parallelism](https://engineering.fb.com/2021/07/15/open-source/fsdp/), which reduces memory usage in exchange for slower training (the amount of slowdown depends on your particular setup). If you are still running out of memory, you may want to consider disabling EMA.        |
| Policy server connection errors           | Check that the server is running and listening on the expected port. Verify network connectivity and firewall settings between client and server.                                            |
| Missing norm stats error when training    | Run `scripts/compute_norm_stats.py` with your config name before starting training.                                                                                                          |
| Dataset download fails                    | Check your internet connection. For HuggingFace datasets, ensure you're logged in (`huggingface-cli login`).                                                                                 |
| CUDA/GPU errors                           | Verify NVIDIA drivers are installed correctly. For Docker, ensure nvidia-container-toolkit is installed. Check GPU compatibility. You do NOT need CUDA libraries installed at a system level --- they will be installed via uv. You may even want to try *uninstalling* system CUDA libraries if you run into CUDA issues, since system libraries can sometimes cause conflicts. |
| Import errors when running examples       | Make sure you've installed all dependencies with `uv sync`. Some examples may have additional requirements listed in their READMEs.                    |
| Action dimensions mismatch                | Verify your data processing transforms match the expected input/output dimensions of your robot. Check the action space definitions in your policy classes.                                  |
| Diverging training loss                            | Check the `q01`, `q99`, and `std` values in `norm_stats.json` for your dataset. Certain dimensions that are rarely used can end up with very small `q01`, `q99`, or `std` values, leading to huge states and actions after normalization. You can manually adjust the norm stats as a workaround. |
