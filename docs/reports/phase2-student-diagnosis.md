# Phase 2 student (Shallow-π, 6-layer π0.5): why the success rate is low

Diagnosis report, 2026-10-22.
Code: `vla_distill` HEAD `3425187`. Training ran from the `966da5c` tarball, and `git diff 966da5c 3425187 -- scripts src` is empty, so every line reference below holds for the code that actually trained.
The fix prototyped here has since been merged as commit `ae0c007` (fp32 master weights, see README.md "Precision"). The fix-is-active gate of (e) 2 is committed as `scripts/gate_check.py` (test: `scripts/gate_check_test.py`), and the freeze list of (c) 3 is the default of both 6-layer configs. The follow-up run is [runbook 2](phase2-rerun-runbook.zh-TW.md), with its [laptop dry run](phase2-rerun-dry-run.md).

## Bottom line

- **Reproduced.**
  - The KD student at step 3000 succeeds in **32.0%** of 400 paired LIBERO episodes (Wilson 95% CI 27.6-36.7).
  - On the same episodes, seeds and noise, the teacher scores **96.0%** and the untrained 6-layer student scores 0.0%.
  - The teacher succeeds where the student fails in 261 pairs; the reverse happens in 5 (exact McNemar p ≈ 2e-70).
  - By suite: spatial 37, object 45, goal 21, libero_10 25 (teacher: 98 / 97 / 96 / 93).
  - The task-only control at step 3000 (no teacher loss) succeeds in **34.0%** of 200 paired episodes (5 per task, init states 0-4; Wilson 95% CI 27.8-40.8). On those same 200 episodes KD step 3000 scores 36.0% and the teacher 96.5%. KD vs task-only: 18 vs 14 discordant pairs (exact McNemar p = 0.60), so no difference, as the held-out error also shows.
- **(a) FP32 → BF16: partly yes, and it is the part that matters.** The teammate's guess is correct in substance: most weights were never trained.
  - The problem is not bf16 as such. openpi's PyTorch trainer, run with `--pytorch-training-precision bfloat16`, keeps the master weights and the AdamW state in bf16, so every update smaller than half a bf16 spacing rounds away.
  - At LR ≤ 2.5e-5, **81.5% of the student's transformer weights stayed bit-identical to their initial values** in all four checkpoints.
  - bf16 inference is *not* a cause: fp32 inference of the same checkpoint changes the held-out action error by 0.4%. The bf16 teacher conversion is not a cause either.
- **(b) Root causes, ranked:**
  1. bf16 master weights drop the updates. Strong, direct evidence, plus a laptop counterfactual where fixing only this gives 30% lower loss, 1.7× the held-out improvement, and 86% instead of 12% of weights moving.
  2. 3,000 steps is short (0.7 epoch, a tenth of the paper's run). Real, but it compounds cause 1: under bf16 the decaying LR froze ever more weights. These checkpoints cannot separate the two.
  - The load path, config, norm stats, preprocessing, prompt, KD code, data and init were all checked and are not causes.
- **(c) Fix:** fp32 master weights in `scripts/distill_pytorch.py`, plus a config flag in `src/openpi/training/config.py` (prototyped, unit-tested, resume-tested, and run in the laptop A/B). Also freeze the parameters that never get a gradient, and correct the bf16 recommendation in `README.md`.
- **(d) Recommendation for the next OVX run: option 1 (bf16 compute + fp32 master weights).**
  - Batch 64 and LR stay as before.
  - Cost: +5.5 GB (about 44 GB reserved of the L40S's ~49.5 usable) and about 1% time: KD 6.4 h, task-only 4.0 h per 3k steps.
  - Run 3k steps first, because that isolates the fix.
  - Full fp32 (with gradient accumulation, or CLI-only at batch 16 / 12k steps) also fits but costs about 2× the GPU hours. The CLI-only variant changes the batch and LR at the same time, so the result can't be attributed to precision alone.
- **(e)** The follow-up runbook needs: the fixed code, a gate showing the fix is active (a weight-change check against init), the extended freeze list, a memory/speed gate with master weights, the run matrix, and export rules. Details in (e).
- **Open decision at the time of writing:** which precision fix the follow-up runbook carries. Option 1 is recommended; option 3 (CLI-only full fp32) is the no-new-code alternative. Runbook 2 carries option 1, with option 3 pre-written as its fallback (`P5F`).
- **Work that should ship:** the (c) 1-2 fix plus `scripts/distill_pytorch_test.py`. Option 1 needs it in the next OVX tarball. The runbook and README changes in (c) 3-4 can go with it. (Since merged as commit `ae0c007`, with the README correction; the (c) 3 freeze list is in runbook 2 and in the 6-layer configs.)
- **Laptop-only limitation:** the full 1.07 B-trainable student cannot be trained on the 16 GB laptop. So the success rate a fixed run will reach is a prediction, not a measurement; the first fixed OVX run is the measurement.

---

## 0. What I did

| Step | What | Result |
|---|---|---|
| Inputs | Extracted `pi05_libero_l06_weights_20261022.zip` (16.45 GB, stored) into a local artifacts directory with Python `zipfile` after a path check. Nothing in it was executed | `sha256sum -c SHA256SUMS`: all 19 entries `OK` |
| Norm stats | Compared each checkpoint's `assets/physical-intelligence/libero/norm_stats.json` with the teacher's | All 4 byte-identical to the teacher's (sha256 `b3a44bb2…`, the value the runbook checks) |
| Training logs | Loss, LR and grad-norm curves of both runs | (b) 2 |
| Code path | Training: `scripts/distill_pytorch.py`, `pi0_pytorch.py`, `gemma_pytorch.py`, `config.py`. Eval: `policy_config.py`, `serve_policy_seeded.py`, `libero_eval.py` | (a) 1, (b) 4 |
| dtype census | Built the student exactly as training does with `--pytorch-training-precision bfloat16`; ran one AdamW step | (a) 1 |
| Weight diff | Every tensor of all 4 checkpoints vs the phase-1 student init, per group and per magnitude binade | (a) 3 |
| Offline action check | `scripts/phase1/action_agreement.py` on the 457 held-out frames, against the phase-1 teacher actions | (b) 2 |
| Decomposition | Hybrid checkpoints: the init plus only the fp32-trained or only the bf16-trained changes of KD step 3000 | (a) 5 |
| Simulator | `scripts/phase1/eval_libero.sh`, KD step 3000: 400 paired episodes, the identical protocol to phase 1. Task-only step 3000: 200 paired episodes (5 per task). The WSL restart interrupted it after 4 of 40 tasks; the harness writes one file per finished task and skips those on rerun, so it resumed from task 5 with the same seeds | Bottom line |
| Load path | Strict-loaded a checkpoint exactly as the eval server does; compared every tensor with the file | (b) 3-4 |
| Precision A/B | Inference: the same checkpoint in fp32 vs bf16, and with exact fp32 norms. Training: laptop runs from the same init, seed, data and noise, pure-bf16 AdamW vs the prototyped fp32 master weights | (a) 4, (b) 3 |
| Fix | Prototype, 2 unit tests (CPU + CUDA), resume test, ruff 0.8.6 (the version pinned in pre-commit) | (c) |
| Cost model | Memory slope bf16 vs fp32 and an optimizer-step benchmark on the laptop, combined with the measured OVX numbers | (d) |

Laptop: RTX 5080 Laptop 16 GB under WSL2, the phase-1 environment and eval harness. The A/B training and the first eval ran at the same time on the one GPU; that slowed both but does not affect their numbers (fixed seeds).
The one-off analysis scripts cited below (the `*.py` / `*.sh` names, e.g. `tensor_diff.py`) are not in the repository: a WSL restart on 2026-10-22 deleted them, and their outputs are quoted here. The one script the next run needs (the fix-is-active gate) is reproduced in full in (e) 2 and committed as `scripts/gate_check.py`.
The raw outputs (JSON / npz) were not kept; the hybrid checkpoints stayed on the laptop.

---

## (a) Was FP32 → BF16 the cause?

**Partly yes, and the teammate's guess about the mechanism is right: most of the student's weights were never trained.**
The cause is not "bf16" in general. It is one specific use of it: `--pytorch-training-precision bfloat16` makes openpi's PyTorch trainer keep the *master weights and the AdamW state* in bf16.
Every AdamW step smaller than half a bf16 ulp of the weight rounds back to the old value.
At this learning rate (peak 2.5e-5, decaying to 2.5e-6) that is the case for 81.5% of the 6-layer student's used transformer weights, which are bit-identical to their initial values in all four checkpoints.
The other uses of bf16 are not the cause:

| Use of bf16 | Cause? | Evidence |
|---|---|---|
| bf16 master weights + bf16 AdamW state during training | **Yes** | 1-4 below |
| Teacher converted to bf16 (`convert_teacher.log`: `dtype='bfloat16'`) | No | The phase-1 teacher is the same bf16 conversion and scores 96.0%. The task-only run never uses the teacher and fails just like the KD run. The trained checkpoints' 412 M vision-tower weights and all large layer weights are bit-identical to the laptop init built from the laptop teacher, so the OVX conversion produced the same tensors |
| bf16 compute at inference | No | The same KD step-3000 checkpoint, served entirely in fp32: held-out action MSE 0.0250 vs 0.0251 in bf16 (paired difference −0.0001 [−0.00016, −0.00004], 0.4%). The two precisions' actions differ from each other by MSE 1.3e-5, 1/2000 of the student-to-teacher error |

### 1. Where precision is set, and what the parameters and optimizer state really are

- The runbook passes `--pytorch-training-precision bfloat16` (`phase2-cloud-runbook.zh-TW.md:286`, `run_one.sh`). That overrides the fork's own choice of `pytorch_training_precision="float32"` for these configs (`src/openpi/training/config.py:822, 849`).
- `scripts/distill_pytorch.py:408, 416` copy that value into both the student's and the teacher's `dtype`.
- `src/openpi/models_pytorch/gemma_pytorch.py:63-83` (`to_bfloat16_for_selected_params`) then casts the whole PaliGemma + action-expert stack to bf16. The exceptions are parameters whose names contain `input_layernorm`, `post_attention_layernorm` or `model.norm` (which includes the expert's adaRMS dense layers) and the SigLIP patch/position embeddings.
- `action_in_proj`, `action_out_proj` and `time_mlp_*` are plain `nn.Linear` outside that stack and stay fp32 (`pi0_pytorch.py:100-105`).
- `torch.optim.AdamW` (`distill_pytorch.py:521-527`) creates `exp_avg` and `exp_avg_sq` with `zeros_like(param)` (`torch/optim/adam.py:176-180`), so bf16 parameters get bf16 moments, and `optim.step()` (`:605`) writes the update directly into the bf16 weight.
- openpi's README states this is intended: PyTorch training "supports either full bfloat16 (default) or full float32 … Mixed precision is not yet supported" (`README.md:548`). The JAX trainer that produced the teacher keeps fp32 weights (`README.md:544`).

Built exactly like training with `--pytorch-training-precision bfloat16` (`dtype_check.py`; CPU, one AdamW step at lr 2.5e-5):

```
trainable transformer matrices         torch.bfloat16     764.4M
trainable expert lm_head (unused)      torch.bfloat16     263.3M
trainable adaRMS/RMSNorm               torch.float32       41.0M
trainable time_mlp_in / time_mlp_out   torch.float32        1.0M + 1.0M
trainable action_in_proj/out_proj      torch.float32        0.03M + 0.03M
param bf16 (2048, 2048): exp_avg bf16, exp_avg_sq bf16; fraction of elements changed by one lr=2.5e-5 step: 0.302
param fp32 (3072, 1024): exp_avg fp32, exp_avg_sq fp32; fraction of elements changed by one lr=2.5e-5 step: 1.000
```

On Adam's first step `|m̂/√v̂| = 1`, so every element receives exactly the largest update AdamW ever makes (`lr`).
Even so, 70% of a bf16 q_proj's elements did not move.
No parameter group was frozen beyond the four intended prefixes (vision tower, projector, token embedding, VLM lm_head: 941.5 M, log line `Froze 941.5M params`).

### 2. The arithmetic: update size vs bf16 spacing

bf16 keeps 8 significant bits, so the spacing between neighbouring values at magnitude |w| is 2^(⌊log2|w|⌋−7).
An update smaller than half of that rounds back to w.
An AdamW update is at most about `lr` per element, and smaller when gradients are noisy.
Over the student-init weights (`ulp_analysis.py`):

| Update size | Fraction of VLM-layer elements where it rounds away | Expert-layer elements |
|---|---|---|
| 2.5e-5 (peak LR, step 300) | 81.7% | 81.1% |
| 1.0e-5 (LR at step ~1,950) | 90.7% | 90.4% |
| 2.5e-6 (final LR) | 97.7% | 97.6% |

The median |w| of these layers is 0.023, and the bf16 half-spacing there is 6.1e-5, 2.4× the peak LR.

### 3. Direct evidence: per-tensor diff of every checkpoint against the student init

`tensor_diff.py` / `binade.py` compare every tensor of each checkpoint with `checkpoints/pi05_libero_l06_init` (phase 1; the same layer mapping, verified bit-identical where untrained).
Fraction of elements changed after 3,000 steps (for fp32-trained tensors, "changed" is counted at bf16 resolution, so it undercounts) (KD run; task-only is within 0.01 everywhere), by tensor group:

| Group | dtype in training | Elements | Changed | Relative L2 change |
|---|---|---|---|---|
| Vision tower, projector, token embedding | bf16, frozen | 941 M | 0.0000 | 0 |
| VLM layers 0-4 MLP | bf16 | 503 M | 0.105-0.145 | 1.0-1.3% |
| VLM layers 0-4 attention | bf16 | 47 M | 0.135-0.230 | 1.2-2.4% |
| VLM layer 5 MLP, q/o proj, post-attention norm; VLM final norm | bf16 / fp32 | 109 M | 0.0000 | 0 (no gradient: their output is never read, see (b) 4) |
| Expert layers 0-5 MLP + attention | bf16 | 104 M | 0.129-0.163 | 1.2-1.6% |
| Expert adaRMS dense (input/post) | fp32 | 37.8 M | 0.971-0.982 | 4.2-8.4% |
| Expert final adaRMS norm | fp32 | 3.1 M | 0.653 | 7.0% |
| time_mlp_in / time_mlp_out | fp32 | 2.1 M | 0.947 / 0.979 | 4.8% / 7.2% |
| action_out_proj | fp32 | 0.03 M | 0.986 | 9.9% |

The same bf16 matrices, by magnitude binade (init → step 3000, and step 1500 → step 3000):

| \|w\| at init | bf16 half-spacing | Share of elements | Changed, init → 3000 (LR ≤ 2.5e-5) | Changed, 1500 → 3000 (LR ≤ 1.58e-5) |
|---|---|---|---|---|
| < 1.95e-3 | ≤ 3.8e-6 | 4.6% | ≥ 99.8% | ≥ 98.7% |
| [1.95e-3, 3.91e-3) | 7.6e-6 | 4.7% | 99.2% | 63.0% |
| [3.91e-3, 7.81e-3) | 1.53e-5 | 9.1% | 49.1% | 0.85% |
| [7.81e-3, 1.56e-2) | 3.05e-5 | 17.3% | **0.16%** | 0.00% |
| ≥ 1.56e-2 | ≥ 6.1e-5 | 64.2% | **0.0000%** | 0.0000% |
| All used matrices | | 655 M | 13.8% | 7.1% |

The cutoff is exactly where the half-spacing crosses the learning rate: 1.53e-5 < 2.5e-5 lets about half through, 3.05e-5 > 2.5e-5 lets almost nothing through.
In the second half, where the LR is ≤ 1.58e-5, the cutoff moves down by exactly one binade.
**81.5% of the student's used transformer weights (534 M of 655 M) are bit-identical to the init.**
The set of changed elements is the same at step 1500 and step 3000 (11.85% of all 764 M bf16 matrix elements in both): after step 1500 no new weight started to move.
What did learn is the 43 M fp32 parameters and the smallest-magnitude 14% of the bf16 matrices.

### 4. Counterfactual on the laptop: the same training with fp32 master weights

Setup (`ab_precision.sh`): two runs of the real `scripts/distill_pytorch.py`, identical in everything except `--no-pytorch-fp32-master-weights` vs `--pytorch-fp32-master-weights`.
- Shared settings: config `pi05_libero_l06_taskonly`, the same layer-mapped init, `--pytorch-training-precision bfloat16`, seed 42, so the same batches, noise and flow times; batch 4, 400 steps, LR warmup 40 → 2.5e-5 → cosine to 2.5e-6, 8 local LIBERO episodes.
- The VLM backbone is frozen as well, because the laptop cannot hold the full 1.07 B-trainable student. The 146.9 M trainable parameters are the 6 action-expert layers (bf16 matrices + fp32 adaRMS), the time MLP and the action projections.

| | Pure bf16 (as on OVX) | fp32 master weights (fix) |
|---|---|---|
| Expert matrix elements changed vs init | **12.2%** (0.09% in \|w\| ∈ [0.0078, 0.0156), 0% above: the OVX signature) | **86.4%** (78% in \|w\| ∈ [0.031, 0.062), 59% in [0.062, 0.125)) |
| Training loss at steps 1-2 | 0.5442, 0.2119 | 0.5442, 0.2119 (identical, as expected) |
| Mean loss, steps 51-100 / 101-200 / 201-300 / 301-400 | 0.124 / 0.088 / 0.054 / 0.056 | 0.087 / 0.061 / 0.040 / 0.039 (**−30%**) |
| Paired per-step loss difference, steps 101-400 | | −0.0193 [−0.0223, −0.0168]; the fix is lower on 96-99% of steps |
| Held-out action MSE vs teacher (457 frames; init 0.181) | 0.134 [0.120, 0.149] | **0.100** [0.088, 0.112]; paired −0.034 [−0.044, −0.024] |
| Gripper sign agreement | 85.0% | 90.5% |
| Allocated after backward | 12.75 GB | 13.75 GB (+1.00 GB; predicted from 4 B master + 4 B Adam per bf16 parameter: +1.00 GB) |

With the same steps and data, the fix lets the network use the weights it was supposed to train.
After 400 steps that is a 1.7× larger reduction of the error to the teacher (0.181 → 0.100 instead of 0.181 → 0.134).
The pure-bf16 arm reproduces the OVX weight-change signature exactly, binade for binade.

While building this A/B I hit and fixed a PyTorch bug.
`torch._foreach_copy_` on CUDA (torch 2.7.1+cu128) writes garbage into the fp32 entries of a destination list that mixes bf16 and fp32 tensors: a 5-line repro returned `[8.0158, 2.3e-41, 0.0]` instead of `[7, 8, 9]`.
My first prototype used it to copy the masters back, and that run's loss jumped to 0.93 after step 1.
The fix copies tensor by tensor, and the unit test now covers a mixed bf16/fp32 list on CUDA.
The laptop memory slope and resume test also ran on the fixed code (see (d) and (c)).

### 5. Does it explain the low success rate?

Observed facts:
- The student reproduces the "very low" success: 32.0% vs 96.0%.
- 81.5% of its used transformer weights never left their initial values, because of the optimizer precision.
- In a run where they could move, those weights would have moved substantially. The bf16 weights small enough to record every update (|w| < 0.002) travelled a median 1.1e-3 (RMS 1.6e-3) over the run. A weight's gradient does not depend on its own value, so the frozen weights would have travelled about as far: about 5% of the median |w| of 0.023, and above their bf16 half-spacing in 97% of cases (`delta_mag.py`).
- The weights that could move mattered as much as the fp32 parameters did. Decomposition of KD step 3000 (held-out action MSE):

| Checkpoint | MSE [95% CI] |
|---|---|
| Untrained init | 0.181 [0.167, 0.195] |
| Init + only the fp32 tensors from step 3000 (43 M trained parameters) | 0.095 [0.085, 0.106] |
| Init + only the bf16 changes from step 3000 (the 14% of elements that moved) | 0.089 [0.081, 0.097] |
| Both (the actual checkpoint) | 0.025 [0.021, 0.030] |

  Each half alone recovers about half of the error, and together they reach 0.025, so the two co-adapted.
  The bf16 matrices *had* to change; only a sliver of them could.
- With everything else held fixed, turning on fp32 master weights gives a markedly better student per step (§4).
- The proven path, the fork's fp32 recipe as published, reaches 93-95% at 30k steps.

Inference (not measured, because the laptop cannot train the full student): the bf16 optimizer is the main, directly evidenced cause, and it also caps what more steps could buy.
How much success a 3k-step fixed run recovers is still open.
If it lands near the teacher, precision was essentially the whole cause.
If it improves a lot but stays well below, the budget (cause 2) is the remaining gap and the next lever is 10k+ steps.

---

## (b) Root causes, ranked by strength of evidence

Per the diagnostic procedure, the three facts kept separate:
- **Initiating trigger:** training with openpi's pure-bf16 mode (`--pytorch-training-precision bfloat16`: bf16 master weights and bf16 AdamW state) at learning rates ≤ 2.5e-5.
- **Masking condition:** 43 M parameters are still stored in fp32, and the smallest 14% of the bf16 weights can still move. So the loss falls (task-only 0.546 → 0.022), the held-out action error improves 7× over the untrained student, and the run looks healthy but undertrained. The phase-1 smoke tests (10-20 steps) could not have shown it, because the loss falls in the first steps either way. LR decay then deepens the freeze over the run.
- **Visible symptom:** a 2.5× faster student whose LIBERO success is 32.0% against the teacher's 96.0% on the same 400 episodes.

Proven path for comparison: the recipe the fork ships trains these configs in fp32 (`config.py:822, 849`).
The JAX trainer that produced the π0.5 teacher keeps fp32 weights with bf16 compute (`README.md:544`).
In both, an AdamW update of any size lands in the weight.
The earliest divergence between that path and the failed run is the dtype of the tensor `optim.step()` writes into (`distill_pytorch.py:605`).

### 1. bf16 master weights drop most updates (strong: direct evidence and a reproduction)

- **Where:**
  - `phase2-cloud-runbook.zh-TW.md:286` passes the flag, which overrides `config.py:822, 849`;
  - `scripts/distill_pytorch.py:408, 416` turn it into the model dtype;
  - `src/openpi/models_pytorch/gemma_pytorch.py:63-83` casts all but the norms to bf16;
  - `scripts/distill_pytorch.py:511, 521-527, 605` run AdamW directly on the bf16 parameters, and torch's AdamW creates bf16 moments (`adam.py:176-180`).
- **History:** the bf16 choice came from this project's own phase-1 plan (`README.md:183-184, 227`, "bf16 (`--pytorch-training-precision bfloat16`)"), made to fit memory, and the runbook carried it over. The fork's authors trained in fp32. The upstream README documents that PyTorch bf16 has no mixed precision (`README.md:548`).
- **Evidence:**
  - the dtype census and the one-step test (§(a) 1);
  - the cutoff between moved and frozen weights lands exactly at the binade where the bf16 half-spacing exceeds the LR, in both runs and both checkpoints, and it moves down one binade when the LR halves (§(a) 3);
  - a laptop run from the same init reproduces the same signature, and turning on fp32 master weights removes it (§(a) 4);
  - the hybrid decomposition shows the few bf16 weights that could move carried about half of the gain, which suggests the frozen 81.5% were not idle capacity (§(a) 5).
- **Effect:** the 18 → 6 layer cut removes two thirds of each stack, so the six kept layers have to change to work as a 6-layer network. Instead, 81.5% of their weights stayed exactly at the teacher-layer values. Adaptation happened only through the 43 M fp32 conditioning and projection parameters and the smallest weights.

### 2. Budget and schedule (medium: real, but inseparable from cause 1 in these checkpoints)

3,000 steps × 64 = 0.7 epoch, a tenth of the paper's 30k steps × 64 (Shallow-π Table 1a setting).
Loss, averaged over 500-step windows from the two logs:

| Steps | 0-499 | 500-999 | 1000-1499 | 1500-1999 | 2000-2499 | 2500-2999 |
|---|---|---|---|---|---|---|
| KD (task + KD) | 0.2593 | 0.0519 | 0.0452 | 0.0421 | 0.0410 | 0.0406 |
| Task-only | 0.1320 | 0.0279 | 0.0245 | 0.0230 | 0.0224 | 0.0222 |

Held-out action MSE vs the teacher (457 frames, the same noise; lower is better; teacher-vs-itself noise floor 0.0086, untrained student 0.181):

| Checkpoint | MSE [95% CI] | Median | Gripper sign agreement |
|---|---|---|---|
| KD step 1500 | 0.0282 [0.0236, 0.0333] | 0.0117 | 97.5% |
| KD step 3000 | 0.0251 [0.0206, 0.0302] | 0.0096 | 97.8% |
| Task-only step 1500 | 0.0272 [0.0227, 0.0323] | 0.0110 | 97.6% |
| Task-only step 3000 | 0.0245 [0.0201, 0.0294] | 0.0094 | 97.8% |
| (teacher, other noise seed) | 0.0086 [0.0060, 0.0118] | 0.0005 | 98.6% |

Steps 1500 → 3000 still cut the error by 11% (paired bootstrap: KD −0.0031 [−0.0040, −0.0023], task-only −0.0027 [−0.0035, −0.0019]).
That happened even though the second half moved only 7.1% of the matrix elements, all with |w| < 3.9e-3.
So the student was still improving, slowly, and more steps alone would help somewhat.
These checkpoints cannot separate "too few steps" from "steps that could not move the weights".
The next run separates them: identical recipe, master weights on, the same 3,000 steps.
The student's median per-frame error (0.0096) is still 20× the teacher's own noise-floor median (0.00048).

### 3. Inference rounds the fp32-trained adaRMS/RMSNorm weights through bf16 (checked: negligible)

`policy_config.py:54-55` builds the model with `load_pytorch` and then calls `to_bfloat16_for_selected_params("bfloat16")` again.
That function first does `self.to(bfloat16)` (`gemma_pytorch.py:65`) and only afterwards up-casts the "keep fp32" parameters.
The 37 fp32-trained norm tensors (expert adaRMS dense layers, VLM RMSNorm scales) therefore reach inference rounded to bf16 precision (`load_check.py`: 37 tensors differ from the file after loading).
The fp32 time MLP and action projections are outside that stack and stay exact.
Measured effect: none that matters. Re-loading the exact fp32 values after the cast changes the held-out MSE by −0.00004 [−0.00006, −0.00001], and full fp32 inference by −0.0001. No change is needed; it is listed only because the brief asked about silent load-time conversions.

### 4. What was checked and ruled out

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Eval builds the wrong architecture, or keys are dropped on load | No | `create_trained_policy` → `PI0Pytorch(pi05_libero_l06)` → `safetensors.torch.load_model(strict=True)`: missing set(), unexpected [], 6 VLM + 6 expert layers, action horizon 10, action dim 32, max_token_len 200, pi05 (`load_check.py`). Every bf16 tensor equals the file. The held-out error of 0.025 vs 0.181 for the init shows the trained weights are the ones running |
| Norm stats, action horizon/dim, denoising steps, image preprocessing, prompt differ between train and eval | No | Norm stats in all 4 checkpoints are byte-identical to the teacher's (`b3a44bb2…`), which is what training loaded (`kd_*.log` data_config). Eval builds its transforms from the same `train_config.data` (`policy_config.py:58-89`): `LiberoInputs`, `ResizeImages(224)`, `TokenizePrompt(discrete_state_input=False)`, `PadStatesAndActions(32)`, quantile norm. The prompt is the task language string in both (training `prompt_from_task=True`; client `str(task.language)`). 10 denoising steps; the client executes 5 of each 10-action chunk |
| Teacher conversion to bf16 | No | See (a). Task-only does not use the teacher and behaves the same |
| KD loss or distillation code bug | No (but KD adds nothing yet) | KD and task-only checkpoints are within 0.0006 MSE of each other, so the KD-specific path neither broke nor helped training. The paired difference is 0.0005 [0.0003, 0.0008] in favour of task-only. Closed loop on the same 200 episodes: KD 36.0% vs task-only 34.0%, 18 vs 14 discordant pairs, exact McNemar p = 0.60 |
| Data pipeline | No | Full dataset (1,699 files fetched; 1,693 episodes), teacher norm stats, smooth loss curves, grad norm 0.21-0.27 (KD) / 0.11-0.14 (task-only) after warmup, so the clip at 1.0 never engages. No NaN |
| Student-init procedure | No | Log: mapping [0, 3, 7, 10, 14, 17]; 1 missing key (`embed_tokens`, tied to `lm_head` and filled through it); 240 skipped teacher keys (12 unused layers × 20 tensors). The trained checkpoints' untrained tensors are bit-identical to the phase-1 laptop init |
| Unintended frozen components | No | Only the 4 intended prefixes (941.5 M). Parameters with no gradient by construction: the last VLM layer's q/o projections, MLP and post-attention norm, plus the final VLM norm. The action expert reads only each VLM layer's K/V, so these outputs are never consumed, exactly as in the teacher; the diff shows them bit-identical. Also the expert `lm_head` (263 M, never called). All of them should be frozen explicitly (see (c) 3) |
| LR schedule misapplied | No | Logged LR matches warmup 300 → 2.5e-5 → cosine → 2.5e-6 at step 3000 |

---

## (c) Files to change, and why

The required change is fp32 master weights in the trainer (items 1-2). Items 3-4 are runbook and documentation changes, item 5 needs no change, and item 6 is only for option 2 in (d).
Items 1-2 were prototyped here and later merged as commit `ae0c007`.
Check results:
- the laptop A/B in (a) 4 runs them;
- `scripts/distill_pytorch_test.py` (2 tests × CPU/CUDA) passes;
- a resume test (6 steps, then `--resume` to 8) restored the masters from `master_params.pt` without the fallback warning;
- `ruff` 0.8.6 check and format are clean.

The test file is not in the diff below.
It asserts two things:
- 100 AdamW steps at lr 2.5e-5 leave a bf16 weight of 0.05 bit-identical (the bug);
- with `fp32_master_weight_step` the same steps move it by -2.5e-3 and keep a mixed bf16/fp32 list correct (the fix and the `_foreach_copy_` trap).

1. **`scripts/distill_pytorch.py`: keep an fp32 master copy of every trainable parameter and run AdamW on the masters.**
   Forward and backward still use the bf16 weights, so step time and activation memory do not change, and neither does the checkpoint format that eval loads.
   After each step the masters are rounded back into the bf16 weights.
   Updates now accumulate in fp32 until they cross a bf16 rounding boundary instead of being dropped at every step.
   The masters are also saved as `master_params.pt` next to `optimizer.pt` and restored on `--resume`, so a resumed run continues exactly.
2. **`src/openpi/training/config.py`: new field `pytorch_fp32_master_weights` (default `False`, so upstream configs are unchanged), set to `True` in `pi05_libero_l06` and `pi05_libero_l06_taskonly`.**
   With the field set, the runbook's existing `--pytorch-training-precision bfloat16` gets master weights with no new flag.
   `--no-pytorch-fp32-master-weights` reproduces the broken behaviour (that is how the laptop A/B ran its control arm).
3. **Runbook `run_one.sh` / `FREEZE` (phase2-cloud-runbook.zh-TW.md:219, 286): also freeze the parameters that never receive a gradient.**
   They are `paligemma_with_expert.gemma_expert.lm_head.` (263 M, the action expert's unused LM head), `paligemma_with_expert.paligemma.model.language_model.norm.`, and the last VLM layer's `mlp.`, `self_attn.q_proj.`, `self_attn.o_proj.` and `post_attention_layernorm.` (`…language_model.layers.5.`).
   Their outputs are never read by the action expert, and the diff confirms they stayed bit-identical in both runs.
   Freezing them changes nothing about learning, and with master weights it saves 1.5 GB of useless fp32 copies.
4. **`README.md:183-184, 227`: correct the phase-2 recommendation.**
   It recommended `--pytorch-training-precision bfloat16` on the assumption that this was only a memory/precision trade-off.
   In openpi's PyTorch trainer it means bf16 master weights (upstream README.md:548: "full bfloat16 … Mixed precision is not yet supported").
   This repository's own phase-1 plan made that choice, not the teammate.
5. **Not needed: `src/openpi/policies/policy_config.py:54-55`.** It rounds the fp32-trained norm weights through bf16 at load, but the measured effect is −0.00004 MSE ((b) 3). Leave it.
6. **Only if option 2 in (d) is chosen: gradient accumulation in `scripts/distill_pytorch.py:592-606`.**
   The loop does one `backward`, one `step` and one `zero_grad` per loader batch, and `TrainConfig` has no accumulation field.
   The change: a `pytorch_grad_accum_steps` field, loader batch `batch_size // accum`, `loss / accum`, step and log every `accum` micro-batches.

Full diff of items 1-2 against `3425187`, as prototyped (the merged commit `ae0c007` differs in detail):

```diff
diff --git a/scripts/distill_pytorch.py b/scripts/distill_pytorch.py
index 84085d1..1357f7b 100755
--- a/scripts/distill_pytorch.py
+++ b/scripts/distill_pytorch.py
@@ -147,7 +147,29 @@ def get_model_parameters(model):
     )
 
 
-def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
+def init_fp32_master_weights(params):
+    """(vla_distill) fp32 copies of `params`, for the optimizer to update instead of the (bf16) parameters."""
+    return [p.detach().clone().float() for p in params]
+
+
+def fp32_master_weight_step(optimizer, params, master_params):
+    """(vla_distill) One optimizer step on fp32 master weights, rounded back into the (bf16) parameters.
+
+    Pure-bf16 AdamW drops every update smaller than half a bf16 ulp of the weight; the masters accumulate them.
+    """
+    for p, mp in zip(params, master_params, strict=True):
+        mp.grad = None if p.grad is None else p.grad.float()
+        p.grad = None
+    optimizer.step()
+    optimizer.zero_grad(set_to_none=True)
+    # Per tensor, not torch._foreach_copy_: on CUDA (torch 2.7.1) that writes garbage into the fp32 entries of a
+    # list mixing bf16 and fp32 destinations.
+    with torch.no_grad():
+        for p, mp in zip(params, master_params, strict=True):
+            p.copy_(mp)
+
+
+def save_checkpoint(model, optimizer, global_step, config, is_main, data_config, master_params=None):
     """Save a checkpoint with model state, optimizer state, and metadata."""
     if not is_main:
         return
@@ -169,6 +191,9 @@ def save_checkpoint(model, optimizer, global_step, config, is_main, data_config)
 
         # Save optimizer state using PyTorch format
         torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")
+        # (vla_distill) fp32 master weights (only for --resume; model.safetensors keeps the bf16 weights used for eval)
+        if master_params is not None:
+            torch.save([p.detach().cpu() for p in master_params], tmp_ckpt_dir / "master_params.pt")
 
         # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
         metadata = {
@@ -195,7 +220,7 @@ def save_checkpoint(model, optimizer, global_step, config, is_main, data_config)
             wandb.log({"checkpoint_step": global_step}, step=global_step)
 
 
-def load_checkpoint(model, optimizer, checkpoint_dir, device):
+def load_checkpoint(model, optimizer, checkpoint_dir, device, trainable=None, master_params=None):
     """Load the latest checkpoint and return the global step."""
     checkpoint_steps = [
         int(d.name)
@@ -243,6 +268,19 @@ def load_checkpoint(model, optimizer, checkpoint_dir, device):
 
         optimizer.load_state_dict(optimizer_state_dict)
         del optimizer_state_dict
+
+        # (vla_distill) Re-sync the fp32 master weights: from master_params.pt if saved, else from the bf16 weights.
+        if master_params is not None:
+            master_path = ckpt_dir / "master_params.pt"
+            if master_path.exists():
+                saved = torch.load(master_path, map_location=device, weights_only=True)
+            else:
+                logging.warning(f"No {master_path}; re-initializing fp32 master weights from the bf16 model weights")
+                saved = [p.detach() for p in trainable]
+            with torch.no_grad():
+                for mp, sp in zip(master_params, saved, strict=True):
+                    mp.copy_(sp)
+            del saved
         torch.cuda.empty_cache()
         gc.collect()
         log_memory_usage(device, latest_step, "after_loading_optimizer")
@@ -517,9 +555,19 @@ def train_loop(config: _config.TrainConfig):
     decay_steps = config.lr_schedule.decay_steps
     end_lr = config.lr_schedule.decay_lr
 
+    # (vla_distill) fp32 master weights: AdamW updates an fp32 copy of each trainable bf16 parameter (fp32 moments
+    # too), which is rounded back into the bf16 weight after every step. Pure-bf16 AdamW (the openpi default) drops
+    # any update smaller than half a bf16 ulp: at peak LR 2.5e-5 that is every weight with |w| >= 2^-7, and in the
+    # phase-2 runs 81% of the transformer weights stayed bit-identical to init. Parameters that never get a gradient
+    # (e.g. the unused expert lm_head) should be frozen, or their master copy is wasted memory.
+    master_params = None
+    if config.pytorch_fp32_master_weights and config.pytorch_training_precision == "bfloat16":
+        master_params = init_fp32_master_weights(trainable)
+        logging.info(f"fp32 master weights for {sum(p.numel() for p in master_params) / 1e6:.1f}M trainable params")
+
     # Create optimizer with config parameters
     optim = torch.optim.AdamW(
-        trainable,
+        master_params if master_params is not None else trainable,
         lr=peak_lr,
         betas=(config.optimizer.b1, config.optimizer.b2),
         eps=config.optimizer.eps,
@@ -529,7 +577,7 @@ def train_loop(config: _config.TrainConfig):
     # Load checkpoint if resuming
     global_step = 0
     if resuming:
-        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
+        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device, trainable, master_params)
         logging.info(f"Resumed training from step {global_step}")
 
     def lr_schedule(step: int):
@@ -602,8 +650,11 @@ def train_loop(config: _config.TrainConfig):
             grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=config.optimizer.clip_gradient_norm)
 
             # Optimizer step
-            optim.step()
-            optim.zero_grad(set_to_none=True)
+            if master_params is not None:
+                fp32_master_weight_step(optim, trainable, master_params)
+            else:
+                optim.step()
+                optim.zero_grad(set_to_none=True)
 
             # Clear gradients more aggressively
             for param in model.parameters():
@@ -658,7 +709,7 @@ def train_loop(config: _config.TrainConfig):
 
             global_step += 1
             # Save checkpoint using the new mechanism
-            save_checkpoint(model, optim, global_step, config, is_main, data_config)
+            save_checkpoint(model, optim, global_step, config, is_main, data_config, master_params)
 
             # Update progress bar
             if pbar is not None:
diff --git a/src/openpi/training/config.py b/src/openpi/training/config.py
index 77bc06c..5eabcad 100644
--- a/src/openpi/training/config.py
+++ b/src/openpi/training/config.py
@@ -492,6 +492,11 @@ class TrainConfig:
 
     # Precision for PyTorch training.
     pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"
+    # (vla_distill) With bfloat16 training, keep an fp32 master copy of every trainable parameter (and fp32 AdamW
+    # state) in scripts/distill_pytorch.py; forward/backward still run on the bf16 weights. Without it, an AdamW
+    # step smaller than half a bf16 ulp of the weight (|update| < ~|w|/256, i.e. most weights at LR 2.5e-5) rounds
+    # back to the old value and is lost. Costs 8 extra bytes per trainable parameter.
+    pytorch_fp32_master_weights: bool = False
     # (vla_distill) State-dict name prefixes of student parameters to freeze in scripts/distill_pytorch.py,
     # e.g. the SigLIP vision tower, to fit a smoke-test step on a small GPU. Empty = train everything.
     pytorch_freeze_prefixes: tuple[str, ...] = ()
@@ -820,6 +825,7 @@ _CONFIGS = [
         batch_size=64,
         save_interval=10_000,
         pytorch_training_precision="float32",
+        pytorch_fp32_master_weights=True,
     ),
     TrainConfig(
         # Control: identical student and data, but trained on the task (flow-matching) loss only.
@@ -847,6 +853,7 @@ _CONFIGS = [
         batch_size=64,
         save_interval=10_000,
         pytorch_training_precision="float32",
+        pytorch_fp32_master_weights=True,
     ),
     #
     # Fine-tuning Aloha configs.
```

## (d) Improvement options for the next OVX run (one L40S-48Q)

Baseline for all estimates, measured on the OVX card: pure bf16, batch 64, peak 36.03 GB allocated / 38.88 GB reserved, KD 7.66 s/step, task-only 4.69 s/step (`speed.txt`, `timed50.log`, the two run logs).
Memory components from [phase2-cloud-findings.md](phase2-cloud-findings.md) §3.2:
- teacher 7.47 GB, student 4.11 GB, AdamW 2.98 GB;
- 702.7 M parameters receive a gradient (659.5 M bf16 + 43.2 M fp32);
- student transient 0.305 GB per sample plus 1.31 GB fixed.
The card shows 49,152 MiB = 51.5 GB, and the desktop session already holds 1.4 GiB (ovx_probe.txt), so about 49.5 GB is usable after the CUDA context.

### Precision: three ways to fix it

Laptop measurements used below (real model code, gradient checkpointing, the phase-2 freeze set; `memslope.py`, `optstep_bench.py`):
- Student forward+backward transient: bf16 **0.31-0.33 GB/sample** (this reproduces the 0.305 that predicted the OVX peak to within 2%), fp32 **0.50 GB/sample**. The fixed part is 1.54 GB (bf16) / 2.93 GB (fp32), which includes the gradients (1.49 / 2.81 GB).
- fp32 forward+backward is **1.70× (batch 1) to 1.88× (batch 2)** slower than bf16. The L40S's TF32 tensor rate is half its bf16 rate (183 vs 362 TFLOPS dense), so ×1.9 is used (range 1.7-2.1). openpi already enables TF32 for fp32 matmuls (`pi0_pytorch.py:111`).
- Optimizer step for 702.7 M parameters: pure-bf16 AdamW 52 ms, fp32 master weights 109 ms (+57 ms, about +1% of a 7.66 s step).
- Master weights cost +1.00 GB in the laptop A/B, exactly the predicted 4 B (master) + 4 B (fp32 instead of bf16 Adam moments) per bf16 parameter.

| | **Option 1 (recommended): bf16 compute + fp32 master weights** (the prototype) | Option 1b: fp32 trainable weights + `torch.autocast(bf16)` | Option 2: full fp32 + gradient accumulation 4 × 16 | Option 3: full fp32, CLI only, batch 16 × 12,000 steps |
|---|---|---|---|---|
| Recipe vs the failed run | Identical (batch 64, LR, 3k steps). One change | Identical | Identical batch/LR/steps; compute precision changes too | Batch 16, LR retuned, 12k steps, precision: four changes at once |
| Code change | ~40 lines in `distill_pytorch.py` + 1 config field (done, tested) | More: build the student in fp32, cast the frozen modules to bf16, wrap the student forward in autocast. The model has explicit bf16 dtype branches (`pi0_pytorch.py:332-337, 553-558`, `gemma_pytorch.py:220-230`) that take the other path with fp32 weights, and that path is untested here | ~25 lines (accumulation loop + config field); optional: keep the teacher in bf16 (`distill_pytorch.py:416`) | None (`966da5c` as shipped) |
| Peak GPU memory, allocated → reserved (est.) | 36.03 + 5.45 = **41.5 → ~44.3 GB** with the unused parameters frozen ((c) 3); 43.0 → ~45.8 GB if not | +5.3 GB static, up to +1.3 GB more during the forward for autocast's bf16 weight copies: ~41.3-42.6 → ~44-46 GB | fp32 teacher 14.47 + fp32 student 8.06 + fp32 Adam 5.62 + 2.93 + 16 × 0.50 = **39.1 → ~42.2 GB**. 2 × 32 does not fit (47.2) unless the teacher stays bf16 (40.2 → ~43.4) | Same as option 2: **39.1 → ~42.2 GB** at batch 16; 37.1 → ~40.0 at batch 12; 41.1 → ~44.4 at batch 20 |
| s/step on one L40S (KD / task-only) | **7.73 / 4.77** (7.66 / 4.69 measured + optimizer) | ≈ 7.75 / 4.8 | ≈ 14.7 / 9.0 per batch-64 step (≈ 11.9 for KD with a bf16 teacher) | ≈ **3.74 / 2.33** per batch-16 step |
| Hours for 192 k samples (= the failed runs' 3k × 64): KD / task-only / both on one card / both in parallel on two cards | **6.4 / 4.0 / 10.4 / 6.4** | ≈ 6.5 / 4.0 / 10.5 / 6.5 | ≈ 12.3 / 7.5 / 19.8 / 12.3 (KD 9.9 h with a bf16 teacher) | ≈ **12.5 / 7.8 / 20.2 / 12.5** (KD 11.2-13.8 h for a slowdown of 1.7-2.1×) |
| Train/eval numerics | bf16 forward = exactly what eval runs | Close to openpi's JAX recipe | fp32/TF32 in training, bf16 in eval (openpi's JAX path has the same gap; small) | Same as option 2 |
| Exported checkpoint | 4.11 GB, unchanged; `master_params.pt` (2.8 GB) stays on OVX for resume | ~5.4 GB (fp32 trainable weights) | 8.06 GB (fp32), or cast to bf16 before export | 8.06 GB, or cast before export |
| Main risk | New code; mitigated by the unit tests, the laptop A/B, the resume test and the fix-is-active gate in (e) | Autocast path untested with this model's dtype branches | New code and 2× cost | See below |

Exact overrides for option 3 (no code change). Also add the unused parameters to `FREEZE` ((c) 3); with fp32 they would get no AdamW state anyway, but they would hold 1.5 GB of fp32 weights in the optimizer's param list:

```bash
--pytorch-training-precision float32 --batch-size 16 --num-train-steps 12000 --save-interval 6000 \
--lr-schedule.peak-lr 1.25e-5 --lr-schedule.warmup-steps 1200 --lr-schedule.decay-steps 12000 --lr-schedule.decay-lr 1.25e-6 \
--pytorch-freeze-prefixes $FREEZE
```

Why these values:
- **12,000 steps** sees the same 192 k samples as 3,000 × 64, so per-sample cost is comparable.
- **Peak LR 1.25e-5** follows the square-root scaling rule for Adam: the batch drops by κ = 4, so the LR is divided by √4 (Malladi et al., NeurIPS 2022, "On the SDEs and Scaling Rules for Adaptive Gradient Algorithms").
  - Linear scaling (6.25e-6) is the conservative alternative.
  - The full rule would also set β1 → 0.975 and β2 → 0.9875 (`--optimizer.b1 0.975 --optimizer.b2 0.9875`), so the moment averages span the same number of samples. I would leave the betas at default for a first run, to limit changes.
- **Warmup 1,200 steps** covers the same 76.8 k samples as the original 300 × 64.
- **Cosine to 1.25e-6** keeps the original 10:1 peak-to-final ratio.
- Gradient noise at batch 16 is about 2× that at batch 64, so expect grad norms of about 0.4-0.55 (0.21-0.27 measured at batch 64). The clip at 1.0 should stay mostly inactive; if the logged `grad_norm` sits near 1.0, the LR is too high.
- `--pytorch-training-precision float32` also makes the teacher fp32 (`distill_pytorch.py:416`): 14.5 GB instead of 7.5 GB and about 1.9× slower teacher forwards. Both are included in the numbers above.

Risks of option 3 compared with options 2 and 1:
1. **Confounded result.** Precision, batch, LR and step count all change at once. A good result cannot be credited to the precision fix; a mediocre one cannot tell "fp32 did not help" from "the LR heuristic was off". Options 1 and 2 keep the recipe and change only precision.
2. **Cost.** About 2× the GPU-hours of option 1 (KD 12.5 h vs 6.4 h for the same samples), because fp32/TF32 halves the tensor-core rate and doubles the bytes moved, including for the teacher, which gains nothing from fp32.
3. **Memory headroom.** About 42 GB reserved of ~49.5 usable; 7 GB of it is the fp32 teacher, which has no use for fp32. Many small steps make allocator fragmentation more likely over 12k steps. Resume works if it OOMs, but costs a restart.
4. **Recipe drift from the paper** (batch 64), which matters if results are compared with Shallow-π's numbers.
5. Upside: no new code, an upstream-supported mode ("full float32", `README.md:548`), and it is how the fork's authors configured these runs (`config.py:822, 849`). It also runs on the teammate's existing workspace without a new tarball.

Gradient accumulation (option 2) removes risks 1 and 4 at the same cost as option 3, but needs a code change.
Mixed precision via master weights (option 1) removes risks 1-4 and halves the cost, for a code change that is already written and tested.
**Recommendation: option 1.** Use option 3 only if running new code on OVX is not acceptable, and say in the results that the recipe changed.

### Other levers (after the precision fix)

| Lever | Expected effect | Cost on one L40S | Recommendation |
|---|---|---|---|
| **Steps.** 3k → 10k (or the paper's 30k) | With master weights every step moves every weight, so more steps should keep lowering the error. Without the fix, extra steps buy little (the second half of this run changed 7% of the weights). The paper reaches 93-95% at 30k × 64 | Option 1: KD 2.14 h and task-only 1.32 h per 1k steps. 10k: 21 h + 13 h; 30k: 64 h + 40 h | First a 3k validation run (the same steps as now, so the result isolates the fix), then 10k if it confirms |
| **KD vs task-only.** At 3k steps KD helped nothing (held-out MSE 0.0251 vs 0.0245; closed loop 36.0% vs 34.0% on 200 paired episodes, p = 0.60) and costs +63% time per step | The paper's KD gain is +0.9 pt at 30k steps. A gain this small is invisible at 400 episodes | KD 7.7 s/step vs task-only 4.7 s/step | Keep the pair only if the experiment's question (KD vs no KD) still matters. If OVX time is tight, run the fixed task-only first (cheapest test of the fix), then KD |
| **LR schedule** | Cosine to 10% of peak is fine once updates are not lost. With bf16 weights the decay was harmful because it froze more weights as the LR fell | none | Keep peak 2.5e-5, warmup 300-500, cosine to 2.5e-6 over the run length |
| **EMA of weights** | openpi trains π0.5 with EMA 0.999 (`config.py:785`). The PyTorch trainer has none (`distill_pytorch.py:562` logs "EMA is not supported"). EMA usually smooths closed-loop behaviour of flow/diffusion policies; the size of the effect here is unknown | +2.8 GB if fp32 on the GPU (or keep it on the CPU), <1% time; ~20 lines of code | Second priority, after the fix is validated |
| **Compute savings.** (i) Run the frozen SigLIP + embeddings once and share them between teacher and student. (ii) Skip the always-masked, all-zero third camera in SigLIP. (iii) Freeze, and later prune, the dead last-layer VLM parameters | (i) saves 1.9 GB and one SigLIP pass per sample; (ii) saves a third of SigLIP compute in both models; together roughly 5-10% of a KD step (estimate from the vision-share measurement in [phase2-cloud-findings.md](phase2-cloud-findings.md) A.3). (iii) frees master-weight memory, and pruning would also speed up inference slightly (the last VLM layer's MLP output is never read at inference either) | code changes in `pi0_pytorch.py` / `distill_pytorch.py` | Optional; worth it only for long runs |
| **Attention-distillation loss** (the paper's third loss) | Paper: +0.7 pt at 30k | code + a hyperparameter | Low priority |
| **Evaluation** | 400 paired episodes give ±4 pt per suite. Fine for judging the fix (expected effect is large), too coarse for KD vs task-only | laptop time only | Unchanged |

## (e) What a follow-up OVX runbook must contain

The runbook itself is out of scope here; these are its requirements.

1. **Code delivery.** A new tarball (or a patch on `966da5c`) containing (c) 1-2, with its sha256. The teammate's workspace from last time (`/data/vla_work`: uv env, 35 GB dataset, converted teacher) can be reused. The runbook should re-verify it with the existing `check_env.sh` / `check_data.sh` and skip setup if they pass. Re-running `uv sync` is not needed: the fix is pure Python.
2. **A check that the fix is active.** This is the reproduction turned into a regression test (committed as `scripts/gate_check.py`).
   - (a) The training log must contain `fp32 master weights for … M trainable params`.
   - (b) A smoke run at a real learning rate must save a checkpoint: 20 steps, batch 2, constant LR 2.5e-5 (`--num-train-steps 20 --save-interval 20 --lr-schedule.warmup-steps 1 --lr-schedule.decay-steps 100000`).
     The script below compares it with the student init. Among the action-expert layer matrices (bf16, and every one of them gets a gradient) with |w| ≥ 2^-6, it reports the fraction of elements that changed. Pass if ≥ 0.25.
     Measured on the laptop with exactly this smoke (task-only config, VLM also frozen to fit 16 GB, which does not affect the expert metric):
     - pure bf16 (`--no-pytorch-fp32-master-weights`): **0.0000** (0 of 65.8 M) → FAIL;
     - fp32 master weights: **0.5520** → PASS;
     - the OVX KD step-3000 checkpoint, after 3,000 real steps: **0.0000** → FAIL.
     The threshold sits between the two with margin on both sides; a longer smoke only raises the fixed value.

     ```python
     """Fix-is-active gate: fraction of action-expert layer weights (bf16 matrices; every one gets a gradient) with
     |w| >= 2^-6 that changed from the student init. Pure-bf16 AdamW at LR <= 2.5e-5 leaves it at exactly 0; with fp32
     master weights it rises within a few steps.
     usage: python -I gate_check.py INIT.safetensors CHECKPOINT.safetensors"""
     import sys
     from safetensors import safe_open

     A, B = safe_open(sys.argv[1], "pt"), safe_open(sys.argv[2], "pt")
     n = c = 0
     for k in A.keys():
         if "gemma_expert.model.layers." in k and "layernorm" not in k:
             a, b = A.get_tensor(k).float(), B.get_tensor(k).float()
             m = a.abs() >= 2.0**-6
             n, c = n + int(m.sum()), c + int((m & (a != b)).sum())
     f = c / max(n, 1)
     print(f"changed fraction of |w|>=2^-6 expert-layer weights: {f:.4f} ({c}/{n}) -> {'PASS' if f >= 0.25 else 'FAIL'}")
     ```

   - Without both lines the run must not start.
3. **Freeze list.** Add the never-trained parameters to `FREEZE`: `gemma_expert.lm_head.`, `language_model.norm.`, and layer 5's `mlp.`, `self_attn.q_proj.`, `self_attn.o_proj.`, `post_attention_layernorm.`. The layer-5 prefixes are specific to the 6-layer student; state that.
4. **Memory and speed gate (P5 equivalent) with master weights at batch 64.**
   - Pass if `peak_reserved_GB` ≤ 46 and s/step is within 10% of the predicted 7.7 (KD) / 4.7 (task-only).
   - Pre-written fallback if it does not fit, so the teammate need not improvise: first option 3's flags from (d), then the decision rule.
   - Keep the bandwidth probe out (already measured).
5. **Run matrix and flags.** Which runs (recommended: task-only + master weights, 3k steps, first; then KD; then a longer run if confirmed), steps, `--save-interval`, LR flags, seed 42 (same data order as the failed runs, so the comparison is paired), one GPU or two in parallel, and the time per run from P5.
6. **Checkpoint export.** Export only `model.safetensors` + `assets/` (4.11 GB each, format unchanged). `master_params.pt` (~2.8 GB) and `optimizer.pt` stay on OVX: they are needed only for `--resume`, which now restores the masters exactly (if `master_params.pt` is missing it re-initializes from the bf16 weights and logs a warning).
7. **Monitoring.** The existing P7, plus the checkpoint-vs-init check from item 2 run on the 1.5k checkpoint, so a silent regression is caught mid-run.
8. **Return package.** Checkpoints, logs, `runs.txt`, `speed.txt`, the item-2 check output, `SHA256SUMS`.
9. **Ask for the teammate's own evaluation details.** Not a runbook step, but needed alongside it: which script and config they evaluated with, how many episodes and suites, and their numbers, so they can be reconciled with the laptop numbers in §0 / (a) 5.

---

## 中文摘要（可轉寄給負責 OVX 的同事）

**結論：你猜的第二個原因（FP32 → BF16，有些權重沒被訓練到）基本上是對的，而且是主要原因。** 不過問題不是「用 bf16」本身，而是 openpi 的 PyTorch 訓練器在 `--pytorch-training-precision bfloat16` 下，**把 master weights 和 AdamW 的狀態也都存成 bf16**。每一步的更新只要小於該權重 bf16 間距的一半，就會被四捨五入掉，權重完全不動。

- **重現結果：** 我們在筆電上用 phase 1 的同一套 paired LIBERO 評估。KD step 3000 在 400 個 episodes 中成功率 **32.0%**（95% CI 27.6-36.7），同樣的 episodes 上 teacher 是 96.0%。task-only step 3000 在 200 個 episodes（每個任務 5 次）中為 **34.0%**；同樣這 200 個 episodes 上 KD 為 36.0%、teacher 為 96.5%，KD 和 task-only 沒有顯著差異。
- **直接證據：** 學習率最高 2.5e-5、最後降到 2.5e-6，比大多數權重的 bf16 半間距還小。四個 checkpoint 中，**81.5% 的 transformer 權重和初始值逐位元完全相同**；有更新的只有 4,300 萬個 fp32 參數（adaRMS、time MLP、action projection）和數值最小的約 14% bf16 權重。step 1500 到 3000 之間，沒有任何新的權重開始更新。
- **對照實驗（筆電）：** 同一個初始化、同樣的資料、seed 和步數，只打開 fp32 master weights：loss 低 30%，有更新的權重從 12% 變成 86%，held-out 動作誤差的改善幅度是原本的 1.7 倍。
- **不是原因的部分：** 推論用 bf16（同一個 checkpoint 用 fp32 推論，誤差只差 0.4%）、teacher 轉成 bf16、checkpoint 載入、norm stats、影像前處理、prompt、KD 程式、資料、student 初始化，都檢查過，沒有問題。
- **步數不足：** 是次要原因。3000 步 × batch 64 只有約 0.7 個 epoch，是論文 30k 步的十分之一；step 1500 → 3000 的 held-out 誤差仍下降 11%。但在 bf16 權重下，學習率越降，能更新的權重越少，所以單純加步數效果有限。

**需要改的檔案：**
1. `scripts/distill_pytorch.py`：每個可訓練參數保留一份 fp32 master copy，AdamW 對 master 做更新，每步再寫回 bf16 權重。forward/backward 仍是 bf16，checkpoint 格式不變。程式已寫好並通過單元測試、resume 測試，也在筆電對照實驗中跑過。
2. `src/openpi/training/config.py`：新增 `pytorch_fp32_master_weights` 欄位，兩個 6 層 student config 預設打開。
3. Runbook 的 `FREEZE`：另外凍結從來收不到梯度的參數（expert 的 `lm_head`、VLM 最後一層的 MLP / q / o 等），可省 1.5 GB。
4. `README.md`：更正 phase 1 當初建議用 bf16 訓練的說明（這是我們自己計畫的選擇，不是你操作的問題）。

**下一次 OVX 建議：** 跟上次完全相同的設定（batch 64、3000 步、相同學習率），只打開 fp32 master weights，這樣結果能單獨歸因於這個修正。在一張 L40S 上約多用 5.5 GB（約 44 GB reserved），時間只多約 1%：KD 約 6.4 小時、task-only 約 4.0 小時。確認有效後再跑 10k 步。也可以完全不改程式，改用全 fp32（batch 16 × 12,000 步），但 GPU 時間大約兩倍，而且 batch 和學習率同時改變，結果無法單獨歸因於精度。

**想請你提供：** 你這次評估用的腳本與設定、跑了幾個 suite / 幾個 episodes，以及實際數字，方便和我們的結果對照。新的 runbook 會另外再寫給你。
