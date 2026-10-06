# Phase 2 on the institute's OVX cloud: findings report

Date: 2026-10-03. Task: `vla-cloud-runbook`. Code: `vla_distill` at commit `966da5c`.
Companion document for the mentor (Traditional Chinese): `runbook.md` in the same folder.

## 1. Summary

The phase 2 plan can run on the institute's OVX nodes as planned, on one 48 GB GPU, with the mentor typing about ten short commands.
Four things in the existing plan are wrong and are corrected here.

| Plan said | Finding | Evidence |
|---|---|---|
| One 80 GB GPU is needed; a 48 GB card forces batch 32 | Batch 64 is predicted to peak at 35.4 GB allocated (about 37-41 GB reserved), which fits a 48 GB card with 7-11 GB to spare. This is an extrapolation from laptop measurements at batch 1-24, not a measurement on a 48 GB card. The runbook's 50-step timed run confirms or refutes it before any long run starts | Section 3 |
| The LeRobot LIBERO download is 54 GB | The download is 34.94 GB. The first load writes a second copy of the same size as an Arrow cache, so the dataset occupies about 70 GB on disk | Section 4 |
| Each checkpoint is 4.1 GB of weights plus about 8 GB of optimizer state | Weights are 4.11 GB (measured) and optimizer state is about 3.0 GB (predicted from the measured AdamW state on the GPU). The script also writes one extra checkpoint at step 2,999, so each run leaves 3 checkpoints, about 21 GB | Section 7 |
| About 70 GB of downloads | About 52 GB inbound if the teacher is converted in the cloud, about 47 GB if the converted teacher is carried in. 16.5 GB comes back out | Section 6 |

Answers to the questions in the ask, in one line each:

- **Exact training data.** Hugging Face `physical-intelligence/libero`, branch `v2.0` (commit `9dfa6951`), 1,693 episodes, 273,465 frames, 40 tasks, two 256×256 cameras, 10 fps, CC-BY-4.0. It is the OpenVLA re-rendered and filtered version of the original human-teleoperated LIBERO demonstrations. Details and the full chain are in section 4.
- **Synthetic data for testing.** The code base has a fake dataset, but no distillation config uses it and it cannot be switched on from the command line. The practical end-to-end test without the full dataset is the 8-episode subset (256 MB), which the runbook uses as its smoke run. Teacher-generated simulator rollouts are not recommended for this comparison. No extra real data is recommended either. Section 5.
- **Bandwidth.** GPU compute bounds the run by a wide margin. The data loader delivers 194-223 samples/s on the laptop with 2 workers against a requirement of 7-12 samples/s. Storage reads during training are about 1-1.5 MB/s. Network speed matters only for setup: the download takes longer than both training runs together only below about 1.2-2.1 MB/s. Section 6.
- **Read versus write.** Training reads about 8.2 MB per step and writes about 7.1 GB three times per run. The setup writes far more than training reads: about 180 GB in total. Network mounts are the risk, for small-file and lock behaviour rather than throughput. All cache paths default to the home directory and must be redirected on a machine where the mentor's writable area is limited. The runbook's environment file does that. Section 7.
- **The ViT.** It is the SigLIP So400m/14 encoder inside Pi0.5, not a separate model. It processes 3 images per sample, is frozen in both teacher and student, and is identical in the two. It is not a problem for training on a 48 GB card as long as it stays frozen. It does limit the compression result: it is 21% of the student's parameters and about 19% of the student's eager inference time, against 11% and 10% for the teacher. Section 8.

What I could not establish is in section 13. What only the mentor can answer is in section 12.

## 2. What I did

This task was started by an earlier worker, interrupted by a restart of the whole WSL machine at about 15:31 CDT, and resumed by me.
I reused every result the earlier worker had produced.
Its measurement outputs for memory and the vision encoder existed only in its session transcript, not in log files; I recovered them from the transcript and they are reproduced verbatim in Appendix A.

Work done, by both workers together:

1. Read the plan (`README.md` experiment log, `results/phase1/SUMMARY.md`, survey report section 7) and the training code path end to end.
2. Built a "simulated cloud" on the laptop under `_cloudsim/`: an empty `HOME`, an environment with no compiler and none of the laptop's caches (`env -i`, `PATH=/usr/local/bin:/usr/bin:/bin:/usr/lib/wsl/lib`), and a self-contained work directory. The code arrives there only as a `git archive` tarball.
3. Installed the Python environment from scratch in that simulated cloud with network (`net_setup*.log`), and again without network from a carried-in bundle inside a network-isolated namespace (`unshare -rn`, `offline_setup.log`).
4. Ran the 10-step distillation smoke run in the online simulated cloud. It passed (`t_smoke.log`: exit 0, 3 min 9 s wall, peak host RSS 17.9 GB) and its ten loss values are identical to the phase 1 values in `results/phase1/SUMMARY.md` (1.1697, 0.5464, 2.4015, …, 0.9379).
5. Measured GPU memory and time versus batch size, the teacher's forward cost, the vision encoder's share, and data-loader throughput (Appendix A).
6. Queried primary sources for the OVX hardware, the dataset, its upstream datasets, and candidate extra datasets.
7. Wrote the runbook, then extracted each of its command blocks from the finished file with the runbook's own `step` helper and executed them in the simulated cloud. Section 10 lists what each block's status rests on.

Not done, by instruction: phase 2 training, a download of the full dataset, any change to the project's code.

## 3. OVX hardware and what the plan needs on a 48 GB card

### 3.1 What an OVX node is

All generations of NVIDIA OVX carry 48 GB cards.

| Generation | GPUs per server | GPU memory | Source |
|---|---|---|---|
| Current spec | 4 or 8 × L40S | 48 GB GDDR6 with ECC | NVIDIA OVX product page (specification table: "4x NVIDIA L40S", "8x NVIDIA L40S"); NVIDIA press release 2023-08-08: "up to eight NVIDIA L40S GPUs per server, each equipped with 48GB of memory"; L40S product page |
| Third generation (2023-03) | 4 × L40 | 48 GB GDDR6 with ECC | NVIDIA blog 2023-03-21: "four NVIDIA L40 GPUs"; L40 product page |
| First generation (2022-03) | 8 × A40 | 48 GB GDDR6 with ECC | NVIDIA press release 2022-03-22: "eight NVIDIA A40 GPUs, three NVIDIA ConnectX-6 Dx 200Gbps NICs, 1TB system memory and 16TB NVMe storage"; A40 product page |

The current product page also lists, per server: 2 × 32-core CPUs and at least 384 GB RAM (4-GPU server) or 2 × 56/64-core CPUs and at least 768 GB RAM (8-GPU server), a 1 TB NVMe boot drive and 2 × 4 TB NVMe data drives, and 200 Gb networking.
So host RAM, CPU cores and local NVMe are not expected to be limits on a bare node.
They can still be limits inside a container, a VM or a scheduler allocation, which is why the probe reads the cgroup limits.

Which generation the institute has is unknown. The probe reports it.
An A40 node would fit the same memory budget but run a multiple slower than an L40S node (NVIDIA claims up to 5× for L40S over A40 on inference; no training figure was found).

L40S figures used below (NVIDIA L40S page): 48 GB, memory bandwidth 864 GB/s, FP16 tensor-core throughput 362 TFLOPS dense (733 with sparsity), 350 W.
For comparison: A100 80GB 312 TFLOPS and 2,039 GB/s; H100 SXM 989 TFLOPS dense (1,979 with sparsity) and 3.35 TB/s.

### 3.2 Memory at batch 64

Measured on the laptop with the real model code (`_cloudsim/m1_mem_slope.py`, `m1b_teacher.py`; Appendix A.1, A.2). Settings: bf16, gradient checkpointing on, vision tower + projector + token embedding frozen, everything else trainable. Units are GB = 10⁹ bytes, as the training script logs them.

| Component | GB | How obtained |
|---|---|---|
| Teacher weights on the GPU | 7.47 | measured |
| Student weights | 4.11 | measured |
| Teacher + student after model creation | 11.59 | measured; the training script logs the same value (`t_smoke.log`: "after_model_creation … allocated: 11.59GB") |
| AdamW state | 2.98 | measured (allocated memory before and after the first optimizer steps) |
| Student forward + backward transient, per sample | 0.305 | measured slope over batch 1, 2, 4, 8, 12, 16, 20, 24; the eight points are linear to within 0.01 GB |
| Student transient, batch-independent part (mostly gradients, 1.49 GB) | 1.31 | measured intercept |
| Teacher no-grad forward transient, per sample | 0.142 | measured over batch 1-32 |
| **Peak at batch 64** = 11.59 + 2.98 + 1.31 + 64 × 0.305 | **35.4** | extrapolated |
| With allocator overhead of 5-15% (observed 6-13% at batch 16-24) | 37-41 | extrapolated |
| L40S usable | about 48.3 | 46,068 MiB is what 48 GB cards with ECC commonly report in `nvidia-smi`; this figure is from memory, not from a source fetched for this report. The probe shows the real value |

Two details explain why the earlier 45-55 GB estimate was too high.
Only 702.7 M parameters receive a gradient, not the 1,070.9 M that are nominally trainable: the unused 263 M expert `lm_head` gets none, and neither do about 105 M more, which I take to be the parts of the last VLM layer whose output is never read (inferred from the count, not checked parameter by parameter). AdamW keeps no state for parameters without a gradient.
The teacher's forward runs before the student's graph exists (`src/openpi/models_pytorch/pi0_pytorch.py:539-549`, then the student forward from line 551), so its 9.1 GB transient at batch 64 does not add to the student's backward peak.

A cross-check against phase 1: the same formula gives 11.59 + 1.92 = 13.51 GB at batch 2, and phase 1 measured 13.53 GB peak for that configuration.

By the same model, batch 32 would peak at about 25.6 GB and the largest batch that fits 48 GB is about 85.
A 40 GB card is marginal and a 24 GB card does not fit batch 64.

### 3.3 Gradient accumulation and multi-GPU

`scripts/distill_pytorch.py` does not support gradient accumulation.
The loop is one `loss.backward()`, one `optim.step()`, one `optim.zero_grad()` per loader batch (`scripts/distill_pytorch.py:592-606`), and there is no accumulation option in `TrainConfig` (`src/openpi/training/config.py:473-546`).
So "batch 32 with accumulation" is not available without a code change, and plain batch 32 is a different recipe.
By the estimate above it is not needed.

The script has a multi-GPU path: `torchrun` sets `WORLD_SIZE`, the script wraps the student in `DistributedDataParallel` and divides the batch across ranks (`scripts/distill_pytorch.py:95-110, 353-357, 459-466`; `src/openpi/training/data_loader.py:311-319`).
I could not exercise it on a one-GPU laptop.
It has untested interactions with this project's additions: parameters are frozen after the DDP wrapper is built, and with 8 or more GPUs the script switches to `static_graph=True` and TF32 (`:451-457, 465`).
The runbook therefore does not use it.

With two free cards the runbook instead runs the KD run and the control run at the same time, one per card (block `P6P`).
Each is then an ordinary single-GPU run on the verified code path, and the wall time drops to that of the longer run.

### 3.4 Time

Two independent estimates for one L40S, both for the KD run at batch 64:

- From FLOPs: the README's 0.77 PFLOP per step at 30-40% of the L40S's 362 TFLOPS gives 5.3-7.1 s/step.
- From the laptop: 0.293 s per sample measured (student forward 0.053 + backward 0.122 + teacher forward 0.119 at batch 16), so 18.8 s per batch-64 step on the laptop GPU. Assuming an L40S is 2-2.5× the 92 W laptop part gives 7.5-9.4 s/step. The 2-2.5× factor is my assumption, not a measurement.

Taking the union, 5.3-9.4 s/step:

| | s/step | 3,000 steps |
|---|---|---|
| KD run | 5.3-9.4 | 4.4-7.8 h |
| Task-only control (0.6× the KD step: no teacher forward; measured ratio on the laptop 0.595) | 3.2-5.6 | 2.6-4.7 h |
| Both, one after the other | | 7.1-12.5 h |
| Both, in parallel on two cards | | 4.4-7.8 h |
| Setup (environment, data, teacher, smoke run, timed run) | | 1-2 h |

The 50-step timed run in the runbook replaces these estimates with a measurement and prints the projected hours.
The control run still loads the teacher onto the GPU (`scripts/distill_pytorch.py:415-436` is unconditional), so its memory is the same; only its step time is lower.

## 4. Training data: what it is and where it came from

### 4.1 The dataset the configs load

Both phase 2 configs set `repo_id="physical-intelligence/libero"` (`src/openpi/training/config.py:807-816` for `pi05_libero_l06`, `:836-844` for `pi05_libero_l06_taskonly`).

| Property | Value | Source |
|---|---|---|
| Repository | `physical-intelligence/libero` on Hugging Face, public, not gated | HF API |
| Revision actually loaded | branch `v2.0`, commit `9dfa69510ea9e1613fc54112bc706444b686a231` (2025-02-02). The pinned LeRobot asks for `v2.1`, finds none and falls back; the log says "Revision v2.1 … not found, using version v2.0". `main` is `a4336d58…` and differs only by README edits | `t_smoke.log`; HF commits API; `lerobot_dataset.py:77`, `utils.py` `get_safe_version` |
| Episodes / frames / tasks | 1,693 / 273,465 / 40 | `meta/info.json` |
| Features | `image` and `wrist_image`: 256×256×3, stored as PNG images inside parquet (no video). `state`: 8 floats. `actions`: 7 floats. Plus timestamp and index columns | `meta/info.json` |
| Frame rate | 10 fps | `meta/info.json` |
| Robot | `panda` | `meta/info.json` |
| Files | 1,693 parquet files (one per episode) + 4 metadata files | HF API |
| Download size | 34,938,730,091 bytes of parquet (34.94 GB), 127.8 KB per frame on average | HF API (`?blobs=true`) |
| On disk after first load | about 70 GB: the parquet files plus an Arrow cache of the same size under `$HF_HOME/datasets` (measured on the 8-episode subset: 267.7 MB parquet, 267.8 MB Arrow) | `~/.cache/huggingface` on the laptop |
| License | CC-BY-4.0 | dataset card |
| Split | one `train` split, `0:1693`. No validation split | `meta/info.json` |

Per-suite split, computed by matching each episode's task string against the LIBERO task map in `third_party/libero`:

| Suite | Tasks | Episodes | Frames | Demonstrations per task |
|---|---|---|---|---|
| LIBERO-Spatial | 10 | 432 | 52,970 | 35-47 |
| LIBERO-Object | 10 | 454 | 66,984 | 42-50 |
| LIBERO-Goal | 10 | 428 | 52,042 | 33-50 |
| LIBERO-10 (Long) | 10 | 379 | 101,469 | 29-49 |
| Total | 40 | 1,693 | 273,465 | |

Episode length is 75 to 505 frames, median 140, mean 161.5.
Two task strings (one LIBERO-10, one LIBERO-Goal; 91 episodes) also occur as LIBERO-90 instructions; they are counted in the suite the dataset was built from.

The README and the survey plan give 54 GB. That figure is 8 episodes × 1,693 / 8, and the first 8 episodes are long LIBERO-10 episodes, so it overstates the size.

### 4.2 Provenance chain

1. **Original demonstrations.** LIBERO (Liu, Zhu, Gao, Feng, Liu, Zhu, Stone; arXiv 2306.03310). 130 tasks in four suites, built on robosuite and MuJoCo. The paper states the demonstrations were "collected by human experts through teleoperation with 3Dconnexion Spacemouse", 50 per task. The four suites used here have 4 × 10 × 50 = 2,000 demonstrations. Images in the original release are 128×128 (per the OpenVLA script's note below). The paper does not name the individual operators.
2. **OpenVLA regeneration** (Kim et al., arXiv 2406.09246; `experiments/robot/libero/regenerate_libero_dataset.py` in the OpenVLA repository). The demonstrations were replayed in the simulator and re-recorded. The script's docstring lists the changes: images saved at 256×256 instead of 128×128; transitions with no-op (zero) actions removed; unsuccessful replays removed; images rotated by 180 degrees during the conversion to RLDS because the environment returns them upside down. The result is `openvla/modified_libero_rlds` on Hugging Face (MIT, about 10 GB), four datasets named `libero_{spatial,object,goal,10}_no_noops`. The dataset card itself defers to the paper's Appendix E for the list; I could not retrieve the appendix text and rely on the script.
3. **Conversion to LeRobot format** by Physical Intelligence. The dataset card says: "This dataset combines four individual Libero datasets: Libero-Spatial, Libero-Object, Libero-Goal and Libero-10. All datasets were taken from [openvla/modified_libero_rlds] and converted into LeRobot format." The conversion script is in this repository: `examples/libero/convert_libero_data_to_lerobot.py:29-34` (the four `_no_noops` sets), `:49` (10 fps), `:53, 58` (256×256 images). Nothing is filtered at this step.

The filtering in step 2 is why there are 1,693 episodes rather than 2,000: 307 demonstrations (15%) failed on replay and were dropped, unevenly across tasks (29 to 50 remain per task).

### 4.3 What training does to it

- Each sample is one frame plus the next 10 actions (action horizon 10, one second at 10 fps): `src/openpi/training/data_loader.py:140-147`, `config.py:803`.
- The prompt is the episode's task string (`config.py:814`, `data_loader.py:149-150`).
- Three image slots are filled: the third-person camera, the wrist camera, and an all-zero image for the absent right wrist camera, which is masked out (`src/openpi/policies/libero_policy.py:55-69`).
- State and actions are normalized with quantile normalization (`config.py:193`: every model type except `PI0`), using **the teacher checkpoint's own statistics**: `gs://openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json` (`config.py:810-813, 838-841`). sha256 `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84`. The file holds mean, std, 1st and 99th percentile for the 8 state and 7 action dimensions. The teacher, both students and the evaluation all use this one file.
- Images are resized to 224×224, the prompt is tokenized to 200 tokens, state and actions are padded to 32 dimensions (`config.py:129-141`).
- Augmentation runs on the GPU inside the model, not in the loader: a 95% random crop and a ±5° rotation on the non-wrist camera, and brightness, contrast and saturation jitter on all cameras (`src/openpi/models_pytorch/preprocessing_pytorch.py:53-129`). The teacher receives the student's already-augmented images (`pi0_pytorch.py:543-549`).
- Shuffling is seeded (`seed=42`, `config.py:515`, `data_loader.py:433-434`), so the KD run and the control see the same batches in the same order, as the plan requires.

3,000 steps at batch 64 is 192,000 samples, 70.2% of the frames, each seen once.

### 4.4 The teacher

`gs://openpi-assets/checkpoints/pi05_libero`: Physical Intelligence's π0.5 fine-tuned on this same dataset with the `pi05_libero` config (`examples/libero/README.md`: "This checkpoint was trained in openpi with the `pi05_libero` config", 96.85% average).
It is a public bucket, 16 objects, 12.44 GB, uploaded 2025-09-08.
That config has no freeze filter (`config.py:769-789`), so the teacher's vision encoder was fine-tuned along with the rest.

Phase 1 converted it to PyTorch bf16 with `examples/convert_jax_model_to_pytorch.py`: `/home/shayn/vla_distill_artifacts/checkpoints/pi05_libero_pytorch/model.safetensors`, 7,233,650,408 bytes, sha256 `d9ef6bb72672509355ed19ebf08909e04e9af0ec4918a5d5ce9fa7c25aaf7ea4`.
That file scored 96.0% [93.6, 97.5] on 400 episodes in phase 1.
The same file is both the KD teacher and the source of the student's initial weights (teacher layers 0, 3, 7, 10, 14, 17; `config.py:818`, `scripts/distill_pytorch.py:469-500`).

### 4.5 Overlap between training and evaluation data

There is no held-out task set. This is the standard LIBERO protocol and it applies equally to the teacher, both students and every published LIBERO number.

- The evaluation runs simulator rollouts on the same 40 tasks the demonstrations cover. It tests whether the policy completes known tasks from the benchmark's fixed initial states (indices 0-9 in phase 1, 0-24 planned), not whether it generalizes to new tasks.
- No evaluation data is a file that could be in the training set: evaluation is live simulation. Whether the benchmark's fixed initial states coincide with the initial states of the training demonstrations I did not verify; it needs a simulator replay to compare them.
- The 457 held-out frames used for the action-agreement metric were recorded from teacher rollouts on initial state 40. They are not frames from the dataset, but they come from the same tasks.
- Both student runs train on identical data, so any overlap affects them equally and the paired comparison is unaffected.

## 5. Synthetic data for testing, and additional data

### 5.1 What the code base provides

- **A fake dataset.** `FakeDataset` returns random tensors with the model's input shapes (`src/openpi/training/data_loader.py:98-127`), selected when `repo_id == "fake"` (`:137-138`) through `FakeDataConfig` (`config.py:210-216`). The `debug`, `debug_restore` and `debug_pi05` configs use it with tiny "dummy" model variants (`config.py:1023-1054`).
- **It is not usable for distillation as the code stands.** No distillation config uses `FakeDataConfig`. Forcing it from the command line (`--data.repo-id fake` on `pi05_libero_l06`) fails with `KeyError: 'image'`, because the LIBERO key remapping is still applied to the fake samples. I ran this on the CPU (`_cloudsim/v/c1_flags.py`, output in Appendix B). A fake-data distillation run needs a new config entry of about ten lines, which is a code change and out of scope.
- **An episode subset.** `--data.lerobot-episodes 0 1 2 3 4 5 6 7` (`config.py:92-94, 177-179`; `data_loader.py:146`) loads and downloads only those episodes: 2,197 frames, 256 MB instead of 34.94 GB. This is real data, and it is what the phase 1 smoke test and the runbook's smoke run use.

### 5.2 What each kind of test proves

| Test | Proves | Does not prove |
|---|---|---|
| Fake tensors (needs the new config) | Environment, GPU, model build, teacher load, forward, backward, optimizer, checkpoint write. Also memory and step time at any batch size, because compute does not depend on pixel values; the memory measurements in section 3 were taken this way | Download, Arrow cache, PNG decode, tokenizer, normalization statistics, loader throughput, dataset integrity. Loss values are meaningless |
| 8-episode real subset (available now) | Everything above plus the real data path end to end, with known reference loss values | Behaviour with the full 35 GB (cache build time, storage latency), memory at batch 64 on the target card |
| 50 steps on the full data at batch 64 (the runbook's timed run) | Everything | Nothing relevant remains except a 3,000-step soak |

The runbook uses the second and third. A fake-data path would add one thing: a test that needs no data at all, which helps only if the dataset is slow to arrive.

### 5.3 Teacher-generated rollouts as extra distillation data

Not recommended for this experiment, for four reasons.

1. The KD loss does not need them. The teacher's velocity target is computed online for every training batch (`pi0_pytorch.py:541-549`). Stored teacher rollouts would add new states, not new labels.
2. They change the control. A task-only run trained on teacher rollouts is behaviour cloning of the teacher, which is itself distillation. The KD-versus-task-only contrast would blur.
3. They risk contaminating the evaluation. The evaluation uses initial states 0-24 and the action-agreement frames come from state 40. LIBERO ships 50 fixed initial states per task, so rollout data would have to be restricted to the remaining ones, and the plan's LIBERO-Plus robustness test adds further constraints.
4. They need tooling that does not exist: a rollout recorder that writes LeRobot format. `scripts/phase1/collect_heldout_frames.sh` records policy inputs for 457 frames, not a dataset. Laptop rollouts run at about 18 simulator steps per second.

Their real use is different: on-policy data covers the states a weak student actually reaches, which the human demonstrations do not. That is worth considering later if the trained student fails by drifting off the demonstrated paths.

### 5.4 Additional real data

| Dataset | Size, license | Compatible with the current config? | Effect on the comparison |
|---|---|---|---|
| `jesbu1/libero_90_lerobot` (LIBERO-90) | 3,959 episodes, 574,571 frames, 69.5 GB, Apache-2.0 | Yes: same keys, 256×256 images, 10 fps, LeRobot v2.0 | The teacher was not fine-tuned on these 90 tasks, so its targets there are unreliable, and the evaluation does not test them. Adds 70 GB and no clear benefit |
| `Sylvest/libero_plus_lerobot` (LIBERO-Plus) | 14,347 episodes, 2,238,036 frames, 16 GB, MIT | No: video instead of images, different keys, 20 fps | It is the planned robustness evaluation distribution (hypothesis H4). Training on it would contaminate that test |
| `lerobot/libero` | Same 1,693 episodes re-encoded as video, 1.9 GB, Apache-2.0 | No: LeRobot v3.0 format, which the pinned LeRobot cannot read; lossy video | Would shrink the transfer from 35 GB to 2 GB, at the price of a code change and pixels that differ from what the teacher was trained on |
| `IPEC-COMMUNITY/libero_90_no_noops_lerobot` | 3,921 episodes, 3.4 GB, Apache-2.0 | No: video, different keys, 20 fps | As LIBERO-90 |
| `openvla/modified_libero_rlds` | about 10 GB, MIT | The upstream of the training set, in RLDS | Same content; converting it reproduces the training set |
| `yifengzhu-hf/LIBERO-datasets` | 100.4 GB, tagged Apache-2.0 | Original HDF5 at 128×128 | Different rendering from the teacher's training data |
| DROID, ALOHA (other openpi configs) | very large | Different robots | Irrelevant to a LIBERO comparison |

Sizes and schemas are from the Hugging Face API on 2026-10-03.

**Recommendation.** Train both runs on `physical-intelligence/libero` only, as planned. It is the teacher's own training set and the paper's setup, and it keeps the comparison paired. Use the 8-episode subset for smoke testing. Add a fake-data distillation config only if a data-free test is wanted; it is a small follow-up, not a prerequisite.

## 6. Bandwidth: every throughput limit on the path

The run is bounded by GPU compute. Every other stage has at least a tenfold margin, except the one-time network transfer, whose speed is unknown.

| Stage | Required | Available | Margin | Basis |
|---|---|---|---|---|
| GPU compute | 0.77 PFLOP per KD step | 362 TFLOPS peak (L40S, bf16) | This is the bound: 5.3-9.4 s/step estimated | Section 3.4 |
| Samples from the data loader | 6.8-12.1 samples/s (64 per 5.3-9.4 s). At the plan's H100 rate it would be 25-32 | Laptop, measured: 194-223 samples/s with 2 workers, 241-375 with 4, 214-401 with 8, 541 with 16 (one run) | 16-33× with 2 workers | Appendix A.4 |
| CPU decode and transforms | as above | 9.2 ms per sample per worker process (5.8 ms parquet read + 2 PNG decodes, 3.4 ms transforms), i.e. 108 samples/s per worker | 4 workers ≈ 35× | `_cloudsim/v/c2_t1.log` |
| Storage read during training | 127.8 KB per sample, 8.2 MB per step, 0.9-1.5 MB/s, 24.5 GB per run | Any disk. Even a slow network mount gives 50 MB/s | > 30× | 34.94 GB / 273,465 frames |
| Host-to-GPU transfer | 115.9 MB per batch (three 224×224 float32 images per sample), 12-22 MB/s | PCIe, several GB/s | > 100× | Appendix A.4 (`host_to_gpu_MB_per_batch`) |
| GPU memory bandwidth | Order of 100-200 GB moved per step (19.5 GB of live activations written once and read a few times, plus weights and optimizer state) | 864 GB/s | About 0.1-0.25 s of a 5-9 s step. My estimate, not a measurement | L40S page |
| Checkpoint write | 7.1 GB, three times per run, training paused while it writes | Laptop: 4.87 GB in 20.3 s and in 10.7 s (240 and 455 MB/s) | About 15-70 s per checkpoint, under 1.5% of the run | `t_smoke.log` 14:44:11 → 14:44:31; resume test 16:19:02 → 16:19:13 |
| Network in (path A) | about 52 GB once: dataset 34.94, teacher (JAX) 12.44, Python packages about 5 (estimated from the wheel sizes in the install log) | Unknown. Laptop: 18-30 MB/s single stream | See below | `v/P0B.out` |
| Network or media out | 16.45 GB once | Unknown | See below | 4 × 4,112,924,656 bytes |

Notes on the measurements:

- The loader figures were taken on the 8-episode subset, which fits in the page cache. They measure CPU decode, not storage. On the full dataset the first pass reads from disk, but at 1-1.5 MB/s that does not change the conclusion.
- The loader runs were noisy (4 workers: 241 and 354-375 in different runs) because orphaned worker processes from earlier runs were still alive during some of them. The lowest value seen for any setting is 194 samples/s, which is still 16× the requirement.
- A single process with PyTorch's default 16 threads takes 39.8 ms per sample (25 samples/s), four times slower than a worker process, which runs single-threaded. The earlier worker's `m3b.log` (34.9 ms) measured that case. This matters only for the no-worker fallback in the runbook's appendix, where it adds about 2.5 s to each step.
- Relative to its compute, the L40S has the least memory bandwidth of the three cards (2.4 GB/s per TFLOPS, against 3.4 for H100 and 6.5 for A100), so it is more likely to land in the slower half of the FLOP-based estimate.

**Break-even network speed.** Both runs together take 7.1-12.5 h. Downloading 52 GB takes that long at 1.2-2.1 MB/s (9-17 Mbit/s). Above that, the download is shorter than the training; at the laptop's measured 18-30 MB/s it is 30-50 minutes. For the 16.45 GB going out, the same break-even is 0.4-0.65 MB/s.

The runbook's probe measures the cloud side: disk write and read of a 1 GiB file, creation of 2,000 small files, a 200 MB download from the Google bucket and a 37 MB download from Hugging Face (block `P0B`). The timed run measures loader plus GPU together, and checkpoint write time.

## 7. Read versus write

### 7.1 Storage

| | Read | Write |
|---|---|---|
| During training, per step | 8.2 MB (random reads into the Arrow file, memory-mapped) | none |
| During training, per run | 24.5 GB of dataset; 7.2 GB teacher file at start, read twice (teacher load and student initialization) | 3 checkpoints × 7.1 GB = 21.3 GB at steps 1,500, 2,999 and 3,000; log under 2 MB |
| Setup, once | 34.9 GB to build the Arrow cache | Python environment 8.9 GB (52,447 files) + package cache 9.6 GB (55,462 files); dataset 34.9 GB; Arrow cache 34.9 GB; teacher 7.2 GB (plus 12.4 GB if converted in the cloud); possibly up to 10 GB of Hugging Face Xet chunk cache (observed: 256 MB for a 256 MB download; the 10 GB cap is the library default, not verified here) |
| Total on disk | | About 180 GB with both runs' checkpoints, the timed-run checkpoint and the export copy. The runbook asks for 200 GB and lists what can be deleted |

Points that matter:

- **The step-2,999 checkpoint.** `save_checkpoint` saves when `step % save_interval == 0` or when `step == num_train_steps - 1` (`scripts/distill_pytorch.py:156`). With 3,000 steps and interval 1,500 that is 1,500, 2,999 and 3,000. The 2,999 copy is redundant and costs 7.1 GB and one write per run. The runbook keeps the plan's flags and tells the mentor to ignore that directory.
- **Checkpoint size.** `model.safetensors` is 4,112,924,656 bytes (measured). `optimizer.pt` was 0.76 GB in the smoke run, where only 147 M parameters train. With the phase 2 freeze set 702.7 M parameters have optimizer state, measured at 2.98 GB on the GPU, so about 3.0 GB on disk. Only `model.safetensors` and `assets/` are needed for evaluation; `optimizer.pt` is needed only to resume.
- **Resume needs a little extra GPU memory.** The checkpoint is loaded onto the GPU beside the freshly built student, a transient 4.1 GB (resume test: peak 15.70 GB against 11.59 GB after model creation). That is well inside the batch-64 budget.
- **Checkpoints are written atomically.** The script writes to `tmp_<step>` and renames (`:158-189`), so an interrupted save leaves no half-written checkpoint that resume could pick up (`:200-204` ignores `tmp_` directories).
- **Risky filesystems.** Network mounts (NFS, Lustre, GPFS, CIFS) are a risk for three reasons unrelated to throughput. The environment is 108,000 small files across the virtual environment and the package cache, so installation and every Python start are slow; the loader's worker processes are started with `spawn` (`data_loader.py:429-431`) and each re-imports everything. The code takes file locks in the caches (`src/openpi/shared/download.py:71-72`; Hugging Face and uv do the same), which hang or fail on mounts without lock support. And quotas on such mounts often count files as well as bytes. A local NVMe disk avoids all three; OVX nodes are specified with two 4 TB NVMe drives. The probe reports the filesystem type, the inode count and a small-file timing.
- **Shared memory is a write path too.** Worker processes hand each 116 MB batch to the main process through `/dev/shm`. With 4 workers up to about 1 GB is in flight. Docker's default 64 MB `/dev/shm` breaks this. The probe reports its size and the runbook has a fallback.

### 7.2 Network

Inbound is about 52 GB (path A) and outbound is 16.45 GB. Inbound is public, anonymous HTTPS from five host groups: Hugging Face and its CDN, Google Cloud Storage, PyPI and its file host, the PyTorch wheel index, and GitHub and Astral for `uv`, the managed Python and the LeRobot source. Outbound has no public destination: the results go back to the laptop, and how is the mentor's and the institute's decision. Institutes commonly allow the first and restrict the second.

### 7.3 Paths the code assumes are writable

On a machine where the mentor can write only in one area, these defaults break. The runbook's environment file (`$W/env.sh`) redirects each one into the work directory.

| Path, default | Used for | Reference | Redirected by |
|---|---|---|---|
| `~/.cache/openpi` | Tokenizer, normalization statistics, teacher download. The code also runs `chmod 0777` on this directory and its contents, which fails if someone else owns it | `src/openpi/shared/download.py:18-28, 129-131` | `OPENPI_DATA_HOME` |
| `~/.cache/huggingface` | `lerobot/` (dataset), `datasets/` (Arrow cache), `hub/`, `xet/` | `lerobot/common/constants.py:38-39` | `HF_HOME` |
| `./checkpoints/<config>/<exp>` | Checkpoints; relative to the current directory | `config.py:512, 554-558` | The runbook always runs from the repository root inside `$W` |
| `checkpoints/pi05_libero_pytorch` | Teacher weights; also relative to the current directory | `config.py:818, 845` | same |
| `./.venv` inside the code directory | Python environment | uv default | The code is unpacked inside `$W` |
| `~/.cache/uv`, `~/.local/share/uv/python`, `~/.local/bin` | Package cache, managed Python, the `uv` binary | uv defaults | `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `UV_INSTALL_DIR` |
| `~/.config/uv/uv-receipt.json`, `~/.nv/ComputeCache` | uv installer receipt; NVIDIA driver cache | Observed in the simulated cloud's `HOME` | If `HOME` is not writable, `env.sh` points it at `$W/home` |
| `~/.cache` (other) | Miscellaneous | | `XDG_CACHE_HOME` |
| `./wandb` | Would be written if wandb were enabled | `scripts/distill_pytorch.py:73-92` | Both configs set `wandb_enabled=False` (`config.py:799, 827`); `WANDB_MODE=disabled` as well |
| `/dev/shm`, `/tmp` | Loader IPC, temporary files | | Reported by the probe |

In the simulated cloud, with `HOME` set to an empty directory, the only files that appeared under it after a full install and a smoke run were the uv receipt and the NVIDIA compute cache. Everything else landed in the work directory.

Two read-side assumptions also matter:

- **Cache expiry by file date.** `download.py:173-194` treats any cached path under `openpi-assets/checkpoints/` whose modification time is before 2025-02-03 as expired, deletes it and downloads again. A carried-in file with a preserved old date, or a tool that resets dates to 1970 or 1980, would trigger a download attempt on an offline machine. The bundle's files are dated 2026, and `tar` preserves that.
- **Resume changes the data order.** On `--resume` the model and optimizer state are restored (`scripts/distill_pytorch.py:198-260`), but the loader's shuffle and the noise generator restart from the seed (`:119-123`, `data_loader.py:433-434`). A run resumed from step 1,500 trains its second half on the same batches, with the same noise, as its first half, and never sees the batches an uninterrupted run would have seen. If only one of the two runs is resumed, the pair is no longer matched in data order. The resume test shows the effect directly: the resumed step 9 logged a loss of 1.1688, almost exactly the 1.1697 that the original run logged at step 0 on the first batch, whereas the original run's own step 9 logged 0.9379. The launch script records every resume in `runs.txt` and the runbook asks the mentor to report it. What to do then is a decision for the experiment owner: accept and document it, or rerun that one run from scratch.

## 8. The ViT in front of the VLA

### 8.1 What is there

The vision encoder is inside Pi0.5: SigLIP So400m/14 as PaliGemma's vision tower, followed by a linear projector into the language model's 2,048-dimensional space.
Measured from the loaded model (`_cloudsim/m2_vit_share.py`, Appendix A.3): 27 layers, width 1,152, MLP 4,304, 16 heads, patch 14, input 224×224, 256 tokens per image.
There is no other vision model anywhere in the pipeline: images go from the dataset (or the simulator) through a resize straight into SigLIP (`src/openpi/models_pytorch/gemma_pytorch.py:85-86`, `pi0_pytorch.py:186-210`).

- **Images per sample:** three pass through SigLIP. Two are real (third-person and wrist). The third is an all-zero placeholder that is masked out of attention but is still encoded (`libero_policy.py:55-69`; the loop at `pi0_pytorch.py:197-210` has no skip). So 768 of the 968 prefix tokens are image tokens, and 256 of them carry nothing.
- **Frozen?** The teacher is entirely frozen (evaluation mode, `torch.no_grad`). The student's vision tower and projector are frozen by the plan's `--pytorch-freeze-prefixes`. That flag is this project's addition; the fork's default trains everything, and the paper does not say what it did. The student's SigLIP starts as a copy of the teacher's and never changes, so the two encoders are identical throughout.

### 8.2 Its share

| | Teacher (18+18 layers) | Student (6+6 layers) |
|---|---|---|
| Parameters, SigLIP + projector | 414.8 M of 3,616.8 M = 11.5% (12.4% of the 3,353 M used at inference) | 414.8 M of 2,012.3 M = 20.6% (23.7% of the 1,749 M used at inference) |
| Weight bytes | 0.83 GB of 7.47 GB | 0.83 GB of 4.11 GB |
| SigLIP time for 3 images, batch 1 | 30.8 ms | 29.8 ms |
| Share of one training-style forward, batch 1 / 8 / 16 | 21.7% / 16.4% / 15.9% | 43.6% / 36.8% / 36.3% |
| Share of 10-step inference, batch 1, eager | 9.6% of 322 ms | 19.5% of 153 ms |
| Activation memory, no gradient | 8.9 MB per sample | same |

Laptop, bf16, eager mode. The compiled latencies in phase 1 (145.7 and 58.6 ms) were not broken down, so the share under `torch.compile` is not known.

### 8.3 Is it a problem?

**(i) Training throughput and memory: no, while it stays frozen.**
In a KD step SigLIP runs twice per sample, once in the teacher and once in the student, about 19 ms each on the laptop, 13% of the step. It has no backward pass and stores no activations, because its parameters are frozen and its input needs no gradient. Its memory cost is its 0.83 GB of weights per model.
The two passes compute the same thing, since the encoders are identical. Reusing the teacher's image embeddings in the student would save about 6.5% of the KD step. That is an optional optimization and needs a code change.
Unfreezing it would be a problem: 412 M more parameters with gradients and optimizer state, and stored activations for 27 layers of image tokens. My rough estimate is 10-15 GB more at batch 64, which would not fit 48 GB. I did not measure this.

**(ii) The compression result: it is a limit, and should be reported as one.**
Depth distillation removes 67% of the transformer layers but none of the vision encoder. That is why total parameters fall only 44% while latency falls 60% (phase 1), and why the encoder's share of inference time doubles from 10% to 19%. It sets a floor: with no transformer layers at all, the student would still spend about 30 ms per query in SigLIP on the laptop. The paper makes a related remark, that reducing visual tokens "yields only modest latency improvements" compared with reducing layers.
It does not affect the planned comparison. All four models share the same encoder, so teacher versus student and KD versus task-only are unaffected.
For the final report this means stating compression as "layers −67%, parameters −44%, latency −60%" with the reason, rather than a single ratio.

**(iii) Anything else.**
- The encoder depends on a patched `transformers`. If `src/openpi/models_pytorch/transformers_replace` is not copied into the installed package the model does not behave as trained. The runbook's environment check asserts the patch is in place.
- The dataset's 256×256 images are resized to 224×224 before the encoder, in training and in evaluation alike.
- The zero third image costs a third of the encoder's time and a quarter of the prefix tokens. Removing it would change the model's input format relative to the teacher, so it is not something to do inside this experiment.

### 8.4 The other reading of the question

"A ViT before the VLA part in the pipeline" could also mean a separate vision model placed ahead of the policy: a perception front end such as a detector or a segmenter, or a standalone encoder whose features are computed first and stored.
Nothing of that kind exists in this code base, in the dataset, or in the evaluation: there is no precomputed feature cache and no second vision network.
If the mentor has such a component in mind, for example one used on the institute's own robots, it would be an addition to both teacher and student that this plan does not contain, and it should be raised as a separate question. I have answered for the encoder that exists.

## 9. Getting code and data onto the machine

### 9.1 The code

The repository is private. The runbook does not clone it.
The code travels as a `git archive` tarball of commit `966da5c`: `vla_distill-966da5c.tar.gz`, 1,009,886 bytes, sha256 `608237d052783be3f59bfe730af0d67576f3442e6576d37b7889959c2de948a0`.
At 1 MB it goes as an attachment to the same message as the runbook. No account, token or deploy key is involved in either path.
The tarball has no git submodules; training does not need them (the LIBERO simulator is used only for evaluation on the laptop). The smoke run passed from this tarball.

The tarball is not stored next to this report: it lives only in the worktree, which is discarded. It has to be regenerated before sending, from a checkout of the repository at `966da5c`. Regenerating it on this laptop (git 2.43.0) reproduced the same checksum, so the hash written in the runbook stays valid. The same command, with the new hash, applies after the default branch moves:

```bash
git archive --format=tar.gz --prefix=vla_distill/ -o vla_distill-<sha>.tar.gz HEAD && sha256sum vla_distill-<sha>.tar.gz
```

The checksum line in runbook blocks `P2A`, `P2B` and Appendix C must then be updated, along with the file name.

### 9.2 Path A: the cloud has outbound internet

Everything else is public and anonymous: `uv` from astral.sh, Python 3.11 from GitHub releases, packages from PyPI and the PyTorch index, LeRobot from GitHub at a pinned commit, the dataset from Hugging Face, the tokenizer and normalization statistics from Google Cloud Storage.

The teacher has two routes. Carrying in the laptop's converted file (7.23 GB) is preferred because the KD teacher is then provably the model evaluated in phase 1; the runbook checks its sha256. If there is no way to get 7 GB in, the runbook downloads the public JAX checkpoint (12.44 GB) and converts it in the cloud, which needs about 30 GB of host RAM. Whether that conversion reproduces the laptop's file byte for byte is not known; the runbook prints the hash either way and asks for it back if it differs.

### 9.3 Path B: no outbound internet

Everything is prepared outside and carried in as one folder of about 52.5 GB (runbook Appendix C). The owner builds it; these are the commands, verified in the simulated cloud except for the full-dataset download:

```bash
# in a work directory $W prepared with runbook blocks P1 and P2A (so that it contains bin/, cache/uv, cache/uv-python)
mkdir -p bundle && tar -cf bundle/vla_env_cache.tar -C "$W" bin cache/uv cache/uv-python        # 10.28 GB, 54 s on the laptop
tar -cf bundle/openpi_cache.tar -C "$W/cache/openpi" big_vision/paligemma_tokenizer.model \
    openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json
cp vla_distill-966da5c.tar.gz bundle/
cp -r /home/shayn/vla_distill_artifacts/checkpoints/pi05_libero_pytorch bundle/                 # 7.23 GB
# full dataset, 34.94 GB (not run here):
"$W"/vla_distill/.venv/bin/huggingface-cli download physical-intelligence/libero --repo-type dataset --revision v2.0 \
    --local-dir bundle/libero_lerobot --max-workers 8
rm -rf bundle/libero_lerobot/.cache
(cd bundle && find -L . -type f ! -name SHA256SUMS.bundle | sort | xargs sha256sum > SHA256SUMS.bundle)
```

The environment cache was built on Ubuntu 24.04 x86_64 and contains portable `manylinux` wheels; the target needs glibc 2.28 or newer.

### 9.4 What the phase 1 artifacts save

`/home/shayn/vla_distill_artifacts` (inspected read-only): `checkpoints/` 19 GB, `data/` 111 MB, `logs/` 1.4 MB.

| Artifact | Size | Use for phase 2 |
|---|---|---|
| `checkpoints/pi05_libero_pytorch/` (converted teacher, `config.json`, `assets/`) | 7.23 GB | Carry it in. Saves the 12.44 GB download and the conversion with its 30 GB RAM peak, and guarantees the KD teacher is the evaluated teacher |
| `checkpoints/pi05_libero_l06_init/` (untrained student) | 4.03 GB | Not needed in the cloud. The training script rebuilds the student from the teacher file with the same layer mapping |
| `checkpoints/pi05_libero_l06_init_forkmapping/`, `pi05_libero_l06/laptop_smoke_b2_frozenvlm/` | 4.03 GB, 4.87 GB | Not needed |
| `data/phase1/heldout_frames.npz` | 111 MB | Evaluation only; stays on the laptop |
| `logs/` | 1.4 MB | Reference values: conversion took 2 min 38 s and 28.8 GB RSS; smoke-run losses |
| The laptop's Hugging Face cache | 256 MB | Holds only the 8 smoke-test episodes, not the dataset |

So the artifacts supply the teacher and the reference numbers. They do not supply the training data: the full dataset has never been on the laptop.

### 9.5 Why the cloud needs no compiler

`scripts/phase1/env.sh` exists to give the laptop a user-local gcc, a PATH free of Windows binaries, and the simulator's EGL settings. None of that applies in the cloud. What does apply is that two locked packages have no prebuilt wheel:

- `av` 14.4.0 (source only on PyPI; needs FFmpeg 7 headers and a C compiler). A plain `uv sync --frozen` fails on it in a clean environment (`net_setup.log`). The runbook skips it and installs the `av` 15.1.0 wheel, which satisfies LeRobot's `av>=14.2.0`. The smoke run's losses are identical with either version.
- `evdev` 1.9.2 (pulled in by `pynput` for keyboard teleoperation). Training never imports it. The runbook skips it.

One more trap: `src/openpi/models_pytorch/gemma_pytorch.py:3` imports `pytest`, which is a development dependency. Installing with `--no-dev` makes the training script fail at import (`ModuleNotFoundError: No module named 'pytest'`, seen in the first clean smoke attempt). The runbook installs the default groups.

## 10. Runbook verification

The blocks were extracted from the finished `runbook.md` with its own `step` helper and run in the simulated cloud (`_cloudsim/v/rb.sh`). "Stand-in" means a small script that replaces the training program, emits log lines in the real format and creates checkpoint directories by the real save and resume rules, so that the shell logic around the training command can be executed without a GPU.

| Block | Status | What it rests on |
|---|---|---|
| `step` helper | Verified | Used to run every block below |
| `P0` probe | Verified | Ran as written. Output in `_cloudsim/W3/ovx_probe.txt` |
| `P0B` throughput | Verified | Ran as written: 1.4 GB/s write, 3.3 GB/s read, 2,000 small files in 53 ms, 25.9 MB/s from Google storage, 18.4 MB/s from Hugging Face |
| `P1` environment and helper scripts | Verified | Ran as written in a new directory; all five generated scripts pass `bash -n` and were then used |
| `P2A` online setup | Verified, with a limit | Ran as written against the existing online workspace (uv installer, unpack, sync, `av`, patch copy, `ENV_OK` including a bf16 GPU matmul). The from-scratch behaviour of the same commands is shown by `net_setup2.log` (2 min 21 s) and `offline_setup.log`, not by this rerun |
| `P2B` offline setup | Verified, with a limit | Ran as written in a new directory from the bundle with uv and Hugging Face in offline mode: 70 s, `ENV_OK`. That run had a network interface available. True network isolation was shown earlier for the same command sequence (`offline_setup.log`, inside `unshare -rn`: "Could not resolve host", then `ENV_OK`) |
| `P3A` online data and teacher | Partly verified | Not run as written: it downloads 34.94 GB. Verified separately: the `huggingface-cli` command with `--include "meta/*"`; anonymous directory download through openpi's `maybe_download` on the 2 KB assets folder (1.7 s); both `curl` seeds with matching checksums; `check_data.sh` on the 8-episode subset. The teacher conversion command ran in phase 1 (`logs/convert_teacher.log`) but not in the clean environment |
| `P3B` offline data and teacher | Verified on the subset | Ran as written with the 8-episode bundle: checksums pass, `TEACHER_OK`, `DATA_OK episodes 8 frames 2197`, `CHECK_DATA PASS` |
| `P4` smoke run | Training command verified; wrapper verified with the stand-in | The identical Python command passed in the online workspace (`t_smoke.log`). The block's surrounding lines ran as written with the stand-in, and its log filter was run on the real log |
| Smoke run, offline workspace | Verified up to data-loader creation only | The earlier worker's offline smoke run got as far as "local_batch_size: 2": configuration, normalization statistics and tokenizer from the carried cache with no download, and the dataset loaded offline. The machine restarted during model loading. Model loading and training steps were not exercised in the offline workspace. They use the same code and the same package versions as the online workspace |
| `P5` timed run | Not verifiable here | Needs a GPU of 40 GB or more and the full dataset. The flags were parsed on the CPU into the intended configuration (Appendix B). The block ran as written with the stand-in, including the failure branch |
| `P6`, `P6P` launch | Shell logic verified with the stand-in | Ran as written with tmux on an isolated socket: start, refusal while a run is active, skip when finished, two parallel sessions. The `nohup` fallback line was run on its own |
| `run_one.sh` fresh / resume / done | Resume verified on the GPU with the real training script; the other two branches with the stand-in | `_cloudsim/v/resume_test.out` and `W/logs/kd_20261003_161704.log`. The script as generated by `P1`, with batch 2, 8 episodes and 10 steps substituted through its override variables, found checkpoint 9, added `--resume`, loaded the model, the optimizer state and the step counter, ran the remaining step, saved checkpoint 10 and reported `RUN kd PASS` in 2 min 13 s (peak host RSS 17.1 GB). Restart-from-scratch after a crash before the first checkpoint, and skip-when-finished, ran with the stand-in |
| `P7` status | Verified | Ran as written during and after stand-in runs; the log filter was also run on the real smoke log |
| `P8` packaging | Verified | Ran as written on four real-size files (hard links to the two real 4.11 GB checkpoints): 13 files, 16,451.7 MB, `sha256sum -c` all OK. The split and reassemble commands were run on a 760 MB file and gave identical checksums |
| Appendix A (Slurm, containers) | Not verifiable here | No scheduler on the laptop; the container shared-memory limit was not simulated. The single-process loader rate behind the no-worker fallback is measured (`v/c2_default.log`) |

Problems the verification found and fixed in the runbook: a probe URL that returns 404 when reachable; a proxy line that printed nothing when no proxy is set; log lines hidden behind the progress bar in the status output; a running-process check that matched unrelated processes; and a log file that a restart within the same second would overwrite.

## 11. The laptop restart

At about 15:31 CDT the whole WSL virtual machine stopped and restarted, ending the earlier worker.
What the logs show:

- The previous boot's journal ends at 15:30:30 with `dxgk: dxgkio_query_adapter_info: Ioctl failed: -22`. The same line appears at every CUDA start on this machine (15:15:28, 15:30:11) and is not an error in itself.
- There is no out-of-memory kill, no filesystem error and no shutdown sequence in that journal. `free` showed 24 GB free a quarter of an hour earlier.
- The interrupted smoke run's log ends in zero padding, which is what an unflushed file looks like after a virtual machine is cut off rather than shut down.
- The job running at the time was the offline smoke run, 27 s after start, while it was loading the teacher and student onto the GPU inside a network-isolated user namespace (`unshare -rn`). The same job outside that namespace had passed 45 minutes earlier, and GPU model loads had run about ten times that afternoon without incident.

The cause is not established. The stop came from outside the Linux guest, which points at the WSL virtual machine or the Windows host (memory pressure on the host, or the GPU paravirtualization driver) rather than at the training code. The namespace is the one thing that differed from the runs that passed, but one event does not show it was the trigger.

Bearing on the cloud run: none directly. A bare-metal or containerized Linux node has no WSL layer and the runbook never uses `unshare`; that was only the laptop's way of simulating a machine without internet. Two general lessons are already built into the runbook: runs go through tmux with a log file and can resume, and only one heavy job runs per GPU.

Since the restart I have run GPU work only as a 2,048×2,048 matmul inside the environment check, plus one resume test of the real training script (batch 2, run alone, outside any namespace, after this report had been saved). It ran from 16:17 to 16:19 CDT and passed. So the same kind of job that was running when the machine went down has now completed twice outside the namespace and never inside it.

A second laptop-side observation: free space on the Windows drive fell from about 98 GB to 48 GB during this session. The WSL virtual disk grows when files are written and does not shrink when they are deleted. This worktree's `_cloudsim/` directory peaked at about 60 GB and holds 12 GB after cleanup. Reclaiming the space needs a host-side compaction of the virtual disk after the worktree is torn down; that is outside what a worker can do.

## 12. What only the mentor can answer

1. Which GPUs the OVX nodes have (L40S, L40 or A40), how many are free, and whether they are shared with other users' jobs.
2. Whether the node can reach the internet, through a proxy or not, and which hosts are allowed.
3. Whether work runs directly on the node, in a container, in a VM or through a scheduler, and whether there is a time limit per job or per session.
4. Where the writable area is, how large it is, whether it is a local disk or a network mount, whether it has a file-count quota, and whether scratch space is purged on a schedule.
5. How files can get in: 1 MB (the code), 7.2 GB (the teacher), and for path B about 52.5 GB. Whether USB media are allowed.
6. How 16.5 GB of results can get out, whether there is a per-file limit, and whether institute policy allows model weights to leave. If only small files can leave, the logs and measured speed still can.
7. Whether the mentor can leave a run unattended for 7-13 hours, and whether background processes survive logout on that machine.
8. Whether a memory or CPU limit applies to the mentor's session (the probe reads the cgroup values).
9. If the question about a ViT meant a separate vision model (section 8.4), which one.

Items 1-4 and 8 are answered by the probe output. Items 5-7 and 9 need a reply in words.

## 13. Still unknown

- Peak memory at batch 64 on a real 48 GB card. Predicted 35.4 GB; measured nowhere.
- Step time on the actual card. Estimated 5.3-9.4 s for an L40S; an A40 would be slower by an unknown factor.
- Whether the multi-GPU (DDP) path works with this project's freezing and distillation changes.
- Whether converting the teacher in the cloud reproduces the laptop's file exactly.
- Size of `optimizer.pt` with the phase 2 freeze set (predicted 3.0 GB).
- Time to build the 35 GB Arrow cache and to start the loader on the full dataset.
- Whether PyTorch's cu128 wheel runs on the institute's driver. CUDA's compatibility rule needs driver 525 or newer; the environment check tests it directly.
- Whether the benchmark's evaluation initial states coincide with training-demonstration initial states (section 4.5).
- The cause of the laptop restart.
- The share of the vision encoder under `torch.compile`.
- The text of OpenVLA's Appendix E; the list of dataset modifications is taken from the regeneration script's docstring.

## 14. Ambiguities I resolved, and how

| Ambiguity | Resolution |
|---|---|
| "Bandwidth" | Treated as every throughput limit from network to GPU, as the brief specifies, with the binding one named |
| "A ViT before the VLA part" | Answered for SigLIP inside Pi0.5, which is the only vision model present; the other reading is stated in section 8.4 |
| "Read vs write" | Covered for storage, for the network, and for path permissions |
| "Synthetic data for testing" | Read as two questions: fake inputs to test the pipeline, and generated data to train on. Both answered |
| What "the cloud" is | Assumed OVX nodes with 48 GB cards on the evidence of NVIDIA's specifications; the runbook assumes nothing and branches on the probe |
| How the mentor receives commands | As a markdown file with named blocks and a three-line helper, so each step is one short command; copy-paste works as a fallback |
| Privileges | Designed for no sudo, no Docker, no compiler |
| Private repository | Tarball instead of a clone, so no credential exists to mishandle |
| Use of several GPUs | The two runs in parallel rather than DDP, because only the single-GPU path is verified |
| What comes back | Model weights and normalization assets for steps 1,500 and 3,000 of both runs, plus logs. Not optimizer state, not the step-2,999 copies |
| The plan's 54 GB and 80 GB figures | Corrected from primary data and measurement rather than carried forward |
| Fallback if batch 64 does not fit | Not pre-decided. The runbook tells the mentor to stop and report, because the alternatives change the experiment |
| Worker count | 4, which gives at least 20× headroom on the laptop's measurements at low startup cost |

## 15. Recommended follow-up work

None of this is needed before the runbook is used. All of it is a change to the repository, which this task was told not to make. I recommend shipping items 1-4 as one small pull request.

1. **Correct the README and the survey plan**: dataset 34.94 GB (70 GB on disk), 48 GB card sufficient by estimate, optimizer state about 3 GB, three checkpoints per run.
2. **Commit the helper scripts** now embedded in the runbook as `scripts/phase2/`: probe, cloud environment file, environment and data checks, launch script, packaging. The runbook would shrink to a handful of one-line commands and the scripts would be under version control and CI lint.
3. **Make the lock file installable without a compiler**: pin `av` to a version with wheels and drop or mark optional `evdev`. Move the `pytest` import out of `gemma_pytorch.py` so training does not depend on a development package.
4. **Fix the double save** at steps N−1 and N in `save_checkpoint`.
5. **Make resume continue the data order and random state** instead of restarting them, or at least log a warning.
6. **Add a fake-data distillation config** for a test that needs no dataset.
7. **Export only the trained tensors.** 2.41 GB of each 4.11 GB checkpoint is frozen or unused and identical to the teacher's. A delta export would cut the return trip from 16.5 GB to about 6.8 GB.
8. **Reuse the teacher's image embeddings in the student** during KD, about 6.5% off the step time.
9. **Verify the DDP path** if a larger run is ever planned.

## 16. Decisions for the captain

1. **Send the runbook and the 1 MB code tarball to the mentor now**, asking for the probe output first. The tarball must first be regenerated with the command in section 9.1. Recommended: the probe takes five minutes and settles path A or B, the GPU, and the storage before anything large moves.
2. **After the probe: path A or path B, and the teacher's route.** If the cloud has internet, path A needs nothing more from outside except, preferably, the 7.2 GB teacher file. If not, the 52.5 GB bundle has to be built (section 9.3; it needs the 35 GB dataset downloaded onto a disk with room, which the laptop's Windows drive currently lacks) and physically carried in. Both depend on the mentor's answers to items 5 and 6 of section 12.
3. **How the 16.5 GB of weights leaves the institute.** This is a question for the mentor (section 12, item 6) that the captain has to ask.
4. **Whether to ship the follow-up repository changes** (section 15, items 1-4) before or after the cloud run.

All four gate the backlog item `vla-phase2`. Its current hold text still says the plan needs an 80 GB card, that a 48 GB card forces batch 32, and that about 70 GB must be downloaded; sections 3 and 6 supersede those three statements.

## Sources

- NVIDIA OVX product page (specification table): https://www.nvidia.com/en-us/data-center/products/ovx/
- NVIDIA press release, 2023-08-08 (OVX with L40S): https://nvidianews.nvidia.com/news/nvidia-global-data-center-system-manufacturers-to-supercharge-generative-ai-and-industrial-digitalization
- NVIDIA press release, 2022-03-22 (first OVX, A40): https://nvidianews.nvidia.com/news/nvidia-launches-data-center-scale-omniverse-computing-system-for-industrial-digital-twins
- NVIDIA blog, 2023-03-21 (third-generation OVX, L40): https://blogs.nvidia.com/blog/third-generation-ovx-computing-systems/
- NVIDIA product pages: https://www.nvidia.com/en-us/data-center/l40s/ , https://www.nvidia.com/en-us/data-center/l40/ , https://www.nvidia.com/en-us/data-center/a40/ , https://www.nvidia.com/en-us/data-center/a100/ , https://www.nvidia.com/en-us/data-center/h100/
- CUDA release notes (driver compatibility): https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- Dataset card and API: https://huggingface.co/datasets/physical-intelligence/libero
- Upstream dataset: https://huggingface.co/datasets/openvla/modified_libero_rlds
- OpenVLA regeneration script: https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py
- LIBERO: arXiv 2306.03310. OpenVLA: arXiv 2406.09246. Shallow-π: arXiv 2601.20262.
- Project files cited by `file:line` above are at commit `966da5c`. `lerobot` paths are in the installed package at commit `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`.

## Appendix A. Raw laptop measurements (RTX 5080 Laptop 16 GB, WSL2, 2026-10-03)

### A.1 Student training step vs batch size (`_cloudsim/m1_mem_slope.py 1,2,4,8,12,16,20,24`, teacher not loaded, bf16, gradient checkpointing, phase-2 freeze set)
```
2718 MiB, 16303 MiB
{"batch": 1, "grad_bytes_GB": 1.491599488, "params_with_grad_M": 702.677024, "base_alloc_GB": 4.131748352, "peak_alloc_GB": 5.741744128, "peak_reserved_GB": 5.941231616, "peak_minus_base_GB": 1.6099957759999999, "fwd_s": 0.07020211600077648, "bwd_s": 0.13164711166367246, "opt_s": 0.000731536332750693, "loss": 7.813653469085693}
{"batch": 2, "grad_bytes_GB": 1.491599488, "params_with_grad_M": 702.677024, "base_alloc_GB": 4.133557248, "peak_alloc_GB": 6.05535488, "peak_reserved_GB": 6.150946816, "peak_minus_base_GB": 1.9217976320000005, "fwd_s": 0.12030529900221154, "bwd_s": 0.2505281423315561, "opt_s": 0.0007710733383040255, "loss": 8.663565635681152}
{"batch": 4, "grad_bytes_GB": 1.491599488, "params_with_grad_M": 702.677024, "base_alloc_GB": 4.137176064, "peak_alloc_GB": 6.66863872, "peak_reserved_GB": 6.880755712, "peak_minus_base_GB": 2.5314626559999995, "fwd_s": 0.23390826266161943, "bwd_s": 0.4956931933362891, "opt_s": 0.0006926373316673562, "loss": 7.171790599822998}
{"batch": 8, "grad_bytes_GB": 1.491599488, "params_with_grad_M": 702.677024, "base_alloc_GB": 4.144414208, "peak_alloc_GB": 7.895206912, "peak_reserved_GB": 8.183087104, "peak_minus_base_GB": 3.750792704, "fwd_s": 0.4339934226688153, "bwd_s": 0.9912190309987636, "opt_s": 0.0007034116715658456, "loss": 6.925825595855713}
{"batch": 12, "grad_bytes_GB": 1.491599488, "params_with_grad_M": 702.677024, "base_alloc_GB": 4.151652352, "peak_alloc_GB": 9.121775616, "peak_reserved_GB": 10.009706496, "peak_minus_base_GB": 4.970123264000001, "fwd_s": 0.6254160263342783, "bwd_s": 1.4897737886640243, "opt_s": 0.0007385176674385244, "loss": 6.658336639404297}
{"batch": 16, "grad_bytes_GB": 1.491599488, "params_with_grad_M": 702.677024, "base_alloc_GB": 4.158890496, "peak_alloc_GB": 10.34834432, "peak_reserved_GB": 10.930356224, "peak_minus_base_GB": 6.189453824000001, "fwd_s": 0.8456416473345598, "bwd_s": 1.9452976813336136, "opt_s": 0.0007808873342582956, "loss": 7.486802577972412}
{"batch": 20, "grad_bytes_GB": 1.491599488, "params_with_grad_M": 702.677024, "base_alloc_GB": 4.16612864, "peak_alloc_GB": 11.574913536, "peak_reserved_GB": 12.505317376, "peak_minus_base_GB": 7.408784896, "fwd_s": 1.0336264213304578, "bwd_s": 2.424816082333564, "opt_s": 0.0007875823309101785, "loss": 5.00400447845459}
{"batch": 24, "grad_bytes_GB": 1.491599488, "params_with_grad_M": 702.677024, "base_alloc_GB": 4.173366784, "peak_alloc_GB": 12.807538688, "peak_reserved_GB": 14.478737408, "peak_minus_base_GB": 8.634171903999999, "fwd_s": 1.5251060750015313, "bwd_s": 3.3203240190002057, "opt_s": 0.0006869196658954024, "loss": 9.489899635314941}
RESULT {"frozen_M": 941.450992, "trainable_M": 1070.862368, "trainable_bytes_GB": 2.227978368, "all_param_bytes_GB": 4.112827232, "after_model_alloc_GB": 4.112896, "runs": [{"batch": 1, "grad_bytes_GB": 1.491599488, "params_with_grad_M": 702.677024, "base_alloc_GB": 4.131748352, "peak_alloc_GB": 5.741744128, "peak_reserved_GB": 5.941231616, "peak_minus_base_GB": 1.6099957759999999, "fwd_s": 0.07020211600077648, "bwd_s": 0.13164711166367246, "opt_s": 0.000731536332750693, "loss": 7.813653469085693}, {"batch": 2, "grad_bytes_GB": 1.491599488, "params_with_grad_M": 702.677024, "base_alloc_GB": 4.133557248, "peak_alloc_GB": 6.05535488, "peak_reserved_GB": 6.150946816, "peak_minus_base_GB": 1.9217976320000005, "fwd_s": 0.12030529900221154, "bwd_s": 0.2505281423315561, "opt_s": 0.0007710733383040255, "loss": 8.663565635681152}, {"batch": 4, "grad_bytes_GB": 1.491599488, "params_with_grad_M": 702.677024, "base_alloc_GB": 4.137176064, "peak_alloc_GB": 6.66863872, "peak_reserved_GB": 6.880755712, "peak_minus_base_GB": 2.5314626559999995, "fwd_s": 0.2
```

### A.2 Teacher no-grad forward vs batch, and optimizer (AdamW) footprint (`m1b_teacher.py`, `m1_mem_slope.py 2,8,16 adam`)
```
{"teacher_alloc_GB": 7.473025024, "param_bytes_GB": 7.472956256}
{"batch": 1, "teacher_fwd_s": 0.14206597466545645, "per_sample_s": 0.14206597466545645, "transient_GB": 0.14173695999999936, "peak_alloc_GB": 7.626904064}
{"batch": 2, "teacher_fwd_s": 0.2636918430022585, "per_sample_s": 0.13184592150112925, "transient_GB": 0.283469824, "peak_alloc_GB": 7.772254208}
{"batch": 4, "teacher_fwd_s": 0.4891197993371558, "per_sample_s": 0.12227994983428896, "transient_GB": 0.5669355520000003, "peak_alloc_GB": 8.062956544}
{"batch": 8, "teacher_fwd_s": 0.97014123366777, "per_sample_s": 0.12126765420847126, "transient_GB": 1.13386752, "peak_alloc_GB": 8.64436224}
{"batch": 16, "teacher_fwd_s": 1.9012485636679533, "per_sample_s": 0.11882803522924708, "transient_GB": 2.2677299200000007, "peak_alloc_GB": 9.807172096}
{"batch": 24, "teacher_fwd_s": 2.8592851416760823, "per_sample_s": 0.1191368809031701, "transient_GB": 3.40159232, "peak_alloc_GB": 10.969981952}
{"batch": 32, "teacher_fwd_s": 3.92572182432923, "per_sample_s": 0.12267880701028844, "transient_GB": 4.535455232, "peak_alloc_GB": 12.13279232}
==== ADAM
{'batch': 2, 'grad_bytes_GB': 1.492, 'params_with_grad_M': 702.677, 'after_adam_alloc_GB': 8.608, 'peak_incl_adam_GB': 10.1, 'base_alloc_GB': 7.117, 'peak_alloc_GB': 9.039, 'peak_reserved_GB': 10.19, 'peak_minus_base_GB': 1.922, 'fwd_s': 0.127, 'bwd_s': 0.246, 'opt_s': 0.077, 'loss': 14.698}
{'batch': 8, 'grad_bytes_GB': 1.492, 'params_with_grad_M': 702.677, 'after_adam_alloc_GB': 8.619, 'peak_incl_adam_GB': 10.878, 'base_alloc_GB': 7.128, 'peak_alloc_GB': 10.878, 'peak_reserved_GB': 11.551, 'peak_minus_base_GB': 3.751, 'fwd_s': 0.424, 'bwd_s': 0.971, 'opt_s': 0.076, 'loss': 8.102}
{'batch': 16, 'grad_bytes_GB': 1.492, 'params_with_grad_M': 702.677, 'after_adam_alloc_GB': 8.634, 'peak_incl_adam_GB': 13.332, 'base_alloc_GB': 7.142, 'peak_alloc_GB': 13.332, 'peak_reserved_GB': 14.72, 'peak_minus_base_GB': 6.189, 'fwd_s': 1.183, 'bwd_s': 3.049, 'opt_s': 0.137, 'loss': 5.804}
```

### A.3 Vision encoder share (`m2_vit_share.py teacher|student 1,8,16`)
```
{"model": "teacher", "params_M": {"vision_tower": 412.4, "multi_modal_projector": 2.4, "vlm_layers": 1981.9, "embed_tokens(tied lm_head)": 526.6, "expert_layers": 424.8, "expert_lm_head(unused)": 263.3, "other": 5.3}, "total_M": 3616.8, "bytes_GB": {"vision_tower": 0.827, "multi_modal_projector": 0.005, "vlm_layers": 3.964, "embed_tokens(tied lm_head)": 1.053, "expert_layers": 1.076, "expert_lm_head(unused)": 0.527, "other": 0.021}}
{"siglip_layers": 27, "hidden": 1152, "mlp": 4304, "patch": 14, "image": 224, "heads": 16}
{"model": "teacher", "batch": 1, "image_tokens_per_image": 256, "emb_dim": 2048, "vit_3img_s": 0.030801451997831464, "vit_transient_GB": 0.008863744, "train_style_fwd_s": 0.14203498400456738, "vit_share_of_fwd": 0.21685820724872254, "infer_10step_eager_s": 0.3218439180054702, "vit_share_of_infer": 0.0957030730570088}
{"model": "teacher", "batch": 8, "image_tokens_per_image": 256, "emb_dim": 2048, "vit_3img_s": 0.15788927000539843, "vit_transient_GB": 0.070909952, "train_style_fwd_s": 0.9623267899878556, "vit_share_of_fwd": 0.1640703258478245}
{"model": "teacher", "batch": 16, "image_tokens_per_image": 256, "emb_dim": 2048, "vit_3img_s": 0.3033596159948502, "vit_transient_GB": 0.141819904, "train_style_fwd_s": 1.9100083439989248, "vit_share_of_fwd": 0.15882633023461856}
{"model": "student", "params_M": {"vision_tower": 412.4, "multi_modal_projector": 2.4, "vlm_layers": 660.6, "embed_tokens(tied lm_head)": 526.6, "expert_layers": 141.6, "expert_lm_head(unused)": 263.3, "other": 5.3}, "total_M": 2012.3, "bytes_GB": {"vision_tower": 0.827, "multi_modal_projector": 0.005, "vlm_layers": 1.321, "embed_tokens(tied lm_head)": 1.053, "expert_layers": 0.359, "expert_lm_head(unused)": 0.527, "other": 0.021}}
{"siglip_layers": 27, "hidden": 1152, "mlp": 4304, "patch": 14, "image": 224, "heads": 16}
{"model": "student", "batch": 1, "image_tokens_per_image": 256, "emb_dim": 2048, "vit_3img_s": 0.02976386999944225, "vit_transient_GB": 0.008863744, "train_style_fwd_s": 0.06819687499955762, "vit_share_of_fwd": 0.43644037941086483, "infer_10step_eager_s": 0.15291564400831703, "vit_share_of_infer": 0.19464241341993369}
{"model": "student", "batch": 8, "image_tokens_per_image": 256, "emb_dim": 2048, "vit_3img_s": 0.15821775500080548, "vit_transient_GB": 0.070909952, "train_style_fwd_s": 0.4296398360020248, "vit_share_of_fwd": 0.3682567158415446}
{"model": "student", "batch": 16, "image_tokens_per_image": 256, "emb_dim": 2048, "vit_3img_s": 0.3047442449897062, "vit_transient_GB": 0.141819904, "train_style_fwd_s": 0.8399521599931177, "vit_share_of_fwd": 0.362811430822683}
```

### A.4 Data loader throughput (`m3_loader.py <workers> 64 <batches> 0..7`, `m3b_persample.py`)
```
LOADER {"workers": 4, "batch": 64, "batches": 200, "build_s": 0.3, "first_batch_s": 16.7, "samples_per_s": 241.4, "s_per_batch": 0.265, "cpu_count": 16, "image_shape": {"base_
LOADER {"workers": 2, "batch": 64, "batches": 40, "build_s": 0.4, "first_batch_s": 8.3, "samples_per_s": 223.2, "s_per_batch": 0.287, "cpu_count": 16, "image_shape": {"base_0_
LOADER {"workers": 8, "batch": 64, "batches": 200, "build_s": 0.2, "first_batch_s": 32.0, "samples_per_s": 214.2, "s_per_batch": 0.299, "cpu_count": 16, "image_shape": {"base_
LOADER {"workers": 4, "batch": 64, "batches": 200, "build_s": 0.2, "first_batch_s": 15.8, "samples_per_s": 354.4, "s_per_batch": 0.181, "cpu_count": 16, "image_shape": {"base_
LOADER {"workers": 8, "batch": 64, "batches": 200, "build_s": 0.3, "first_batch_s": 31.2, "samples_per_s": 358.9, "s_per_batch": 0.178, "cpu_count": 16, "image_shape": {"base_
LOADER {"workers": 2, "batch": 64, "batches": 200, "build_s": 0.2, "first_batch_s": 8.6, "samples_per_s": 194.0, "s_per_batch": 0.33, "cpu_count": 16, "image_shape": {"base_0_
LOADER {"workers": 2, "batch": 64, "batches": 100, "build_s": 0.2, "first_batch_s": 8.3, "samples_per_s": 215.4, "s_per_batch": 0.297, "cpu_count": 16, "image_shape": {"base_0
PERSAMPLE {"n_frames": 2197, "read_decode_ms": 33.83, "full_ms": 34.87, "transform_ms": 1.03, "single_core_samples_per_s": 28.7, "raw_keys": {"image": [3, 256, 256], "wrist_im
```

### A.4b Earlier 30-batch loader runs, recovered from the first worker's session transcript (no log file; same script, `m3_loader.py <workers> 64 30 0..7`)
```
LOADER {"workers": 2, "batch": 64, "batches": 30, "first_batch_s": 8.4, "samples_per_s": 210.7, "s_per_batch": 0.304}
LOADER {"workers": 4, "batch": 64, "batches": 30, "first_batch_s": 15.9, "samples_per_s": 375.1, "s_per_batch": 0.171}
LOADER {"workers": 8, "batch": 64, "batches": 30, "first_batch_s": 31.1, "samples_per_s": 401.4, "s_per_batch": 0.159}
LOADER {"workers": 16, "batch": 64, "batches": 30, "first_batch_s": 65.0, "samples_per_s": 541.4, "s_per_batch": 0.118}
```

### A.5 Single-process sample cost, default threads versus one thread (`_cloudsim/v/c2_persample.py`, CPU only, this session)
```
PERSAMPLE {"torch_threads": 16, "read_decode_ms": 37.19, "full_ms": 39.81, "samples_per_s_one_process": 25.1}
PERSAMPLE {"torch_threads": 1, "read_decode_ms": 5.8, "full_ms": 9.24, "samples_per_s_one_process": 108.2}
```

## Appendix B. Verification outputs

### B.1 The runbook's training flags parsed on the CPU (`_cloudsim/v/c1_flags.py`)
```
CFG pi05_libero_l06 exp kd_b64_s3000 bs 64 steps 3000 save 1500 log 10 workers 4 prec bfloat16 resume False overwrite False wandb False seed 42 lr (300, 2.5e-05
, 3000, 2.5e-06) opt ('AdamW', 0.9, 0.95, 1e-08, 1e-10, 1.0) w_gt/w_teacher (1.0, 1.0) depth 6 teacher pi05_libero nfreeze 4 weights checkpoints/pi05_libero_pyt
orch ckpt_dir checkpoints/pi05_libero_l06/kd_b64_s3000 episodes None repo physical-intelligence/libero assets gs://openpi-assets/checkpoints/pi05_libero/assets 
horizon 10 max_token_len 200 action_dim 32
CFG pi05_libero_l06_taskonly exp taskonly_b64_s3000 bs 64 steps 3000 save 1500 log 10 workers 4 prec bfloat16 resume False overwrite False wandb False seed 42 l
r (300, 2.5e-05, 3000, 2.5e-06) opt ('AdamW', 0.9, 0.95, 1e-08, 1e-10, 1.0) w_gt/w_teacher (1.0, 0.0) depth 6 teacher pi05_libero nfreeze 4 weights checkpoints/
pi05_libero_pytorch ckpt_dir checkpoints/pi05_libero_l06_taskonly/taskonly_b64_s3000 episodes None repo physical-intelligence/libero assets gs://openpi-assets/c
heckpoints/pi05_libero/assets horizon 10 max_token_len 200 action_dim 32
CFG pi05_libero_l06 exp kd_b64_s3000 bs 64 steps 3000 save 1500 log 10 workers 4 prec bfloat16 resume True overwrite False wandb False seed 42 lr (300, 2.5e-05,
 3000, 2.5e-06) opt ('AdamW', 0.9, 0.95, 1e-08, 1e-10, 1.0) w_gt/w_teacher (1.0, 1.0) depth 6 teacher pi05_libero nfreeze 4 weights checkpoints/pi05_libero_pyto
rch ckpt_dir checkpoints/pi05_libero_l06/kd_b64_s3000 episodes None repo physical-intelligence/libero assets gs://openpi-assets/checkpoints/pi05_libero/assets h
orizon 10 max_token_len 200 action_dim 32
TIMED timed50 64 50 True None
FAKE repo_id -> fake LeRobotLiberoDataConfig
FAKE with LIBERO transforms FAILS: KeyError 'image'
```

### B.2 Clean online install and smoke run (`net_setup2.log`, `t_smoke.log`)
```
real	2m21.446s
UV_SYNC_EXIT=0
 + av==15.1.0
real	0m1.242s
AV_EXIT=0
CP_EXIT=0
8.8G	.venv
9.5G	_cloudsim/W/cache/uv
94M	_cloudsim/W/cache/uv-python
49M	_cloudsim/W/bin
14:43:37.953 [I] Loaded teacher pyTorch weights from checkpoints/pi05_libero_pytorch/model.safetensors (27804:distill_pytorch.py:435)
14:43:37.955 [I] Step 0 (after_model_creation): GPU memory - allocated: 11.59GB, reserved: 11.59GB, free: 0.01GB, peak_allocated: 11.59GB, peak_reserved: 11.59GB (27804:distil
14:43:37.968 [I] Initialized student layers from teacher layers [0, 3, 7, 10, 14, 17] (backbone + action expert) (27804:distill_pytorch.py:498)
14:43:38.789 [I] Froze 1865.4M params with prefixes ('paligemma_with_expert.paligemma.model.vision_tower.', 'paligemma_with_expert.paligemma.model.multi_modal_projector.', 'pa
14:43:38.790 [I] Trainable params: 146.9M                                                         (27804:distill_pytorch.py:512)
14:44:05.639 [I] step=0 loss=1.1697 lr=2.50e-08 grad_norm=12.15 time=26.8s                        (27804:distill_pytorch.py:638)
14:44:06.303 [I] step=1 loss=0.5464 lr=5.00e-08 grad_norm=7.32 time=0.7s                          (27804:distill_pytorch.py:638)
14:44:07.011 [I] step=2 loss=2.4015 lr=7.49e-08 grad_norm=26.04 time=0.7s                         (27804:distill_pytorch.py:638)
14:44:07.675 [I] step=3 loss=0.6410 lr=9.99e-08 grad_norm=11.56 time=0.7s                         (27804:distill_pytorch.py:638)
14:44:08.348 [I] step=4 loss=0.2641 lr=1.25e-07 grad_norm=4.27 time=0.7s                          (27804:distill_pytorch.py:638)
14:44:09.021 [I] step=5 loss=1.1945 lr=1.50e-07 grad_norm=12.50 time=0.7s                         (27804:distill_pytorch.py:638)
14:44:09.687 [I] step=6 loss=0.6005 lr=1.75e-07 grad_norm=8.73 time=0.7s                          (27804:distill_pytorch.py:638)
14:44:10.355 [I] step=7 loss=1.1265 lr=2.00e-07 grad_norm=9.49 time=0.7s                          (27804:distill_pytorch.py:638)
14:44:11.041 [I] step=8 loss=1.0393 lr=2.25e-07 grad_norm=12.24 time=0.7s                         (27804:distill_pytorch.py:638)
14:44:31.363 [I] Saved checkpoint at step 9 -> _cloudsim/W/vla_distill/checkpoints/pi05_libero_l06/smoke/9 (27804:distill_pytorch.py:191)
14:44:32.002 [I] step=9 loss=0.9379 lr=2.50e-07 grad_norm=12.95 time=21.0s                        (27804:distill_pytorch.py:638)
	Elapsed (wall clock) time (h:mm:ss or m:ss): 3:08.75
	Maximum resident set size (kbytes): 17870200
SMOKE_EXIT=0
```

### B.3 Offline install inside a network-isolated namespace (`offline_setup.log`), and how far the offline smoke run got (`t_smoke_w2.log`)
```
network check (expect failure):
curl: (6) Could not resolve host: pypi.org
000
Sat Oct  3 15:28:56 CDT 2026

real	0m12.921s
user	0m0.280s
sys	0m6.291s
uv 0.12.19 (x86_64-unknown-linux-gnu)
 + xxhash==3.5.0
 + yarl==1.20.0
 + zarr==3.0.8
 + zipp==3.22.0

real	0m52.789s
user	0m0.656s
sys	0m10.928s
Installed 1 package in 3.15s
 + av==15.1.0
torch 2.7.1+cu128 cuda True 1 NVIDIA GeForce RTX 5080 Laptop GPU
transformers 4.53.2 patched True
jax 0.5.3 [CpuDevice(id=0)] av 15.1.0
ENV_OK
Sat Oct  3 15:30:12 CDT 2026
EXIT=0
--- t_smoke_w2.log: first and last readable lines
Sat Oct  3 15:30:18 CDT 2026
15:30:30.978 [I] Created experiment checkpoint directory: _cloudsim/W2/vla_distill/checkpoints/pi05_libero_l06/smoke (114544:distill_pytorch.py:341)
15:30:31.052 [I] Loaded norm stats from gs://openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero (114544:config.py:203)
15:30:31.492 [I] local_batch_size: 2                                                              (114544:data_loader.py:325)
```

### B.4 Probe throughput block on the laptop (`_cloudsim/v/P0B.out`)
```
=== ovx_probe throughput 2026-10-03T21:03:11Z pwd=_cloudsim/W3
--- DISK WRITE 1GiB (fdatasync)
1073741824 bytes (1.1 GB, 1.0 GiB) copied, 0.75578 s, 1.4 GB/s
--- DISK READ 1GiB (direct if supported)
1073741824 bytes (1.1 GB, 1.0 GiB) copied, 0.321776 s, 3.3 GB/s
--- SMALL FILES: create and delete 2000
small_files_ms=53
--- NET DOWNLOAD
GCS 200MB: 209715200 bytes 25883847 B/s 8.102165s http=206
HF 37MB: 37414512 bytes 18436724 B/s 2.029347s http=200 host=https://us.aws.cdn.hf.co/xet-bridge-us/679dfaa536b51abda2a01b06/a4a9dffc6a6f89c36cd1bdd6b7661db89b86a5977f009
```
