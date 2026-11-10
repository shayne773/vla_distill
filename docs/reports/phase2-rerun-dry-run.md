# Second OVX runbook (fp32 master weights): laptop dry-run report

Date: 2026-11-06. Code: `vla_distill` main at `3d11c3e` (merge of https://github.com/shayne773/vla_distill/pull/3).
Started 2026-10-22/23 (draft and tarball), finished 2026-11-06 (laptop dry run, fixes, this report).
The runbook is [phase2-rerun-runbook.zh-TW.md](phase2-rerun-runbook.zh-TW.md) in this folder (written and sent as `runbook.md` / `runbook2.md`); the diagnosis it implements is [phase2-student-diagnosis.md](phase2-student-diagnosis.md).

## 1. Summary

- `runbook.md` is the second OVX runbook for the teammate, in Traditional Chinese, with every command and technical term in English. It keeps the first runbook's structure: named blocks run with `step <ID>`, and per step the command, the time, the success line and what to send back on failure. It implements items 1-9 of section (e) in [phase2-student-diagnosis.md](phase2-student-diagnosis.md) (mapping in section 3).
- `vla_distill-3d11c3e.tar.gz` (1,141,294 bytes, sha256 `18b95943e25253b8b7c99b1b7360b561beea8404a1dc2a5f904965c0e52cc41a`) is `git archive --format=tar.gz --prefix=vla_distill/ 3d11c3e`. It is reproducible byte for byte, and the runbook's `P1` and `R1` check that hash.
- Every block was run on the laptop, in order, through the runbook's own `step` helper, in a simulated `/data/vla_work` (`_cloudsim/W`). That workspace was rebuilt in the state the first OVX run left behind: old 966da5c code, the teammate's four real returned weights hard-linked into place, last run's `runs.txt`/`speed.txt`, and junk files for the cleanup step.
- The fix-is-active gate gives exactly the report's laptop numbers. `P4` with fp32 master weights: **0.5520 → `GATE PASS`**. The same block with `--no-pytorch-fp32-master-weights`: **0.0000 (0/65,811,426) → `GATE FAIL`**, and afterwards `P5`, `P6` and `run2.sh` all refuse to start. On a step-1500 checkpoint trained with the real LR schedule (warmup 300, decay 3000; batch 2, VLM frozen), the gate reads **0.8972 → PASS**. Last run's two real OVX step-1500 checkpoints read **0.0000 → FAIL**, so the mid-run check in `P7` separates the two cases cleanly.
- The dry run found three problems; all are fixed in `runbook.md` (section 5). None was in the training command or the gates.
- The runbook still needs, and only OVX can provide: batch 64 with master weights on the 48 GB card (predicted 44-45 GB reserved, gate at 46), the real s/step, the full-size `P4`/`P7` numbers (698.5M trainable rather than the laptop's 146.9M), and the fp32 fallback `P5F` (it needs more than 16 GB even at batch 2).
- Nothing needs to ship for this runbook, and no decision is pending. Open questions are for the teammate (section 7), and the runbook asks them in its section 13.

## 2. Deliverables

| File | What |
|---|---|
| `runbook.md` (in this folder: `phase2-rerun-runbook.zh-TW.md`) | The runbook (55 KB). Send it under the name **`runbook2.md`**: its `step` helper defaults to `RB=~/runbook2.md`. Section 2 now also says to change `RB=` if the file name differs |
| `vla_distill-3d11c3e.tar.gz` (not in the repository; rebuilt byte for byte by the `git archive` command below) | Code, 1,141,294 bytes, sha256 `18b95943e25253b8b7c99b1b7360b561beea8404a1dc2a5f904965c0e52cc41a` |
| `report.md` | This file |

Tarball checks (2026-11-06): `git archive --format=tar.gz --prefix=vla_distill/ 3d11c3e | sha256sum` gives the same hash. The unpacked tree is `diff -r` identical to a plain `git archive` extraction. It contains `def fp32_master_weight_step` at `scripts/distill_pytorch.py:155`. `git diff --diff-filter=D 966da5c 3d11c3e` is empty, so unpacking over the old code leaves no stale file behind.

## 3. What changed from the first runbook, and why

| Area | First runbook (`966da5c`) | Second runbook (`3d11c3e`) | Why / report (e) item |
|---|---|---|---|
| Setup | Probe, environment install, 35 GB download, teacher conversion (`P0B`, `P2A/B`, `P3A/B`) | `P0` read-only state check, then `P1` unpacks the new tarball over `$W/vla_distill` and re-runs the existing `check_env.sh` and `check_data.sh`. No `uv sync`. The install and download blocks survive only as appendix R (`R1`, `R2`), used only if a check fails | (e) 1: reuse the workspace, re-verify, skip setup, pure-Python change |
| Code check | none | `P1` asserts that Python imports `openpi` from `$W/vla_distill/src`, that `TrainConfig` has `pytorch_fp32_master_weights`, that both l06 configs default it to True, and that `fp32_master_weight_step` exists (`CODE_OK`) | Makes sure the editable install picks up the new code |
| Disk | n/a | `P2` deletes only last run's dead files (smoke/timed50 checkpoints, `2999`, `optimizer.pt`, xet cache, JAX teacher params). `P2B` deletes last run's four weights only if each sha256 matches what we received | Checkpoints grow from 7.1 to 12.5 GB (masters + fp32 Adam); `/data` is about 147 GB |
| Student init | n/a | `P3` builds the layer-mapped init and compares it with the laptop init hash | Reference for the gate |
| Fix-is-active gate | none | `P4`: 20-step, batch-2, constant-LR (`--lr-schedule.warmup-steps 1 --lr-schedule.decay-steps 100000`) smoke on 8 episodes, then `gate_check.py` (verbatim from (e) 2). `GATE PASS` needs both the `fp32 master weights for … M trainable params` log line and a fraction ≥ 0.25. `run2.sh`, `P5`, `P5F`, `P6` and `P6P` refuse to run without a `GATE PASS` | (e) 2 |
| Freeze list | 4 prefixes | 10 prefixes in `$W/r2/env2.sh`: the 4 old ones plus expert `lm_head`, final VLM `norm`, and layer 5's `mlp.`, `self_attn.q_proj.`, `self_attn.o_proj.`, `post_attention_layernorm.`. The comment states that `layers.5.` is the last VLM layer of the 6-layer student only | (e) 3 |
| Speed/memory gate | `P5` timed kd only, plus a bandwidth probe | `P5` times both runs at batch 64 with master weights: PASS if `peak_reserved_GB` ≤ 46, the master-weights line is present and s/step ≤ 1.1 × 4.7 / 7.7. SLOW at 1.1-1.5×, FAIL beyond that. Section 8.1 gives the decision rule; `P5F` holds option 3's flags pre-written (fp32, batch 16, 12k steps, peak 1.25e-5, warmup 1200, decay 12000 → 1.25e-6), selected with `VLA_RECIPE=fp32`. No bandwidth probe | (e) 4 |
| Runs | kd then task-only, `run_one.sh` | `run2.sh`: task-only first, then kd; 3000 steps, batch 64, seed 42, warmup 300, decay 3000, save 1500; `P6` (one GPU, sequential) or `P6P` (two GPUs, parallel). Auto fresh / `--resume` / skip-if-done as before. After success it deletes `2999` and the step-1500 `optimizer.pt` + `master_params.pt` | (e) 5 |
| Export | model + optimizer | `P8` exports `model.safetensors` + `assets/` only (hard links where possible) plus logs, `runs.txt`, `speed.txt`, `gate.txt`, `ovx_state.txt` and `SHA256SUMS`. `optimizer.pt` and `master_params.pt` of step 3000 stay on OVX for a later `--resume` | (e) 6, 8 |
| Monitoring | `P7` | `P7` as before, plus a one-time `gate_check.py` on each run's step-1500 checkpoint, cached in `gate.txt`, with a stop rule on `GATE FAIL` | (e) 7 |
| Return route | "your choice" | The same Dropbox route the teammate used last time; a priority order if the volume is limited | (e) 8 |
| Teammate's evaluation | n/a | Section 13 asks for the script and config, suites and episode counts, the numbers, the machine and the inference precision | (e) 9 |

Time on one L40S: task-only about 4.0 h, kd about 6.5 h, so about 10.5 h sequential, or about 6.5 h on two GPUs. These numbers rest on the measured OVX logs: mean s/step over steps 20-2990 was **4.699** (task-only) and **7.692** (kd) in the returned `taskonly_20261019_153941.log` / `kd_20261019_091114.log`. The report's estimate adds about 0.06 s for the larger optimizer step.

## 4. Laptop verification

Machine: RTX 5080 Laptop 16 GB under WSL2, `scripts/phase1/env.sh` sourced. The simulated workspace uses the first runbook's generated `env.sh` / `check_*.sh` / `run_one.sh` / `diag.sh` with `W=_cloudsim/W`, the 8-episode dataset subset and the real converted teacher (`TEACHER_OK`).
To fit 16 GB, the training blocks ran with the runbook's test-only override `VLA_EXTRA`, which appends the whole VLM (`paligemma_with_expert.paligemma.model.language_model.`) to the freeze list and, for P5/P6, adds `--batch-size 2 --data.lerobot-episodes 0..7`. `VLA_STEPS` shortened P6 to 40 steps. The override does not affect the gate metric, which only counts action-expert layers.

| Block / command | Status | Evidence |
|---|---|---|
| `step` helper (section 2) | Verified | Used for every block below. All 14 blocks extract (14 `>>>`, 14 `<<<`), pass `bash -n`, and are clean under shellcheck 0.11.0 at warning level (SC2086 excluded because the word-splitting of `$FREEZE`/`$ARGS` is intentional). The five scripts generated by P1 pass both checks too |
| `P0` | Verified | Ran as written: all workspace files `present`, `code: last run's code (966da5c), to be replaced in P1`, last run's `runs.txt`/`speed.txt` shown, `state written to …` |
| `P1` | Verified on the subset | Unpack and sha256, the five `r2/` scripts, `CODE_OK`, `ENV_OK` (patched transformers, bf16 GPU matmul), `TEACHER_OK`, `CHECK_DATA PASS`, `P1 OK`. In the sim this needed `VLA_EPISODES="0 … 7"`, because the sim holds 8 of 1,693 episodes: without it, `check_data.sh` started downloading the missing episodes and hit a Hugging Face 429. On OVX the full dataset is present (last run's `CHECK_DATA PASS`), so this is a sim artifact |
| FREEZE prefixes | Verified | All 10 prefixes match real parameter names. On the meta device, with the trainer's own `named_parameters` + `startswith` logic: `Froze 1313.8M`, `Trainable 698.5M` (655.4M bf16 + 43.1M fp32), masters 2.79 GB, AdamW 5.59 GB, the same for both configs. These are the runbook's reference values. `embed_tokens` is tied to the VLM `lm_head`, so it is not a separate tensor in the safetensors file but is frozen either way |
| `P2` | Verified | Deleted exactly the listed dead files (including a real 10 GB xet cache left by the 429 above), kept the weights; `DISK OK`; `VLA_NEED_GB=2000` → `DISK LOW … -> step P2B` |
| `P2B` | Verified, including the hashes | With the teammate's real returned weights hard-linked in place: three files matched the hard-coded sha256 and were deleted. A deliberately altered fourth was `KEEP (differs …)`, so `$W/out` was kept; `cache/uv` was deleted |
| `P3` | Verified | 35 s; `layer mapping (student <- teacher): [0, 3, 7, 10, 14, 17]`; `INIT_OK identical to the laptop init` (sha256 `bdd4540d…`) |
| `P4` fix active | Verified (VLM frozen) | `Froze 1865.4M`, `Trainable params: 146.9M`, `fp32 master weights for 146.9M trainable params`; `GATE PASS p4_gate_smoke step20 \| … \| 0.5520 (36329763/65811426) -> PASS`; 2 min 19 s; the temporary checkpoint was deleted |
| `P4` fix off | Verified | `VLA_EXTRA+=--no-pytorch-fp32-master-weights`: `GATE FAIL … NO fp32 master weights LINE \| 0.0000 (0/65811426) -> FAIL`. Then `P5 FAIL: step P4 has not passed`, `P6 REFUSED: …`, `RUN taskonly REFUSED: …`. Re-running P4 with the fix restored PASS (0.5520 again) |
| Flag sets | Verified | Every training command of P4, `timed.sh` (mw, fp32) and `run2.sh` (mw, fp32, fresh and resume) parsed with the project's tyro CLI into the intended config: precision, master weights, batch, steps, save interval, seed 42, warmup/peak/decay/decay_lr, episodes, 10 freeze prefixes. With `float32` the trainer creates no masters (`distill_pytorch.py:563` requires bfloat16), so the fallback correctly has no master-weights line |
| `P5` parsing | Verified at batch 2 | `TIMED taskonly PASS … s_per_step=0.49 … peak_reserved_GB=14.62 master_weights_line=yes`, the same line for kd, `SPEED_GATE PASS recipe=mw …`; batch-2 override confirmed in the log; temporary checkpoints deleted. The peak is read from `after_backward` at step 4, a running peak that already includes the optimizer steps 0-3 (AdamW state, masters, fp32 grad copies) |
| `P5` judgement | Verified with synthetic logs | Stand-in training runs in the real log format: PASS (4.70), SLOW (5.30, `10-50%_slower`), FAIL (8.00, `>50%_slower`), FAIL (46.3 GB, `peak_reserved>46GB`), FAIL (no master line), FAIL (OOM, rc=1), and fp32 PASS |
| `P5` at batch 64, `P5F` | **Not verifiable here** | Batch 64 needs about 44 GB. The fp32 recipe's teacher alone is 14.5 GB in fp32, so not even batch 2 fits 16 GB. Only the flags and the judgement logic are verified |
| `P6` | Verified (40-step stand-in) | Isolated tmux socket. A fresh start (`started in tmux session 'vla2'`), killed after checkpoint 20; `P7` on the dead run; `step P6` again → `START taskonly … mode=--resume` → then kd `mode=fresh` → `RUN kd PASS`. After success: `39` removed, `20` stripped to `model.safetensors` + `assets`, `40` complete. Refusals when the gate or the speed gate is missing were exercised (the latter by accident, see section 5). `ALREADY DONE` for finished runs |
| `--resume` | Verified on CPU, real script | On the GPU the resumed taskonly run failed twice with `CUDA driver error: device not ready` in `load_checkpoint`. A standalone repro shows this is the laptop's 16 GB, not the code: loading the 4.11 GB `model.safetensors` to CUDA works with 4.77 GB free and fails with exactly this error with 3.57 GB free; the resumed process held 12.17 GB before the load. On OVX the same moment is about 13.7 GB resident + 4.1 GB + 5.6 GB optimizer, of 48 GB, before any activations. The same `run2.sh` resume on CPU (`CUDA_VISIBLE_DEVICES=`, 20 → 22): `Successfully loaded all checkpoint components from step 20`, `Resumed training from step 20`, no master re-init warning, lr at step 20 = 1.74e-6 = the schedule's value, `RUN taskonly PASS` (7 min, peak RSS 22 GB) |
| `P6P`, nohup fallback | Shell logic verified (`DRY=1`) | `P6P` started `vla2_taskonly` on `gpu=0` and `vla2_kd` on `gpu=1`, with the right `--resume` training commands in each log. With tmux hidden from `PATH`: `P6` → `started with nohup pid=…`, sequential taskonly → kd; `P6P` → `started two runs with nohup`, one per GPU. The new guard (section 5, item 1) refused correctly with a leftover `vla2_kd` session and with only a `run2.sh` wrapper process alive. `VLA_RECIPE=fp32` is refused by `P6`/`P6P` until `P5F` passes. `DRY` only echoes the command, so these runs end in `RUN … FAIL rc=0`, as expected |
| `P7` | Verified | Parsed the real OVX logs (ETA lines, last loss lines). In the sim: master-weights line, `Resumed` line, ETA, the one-time gate on both "1500" checkpoints written to `gate.txt` and re-shown from the cache on the next call. The stand-in "1500" (step 20 of the warmup-300 schedule) correctly read 0.0000 → `GATE FAIL p7`: summed LR by step 20 is about 2e-5, below half a bf16 ulp at the gate's \|w\| ≥ 2⁻⁶ (6e-5). The real step-1500 case was measured separately: 1,500 steps with the real LR flags (batch 2, VLM frozen, 12 min) → `0.8972 (59043674/65811426) -> PASS`. Last run's OVX `taskonly_b64_s3000_step1500` and `kd_b64_s3000_step1500` → `0.0000 -> FAIL` |
| `P8` | Verified | Four real-size weights (4,112.9 MB each, hard links), `assets/`, logs, `runs.txt`, `speed.txt`, `gate.txt`, `ovx_state.txt`: 21 files, 16,451.8 MB, `PACKAGE READY`; `sha256sum -c SHA256SUMS` all OK |
| `diag2.sh` | Verified | Writes one 7 KB file with GPU, disk, runs, gate, speed and the last three log tails |
| `R1`, `R2` | Not run | Identical to the first runbook's verified `P2A`/`P3A` except for the tarball name and hash; needed only if `check_env.sh` / `check_data.sh` fail on OVX. `bash -n` and shellcheck clean |
| Dropbox upload | Not verifiable here | The teammate's own route |

Longest GPU job in this session: the 12-minute, 1,500-step batch-2 run behind the step-1500 gate figure. No full-size or long training ran. The simulated workspace `_cloudsim/W`, the block extractions and the test harnesses were temporary and are not in the repository.

## 5. Problems the dry run found, and the fixes in `runbook.md`

1. **`P6` / `P6P` re-run right after launch.** The "already running" check only looked for the Python process. Run `step P6` again within the roughly 70 s before Python starts, or in the gap between taskonly and kd, and the block went on to `tmux new-session` and printed a bare `duplicate session: vla2`. Nothing was started twice, but the message would have confused the teammate. Both blocks now also detect the `run2.sh` wrapper (`pgrep -f "^bash [^ ]*/run2\.sh (taskonly|kd)"`, anchored so it matches neither the tmux server nor itself) and any `vla2`, `vla2_taskonly` or `vla2_kd` tmux session, and print `training is already running (or its tmux session still exists); use: step P7`. Re-checked against a live run.
2. **File name.** The runbook calls itself `runbook2.md` and its helper reads `~/runbook2.md`, but the deliverable is `runbook.md`. With the wrong name, `step` would do nothing or fail on a missing file. Section 2 now says to set `RB=` to the actual path. Send the file as `runbook2.md`.
3. **The step-1500 gate expectation.** `P7` said the step-1500 fraction is "far above" P4's, with no measurement behind it. It now gives the measured reference values: 0.8972 for the fixed recipe at step 1500 on the laptop, 0.0000 for last run's two OVX step-1500 checkpoints.
4. Cosmetic: version date 2026-11-06, file size 55 KB.

Laptop-only issues, not runbook bugs: `check_data.sh` downloading on the 8-episode sim; the WSL `device not ready` error at 16 GB; and my own test harness once appending synthetic `SPEED_GATE FAIL` lines to the sim's `speed.txt`, because `env2.sh` resets `R`. That last one showed that `P6` then refuses as it should; the lines were removed before the real runs.

## 6. What remains unverified, and the risks

- **Memory at batch 64 with master weights** (predicted 41.5 → about 44.3 GB reserved for kd and about 44.8 GB for task-only; gate at 46 GB). This comes from the report's measured +1.00 GB laptop slope and last run's 38.9-39.4 GB. If it fails, 8.1 point 4 sends the teammate to `P5F` without waiting for us.
- **The fp32 master-weight step's transient** (the fp32 copy of each gradient, about 1.4 GB more than the bf16 grads) is inside P5's measured peak, as explained in section 4, but has never been measured at full size.
- **Full-size P4 numbers**: 698.5M trainable, and the fraction with the VLM trainable. The metric counts only expert-layer weights, which get the same gradients either way; the laptop number is 0.5520, threshold 0.25.
- **`P5F` / option 3** was never run anywhere. Its flags are parsed and its time and memory are the report's estimates (2.33 / 3.74 s/step, about 42 GB).
- **The real resume on the GPU** at full size (verified on CPU here, and on the GPU in the report's earlier 6 → 8 resume test).
- **Hugging Face access during `check_data.sh`.** LeRobot contacts the Hub even when the data is local (the `Revision v2.1 … not found` warning). On the laptop an anonymous IP hit a 429 only when it actually downloaded files. On OVX the dataset is complete, and last time `CHECK_DATA PASS` came through.

## 7. What only the teammate can answer

1. Their own evaluation last time: script and config, checkpoint path, suites, tasks and episodes per suite, the success numbers per checkpoint, latency method and numbers, the machine, bf16 or fp32 inference (runbook section 13).
2. Whether `/data/vla_work` is intact, and whether they want to keep last run's four weights for their own evaluation (if so, they should not run `P2B`; the runbook says so).
3. Whether one or two L40S GPUs are free for about 10.5 h / 6.5 h, and whether the GPU is shared (`P0` shows it).
4. Whether the Dropbox route can carry about 16.5 GB again, and whether model weights may still leave the institute.
5. Whether `tmux` is still available. The runbook falls back to `nohup`.

## 8. Recommendations

- Send `runbook.md` renamed to `runbook2.md`, with `vla_distill-3d11c3e.tar.gz`, to the teammate. Point them at section 13 for the evaluation questions, which they can answer before training.
- On receipt, verify with `sha256sum -c SHA256SUMS`. Then run `gate_check.py` on each step-3000 checkpoint against the init (it should PASS, against 0.0000 for last run's), and only then the paired LIBERO evaluation.
- No code change is needed for this runbook. Not done, by design: option 1b/2/3 implementations, EMA, the compute-saving levers.
