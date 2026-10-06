# Memory bandwidth of the Pi0.5 teacher and the 6-layer student

Date: 2026-10-03. Task: `vla-mem-bandwidth`. Code: `vla_distill` at commit `966da5c`.
Follows `data/vla-cloud-runbook/report.md` ("the earlier report"), which read "bandwidth" as data throughput.
This report answers the question that was asked: GPU memory bandwidth.

## 摘要（可直接轉給 mentor）

**問題。** Pi0.5 teacher → 6-layer student 的 distillation，模型在 GPU memory 上搬多少資料、對上 GPU 的 memory bandwidth 會不會成為瓶頸、read 與 write 的比例、以及 VLA 前面的 ViT 有沒有影響。
先前的報告把 bandwidth 讀成網路／磁碟／data loader 的 throughput，並估計每個 KD step 只搬 100–200 GB，這個估計低了一個數量級，以本報告為準。

**方法。** 在真實的模型程式上追蹤每一個 GPU operator 的輸入與輸出 tensor 大小（teacher forward、student forward、backward、optimizer、inference），得到 read／write bytes 與 FLOPs；batch 64 由 batch 4–16 的追蹤結果線性外推（外推誤差為 0）。再在筆電（RTX 5080 Laptop，16 GB）上實測 memory 速率、GEMM 速率、每個 operator 的時間，以及 `utilization.memory`。

**結論。**

| | 每次搬移的資料量 | Read : Write | 運算量 |
|---|---|---|---|
| 訓練，KD step（batch 64，bf16） | 2,723 GB（讀 1,501、寫 1,222） | 1.23 : 1 | 739 TFLOP |
| 訓練，task-only step | 1,636 GB（讀 926、寫 710） | 1.30 : 1 | 441 TFLOP |
| 推論，teacher（batch 1，10 個 denoising step） | 34.6 GB（讀 26.6、寫 8.1），其中 weights 17.4 GB | 3.3 : 1 | 4.7 TFLOP |
| 推論，student | 15.0 GB（讀 11.5、寫 3.4），其中 weights 7.6 GB | 3.4 : 1 | 2.0 TFLOP |

1. **訓練：memory bandwidth 在 L40S 上是較大的一項限制，在 A100 與 A40 上不是。**
   以原廠規格計算，L40S（864 GB/s，BF16 362 TFLOPS）搬完一個 KD step 的資料要 3.2 秒，而運算只要 2.0 秒，memory traffic 約佔每步時間的 55–60%。
   A100 80GB（2,039 GB/s，312 TFLOPS）則是 traffic 1.3 秒、運算 2.4 秒，以 compute-bound 為主（traffic 約 30–36%）。
   A40（696 GB/s，150 TFLOPS）是 traffic 3.9 秒、運算 4.9 秒（traffic 約 37–44%）。L40 的原廠 BF16 數字（181 TFLOPS）與 L40S 相差兩倍，若照公布值計算則與 A40 相近，若實際與 L40S 相同則結論同 L40S。
   預估每個 KD step：L40S 4.5–7.3 秒，A100 80GB 3.4–5.6 秒，A40 7.9–13.1 秒。每一種卡都能跑，只影響時間，不需要更改實驗設定。
2. **哪些部分 bandwidth-bound。** 75% 的 bytes 來自 elementwise operator（dtype cast、add、mul、GELU、RMSNorm、softmax），在所有的卡上都是 bandwidth-bound；其中 978×978 的 attention map（eager attention，softmax 用 float32）相關的 operator 佔 24%。MLP 與 projection 的 GEMM 是 compute-bound。
3. **推論：denoising loop 是 bandwidth-bound，prefix 是 compute-bound。** Action expert 的 weights 每個 denoising step 都要重讀一次，10 步共 11.0 GB（teacher）／3.8 GB（student）。筆電實測每個 denoising step 3.4 ms（teacher）、1.0–1.5 ms（student），接近「該步的 bytes ÷ 實測 memory 速率」（2.7、0.9 ms），運算本身只需 0.16、0.05 ms。SigLIP 加 VLM 的 prefix 只算一次並以 KV cache 重複使用，是 compute-bound。
4. **Read vs write。** 訓練約 1.2–1.3 : 1，推論約 3.3 : 1（weights 只讀不寫）。筆電實測 write-only 571 GB/s 不比 read-only 501–555 GB/s 慢，所以 read／write 的比例不改變結論，決定時間的是總 bytes。
5. **ViT（SigLIP）。** 訓練中佔 memory traffic 的 10.9%（KD）、9.0%（task-only）；它是 frozen 的，沒有 backward、沒有 recomputation、不進 optimizer。全零的第三張 image 仍然被編碼，正好佔 SigLIP traffic 的三分之一（KD step 中 98.5 GB，3.6%）。推論中它佔 teacher 的 14%、student 的 32%，因為 distillation 沒有縮小它。它不是 memory bandwidth 問題的來源：每個 token-layer 的 traffic 只有 Gemma layer 的 1/7.5。
6. **筆電驗證。** 實測 memory 速率 463–571 GB/s、bf16 GEMM 50 TFLOPS。只用這兩個實測值加上追蹤到的 bytes 與 FLOPs，預測值與實測相符：KD step（換算成 batch 64）預測 19.2–20.7 秒、實測 19.4 秒；teacher inference 預測 144.7 ms、實測 145.7 ms；student 預測 59.3 ms、實測 58.6–62.1 ms。`utilization.memory`：純 copy 78–94%，純 GEMM 13–21%，訓練 step 31–32%。
7. **尚未確認的事。** 雲端實際的卡型，以及該卡實際能達到的 memory 速率與 GEMM 速率（以上對雲端卡的數字用的是原廠規格，不是量測）。已準備好可加進 runbook `P5` 的指令（多花約 25 秒，已在筆電上以真實的訓練程式測過），會直接印出卡名、copy／read／write GB/s、GEMM TFLOPS、訓練中的 `utilization.memory`，以及「memory traffic 佔實測 step 時間的比例」。
8. **無法量測的部分。** WSL2 下讀不到 GPU 的 hardware counter（Nsight Compute 回報 `ERR_NVGPUCTRPERM`），所以 bytes 是 operator 層級的計數，不是硬體計數；16 GB 的筆電放不下 batch 64，也放不下 teacher + student + optimizer 同時存在。
9. **附帶發現。** Student 的 transformer layers 每個 step 被計算三次而不是兩次：gradient checkpointing 套了兩層，外層重做了內層已經會做的 recomputation。這佔 KD step 的 11.5%（313 GB、85 TFLOP），task-only step 的 19%，不影響結果。本次沒有修改程式，列為建議。

---

## 1. Summary

Legend used throughout: **[M]** measured on this laptop in this task (timing, NVML, or an operator trace of the executed code), **[M1]** measured in phase 1 or in the earlier report, **[D]** derived by arithmetic from measured or published values, **[V]** vendor-published, **[A]** assumed.

**Demand.**
One KD step at batch 64 moves 2,723 GB through GPU memory, 1,501 GB read and 1,222 GB written, for 739 TFLOP of matrix arithmetic [D from M].
One task-only step moves 1,636 GB for 441 TFLOP [D from M].
One inference query moves 34.6 GB for the teacher and 15.0 GB for the student [M], half of it weights.
The earlier report's row for this was 100-200 GB per KD step, an estimate that is low by a factor of 14 to 27.

**Supply.**
L40S and L40 864 GB/s, A40 696 GB/s, A100 40GB 1,555 GB/s, A100 80GB 1,935 (PCIe) or 2,039 (SXM) GB/s, H100 SXM 3,350 GB/s [V].
The laptop delivered 463 GB/s for mixed read-write operators, 501-555 GB/s read-only and 571 GB/s write-only [M].

**Verdict for training.**
A card is bandwidth-leaning for this step when its compute-to-bandwidth ratio exceeds the step's arithmetic intensity of 271 FLOP per byte [D].
The L40S ratio is 419, the A100 80GB ratio is 153 to 161, the laptop's measured ratio is 108.
So the same step is mostly bandwidth-bound on an L40S, mostly compute-bound on an A100 and on the laptop.

| KD step, batch 64 | Traffic alone | Compute alone | Predicted step | Traffic share |
|---|---|---|---|---|
| L40S, vendor figures [D] | 3.15 s | 2.04 s | 4.5-5.2 s | 61% |
| A100 80GB SXM, vendor figures [D] | 1.34 s | 2.37 s | 3.4-3.7 s | 36% |
| A40, vendor figures [D] | 3.91 s | 4.93 s | 7.9-8.9 s | 44% |
| Laptop, measured supply [D], against 19.4 s measured-equivalent [D from M] | 5.88 s | 14.78 s | 19.2-20.7 s | 28% |

**Verdict for inference.**
The prefix pass (SigLIP and the VLM over 968 tokens) is compute-bound on the laptop.
The denoising loop is bandwidth-bound: each step re-reads the whole action expert, and its measured time is 1.1 to 1.6 times the pure streaming time of the bytes it touches and about 20 times its compute time [M].
On any of the cloud cards the whole query would be bandwidth-leaning, because their compute grows faster than their bandwidth.

**Read versus write.**
Training is 1.23 reads per write, inference 3.3.
On the laptop a written byte costs no more than a read byte, so the ratio does not change the verdict.

**The ViT.**
SigLIP plus projector is 10.9% of a KD step's traffic and 9.0% of a task-only step's.
In inference it is 14% for the teacher and 32% for the student.
A third of it is the all-zero third image.
It is the cheapest part of the model per token and is not a memory-bandwidth problem; it is a limit on how far the student's per-query traffic can fall.

**What I also found.**
The student's transformer layers are computed three times per step, not twice: gradient checkpointing is nested, and the outer level repeats work the inner level already does (section 4.3).
This is 313 GB and 85 TFLOP per step that a one-line change would remove.
I did not make the change; it is a recommendation.

What I could not measure is in section 10. What remains unknown is in section 14.

## 2. What was asked, and how I read it

The questions, in the original words: "I want to know the bandwidth, and will it be a bottleneck?", "THere should also be a Vit before the vla part in the piplien, would thtat be an issue?", "Check also on read vs write."
The correction: "im not askin gnetwork bandwidth, im asking about model bandwidth", then "memory".

I read this as GPU memory bandwidth: the rate at which bytes travel between the GPU's memory (GDDR or HBM) and its compute cores, against the number of bytes the model needs to move.
Whether the question concerned training in the cloud or inference of the finished student was not settled, so both are answered.

## 3. Method

### 3.1 What counts as memory traffic

For every operator the model executes on the GPU:

- bytes read = the sizes of its input tensors, weights included;
- bytes written = the size of every output that owns new storage, or of the target of an in-place operator;
- a view (reshape, transpose, slice, expand) moves nothing and counts as zero;
- an embedding or index lookup reads only the rows it selects, not the whole table;
- a broadcast tensor counts its real storage, not its expanded size.

This is an operator-level count, not a hardware count.
It is exact for what the framework asks the GPU to do, and it equals memory traffic when each tensor is too large to stay in the GPU's on-chip cache between the operator that writes it and the operator that reads it.
At batch 64 the backbone's activation tensors are 0.25 to 2 GB each and SigLIP's 38 to 141 MB, and the laptop shows a cache effect at 16 MiB and none at 64 MiB (Appendix A.1), so the condition holds for training.
At batch 1 it holds for weights (every pass cycles through 0.8 to 4 GB of them) but only partly for activations, which are a few MB each; for inference the count is therefore exact for weights and an upper bound for activations.

What the count misses: re-reads inside one kernel when a matrix product's tiles fall out of cache, and cuDNN or cuBLAS workspace.
What it overstates: a tensor passed twice to one operator is counted twice (1.5 GB per step in AdamW, 0.05% of the step).

FLOPs are exact for matrix products, convolutions and fused attention (2·m·n·k).
Elementwise operators are given zero FLOPs and treated as pure traffic.
The laptop measurement supports that: large elementwise operators ran at 366-413 GB/s against 463 GB/s for a pure copy (section 6.1).

### 3.2 The trace

A `TorchDispatchMode` records every ATen operator the real model code executes, including the autograd backward, the recomputation done by gradient checkpointing, and the optimizer.
Components are tagged with forward hooks on the model's modules; gradient operators, which run outside any module, are attributed by tensor shape.
The project's code was not modified; the tracer lives in a scratch directory and its accounting function is reproduced in Appendix B.

Settings, matching `scripts/distill_pytorch.py` and the runbook's flags:

- bf16, gradient checkpointing on, vision tower + projector + token embedding (tied `lm_head`) frozen;
- AdamW (0.9, 0.95, eps 1e-8, weight decay 1e-10), `clip_grad_norm_` 1.0;
- inputs built from the phase 1 held-out frames with the policy's own input transform;
- 702.7 M parameters receive a gradient (1.49 GB of gradients) [M], the same count the earlier report measured.

The student step and the teacher forward were traced in separate processes, because teacher + student + AdamW state do not fit 16 GB.
The teacher's forward runs under `no_grad` before the student's graph exists (`src/openpi/models_pytorch/pi0_pytorch.py:539-549`), so the two are independent and a KD step is their sum.
In the student trace the teacher is a stub that returns a ready tensor.
The task-only step is the student step without the teacher: at batch 2 the two student traces differ by 6 operators and 0.021 MB [M].

Batch 64 does not fit the laptop.
Every tensor dimension other than the batch is fixed, so bytes are an exact linear function of batch size per operator.
I fitted each operator on the two largest batches traced (4 and 8 for the student, 8 and 16 for the teacher) and checked the fit on the others: the relative error of total reads, writes and FLOPs at batch 2 and 4 is 0.0 [M].
Batch 64 values are therefore marked [D from M].

Inference was traced on the eager path (the un-compiled `sample_actions`), because a dispatch-mode tracer cannot see inside a `torch.compile` graph.
Phase 1 timed the compiled path; I timed both.
The compiled graph reads the same weights and may move fewer activation bytes.

### 3.3 Measurements on the laptop

Four, chosen because each can separate bandwidth-bound from compute-bound behaviour on its own:

1. **Supply calibration.** Pure streaming kernels (sum, fill, copy, add, in-place multiply, dtype cast) on 1 GiB tensors give the delivered memory rate; large bf16 matrix products give the delivered FLOP rate. NVML `utilization.memory` was sampled during each, which calibrates that counter for both extremes.
2. **Per-operator timing.** The same trace with a CUDA synchronize around every operator gives the time of each one. An operator is bandwidth-bound if it runs at the calibrated memory rate and compute-bound if it runs at the calibrated FLOP rate.
3. **NVML `utilization.memory`** during training steps, the teacher forward and inference, compared with its two calibration points.
4. **Latency against the number of denoising steps** for the compiled model. The slope is the time of one denoising step, to be compared with what its bytes imply and what its FLOPs imply.

I chose these over the PyTorch profiler because what the profiler adds is kernel timing, which method 2 gives with exact attribution to the traced bytes; and over scaling time with precision, because fp32 doubles the bytes and halves the tensor-core rate at once (25 against 50 TFLOPS measured), so it cannot tell the two regimes apart.

All GPU jobs ran one at a time, outside any network namespace, and each saved its result as it finished.
There was no restart.
The teacher was run three times and the student twice; byte counts were identical in every run [M].
Timings quoted are from the third run, taken with nothing else running, unless stated.

## 4. Demand: bytes moved through GPU memory

### 4.1 Training, batch 64

By pass [D from M]. GB = 10⁹ bytes.

| Pass | Read GB | Write GB | Total GB | Read : write | TFLOP |
|---|---|---|---|---|---|
| Teacher forward, no gradient (KD only) | 575.2 | 511.6 | 1,086.8 | 1.12 | 297.3 |
| Host-to-GPU input copy | 0.0 | 0.1 | 0.1 | | |
| Student forward | 249.8 | 218.6 | 468.4 | 1.14 | 127.3 |
| Backward: recomputation from gradient checkpointing (two passes, section 4.3) | 329.8 | 296.8 | 626.6 | 1.11 | 170.0 |
| Backward: gradient operators, including the 1.49 GB of parameter gradients | 323.6 | 181.3 | 504.9 | 1.78 | 144.0 |
| Gradient clipping | 3.0 | 1.5 | 4.5 | 2.00 | 0 |
| AdamW update | 19.4 | 11.9 | 31.3 | 1.63 | 0 |
| **KD step** | **1,500.8** | **1,221.9** | **2,722.7** | **1.23** | **738.6** |
| **Task-only step** (everything except the teacher forward) | **925.6** | **710.3** | **1,635.9** | **1.30** | **441.3** |

The task-only step is 60% of the KD step in both bytes and FLOPs, which matches the 0.6 factor the runbook uses and the 0.595 time ratio the earlier report measured.

By component, all passes together [D from M]:

| Component | KD step: read / write / total GB | Task-only: read / write / total GB | FLOP per byte | Share of KD bytes |
|---|---|---|---|---|
| SigLIP vision encoder + projector | 162.4 / 133.3 / 295.6 | 81.2 / 66.6 / 147.8 | 286 | 10.9% |
| Embeddings (token lookup, prefix assembly) | 0.7 / 0.7 / 1.4 | 0.4 / 0.4 / 0.7 | 0 | 0.05% |
| VLM backbone layers: projections, MLP, norms, residuals on the 968 prefix tokens | 823.3 / 642.8 / 1,466.1 | 524.4 / 379.0 / 903.4 | 429 | 53.8% |
| Joint attention over 978 tokens (99% of the rows are VLM tokens) | 483.2 / 425.5 / 908.7 | 291.1 / 246.5 / 537.5 | 26 | 33.4% |
| Action expert layers and action/time projections (10 tokens) | 6.6 / 3.5 / 10.1 | 4.1 / 2.3 / 6.4 | 109 | 0.4% |
| Image augmentation on the GPU | 1.9 / 1.6 / 3.5 | 1.9 / 1.6 / 3.5 | 0 | 0.1% |
| Loss, masks, noise | 0.4 / 0.9 / 1.3 | 0.2 / 0.4 / 0.6 | 0 | 0.05% |
| Gradient clipping | 3.0 / 1.5 / 4.5 | 3.0 / 1.5 / 4.5 | 0 | 0.2% |
| AdamW update | 19.4 / 11.9 / 31.3 | 19.4 / 11.9 / 31.3 | 0 | 1.1% |

The backbone and its attention are 87% of the traffic.
The action expert is negligible in training, because it processes 10 tokens against the backbone's 968.

Weights, gradients and optimizer state are a small part at batch 64.
The batch-independent traffic is 7.6 GB for the teacher forward (its weights; SigLIP's are read three times, section 8) and 46.4 GB for the student step (31.3 GB optimizer, 4.5 GB clipping, 10.6 GB of weights and parameter gradients across the forward, recomputation and gradient passes) [D from M].
Everything else scales with the batch: 16.9 GB per sample for the teacher forward and 24.8 GB per sample for the student step.

### 4.2 Where the bytes come from

75% of the bytes of a KD step (2,035 of 2,723 GB) are moved by elementwise operators, 25% are the operands of matrix products [D from M].

One Gemma layer of the teacher moves 807 MB per sample [D from M], about 200 times the 4 MB of its output:

| Part of the layer | MB per sample | Why |
|---|---|---|
| Attention maps and their glue | 322 | Eager attention materializes the 8×978×978 map (15 MB in bf16), adds the mask, takes the softmax in float32 (31 MB) and casts back (`transformers_replace/models/gemma/modeling_gemma.py:243-250`); rotary embedding and concatenation of prefix and suffix add more |
| MLP | 266 | Three 978×16,384 tensors of 32 MB each (gate, GELU, up) and their product, plus the operands of three matrix products |
| RMSNorm, twice per layer | 155 | Computed in float32: cast up, square, mean, scale, cast back (`modeling_gemma.py:66-71`) |
| Attention projections and residuals | 64 | |

Operators that touch a 978×978 attention map account for 659 GB of the KD step, 24% [D from M].
Of that, 433 GB is the float32 part.

The ten operators that move the most, KD step at batch 64 [D from M]:

| Operator | Total GB | TFLOP |
|---|---|---|
| `mul` | 637 | |
| `mm` (linear layers) | 404 | 629.9 |
| `_to_copy` (dtype casts) | 287 | |
| `add` | 282 | |
| `gelu` | 192 | |
| `bmm` (attention products) | 144 | 24.1 |
| `_softmax` | 141 | |
| `clone` | 132 | |
| `addmm` (SigLIP linear layers) | 114 | 81.3 |
| `pow` (the square in RMSNorm) | 87 | |

### 4.3 What is reused and what is recomputed

- **The teacher forward** runs once per step and stores nothing.
- **SigLIP runs twice per KD step**, once in the teacher and once in the student. The encoders are identical and both receive the student's augmented images (`pi0_pytorch.py:543-549`), so the second pass computes what the first already produced: 147.8 GB and 42.3 TFLOP per step that are not reused.
- **The student's layers are computed three times per step, not twice.** `DistilledPI0Pytorch.forward` wraps the whole layer stack in one checkpoint (`pi0_pytorch.py:581-583`), and inside it each layer is checkpointed again (`gemma_pytorch.py:241-252`). In the backward pass the outer checkpoint first recomputes all six layers in one block, and then each layer is recomputed again when its own gradients are needed. The trace shows it directly [M]: at batch 8 the backward pass opens with one recomputation block of 635 operators, 40.95 GB and 10.625 TFLOP, exactly the forward pass's layer stack, followed by six per-layer blocks of 101 operators, 6.72 GB and 1.771 TFLOP each, interleaved with gradient operators. The forward pass has 18 MLP matrix products in the backbone; the backward pass repeats 36.
  So recomputation costs 626.6 GB and 170.0 TFLOP at batch 64, of which half, 313.3 GB and 85.0 TFLOP, is the outer level repeating what the inner level does. The README's plan counted one recomputation.
- **Checkpointing itself is needed.** With it, the step holds about one layer's intermediates at a time: 0.305 GB per sample [M1, earlier report]. Without it the graph would hold all six layers' worth, on the order of 100 GB at batch 64 [D, rough], which fits no candidate card.
- **SigLIP has no backward pass, no recomputation and no optimizer state.** It is frozen and its input needs no gradient. The trace has no SigLIP operator in the backward phase [M].
- **The last backbone layer does work nobody reads.** Its MLP output for the 968 prefix tokens is never used by the loss, in the teacher or the student. It is computed in the forward pass; autograd then skips its gradients. Checked directly [M]: of the 1,070.9 M trainable parameters, 368.2 M receive no gradient, namely the unused expert `lm_head` (263.3 M) and the last backbone layer's MLP, attention output projection and second norm plus the backbone's final norm (104.9 M). That leaves the 702.7 M that do.
- **Inference reuses the prefix.** SigLIP and the VLM run once per query; their keys and values are cached and each of the ten denoising steps reads 18.0 MB of them for the teacher, 6.0 MB for the student [M] (`pi0_pytorch.py:393-399`, `:406-418`). Without the cache each step would repeat the 17.4 GB backbone pass (teacher).
- **Inference does not reuse the action expert's weights.** Every denoising step reads all of them again: 1.10 GB per step for the teacher, 0.38 GB for the student [M]. Of that, 0.47 GB and 0.16 GB are the float32 dense layers of the adaptive RMSNorm, whose output depends only on the time step and not on the observation (`modeling_gemma.py:88`), so it is the same ten vectors per layer on every query.

### 4.4 Inference, batch 1, 10 denoising steps

Per query, eager operator trace [M]:

| Component | Teacher: read / write / total GB | Student: read / write / total GB | Of which weights, teacher / student GB |
|---|---|---|---|
| SigLIP + projector, three images | 3.7 / 1.0 / 4.8 | 3.7 / 1.0 / 4.8 | 2.49 / 2.49 |
| Embeddings | 0.01 / 0.01 / 0.02 | same | none (0.8 MB of selected rows) |
| VLM backbone, one pass over 968 tokens | 11.0 / 6.4 / 17.4 | 3.7 / 2.2 / 5.9 | 3.96 / 1.32 |
| Action expert, ten denoising steps | 11.8 / 0.6 / 12.4 | 4.1 / 0.2 / 4.3 | 10.98 / 3.80 |
| **Query** | **26.6 / 8.1 / 34.6** | **11.5 / 3.4 / 15.0** | **17.4 / 7.6** |
| Matrix-product FLOPs | 4.7 TFLOP | 2.0 TFLOP | |

Half the bytes of a query are weights.
The action expert is the largest reader of weights although it does 2% of the arithmetic (0.08 of 4.7 TFLOP for the teacher), because ten steps re-read it ten times.
SigLIP's 0.83 GB of weights are read three times, because the three images are encoded in three separate calls (`pi0_pytorch.py:197-210`).

The student moves 43% of the teacher's bytes (15.0 against 34.6 GB) and does 43% of its arithmetic.
Phase 1 measured its latency at 40% of the teacher's (58.6 against 145.7 ms).

## 5. Supply: memory bandwidth of the candidate cards

| Card | Memory | Memory bandwidth, GB/s | BF16 tensor rate, dense, TFLOPS | Ratio, FLOP per byte [D] | Source |
|---|---|---|---|---|---|
| L40S | 48 GB GDDR6 | 864 [V] | 362.05 (733 with sparsity) [V] | 419 | NVIDIA L40S product page |
| L40 | 48 GB GDDR6 | 864 [V] | 181.05 (362.1 with sparsity) [V] | 210 | NVIDIA L40 datasheet, January 2023 |
| A40 | 48 GB GDDR6 | 696 [V] | 149.7 (299.4 with sparsity) [V] | 215 | NVIDIA A40 datasheet, March 2022 |
| A100 40GB, PCIe and SXM | 40 GB HBM2 | 1,555 [V] | 312 (624 with sparsity) [V] | 201 | NVIDIA A100 datasheet |
| A100 80GB PCIe | 80 GB HBM2e | 1,935 [V] | 312 [V] | 161 | NVIDIA A100 datasheet and product page |
| A100 80GB SXM | 80 GB HBM2e | 2,039 [V] | 312 [V] | 153 | same |
| H100 SXM, for reference | 80 GB | 3,350 [V] | 989.5 [D: half of the 1,979 listed with sparsity] | 295 | NVIDIA H100 product page |
| RTX 5080 Laptop GPU | 16 GB GDDR7, 256-bit | not published; see below | not published; 50.0 [M] | 108 [D from M] | NVIDIA laptop comparison page; NVML; calibration |

Three remarks.

**The two L40 documents disagree with the L40S page by a factor of two.**
The L40 datasheet gives 181 TFLOPS dense for BF16 and the L40S page gives 362, for cards that differ by 1% in their FP32 figure (90.5 against 91.6 TFLOPS).
I use each as published and do not know which convention is right.
If the L40 behaves like the L40S, read the L40S rows for it.

**The laptop.**
NVIDIA publishes "16 GB GDDR7" and a 256-bit interface for the RTX 5080 Laptop GPU, but no bandwidth and no tensor rate.
NVML on this machine reports a 256-bit bus and a maximum memory clock of 14,001 MHz [M].
If the data rate is twice the reported clock, as it is for GDDR6 cards [A], that is 28 Gbit/s per pin and 896 GB/s at the maximum clock [D].
Under sustained load the clock dropped to 9,001 MHz in almost every workload [M], which gives 576 GB/s [D].
Delivered [M], 1 GiB tensors:

| Kernel | bf16 | float32 | Memory clock during the test | `utilization.memory` |
|---|---|---|---|---|
| Read only (`sum`) | 501 GB/s | 555 GB/s | 9,001 / 10,009 MHz | 89% |
| Write only (`fill_`) | 571 GB/s | 573 GB/s | 11,101 / 11,941 MHz | 80-86% |
| Copy (read + write) | 473 GB/s | 484 GB/s | 9,001 MHz | 84% |
| Add (two reads, one write) | 473 GB/s | 486 GB/s | 9,001 MHz | 86-87% |
| Multiply in place | 461 GB/s | 465 GB/s | 9,001 MHz | 84% |
| Cast bf16 to float32 | 446 GB/s | | 9,001 MHz | 82% |
| bf16 matrix product, 4,096² to 16,384² | 49.6-50.0 TFLOPS | 24.5-25.7 (TF32) | 9,001 MHz | 13-21% |

For the roofline I use 463 GB/s, the mean of copy, add, in-place multiply and cast in bf16, and 50.0 TFLOPS.
Streaming kernels delivered 75 to 87% of the interface rate implied by the memory clock at the time.

**What the vendor figure is.**
For the GDDR6 cards the published bandwidth equals the interface rate (384 bits × 18 Gbit/s ÷ 8 = 864 GB/s for the L40S).
A card does not deliver all of it; the laptop delivered about 80%.
The verdict below gives both the vendor figures and a derated case.

## 6. Verdict

### 6.1 What the laptop measurements show

**Per-operator timing, batch 8** [M]. Sum of per-operator times against un-instrumented wall time: 1.140 s against 1.031 s for the teacher forward, 1.617 s against 1.447 s for the student step; the difference is the synchronize overhead.

| Operator class | Teacher forward: time, rate | Student step: time, rate | Bound by |
|---|---|---|---|
| Matrix products and fused attention | 743.7 ms (65%), 50.0 TFLOPS | 1,014.9 ms (63%), 54.4 TFLOPS | Compute: they run at the calibrated 50 TFLOPS |
| Elementwise, 64 MB or more per call | 221.0 ms (19%), 366 GB/s | 386.6 ms (24%), 413 GB/s | Bandwidth: 79-89% of the calibrated 463 GB/s |
| Elementwise, under 64 MB per call | 174.9 ms (15%), 93 µs per call | 215.7 ms (13%), 83 µs per call | Launch and synchronize overhead; at batch 64 these tensors are large and move to the row above |

The attention products (`bmm`) sit between the two: 31-37 TFLOPS and 188-223 GB/s, neither rate reached.
Their intensity is 167 FLOP per byte, close to the laptop's ratio of 108, so both terms matter for them.
The AdamW update ran at 483 GB/s, pure streaming.

**NVML `utilization.memory`** [M]:

| Workload | `utilization.gpu` | `utilization.memory` |
|---|---|---|
| Idle | 1-8% | 2-3% |
| Pure streaming kernels, 1 GiB | 92-99% | 78-89% (89-94% in the P5 probe) |
| Pure bf16 matrix product | 98-100% | 13-21% (15% at 8,192²) |
| Matrix product with 10 rows against a 67 MB weight | 88-89% | 70% |
| Teacher KD forward, batch 16 | 99.6% | 30.5% (29.4% and 29.5% in the other runs) |
| Student training step, batch 8 | 93.4% | 31.8% (31.2% in the second run) |
| Compiled inference, teacher / student | 92.6% / 91.0% | 29.2% / 28.3% |
| The real `distill_pytorch.py`, batch 2, backbone frozen (section 9) | 86% | 23% |

The two measurements agree with each other.
Weighting the calibration points (87% for streaming during the time spent in large elementwise operators, 15% for the rest of the time) predicts 32.2% for the student step and 29.0% for the teacher forward, against 31.8% and 30.5% measured.

**The roofline, checked against measured time** [D against M].
Per operator, time is at least max(bytes ÷ bandwidth, FLOPs ÷ rate) and at most their sum.
With the laptop's two measured rates:

| | Predicted, max to sum | Measured |
|---|---|---|
| KD step scaled to batch 64 | 19.2-20.7 s | 19.4 s (64 × (0.1216 s teacher at batch 16 + 0.1808 s student at batch 8)) |
| Task-only step scaled to batch 64 | 11.5-12.4 s | 11.6 s |
| Teacher inference, compiled | 144.7-169.1 ms | 145.7 ms (phase 1: 145.7 ms) |
| Student inference, compiled | 59.3-72.5 ms | 62.1 ms; 59.1 ms in the second run; phase 1: 58.6 ms |

The measured values sit at the "max" end in all four cases.
Two rates and a byte and FLOP count reproduce the laptop's training and inference times to within 1 to 5%.
That is the basis for applying the same model to cards I cannot measure.

**Denoising steps** [M], compiled model, p50 of 100 calls:

| | Teacher | Student |
|---|---|---|
| Latency at 10 / 5 / 1 steps | 145.7 / 129.1 / 115.3 ms | 62.1 / 55.8 / 50.0 ms (second run: 59.1 / 54.6 / 50.2) |
| Time per denoising step | 3.38-3.43 ms | 1.35-1.45 ms (second run: 0.99-1.10) |
| Bytes per step ÷ 463 GB/s [D] | 2.68 ms (1.243 GB) | 0.93 ms (0.429 GB) |
| FLOPs per step ÷ 50 TFLOPS [D] | 0.16 ms (7.9 GFLOP) | 0.05 ms (2.6 GFLOP) |

A denoising step takes 1.1 to 1.6 times as long as streaming its bytes and about twenty times as long as its arithmetic.
It is bandwidth-bound, with launch overhead making up the rest.
The calibration shows the mechanism in isolation: a 10-row product against an expert-sized 8.4 MB weight takes 31.8 µs when the weight is reused and 43.3 µs when 54 different weights are cycled, with `utilization.memory` at 3% and 37% [M].

In eager mode the same step takes 22.3 ms (teacher) and 7.4 ms (student) [M]; that is Python and kernel-launch overhead for 1,066 (teacher) and 394 (student) operators per step, not memory.

### 6.2 Training on the candidate cards

Time the traffic alone implies, time the arithmetic alone implies, and the predicted step, at vendor figures and derated to 80% of the bandwidth and 60% of the tensor rate [D].
The derating factors are assumptions [A]: 80% is what the laptop's streaming kernels delivered against its interface rate; I have no measurement of what fraction of the vendor tensor rate a bf16 matrix product reaches on these cards.

| Card | Traffic alone | Compute alone | KD step, vendor figures (max-sum) | KD step, derated | Task-only step, vendor / derated | Traffic share of the step | Leaning |
|---|---|---|---|---|---|---|---|
| L40S | 3.15 s | 2.04 s | 4.5-5.2 s | 6.5-7.3 s | 2.7-3.1 / 3.9-4.4 s | 61% (54% derated) | Bandwidth |
| L40 as published | 3.15 s | 4.08 s | 6.5-7.2 s | 9.8-10.7 s | 3.9-4.3 / 5.9-6.4 s | 44% (37%) | Compute |
| A40 | 3.91 s | 4.93 s | 7.9-8.9 s | 11.9-13.1 s | 4.8-5.3 / 7.1-7.9 s | 44% (37%) | Compute |
| A100 40GB | 1.75 s | 2.37 s | 3.7-4.1 s | 5.6-6.1 s | 2.2-2.5 / 3.4-3.7 s | 43% (36%) | Compute |
| A100 80GB PCIe | 1.41 s | 2.37 s | 3.4-3.8 s | 5.3-5.7 s | 2.1-2.3 / 3.2-3.4 s | 37% (31%) | Compute |
| A100 80GB SXM | 1.34 s | 2.37 s | 3.4-3.7 s | 5.2-5.6 s | 2.0-2.2 / 3.1-3.4 s | 36% (30%) | Compute |
| H100 SXM | 0.81 s | 0.75 s | 1.4-1.6 s | 2.0-2.3 s | 0.8-0.9 / 1.2-1.4 s | 52% (45%) | Balanced |
| Laptop, measured supply | 5.88 s | 14.78 s | 19.2-20.7 s (19.4 s measured-equivalent) | | 11.5-12.4 s | 28% | Compute |

Read against the earlier report's estimate of 5.3-9.4 s per KD step on an L40S: the range here is 4.5-7.3 s, consistent with it, and it now has an explanation.
The earlier estimate assumed the card would reach 30-40% of its tensor rate.
The reason it reaches no more is that most of the step is not matrix arithmetic.

By component, KD step [D]. A component is bandwidth-leaning on a card whose ratio exceeds the component's intensity.

| Component | Intensity, FLOP per byte (its matrix products alone) | L40S (419): traffic / compute | A100 80GB SXM (153): traffic / compute | Verdict |
|---|---|---|---|---|
| Joint attention | 26 (167) | 1.05 s / 0.07 s | 0.45 s / 0.08 s | Bandwidth-bound on every card |
| VLM backbone layers | 429 (1,570) | 1.70 s / 1.74 s | 0.72 s / 2.02 s | Balanced on the L40S; compute-leaning on the H100; compute-bound on the A100, A40, L40 as published and the laptop. Its matrix products are compute-bound everywhere; its elementwise chain, 73% of its bytes, is bandwidth-bound everywhere |
| SigLIP + projector | 286 (614) | 0.34 s / 0.23 s | 0.15 s / 0.27 s | Bandwidth-leaning on the L40S, balanced on the H100, compute-leaning on the others |
| Action expert | 109 (241) | 0.012 s / 0.003 s | 0.005 s / 0.004 s | Bandwidth-leaning on every cloud card, and too small to matter |
| Embeddings, augmentation, clipping, AdamW | 0 | 0.05 s / 0 | 0.02 s / 0 | Pure traffic, 1.5% of the step |
| Whole KD step | 271 | 3.15 s / 2.04 s | 1.34 s / 2.37 s | Bandwidth-leaning above a ratio of 271 (L40S, H100), compute-leaning below (A40, L40 as published, A100, laptop) |

**So, is memory bandwidth a bottleneck for training?**

- On an L40S: yes, it is the larger of the two limits, a little over half the step. By the same model, doubling that card's tensor rate would shorten the step by about 20%, doubling its bandwidth by about 30% [D].
- On an A100: it is about a third of the step, and compute is the larger limit. By this model an A100 80GB runs the step 1.3 to 1.4 times faster than an L40S despite a lower tensor rate (3.4-3.7 against 4.5-5.2 s), and an A100 40GB 1.2 to 1.3 times faster.
- On an A40 or an L40 at its published rate: both limits are lower, compute a little more so, and the step is the slowest of the candidates.
- In no case does it prevent the run. Three thousand KD steps are 3.8-6.1 h on an L40S and 2.8-4.7 h on an A100 80GB [D]; the control is 60% of that.
- The laptop understates the effect. Its ratio of 108 FLOP per byte is the lowest of all the cards, so it is the most compute-bound of them, and a third of its step is already memory traffic.

The captain's impression was an A100 with 48 GB.
The two halves of that point to different verdicts: an A100 (40 or 80 GB) is compute-leaning, a 48 GB OVX card is the L40S, L40 or A40.
The probe's first line settles which.

### 6.3 Inference on the candidate cards

Per query, eager byte count, vendor figures [D]. The laptop rows use measured supply and can be compared with measured latency.

| Card | Teacher: prefix traffic / compute, denoising traffic / compute, total (max-sum) | Student: same |
|---|---|---|
| L40S | 25.7 / 12.8 ms, 14.4 / 0.2 ms, 43.5-53.1 ms | 12.3 / 5.5 ms, 5.0 / 0.1 ms, 18.4-22.9 ms |
| L40 as published | 25.7 / 25.6, 14.4 / 0.4, 54.0-66.1 | 12.3 / 11.0, 5.0 / 0.2, 21.9-28.4 |
| A40 | 31.9 / 31.0, 17.9 / 0.5, 66.3-81.3 | 15.3 / 13.3, 6.2 / 0.2, 27.0-34.9 |
| A100 40GB | 14.3 / 14.9, 8.0 / 0.3, 30.5-37.4 | 6.9 / 6.4, 2.8 / 0.1, 12.4-16.1 |
| A100 80GB SXM | 10.9 / 14.9, 6.1 / 0.3, 26.7-32.1 | 5.2 / 6.4, 2.1 / 0.1, 10.8-13.8 |
| H100 SXM | 6.6 / 4.7, 3.7 / 0.1, 12.3-15.1 | 3.2 / 2.0, 1.3 / 0.03, 5.1-6.5 |
| Laptop, measured supply | 48.0 / 92.7, 26.8 / 1.6, 144.7-169.1 (measured 145.7) | 23.0 / 39.7, 9.3 / 0.5, 59.3-72.5 (measured 58.6-62.1) |

By component:

| Component | Intensity, FLOP per byte | Verdict |
|---|---|---|
| Action expert, denoising loop | 6 | Bandwidth-bound on every card, by a factor of 17 to 70. On the laptop it is 23% of the teacher's latency (34 of 146 ms) and 17-23% of the student's [M] |
| VLM backbone, prefix pass | 228 | Compute-bound on the laptop (its matrix products ran at 46-47 TFLOPS in the trace [M]). Balanced on the A40, L40 as published and A100 40GB; bandwidth-leaning on the L40S and H100; compute-leaning on the A100 80GB |
| SigLIP + projector | 138 | Compute-leaning on the laptop; bandwidth-leaning on every cloud card except the A100 80GB, where it is balanced, because at batch 1 half of its traffic is weights, read three times |
| Whole query | 136 (teacher), 134 (student) | Compute-leaning on the laptop only |

**So, is memory bandwidth a bottleneck for inference?**

- On the laptop, as evaluated in phase 1: for about a fifth to a quarter of the latency, the denoising loop. The rest is compute. Traffic alone, at the measured rate, would take 75 ms of the teacher's 146 ms if every counted byte reached memory, and 38 ms counting weights only; the truth for the compiled model lies between.
- On a data-center card: yes, for most of the query. Their tensor rate is 3 to 20 times the laptop's and their bandwidth 1.5 to 7 times, so the compute-bound prefix shrinks faster than the weight streaming. On an L40S the floor is set by bytes: 40 ms of traffic against 13 ms of arithmetic for the teacher.
- These cloud figures are lower bounds. At batch 1 on a fast card, per-kernel launch overhead, which the roofline does not model, becomes a third limit. I could not measure it.
- The student's advantage survives in either regime: it moves 43% of the teacher's bytes and does 43% of its arithmetic.

## 7. Read versus write

Demand [D from M for training, M for inference]:

| | Read : write |
|---|---|
| KD step | 1.23 |
| Task-only step | 1.30 |
| Teacher forward | 1.12 |
| Student forward | 1.14 |
| Recomputation | 1.11 |
| Gradient operators | 1.78 |
| Gradient clipping | 2.00 |
| AdamW | 1.63 |
| SigLIP + projector, training | 1.22 |
| VLM backbone layers, training | 1.28 |
| Joint attention, training | 1.14 |
| Action expert, training | 1.89 |
| Inference query, teacher / student | 3.30 / 3.36 |
| SigLIP, inference | 3.57 |
| VLM backbone, inference | 1.73 |
| Action expert, denoising | 19.3 / 19.7 |

Forward-style passes read a little more than they write, because every operator reads its inputs and weights and writes one output.
The gradient pass reads nearly twice what it writes: it reads saved activations and incoming gradients and writes only gradients.
Inference is read-heavy because weights are half its traffic and are never written.
The denoising loop is almost pure reading.

Supply [M]: on the laptop a write-only kernel ran at 571 GB/s and a read-only kernel at 501-555 GB/s.
A copy, which does both, ran at 473-484 GB/s in total.
A written byte costs no more than a read byte; there is no sign of a read-before-write penalty.

**Does it change the verdict? No.**
With reads and writes costing the same, time depends on the total.
If a card wrote markedly slower than it read, training (45% writes) would suffer more than inference (23% writes).
The probe in section 9 prints the read-only and write-only rate of the cloud card, which closes this for that card.

## 8. The ViT

SigLIP So400m/14 plus the linear projector, as described in the earlier report's section 8: 27 layers, 256 tokens per image, three images per sample, frozen, identical in teacher and student.

**Its share of memory traffic.**

| | SigLIP + projector | Share | Of which the all-zero third image |
|---|---|---|---|
| KD step, batch 64 [D from M] | 295.6 GB (162.4 read, 133.3 written), two passes | 10.9% of bytes, 11.4% of FLOPs | 98.5 GB, 3.6% of the step |
| Task-only step [D from M] | 147.8 GB, one pass | 9.0% of bytes, 9.6% of FLOPs | 49.3 GB, 3.0% |
| Teacher inference query [M] | 4.8 GB, of which 2.49 GB weights | 13.8% | 1.6 GB |
| Student inference query [M] | 4.8 GB, of which 2.49 GB weights | 31.9% | 1.6 GB |

The three images are three identical operator sequences of 438 operators each in the trace, moving 49.27 GB each per pass at batch 64 [M].
The zero image is masked out of attention afterwards but is encoded in full (`src/openpi/policies/libero_policy.py:62`; `pi0_pytorch.py:197-210` has no skip).

**Is it an issue for memory bandwidth? No.**

- It is the lightest part of the model per unit of work: 110 KB per token per layer against 825 KB for a Gemma layer [D from M], a factor of 7.5. It uses fused attention, so its attention maps never reach memory, and its layers are narrower (1,152 against 2,048, MLP 4,304 against 16,384).
- Its intensity of 286 FLOP per byte is at the level of the whole step (271). On an L40S it is bandwidth-leaning like the rest, on an A100 compute-leaning like the rest. It does not behave differently from the model around it.
- In training it has no backward pass, no recomputation and no optimizer traffic.

**Does the answer differ between teacher and student? In share, yes; in bytes, no.**
Distillation removes 12 of 18 layers from the backbone and the expert and nothing from SigLIP.
Its 4.8 GB per query is the same in both models, so its share rises from 14% to 32%, and in the student it is as large as the 6-layer backbone pass (5.9 GB) or the ten-step expert loop (4.3 GB).
With no transformer layers at all, a query would still move 4.8 GB.
That is the memory-traffic form of the limit the earlier report stated for parameters and latency.

**Three things about it that are avoidable**, none needed for the experiment, all code changes:

1. In a KD step the student's SigLIP pass repeats the teacher's: 147.8 GB and 42.3 TFLOP, 5.4% of the step's bytes.
2. At batch 1 the weights are read three times because the images are encoded one by one. One batched call would read them once and save 1.66 GB per query, 11% of the student's traffic.
3. The zero image costs a third of the encoder and, through its 256 masked tokens, part of the backbone. Removing it changes the model's input format, so it is not something to do inside this experiment.

## 9. What a measurement on the cloud card settles, and the lines for `P5`

Everything in sections 6.2 and 6.3 for the cloud cards rests on vendor figures and on the card being one of the candidates.
A measurement on the real card settles four things:

1. **Which card it is.** One line.
2. **What it delivers.** Its copy, read-only and write-only rates and its bf16 matrix-product rate replace the vendor figures. This also settles the factor of two between the L40 and L40S documents, and read against write on that card.
3. **How much of the real batch-64 step is memory traffic.** Measured seconds per step against 2,723 GB ÷ the measured copy rate and 739 TFLOP ÷ the measured rate. This is the direct answer to "is it a bottleneck" for that card.
4. **An independent check.** `utilization.memory` during the steps, against its value under a pure copy and a pure matrix product on the same card.

The lines below add these to the runbook's 50-step timed run.
They cost about 25 seconds, write nothing outside `$W/logs`, and append to the `speed.txt` the mentor already sends back.
I ran the complete block with these lines on the laptop against the real `scripts/distill_pytorch.py`; the only substitutions were batch 2 on the 8 cached episodes, 2 workers, and a freeze set that also freezes the backbone so that the run fits 16 GB.
It passed [M]; the output is below.
I did not edit the runbook.

**(A) Insert after the line `export CUDA_VISIBLE_DEVICES="${GPU:-0}"`:**

```bash
# Supply probe (about 25 s, GPU otherwise idle): what this card delivers for a pure copy, read, write and bf16 GEMM,
# and what NVML's utilization.memory reads for a purely bandwidth-bound and a purely compute-bound load.
.venv/bin/python - > "$W/logs/membw_probe.txt" 2>&1 <<'PY'
import time, torch
def util():
    try: return "util_gpu=%d util_mem=%d" % (torch.cuda.utilization(0), torch.cuda.memory_usage(0))
    except Exception as e: return "util=NA(%s)" % type(e).__name__
def rate(fn, secs=5.0):
    fn(); torch.cuda.synchronize(); t = time.perf_counter(); n = 0
    while time.perf_counter() - t < secs:
        fn(); n += 1
        if n % 4 == 0: torch.cuda.synchronize()
    u = util(); torch.cuda.synchronize(); return n / (time.perf_counter() - t), u
G = 1 << 30
x = torch.ones(G // 2, dtype=torch.bfloat16, device="cuda"); y = torch.empty_like(x)
print("PROBE gpu=%s" % torch.cuda.get_device_name(0).replace(" ", "_"))
r, u = rate(lambda: y.copy_(x)); print("PROBE copy_GBps=%.0f %s" % (r * 2 * G / 1e9, u))
r, u = rate(lambda: x.sum()); print("PROBE read_GBps=%.0f %s" % (r * G / 1e9, u))
r, u = rate(lambda: y.fill_(1.0)); print("PROBE write_GBps=%.0f %s" % (r * G / 1e9, u))
del x, y
a = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda"); b = torch.randn_like(a)
r, u = rate(lambda: a @ b); print("PROBE gemm_bf16_TFLOPS=%.0f %s" % (r * 2 * 8192**3 / 1e12, u))
PY
# Sample NVML utilization twice a second while the 50 steps run.
nvidia-smi -i "${GPU:-0}" --query-gpu=utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits -lms 500 > "$W/logs/timed50_gpu.csv" 2>/dev/null &
SMI=$!
```

**(B) Insert after the line `RC=$?`:**

```bash
kill "$SMI" 2>/dev/null
```

**(C) Append after the last line of the block, the one that ends in `| tee "$W/logs/speed.txt"`:**

```bash
{ grep PROBE "$W/logs/membw_probe.txt"
  awk -F', *' '$1 >= 50 { n++; g += $1; m += $2; if ($2 > mx) mx = $2 } END { if (n) printf "MEMBW train_samples=%d util_gpu_mean=%.0f util_mem_mean=%.0f util_mem_max=%d\n", n, g / n, m / n, mx; else print "MEMBW train_samples=0" }' "$W/logs/timed50_gpu.csv"
  awk -v kd=2723 -v fl=739 '/copy_GBps/ { split($2, a, "="); c = a[2] } /gemm_bf16_TFLOPS/ { split($2, a, "="); f = a[2] } /TIMED PASS/ { split($3, a, "="); s = a[2] } END { if (c > 0 && f > 0 && s > 0) printf "MEMBW traffic_s=%.2f compute_s=%.2f measured_s=%.2f traffic_share_of_measured=%.0f%%\n", kd / c, fl / f, s, 100 * kd / c / s }' "$W/logs/membw_probe.txt" "$W/logs/speed.txt"
} | tee -a "$W/logs/speed.txt"
```

Output of the block on the laptop [M] (`speed.txt`):

```
TIMED PASS s_per_step=0.70 peak_reserved_GB=13.39 est_hours_kd=0.6 est_hours_taskonly=0.4
PROBE gpu=NVIDIA_GeForce_RTX_5080_Laptop_GPU
PROBE copy_GBps=489 util_gpu=99 util_mem=89
PROBE read_GBps=526 util_gpu=97 util_mem=94
PROBE write_GBps=566 util_gpu=97 util_mem=89
PROBE gemm_bf16_TFLOPS=55 util_gpu=100 util_mem=15
MEMBW train_samples=66 util_gpu_mean=86 util_mem_mean=23 util_mem_max=33
MEMBW traffic_s=5.57 compute_s=13.44 measured_s=0.70 traffic_share_of_measured=796%
```

How to read it:

- The constants 2,723 GB and 739 TFLOP in the last line are the KD step at batch 64 with the phase 2 freeze set. The last line is meaningful only for that run; on the laptop at batch 2 it printed 796%, which means nothing. The laptop's batch-64 equivalent is 5.57 s ÷ 19.4 s = 29%.
- `traffic_share_of_measured` near or above 50% means the step is mostly bandwidth-bound on that card. Expected from the table in section 6.2: about 55-70% on an L40S, 30-50% on an A100, A40 or L40.
- `copy_GBps` is expected at 80-100% of the vendor bandwidth: 690-860 on an L40S or L40, 560-700 on an A40, 1,240-1,555 on an A100 40GB, 1,550-2,040 on an A100 80GB [D, A].
- `gemm_bf16_TFLOPS` is expected below the vendor rate by an unknown margin. On an L40 it shows which of the two published figures is real.
- `util_mem_mean` during training should fall between the `util_mem` printed for the matrix product and the one printed for the copy. The closer to the copy value, the more bandwidth-bound the step. On the laptop the anchors were 15 and 89 and a batch-8 student step read 32.
- `torch.cuda.utilization` and `memory_usage` need the `pynvml` package, which is in the project's lock file. If it is missing the line prints `util=NA(...)` and the rest still works.
- The `-i` index assumes `nvidia-smi` and CUDA number the cards in the same order, which holds on a node with identical cards.

## 10. What could not be measured

- **Bytes at the hardware level.** NVIDIA's profiler has per-kernel DRAM counters (`dram__bytes_read.sum`, `dram__bytes_write.sum`). Under WSL2 it connects and then refuses [M], Nsight Compute 2026.2.0.0: `ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.` Enabling them is a setting on the Windows host, outside what a worker can do. So the byte counts are operator-level, and the claim that they reach memory rests on the indirect evidence of section 6.1: large elementwise operators running at the measured memory rate, and `utilization.memory` agreeing with the time shares.
- **Batch 64.** A 16 GB card holds the student step up to about batch 16 with AdamW, and the teacher forward to batch 32. Batch 64 is an exact extrapolation of the bytes, and a model prediction of the time.
- **A real KD step with the phase 2 freeze set.** Teacher, student and AdamW state together need more than 16 GB even at batch 2 (phase 1: failed at the first AdamW step, 13.53 GB peak). Teacher forward and student step were measured separately. The P5 test in section 9 ran the real KD step, with the backbone frozen.
- **Any cloud card.** All cloud figures are vendor specifications passed through a model validated on one laptop.
- **The compiled graph's traffic.** Inference bytes are counted on the eager path. The compiled path reads the same weights; its activation traffic may be lower. The agreement of the roofline with compiled latency suggests it is not much lower, or that cache absorbs a similar amount.
- **The cache.** The size of the GPU's on-chip cache is not published and not readable. The calibration only shows that 16 MiB tensors are served from it and 64 MiB tensors are not.
- **A fixed memory clock.** The laptop lowers its memory clock under load, from 14,001 to 9,001 MHz. I could not hold it fixed, so the laptop's delivered bandwidth is that of a power-managed state, about 64% of its design maximum.
- **Fine time resolution from NVML.** Its utilization is an average over a period of up to one second, so it describes a workload, not an operator. Per-operator attribution comes from synchronized timing, which adds 10-12% to the total.

## 11. Recommendations

1. **Forward the summary at the top to the mentor**, as the corrected answer to the bandwidth question.
2. **Add the lines of section 9 to runbook block `P5`** before the runbook is used. They are tested, cost 25 seconds, and turn every cloud figure in this report from a specification into a measurement.
3. **Change nothing in the experiment for this.** The run fits and finishes on any candidate card. Memory bandwidth affects how long it takes, not whether it works or what it produces.
4. **Report the student's efficiency in bytes as well as FLOPs.** For a productization audience the per-query traffic (15.0 against 34.6 GB) is the quantity that sets latency on fast hardware.

Follow-up code changes, none made here, none required, in order of payoff per KD step at batch 64 [D]:

| Change | Saves per KD step | Changes the arithmetic? |
|---|---|---|
| Remove the outer checkpoint around the layer stack (`pi0_pytorch.py:581-583`); each layer is already checkpointed on its own | 313 GB (11.5%) and 85 TFLOP (11.5%); 19% of a task-only step | No. Peak memory is expected to be unchanged, because it occurs in the backward pass where the per-layer boundaries exist either way; not measured |
| Reuse the teacher's image embeddings in the student during KD | 148 GB (5.4%) and 42 TFLOP (5.7%) | No |
| Fused attention for the Gemma layers, as SigLIP already has | Up to 659 GB (24%) | Slightly: the softmax is no longer taken in float32. Needs revalidation against the phase 1 teacher numbers |
| Inference: precompute the adaptive-norm modulation for the ten fixed time steps | 4.7 GB per teacher query (13%), 1.6 GB per student query (11%) | No |
| Inference: encode the three images in one batched call | 1.66 GB per query | No |

The first two are candidates for the pull request the earlier report already proposes (its section 15), if the captain decides to ship changes before the cloud run.
The 10-step smoke run with its known loss values is the test that they change nothing.

## 12. Corrections to earlier documents

| Document | Says | Finding |
|---|---|---|
| Earlier report, section 6, row "GPU memory bandwidth" | 100-200 GB moved per KD step; 0.1-0.25 s of a 5-9 s step | 2,723 GB per KD step; 3.15 s on an L40S at the vendor bandwidth, a little over half the step |
| Earlier report, section 6, note | The L40S "is more likely to land in the slower half of the FLOP-based estimate" because it has the least bandwidth per TFLOPS | Correct in direction. The size of the effect is that traffic, not arithmetic, is the larger term on that card |
| Earlier report, section 8.3 | Reusing the teacher's image embeddings "would save about 6.5% of the KD step" | Consistent: 5.4% of bytes, 5.7% of FLOPs |
| README experiment log, lines 213-215 | Per sample: teacher forward 4.5 TFLOP, student forward 1.9, backward 3.8, checkpoint recompute 1.9, total 12; 0.77 PFLOP per step | Traced [D from M]: teacher forward 4.65, student forward 1.99, gradient operators 2.25, recomputation 2.66 (two passes), total 11.5; 0.739 PFLOP per step. The total is close because two errors cancel |
| README, line 217 | "~30-40% bf16 utilization with eager attention" assumed | The mechanism behind that figure is memory traffic. On the laptop matrix products run at 100% of the calibrated rate and take 63-65% of the time |

## 13. Ambiguities I resolved

| Ambiguity | Resolution |
|---|---|
| "Memory bandwidth" | GPU device memory (GDDR or HBM) to the GPU's cores. Not host RAM, not PCIe, not the control bandwidth of the policy |
| Training or inference | Both |
| What "bytes moved" means | Operator-level operand count, defined in section 3.1, with its limits stated |
| Which training code path | The eager path, as `scripts/distill_pytorch.py` runs it; only `sample_actions` is compiled in this code base (`pi0_pytorch.py:112`) |
| Which inference path | Timed compiled, as in phase 1; traced eager |
| "The phase 2 freeze set" | Vision tower, projector, token embedding and its tied `lm_head`, as in the runbook's `FREEZE` |
| Dense or sparse tensor rates | Dense. The sparsity figures need pruned weights, which this model does not have |
| The L40's published tensor rate | Used as published, with the conflict flagged |
| What "bottleneck" means | The larger of the two terms, traffic time and compute time, per operator and in total. Stated as a share of the step, since both always contribute |
| Where the joint attention belongs | Its own row. It is computed over backbone and expert tokens together; 99% of its rows are backbone tokens |
| How to reach batch 64 | Exact linear extrapolation per operator, checked at the batches not used for the fit |
| Which laptop bandwidth to use | The measured mixed read-write rate, 463 GB/s, because most operators both read and write |
| Inference hardware | The candidate cards and the laptop. The target device is undecided and, per the captain, not to be asked again |

## 14. Still unknown

- The cloud card's identity.
- What fraction of their vendor bandwidth and tensor rate the cloud cards deliver. The derated column assumes 80% and 60%.
- Which of the two published L40 tensor rates is real.
- Whether the cloud cards write as fast as they read.
- Measured batch-64 step time on any card.
- Hardware-counted bytes, anywhere.
- How much of the batch-1 activation traffic the on-chip cache absorbs, and how much the compiled graph removes.
- Per-kernel launch overhead on the cloud cards at batch 1, which bounds inference there from below.
- Whether removing the outer checkpoint leaves peak memory unchanged. Expected, not measured.
- Whether the data rate of this laptop's memory is twice the clock NVML reports. Assumed from GDDR6 practice; it affects only the "interface rate" comparison, not any measured rate.

## 15. Decisions for the captain

1. **Add the capture lines of section 9 to runbook block `P5` before the runbook goes to the mentor.** Recommended. If the runbook has already gone, they can be sent as a replacement for that one block.
2. **Whether the first two follow-up changes of section 11 join the repository follow-ups** that the earlier report's section 16, item 4, already leaves to the captain to ship before or after the cloud run. Together they remove about 17% of a KD step's bytes and FLOPs and 19% of a task-only step's, and change no result. Not required.

Both gate the backlog item `vla-phase2`, which is already held for the captain.

## Sources

- NVIDIA L40S product page, specification table: https://www.nvidia.com/en-us/data-center/l40s/
- NVIDIA L40 product page and datasheet (January 2023): https://www.nvidia.com/en-us/data-center/l40/ , https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/support-guide/NVIDIA-L40-Datasheet-January-2023.pdf
- NVIDIA A40 product page and datasheet (March 2022): https://www.nvidia.com/en-us/data-center/a40/ , https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf
- NVIDIA A100 product page and datasheet: https://www.nvidia.com/en-us/data-center/a100/ , https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
- NVIDIA H100 product page: https://www.nvidia.com/en-us/data-center/h100/
- NVIDIA GeForce laptop comparison page (RTX 5080 Laptop GPU: 16 GB GDDR7, 256-bit): https://www.nvidia.com/en-us/geforce/laptops/compare/
- Nsight Compute release notes (WSL2 support): https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html
- Earlier report and runbook: `data/vla-cloud-runbook/report.md`, `data/vla-cloud-runbook/runbook.md` (block `P5` at lines 498-513).
- Project files cited by `file:line` are at commit `966da5c`. `modeling_gemma.py` is `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py`.

## Appendix A. Raw measurements (RTX 5080 Laptop GPU 16 GB, driver 616.92, torch 2.7.1+cu128, WSL2, 2026-10-03 20:10-20:53 CDT)

### A.1 Supply calibration (`calib.py`, 4 s per workload)

```
gpu_static: mem_bus_width_bits 256, mem_clock_max_MHz 14001, sm_clock_max_MHz 3090
stream, 1 GiB tensors            read GB/s  write GB/s  total GB/s  util_gpu  util_mem  mem clock MHz
read_sum       float32               555.3        0.0       555.3      98.0      89.3   10009
write_fill     float32                 0.0      572.6       572.6      98.0      79.8   11941
copy           float32               242.2      242.2       484.4      98.9      84.3    9001
add_out        float32               323.7      161.8       485.5      99.0      86.9    9085
mul_inplace    float32               232.5      232.5       464.9      98.9      84.4    9001
read_sum       bfloat16              500.8        0.0       500.8      97.8      89.2    9001
write_fill     bfloat16                0.0      570.5       570.5      98.0      86.1   11101
copy           bfloat16              236.4      236.4       472.7      98.8      84.1    9001
add_out        bfloat16              315.6      157.8       473.4      98.9      86.4    9001
mul_inplace    bfloat16              230.6      230.6       461.2      98.7      83.8    9001
cast bf16->f32                       148.5      297.0       445.5      99.0      81.6    9001
clone          bfloat16              230.4      230.4       460.9      92.2      78.2    9001
smaller tensors (float32), total GB/s and time per call:
  64 MiB: read 405.8 (165 us)  write 522.0  copy 458.1  add 459.7  mul 446.2      util_mem 74-84
  16 MiB: read 438.0 (38 us)   write 629.4  copy 1241.8 (27 us)  mul 1233.2       util_mem 2-3 (cache)
   4 MiB: read 123.1 (34 us)   write 224.9  copy 426.4 (20 us)   mul 415.9        util_mem 2-3 (launch overhead)
matrix product, square           TFLOPS   util_gpu  util_mem
bfloat16 4096 / 8192 / 16384     49.9 / 50.0 / 49.6     97.7-99.4   12.7 / 15.2 / 8.6 (the last was sampled partly idle: util_gpu 35)
float16  4096 / 8192 / 16384     50.2 / 49.4 / 48.9     98.2-99.6   12.7 / 15.0 / 21.1
float32  4096 / 8192 / 16384     24.5 / 25.7 / 25.7     82.7-100    19.4-20.5
matrix product, M rows x weight, bf16; cold = 12 or 54 different weights cycled, hot = one weight
VLM MLP 2048->16384 (67 MB)   M=1     196.7 us cold  195.2 us hot   weights at 341 GB/s    util_mem 63
                              M=10    175.6          175.5          382 GB/s               70
                              M=100   257.4          257.9          26 TFLOPS              51
                              M=256   432.7          439.0          40 TFLOPS              34
                              M=968  1395.6         1397.1          46.5 TFLOPS            16
                              M=7744 10346          10439           50 TFLOPS              14
                              M=61952 80819         80872           51.4 TFLOPS            13
expert MLP 1024->4096 (8.4 MB) M=10    43.3 us cold   31.8 us hot                          util_mem 36.8 cold, 3.0 hot
                              M=968   225.8          218.2          36-37 TFLOPS           10.0 cold, 3.0 hot
                              M=61952 10598          10603          49.0 TFLOPS            17
SigLIP MLP 1152->4304 (9.9 MB) M=10    50.1 us cold   33.1 us hot                          util_mem 38.2 cold, 3.0 hot
                              M=61952 12902          12822          47.6-47.9 TFLOPS       18
idle: util_gpu 8, util_mem 2, 34 W, memory clock 11101 MHz
```

### A.2 Traces: totals per batch (identical in every run: teacher three, student two)

```
teacher KD forward (eval_model, no grad)      student training step (fwd + backward + clip + AdamW)
batch  read GB  write GB  wall s              batch  read GB  write GB  fwd s   bwd s   clip s  opt s
1      16.21     7.78     0.1462              1      45.12    25.54     0.0755  0.1414  0.0148  0.0631
2      25.30    15.99     0.2624              2      59.38    36.65     0.1156  0.2479  0.0142  0.0625
4      43.04    31.98     0.4845              4      87.33    58.38     0.2099  0.4777  0.0137  0.0625
8      78.52    63.95     1.0310              8     143.21   101.84     0.3908  0.9757  0.0138  0.0628
16    149.48   127.89     1.9460
operators per pass that move data: teacher 3,231; student step 4,438 (3,159 and 4,336 at batch 1, where some copies become views)
linear fit on the two largest batches, relative error at the others: read 0.0, write 0.0, FLOPs 0.0
teacher: batch-independent 7.56 GB read; per sample 8.87 GB read, 7.99 GB written
student: batch-independent 31.44 GB read, 14.92 GB written; per sample 13.97 GB read, 10.87 GB written
student peak allocated: 10.10 GB at batch 1-4 (set by the AdamW step's temporaries), 10.88 GB at batch 8
task-only against KD student part, batch 2: 4,432 against 4,438 operators, 0.021 MB difference
```

Backward phase of the student step at batch 8, consecutive blocks (recomputation yes/no, operators, GB, TFLOP, backbone MLP products):

```
no    14   0.00   0.000   0
yes  635  40.95  10.625  18      <- outer checkpoint: all six layers at once
no    21   0.03   0.000   0
yes  101   6.72   1.771   3      <- inner checkpoint, one layer
no   142   5.61   0.294   0      <- that layer's gradient operators
yes  101   6.72   1.771   3
no   176  12.28   3.542   0
 ... four more pairs like the last two ...
forward pass, layer stack: 10.625 TFLOP, 18 backbone MLP products; recomputation in total: 21.25 TFLOP, 36
```

### A.3 Per-operator timing, batch 8 (CUDA synchronize around every operator)

```
teacher KD forward: sum 1139.6 ms; wall without the tracer 1031.0 ms
  matrix products          920 calls   41.7 GB  37.16 TFLOP   743.7 ms    56 GB/s  50.0 TFLOPS
  elementwise >= 64 MB     437 calls   80.9 GB                221.0 ms   366 GB/s
  elementwise <  64 MB    1874 calls   19.9 GB                174.9 ms   114 GB/s   93 us per call
  by operator: mm 557.4 ms (55.2 TFLOPS), addmm 137.6 (36.9 TFLOPS), mul 92.4 (324 GB/s), add 68.1 (219),
               clone 53.3 (147), _to_copy 51.4 (276), bmm 36.2 (31.2 TFLOPS, 188 GB/s), gelu 31.4 (383),
               cat 22.5 (138), _softmax 19.7 (447), pow 16.2 (292), native_layer_norm 14.5 (108)
student training step: sum 1617.1 ms; wall without the tracer 1446.6 ms
  matrix products         1139 calls   60.0 GB  55.16 TFLOP  1014.9 ms    59 GB/s  54.4 TFLOPS
  elementwise >= 64 MB     685 calls  159.5 GB                386.6 ms   413 GB/s
  elementwise <  64 MB    2614 calls   25.5 GB                215.7 ms   118 GB/s   83 us per call
  by section: forward products 316.9 ms (50.2 TFLOPS), forward elementwise 166.6 (247 GB/s),
              recomputation products 374.5 (56.7), recomputation elementwise 202.6 (299),
              gradient products 323.4 (55.7), gradient elementwise 150.9 (314),
              clipping 13.7 (327 GB/s), AdamW 64.8 (483 GB/s)
  by operator: mm 817.0 ms (58.7 TFLOPS), mul 137.8 (361 GB/s), addmm 134.6 (37.7 TFLOPS), add 77.4 (262),
               _to_copy 73.5 (296), clone 55.4 (156), bmm 50.6 (37.2 TFLOPS, 223 GB/s), gelu 29.0 (415),
               _foreach_mul_ 21.8 (410), pow 20.4 (300), cat 18.9 (164), _softmax 18.7 (472)
```

### A.4 NVML during model workloads (mean after the first 1.5 s; memory clock 9,001 MHz throughout)

```
teacher KD forward, batch 16        util_gpu 99.6  util_mem 30.5   (runs 1 and 2: 29.5, 29.4)
student training step, batch 8      util_gpu 93.4  util_mem 31.8   (run 2: 31.2; 1.57 s per step in the loop)
teacher inference, eager            util_gpu 52.4  util_mem 16.9
teacher inference, compiled         util_gpu 92.6  util_mem 29.2   (runs 1 and 2: 28.6, 26.3)
student inference, eager            util_gpu 44.0  util_mem 13.9
student inference, compiled         util_gpu 91.0  util_mem 28.3   (run 2: 27.8)
```

### A.5 Inference timing, batch 1 (p50 of 100 calls compiled; mean over 3 s eager)

```
teacher  compiled 10 / 5 / 1 steps: 145.7 / 129.1 / 115.3 ms   (run 1: 152.1 / 133.5 / 120.1; run 2, CPU busy: 161.6 / 132.8 / 118.0)
         eager    10 / 5 / 1 steps: 379.0 / 275.3 / 178.1 ms;  batch 2 / 4 / 8 at 10 steps: 543 / 810 / 1296 ms
student  compiled 10 / 5 / 1 steps:  62.1 /  55.8 /  50.0 ms   (run 2, CPU busy: 59.1 / 54.6 / 50.2)
         eager    10 / 5 / 1 steps: 156.4 / 136.9 /  90.1 ms;  batch 2 / 4 / 8 at 10 steps: 206 / 308 / 520 ms
phase 1 (results/phase1/bench): teacher 145.7 ms, student 58.6 ms
per denoising step, eager trace: teacher 1.243 GB (weights 1.098, of which adaptive-norm dense 0.466; cached keys and values 18.0 MB), 7.91 GFLOP
                                 student 0.429 GB (weights 0.380, of which adaptive-norm dense 0.164; cached keys and values 6.0 MB), 2.64 GFLOP
prefix (everything before the first denoising step): teacher 22.22 GB, 4.64 TFLOP; student 10.67 GB, 1.99 TFLOP
weight bytes on the GPU: SigLIP + projector 0.832 GB, backbone layers 3.964 (teacher) / 1.321 (student),
                         token embedding 1.053, expert layers 1.089 / 0.371, unused expert lm_head 0.527
```

"CPU busy" marks the run during which a package download was running; byte counts are unaffected, eager and small-kernel timings are.

### A.6 Nsight Compute under WSL2

```
$ ncu --target-processes all --metrics dram__bytes_read.sum,dram__bytes_write.sum,... python ncu_probe.py
Version 2026.2.0.0 (build 37790515) (public-release)
==PROF== Connected to process 191687 (.../python3.11)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```

### A.7 The P5 block with the added lines, on the laptop

Batch 2, 8 cached episodes, 2 workers, backbone frozen; otherwise the runbook's block with the lines of section 9. 2 min 39 s.

```
20:51:13.556 [I] step=0 loss=1.1697 lr=8.31e-08 grad_norm=12.15 time=11.5s
20:51:20.658 [I] step=10 loss=0.9249 lr=5.40e-07 grad_norm=11.33 time=7.1s
20:51:27.678 [I] step=20 loss=1.0307 lr=1.37e-06 grad_norm=14.84 time=7.0s
20:51:34.745 [I] step=30 loss=0.8858 lr=2.20e-06 grad_norm=10.04 time=7.1s
20:51:41.755 [I] step=40 loss=0.8907 lr=3.03e-06 grad_norm=10.93 time=7.0s
20:52:00.118 [I] Saved checkpoint at step 49 -> .../checkpoints/pi05_libero_l06/timed50/49
TIMED PASS s_per_step=0.70 peak_reserved_GB=13.39 est_hours_kd=0.6 est_hours_taskonly=0.4
PROBE ... MEMBW ...   (as in section 9)
timed50_gpu.csv: 270 samples; no sampler process left running
```

The first loss, 1.1697, is the phase 1 value for the same first batch.

## Appendix B. The accounting rule, as code

The whole tracer is about 250 lines in the discarded worktree. This is the part that decides what is counted, condensed (`flatten`, `index_bytes` and `inner_dim` stand for a few lines each).

```python
class Tracer(torch.utils._python_dispatch.TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if self.timed: torch.cuda.synchronize(); t0 = time.perf_counter()
        out = func(*args, **(kwargs or {}))
        if self.timed: torch.cuda.synchronize(); dt = time.perf_counter() - t0
        ins  = [t for t in flatten(args, kwargs) if t.is_cuda]          # tensors and lists of tensors
        outs = [t for t in flatten(out) if t.is_cuda]
        name = func._schema.name.split("::")[-1]
        nbytes = lambda t: min(t.numel() * t.element_size(), t.untyped_storage().nbytes())
        in_storages = {t.untyped_storage().data_ptr() for t in ins}
        if func._schema.is_mutable:                                     # in-place or out=
            if name in {"fill_", "zero_", "copy_", "normal_", "uniform_", ...}:
                rd = sum(nbytes(t) for t in ins[1:]); wr = nbytes(ins[0])          # target not read
            elif name.startswith("_foreach"):
                rd = sum(nbytes(t) for t in ins); wr = sum(nbytes(t) for t in flatten(args[0]))
            else:
                rd = sum(nbytes(t) for t in ins); wr = sum(nbytes(t) for t in outs)
        else:
            new = [t for t in outs if t.untyped_storage().data_ptr() not in in_storages]
            if outs and not new: return out                             # a view: nothing moves
            wr = sum(nbytes(t) for t in new)
            rd = wr + index_bytes if name in {"embedding", "index", "index_select", "gather"} \
                 else sum(nbytes(t) for t in ins)
        flops = 2 * outs[0].numel() * inner_dim  # mm, addmm, bmm, matmul, linear, convolution, fused attention; else 0
        self.ops.append((phase, stage, in_checkpoint_recompute, module_tag, name, shapes, rd, wr, flops, dt))
        return out
```

Roofline per operator: `t >= max((rd + wr) / bandwidth, flops / rate)`, `t <= (rd + wr) / bandwidth + flops / rate`, summed over operators.

## Appendix C. Commands

All from the worktree, after `source scripts/phase1/env.sh`, one at a time, with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

```bash
.venv/bin/python _membw/calib.py 4                          # A.1; 6 min
.venv/bin/python _membw/run_fwd_infer.py teacher 1,2,4,8,16 8   # teacher KD forward traces, inference trace and timing, NVML
.venv/bin/python _membw/run_fwd_infer.py student ""             # student inference
.venv/bin/python _membw/run_train.py 1,2,4,8 8 8                # student step traces, timed trace at batch 8, NVML at batch 8
.venv/bin/python _membw/analyze.py                              # tables (CPU only)
W=$PWD/_membw/p5w bash _membw/p5_laptop_test.sh                 # A.7
micromamba create -p _membw/ncu -c nvidia -c conda-forge nsight-compute=2026.2.0.7   # A.6; removed afterwards
```

Models were loaded from `/home/shayn/vla_distill_artifacts/checkpoints/pi05_libero_pytorch` (teacher) and `pi05_libero_l06_init` (student), read-only.
Inputs were 16 of the 457 held-out frames in `data/phase1/heldout_frames.npz`.
Scratch use was 31 MB of results, plus a 1.5 GB profiler install and a 4.6 GB checkpoint from the P5 test, both deleted.
