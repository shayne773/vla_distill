# π0.5 的 Shallow-π 蒸餾：原理與實驗流程（簡版）

- 日期：2026-09-28
- 定位：[完整報告](../vla-survey-plan/report.md)的精簡版，只講 Shallow-π 的原理和本儲存庫的實驗 pipeline。文獻調查、統計檢定力、產品化與認證等內容請看完整報告。
- 來源：Shallow-π 論文（arXiv [2601.20262](https://arxiv.org/abs/2601.20262) v1，下稱「論文」）；程式碼儲存庫 `shayne773/vla_distill` 的 main 分支（Phase 1 合併自 https://github.com/shayne773/vla_distill/pull/1）；完整報告。文中每個數字都標註出處；原文沒有給出的細節寫「論文未說明」，不作推測。

---

## 0. 摘要

| 項目 | 內容 |
|---|---|
| 目標 | 把 π0.5 LIBERO 模型（teacher）的 VLM backbone 和 action expert 都從 18 層壓縮到 6 層（student），測量 knowledge distillation（KD）相比「只用 task loss 訓練同一個淺層模型」到底帶來多少收益 |
| 方法 | Shallow-π：從 teacher 均勻抽取 6 層來初始化 student（`[0, 3, 7, 10, 14, 17]`），再用三個 loss 訓練：task（flow matching）loss、teacher-velocity KD loss、attention-distillation loss |
| 論文結果 | π0.5：18 → 6 層，LIBERO 平均 96 → 95，FLOPs 3.39T → 1.30T，H100 CUDA time 25.5 → 11.3 ms（Table 2）。L6 的 loss 消融：task only 93.0 / +KD 93.9 / +KD+attention 94.6（Table 1a） |
| 程式碼 | 作者的 openpi fork（icsl-Jeon/openpi，Apache-2.0）。本儲存庫修正了其中 3 個 bug；**fork 沒有實作 attention-distillation loss**，所以目前只用前兩個 loss |
| Phase 1（已完成，筆電 RTX 5080 Laptop 16 GB） | Teacher 96.0%（400 個 paired LIBERO episodes）；未訓練的 6 層 student 0.0%；每個 action chunk 的延遲 145.7 → 58.6 ms（2.5×）；訓練流程在筆電上跑通（只能在凍結 VLM 的條件下） |
| Phase 2（等待部門雲端算力） | KD run 與 task-only 對照組各 3,000 steps × batch 64，bf16，1 張 80 GB GPU，合計約 4–5 H100 GPU-hours；訓練完回到筆電做同樣的評估（每個模型 1,000 個 paired episodes） |
| 關鍵結論 | 只刪層、不訓練，policy 完全失效（96% → 0%），student 的能力全部來自訓練。Phase 2 要回答的是：同樣的算力下，KD 是否比只用 task loss 更好 |

---

## 1. Shallow-π 的原理

### 1.1 π0.5 的結構

π0.5 是一個 flow-based VLA（vision-language-action）模型，由兩個 transformer 組成：

| 元件 | 輸入 | 結構（本儲存庫配置） | 作用 |
|---|---|---|---|
| **VLM backbone**（PaliGemma） | 影像經 SigLIP 編碼成 image token（每路 256 個），加上語言指令 token | Gemma 2B：width 2048，18 層（`gemma_2b`，`src/openpi/models/gemma.py`） | 把影像和指令編碼成多模態表徵（prefix） |
| **Action expert** | 10 個帶噪的 action token（一個 action chunk = 10 步動作）；denoising 時間 τ 透過 adaRMSNorm 注入 | Gemma 300M：width 1024，18 層（`gemma_300m`） | 預測 flow-matching velocity，把噪聲逐步變成動作 |

**Action expert 如何讀取 VLM 資訊。** 兩個 transformer 的層數相同，而且逐層配對：在第 i 層，VLM token 和 action token 的 Q/K/V 被拼在一起做一次 attention（`gemma_pytorch.py` 中的 `compute_layer_complete`）。Attention mask 規定：

- VLM token（影像 + 語言）之間是雙向 attention，但**看不到** action token；
- action token 可以看到**同一層**的全部 VLM token，action token 之間也互相可見。

所以 action expert 的第 i 層以 VLM 第 i 層的 K/V 為條件。論文把這種結構稱為「action head mirrors the VLM backbone」（Figure 1）。由此得到一個直接推論：**兩個元件必須同步減層**。程式碼裡兩者共用同一個 `gemma_depth` 引數（`pi0_pytorch.py:90-91`）。

**Flow matching 與 denoising step。** 訓練時從真實動作 $a$ 和高斯噪聲 $\epsilon$ 構造插值 $a_\tau = \tau a + (1-\tau)\epsilon$，模型學習預測 velocity $u = a - \epsilon$（論文 §3.1）：

$$L_{\text{task}} = \mathbb{E}\,\big\|v_\theta(a_\tau, o, l, \tau) - u\big\|_2^2$$

推理時從純噪聲出發，用 Euler 法沿 $v_\theta$ 積分若干步，每一步就是一個 **denoising step**。openpi 預設 10 步（`sample_actions(num_steps=10)`）。VLM 的 K/V 與 τ 無關，因此每個 action chunk 只計算一次並快取；10 個 denoising step 都只重新執行 action expert。

> 記號說明：程式碼採用相反的時間方向（`x_t = t·noise + (1−t)·actions`，`u_t = noise − actions`，從 t=1 積分到 t=0，`pi0_pytorch.py:327-328, 401-417`），和論文的公式等價。

### 1.2 為什麼深度是主要成本

1. **層是序列執行的。** 同一層內的 token 可以在 GPU 上平行計算，層與層之間只能依次執行。論文 Figure 2（π0.5 LIBERO，H100 與 Jetson Orin，`torch.compile`）顯示：減少 transformer 層帶來的延遲下降，遠大於減少 visual token；在 H100 上，減少 token 幾乎不省時間（§3.2）。
2. **Action expert 在每個 denoising step 都要重新執行一次。** 它的深度要乘以 denoising step 數（10），因此只削減 backbone 不夠（§1）。
3. **Layer skipping 在 π 這類模型上行不通。** 論文 §4：相鄰層 cosine similarity 隨 τ 大幅變化，也無法預測某一層的重要性（跳過第 1–2 層的損失遠大於跳過第 16–17 層）；即使以 sensitivity 作為 oracle 依次刪層，刪掉超過 3 層後成功率也會崩潰（Figure 4）。所以論文選擇「結構性刪層 + 蒸餾」，而不是推理時跳層。

### 1.3 Student 如何構成

| 項目 | 做法 | 出處 |
|---|---|---|
| 深度 | backbone 與 action expert 都從 18 層減到 6 層（論文另測了 L9、L4） | 論文 Table 1、2 |
| 保留哪些層 | TinyBERT 式均勻抽樣，保留兩端：`linspace(0, 17, 6).round()` = **[0, 3, 7, 10, 14, 17]**；backbone 與 action expert 使用同一組層號，所以 teacher 中互相 attend 的層對（VLM 第 k 層 ↔ expert 第 k 層）在 student 中仍然配對 | 論文 §5；`layer_mapping.py:21-23` |
| 權重初始化 | 被選中的層直接複製 teacher 的權重；非層級權重（SigLIP、projector、token embedding、final norm、action 輸入/輸出投影、time MLP）形狀相同，也直接複製 | `make_student_init.py`；`distill_pytorch.py` |
| 選層策略 | 論文稱，只要訓練步數足夠，按 sensitivity 選層不比均勻抽樣更好 | 論文 §5 |

> 注意：論文 §5.2 又寫道 「the student is initialized by directly copying the bottom layers of the teacher」，與 §5 的均勻抽樣說法不一致；fork 的原始程式碼（有 bug）恰好讓 backbone 保留了第 0–5 層。本儲存庫按 §5 的說法實作均勻抽樣；在 Phase 1 中，均勻抽樣的初始 student 離 teacher 更近（動作 MSE 0.181 vs 0.266）。論文實際使用的是哪一種，原文無法確定。

### 1.4 三個 loss

| Loss | 公式（論文 Eq. 1–3） | 含義 |
|---|---|---|
| Task loss | $L_{\text{task}} = \mathbb{E}\,\|v_\theta - u\|_2^2$ | 普通的 flow matching：對準由真實動作算出的 velocity |
| KD loss（teacher velocity） | $L_{\text{kd}} = \mathbb{E}\,\|v_\theta - v_\phi\|_2^2$ | 在**同一個** $(a_\tau, o, l, \tau)$ 上對準 teacher 預測的 velocity $v_\phi$。和真實動作相比，teacher 輸出是更平滑、資訊更多的目標 |
| Attention-distillation loss | $L_{\text{attn}} = \mathbb{E}\,\mathrm{KL}\big(\mathrm{Attn}_\phi^{a\to vl}\,\big\|\,\mathrm{Attn}_\theta^{a\to vl}\big)$，其中 $\mathrm{Attn}^{a\to vl} = \mathrm{softmax}(Q^a K^{vl\top})$ | 在**一箇中間層**上，讓 student 中每個 action token 在 vision-language token 上的 attention 分佈去對準 teacher 的分佈 |

論文沒有給出三個 loss 的權重（fork 的程式碼中 task 與 KD 權重都是 1.0，見 §2.4）。

**Attention loss 只作用在 action token 上（action → vision-language），原因（§5.2）：**

- 在 flow-matching VLA 中，只有 action token 是被生成的，VL token 只是條件。對全部 token（$vl \to vl$）做 attention 匹配會過度約束 student，並干擾預訓練表徵。
- 實驗證據（Table 1c）：只對 action token 做時，L9/L6/L4 = 96.8 / 94.6 / 93.9；對全部 token 做時，L9/L6/L4 = **0.0 / 61.5 / 0.0**（論文描述為 「unstable training and near-zero success rates」）。
- 只在**中間層**做（Table 1b，L6：初始層 93.9 / 中間層 94.6 / 後期層 94.1）。理由：前面的層是直接從 teacher 複製的，本來就接近；最後一層的輸出已經由 $L_{\text{task}}$ 和 $L_{\text{kd}}$ 約束。論文沒有說明具體是第幾層、teacher 與 student 的層如何對應，也沒有說明 attention head 如何處理。

**Loss 消融（Table 1a，π0.5 LIBERO，30k steps × batch 64）：**

| Loss 組合 | L9 | L6 | L4 |
|---|---|---|---|
| task | 95.4 | 93.0 | 92.3 |
| task + KD | 96.2 | 93.9 | 93.9 |
| task + KD + attention | 96.8 | 94.6 | 94.2 |

（Table 1a 的 L4 完整組合為 94.2，Table 1c 同一設定為 93.9，論文中兩處數字不一致。Table 2 的 π0.5-L6 平均 95 為四捨五入值。）

### 1.5 Teacher / student 訓練示意圖

```
          one training batch: images + prompt (o, l), ground-truth chunk a,
          noise eps, time tau  ->  noisy chunk a_tau,  target u = a - eps
                     |                                   |
   +-----------------v-----------------+   +-------------v------------------+
   | Teacher pi0.5 (frozen, no_grad)   |   | Student (trainable)            |
   |  VLM backbone   18 layers         |   |  VLM backbone   6 layers       |
   |    | per-layer K/V                |   |    | per-layer K/V             |
   |  action expert  18 layers         |   |  action expert  6 layers       |
   |  init: -                          |   |  init: teacher layers          |
   |                                   |   |        [0,3,7,10,14,17]        |
   +-------+--------------------+------+   +------+---------------+---------+
           | v_phi              | Attn_phi(a->vl)  | v_theta       | Attn_theta(a->vl)
           |                    |  (middle layer)  |               |  (middle layer)
           |                    +-- L_attn = KL ---|---------------+   <- NOT in the fork
           +---- L_kd = ||v_theta - v_phi||^2 -----+
                 L_task = ||v_theta - u||^2 -------+   (u from ground-truth actions)
```

### 1.6 論文的主要數字

| 模型（Table 2，H100） | Spatial | Object | Goal | Long | Avg | FLOPs (T) | CUDA time (ms) |
|---|---|---|---|---|---|---|---|
| π0.5（teacher） | 98 | 96 | 97 | 93 | 96 | 3.39 | 25.5 |
| π0.5-L9 | 99 | 98 | 97 | 93 | 97 | 1.82 | 14.8 |
| π0.5-L6 | 98 | 96 | 94 | 90 | 95 | 1.30 | 11.3 |

- 損失集中在長時程的 Long suite（93 → 90）。
- 實機（Table 3，Jetson Thor，RB-Y1）：E2E 130 → 78 ms，例如 Recycle 12/20 → 17/20。student 表現更好，主要是因為延遲更短（論文 §6.2），不代表動作本身更準確。
- 論文的 teacher 是作者自行訓練的（30k steps，batch 64，只用兩路相機，§6.1）；本儲存庫使用 openpi 官方的 `pi05_libero` checkpoint。

---

## 2. 本儲存庫中的 pipeline

### 2.1 流程圖

```
 LAPTOP (RTX 5080 Laptop 16 GB)                         CLOUD (1x 80 GB GPU, phase 2)
 ----------------------------------------------------   ------------------------------------
 [1] env: source scripts/phase1/env.sh; uv sync;         [1'] same env + LeRobot LIBERO (54 GB)
     transformers_replace copy; setup_libero_client.sh
 [2] teacher: gs://openpi-assets/checkpoints/pi05_libero [2'] same conversion (~30 GB host RAM)
     -> examples/convert_jax_model_to_pytorch.py (bf16)
     -> checkpoints/pi05_libero_pytorch (+ assets/)
 [3] student init: scripts/phase1/make_student_init.py
     layer_mapping.py [0,3,7,10,14,17]
     -> checkpoints/pi05_libero_l06_init (0 steps)
 [4] readiness: scripts/phase1/distill_smoke.sh ---->    [4'] scripts/distill_pytorch.py
     (8 episodes, batch 2/8, frozen VLM)                      pi05_libero_l06           (KD)
                                                              pi05_libero_l06_taskonly  (control)
                                                              3k steps x 64, ckpt @1.5k, 3k
                                                                  |
 [5] evaluate every model identically  <------------------------- + (copy back model weights)
     eval_libero.sh      paired LIBERO success
     bench_model.py      latency / memory / params
     action_agreement.py MSE vs teacher on 457 held-out frames
     make_summary.py  -> results/phase1/SUMMARY.md
```

### 2.2 各步驟說明

| 步驟 | 做什麼 | 指令碼 / 配置 | 在哪裡跑 |
|---|---|---|---|
| 1. 環境 | 本機沒有 gcc，用 micromamba 環境 `vla-tools` 提供編譯器；兩個 venv：`.venv`（Python 3.11，torch 2.7.1+**cu128**，因為 PyPI 的 cu126 版本沒有 sm_120 kernel）與 `examples/libero/.venv`（Python 3.8，LIBERO 模擬客戶端）；`uv sync` 後需要把 `transformers_replace/*` 複製到 transformers 包中 | `scripts/phase1/env.sh`、`setup_libero_client.sh`、`pyproject.toml` | 筆電；雲端需重建 |
| 2. Teacher | 官方 JAX checkpoint 轉成 PyTorch bf16；norm stats（`assets/`）要手動複製（轉換指令碼的已知問題）。轉換耗時 3 min，主機記憶體峰值 28.8 GB | `examples/convert_jax_model_to_pytorch.py --config_name pi05_libero` | 筆電；雲端需重做 |
| 3. Student 初始化 | 複製 teacher 權重，逐層對映 `[0,3,7,10,14,17]`（backbone 與 expert 同時對映） | `scripts/phase1/make_student_init.py`，`src/openpi/models_pytorch/layer_mapping.py` | 筆電 |
| 4. 蒸餾訓練 | 見 §2.3 | `scripts/distill_pytorch.py`，配置 `pi05_libero_l06` / `pi05_libero_l06_taskonly`（`src/openpi/training/config.py:797-850`） | 筆電只做 smoke test；正式訓練在雲端 |
| 5. 評估 | 見 §2.6 | `eval_libero.sh`、`serve_policy_seeded.py`、`libero_eval.py`、`bench_model.py`、`action_agreement.py`、`make_summary.py` | 筆電（所有模型都在同一臺機器上評估） |

### 2.3 蒸餾訓練迴圈（`scripts/distill_pytorch.py`）

每個 step（`DistilledPI0Pytorch.forward`，`pi0_pytorch.py:521-597`）：

1. 對 batch 做一次預處理（random crop / rotate / color jitter），得到的輸入 **student 與 teacher 共用**；
2. 取樣噪聲 ε 與時間 t（Beta(1.5, 1) 分佈），構造帶譟動作 `x_t` 與目標 `u_t`；
3. 若 `loss_weight_teacher > 0`：在 `torch.no_grad()` 下由 teacher 計算 `v_t_teacher`（`eval_model`，同樣的 noise 與 t）；
4. student 前向得到 `v_t`；
5. `loss = loss_weight_gt · MSE(u_t, v_t) + loss_weight_teacher · MSE(v_t_teacher, v_t)`；backward，gradient clip 1.0，AdamW。

| 設定 | `pi05_libero_l06`（KD） | `pi05_libero_l06_taskonly`（對照） |
|---|---|---|
| Student | `DistilledPi0Config(pi05=True, gemma_depth=6, action_horizon=10)`，teacher = `pi05_libero` | 相同 |
| Loss 權重 | `loss_weight_gt=1.0`，`loss_weight_teacher=1.0` | `loss_weight_teacher=0.0`（不跑 teacher forward） |
| 資料 | LeRobot `physical-intelligence/libero`（1,693 episodes / 273,465 frames），使用 teacher 的 norm stats | 相同 |
| 配置預設值（fork） | 30,000 steps，batch 64，fp32，cosine LR（warmup 1,000，peak 2.5e-5，decay 30k → 2.5e-6），AdamW(b1 0.9, b2 0.95)，seed 42 | 相同 |
| Phase 2 覆蓋引數（README） | 3,000 steps，`--save-interval 1500`，`--lr-schedule.warmup-steps 300 --lr-schedule.decay-steps 3000`（否則 3k 步內 LR 不會衰減），`--pytorch-training-precision bfloat16` | 相同 |
| 凍結 | teacher 全部凍結（`eval()` + `no_grad`）。student 預設全部可訓練；新加的 `pytorch_freeze_prefixes` 可以按字首凍結 | 相同 |
| 其他 | gradient checkpointing 開啟；PyTorch 路徑不支援 EMA（日誌中說明） | 相同 |

**精度。** fork 的配置用 fp32 訓練，但單是 fp32 的 teacher 就佔 14.5 GB，所以筆電 smoke test 與 Phase 2 計劃都改用 bf16（teacher 與 student 都用 bf16）；推理評估一律用 bf16。

**凍結的範圍。** 筆電 smoke test 凍結了 SigLIP、projector 和 token embedding（run B/C 還凍結了 VLM backbone，只訓練 147M 引數）。完整報告 §7.3 的 Phase 2 計劃是：除 vision encoder 與 token embedding 外全部訓練；README 的視訊記憶體估算按整個 student 可訓練計算（約 45–55 GB，需要 80 GB 卡）。

### 2.4 修正的三個 fork bug

| # | 問題 | 後果 | 修正 |
|---|---|---|---|
| 1 | 層對映只作用於 action expert：`k.startswith(expert_prefix, k.startswith(backbone_prefix))` 把一個 bool 當成了 `startswith` 的 start 引數，backbone 的 key 從未匹配 | VLM 保留 teacher 第 0–5 層，而不是 `[0,3,7,10,14,17]`；backbone 與 expert 的層對錯位。初始 student 的動作 MSE 為 0.266（修正後 0.181），夾爪方向一致率 49%（修正後 71%） | 統一的 `layer_mapping.py`，訓練與 `make_student_init.py` 共用 |
| 2 | Teacher 與 student 各自重新做隨機資料增強 | KD 目標 $v_\phi$ 是在**另一張圖**上算出來的，KD loss 實際上帶有噪聲 | teacher 複用 student 的預處理結果（`eval_model(..., preprocessed=...)`） |
| 3 | `loss_weight_gt / loss_weight_teacher` 雖然存在但沒有被使用，loss 固定為兩項之和 | 無法做 task-only 對照 | 權重生效；`loss_weight_teacher=0` 時跳過 teacher → 對照配置 `pi05_libero_l06_taskonly` |

另外新增了 `pytorch_freeze_prefixes` 與 `data.lerobot_episodes`（選取部分 episode），只用於筆電 smoke test。

### 2.5 缺少的 attention-distillation loss

- fork 的 `distill_pytorch.py` 只有 task 與 KD 兩個 loss。Phase 2 的兩個 run 都缺少這一項，所以 KD 與 task-only 的比較仍然公平。
- 論文中這一項在 L6 貢獻約 +0.7 pt（93.9 → 94.6，Table 1a）。按完整報告 §4 的統計檢定力（power）表，這個幅度在 2,000 個 episodes 以內都無法可靠檢測。
- 實作細節敏感（對全部 token 做會失敗，Table 1c），論文也沒有給出層號與權重。因此列為選用延伸（完整報告 O2：開發 1–2 天，另需約 1.7–2.2 H100-h）。

### 2.6 評估（每個模型都用完全相同的命令）

| 評估 | 設定 | 指令碼 |
|---|---|---|
| Paired LIBERO | 4 suites × 10 tasks × 10 trials = 400 episodes/模型（Phase 2 改為 25 trials = 1,000）；init state 0–9，env seed 7；每個 chunk 10 個動作，執行 5 個後 replan；10 個 denoising step；224 px；各 suite 步數上限 220/280/300/520。每次 policy query 的 flow-matching 噪聲由固定 seed 生成，所有模型在同一時刻看到相同的噪聲；統計用 Wilson 95% CI 與 exact McNemar | `eval_libero.sh` → `serve_policy_seeded.py`（server）+ `libero_eval.py`（4 個並行 client）→ `results/phase1/libero/<model>/` |
| 延遲 / 視訊記憶體 | 一個 action chunk（10 個動作，10 個 denoising step），batch 1，bf16，`torch.compile`，每次 CUDA synchronize，warm-up 20 次 + 計時 200 次；報告 p50/p95、推理峰值視訊記憶體、引數量 | `bench_model.py` → `results/phase1/bench/` |
| Action agreement | 457 個 held-out frames（來自 teacher 在 init state 40 上的 rollout，不參與成功率評估）；chunk MSE 與首個動作的 MSE（相對 teacher）、夾爪方向一致率；teacher 換噪聲 seed 得到自身的噪聲底線（0.0086） | `collect_heldout_frames.sh`、`action_agreement.py` |

筆電上 MuJoCo 用軟體渲染（WSL2 下 GPU 驅動會卡住），4 個 worker 約 18 env steps/s；teacher 的 400 個 episodes 耗時 57 min，未訓練的 student 為 1 h 43 min（每個 episode 都跑滿步數上限）。筆電 GPU 與 Windows 桌面共用，因此**應比較 teacher/student 的延遲比值，而不是絕對毫秒數**。

---

## 3. 現狀與下一步

### 3.1 Phase 1 結果（`results/phase1/SUMMARY.md`）

| 指標 | Teacher `teacher_l18`（18+18 層） | 未訓練 student `student_l06_init`（6+6 層） |
|---|---|---|
| LIBERO 成功率（400 paired episodes） | **96.0%** [93.6, 97.5] | **0.0%** [0.0, 1.0] |
| spatial / object / goal / libero_10 | 98 / 97 / 96 / 93 | 0 / 0 / 0 / 0 |
| Exact McNemar | 只有 teacher 成功 384，只有 student 成功 0，p = 5.1e-116 | |
| 延遲 p50 / p95（每個 chunk，筆電） | 145.7 / 155.5 ms | 58.6 / 61.0 ms（2.5×） |
| 推理峰值視訊記憶體（allocated） | 6.97 GiB | 3.84 GiB |
| 引數：總計 / 推理實際使用 | 3,617M / 3,353M | 2,012M / 1,749M |
| 動作 MSE vs teacher（457 frames） | 0（噪聲底線 0.0086） | 0.181 [0.167, 0.195]，約為底線的 21 倍；夾爪方向一致率 71.1% |
| 訓練可行性（筆電） | VLM 可訓練（1.07B）時，batch 2 在第一次 AdamW step 時 OOM（峰值 13.5 GB）；凍結 VLM（147M）時 batch 8 可以訓練（2.3 s/step，峰值 15.0 GB），存出的 checkpoint 能重新載入並推理 | |

- Teacher 96.0% 與論文 Table 2（96）和 openpi 公佈值（96.85）一致，說明 PyTorch 轉換與評估流程是正確的。
- 總引數只減少 44%，延遲卻減少 60%：SigLIP（415M）與 token embedding（527M，只是查表）都沒有縮減。

### 3.2 Phase 2 計劃（概要）

| 項目 | 內容 |
|---|---|
| 比較的模型 | `teacher_l18`、`student_l06_init`（已完成）、`student_l06_kd`、`student_l06_taskonly`，以及後兩者在 1.5k step 的 checkpoint |
| 訓練 | 每個 run 3,000 steps × batch 64（約 0.7 epoch，1 epoch = 4,273 steps），bf16，peak LR 2.5e-5，warmup 300，decay 3,000；兩個 run 除 `loss_weight_teacher` 外設定完全相同（seed 與資料順序也相同） |
| 算力 | 1 張 80 GB GPU（優先 H100）。H100：KD 1.7–2.2 h + 對照 1.1–1.4 h + 環境建置約 1 h ≈ **4–5 GPU-hours**；A100 80GB 約 8–10 h。磁碟 ≥ 200 GB，主機記憶體 ≥ 64 GB，下載約 70 GB 公開資料 |
| 評估 | 回到筆電，每個模型 1,000 個 paired LIBERO episodes（25 trials/task），加上 bench 與 action agreement；建議加做 LIBERO-Plus 子集（約 700 episodes） |
| 主要問題 | H1：3k steps 的 KD student 能否恢復大部分 teacher 表現；H2：同樣算力下，KD student 的動作是否比 task-only 更接近 teacher（以動作 MSE 為主要指標，因為約 1 pt 的成功率差距檢測不出來） |
| 現狀 | 等待部門雲端算力 |

以下內容請看完整報告（`data/vla-survey-plan/report.md`）：VLA 壓縮方法調查（§2–3）、paired 設計與統計檢定力（power）表（§4）、產品化與認證現況（§5）、Phase 2 的完整假設、選用延伸、時程與風險（§7）。

---

## 附錄：資料來源與核對說明

**資料來源**

| 內容 | 位置 |
|---|---|
| Shallow-π 原理、公式、Table 1–3 | arXiv 2601.20262v1 HTML 全文（§3.1、§3.2、§4、§5.1–5.3、§6） |
| Phase 1 數字 | `results/phase1/SUMMARY.md`；`README.md` 實驗記錄（2026-09-27） |
| 訓練迴圈與 loss | `scripts/distill_pytorch.py`；`src/openpi/models_pytorch/pi0_pytorch.py:521-597` |
| 層對映 | `src/openpi/models_pytorch/layer_mapping.py`；`scripts/phase1/make_student_init.py` |
| 逐層 joint attention | `src/openpi/models_pytorch/gemma_pytorch.py`（`compute_layer_complete`，Q/K/V 在兩組 token 間拼接） |
| 配置 | `src/openpi/training/config.py:770-850`；LR/optimizer 預設值 `src/openpi/training/optimizer.py:16-74` |
| Smoke test 與凍結字首 | `scripts/phase1/distill_smoke.sh`；`results/phase1/train_smoke/distill_smoke.json` |
| 評估協定 | `scripts/phase1/eval_libero.sh`、`serve_policy_seeded.py`、`bench_model.py`、`action_agreement.py` |
| Phase 2 成本與計劃 | `README.md`（「Phase-2 resource estimate」）；完整報告 §7 |

**核對中發現、與完整報告或論文字身不一致之處**

1. **Table 1c 「All tokens」 並非全部為 0。** 完整報告 §7.3 寫道「把 attention loss 套到全部 token 會失敗（成功率 0.0%）」。論文 Table 1c 實際為 L9 0.0 / **L6 61.5** / L4 0.0。結論（不應對全部 token 做 attention 匹配）不變，但 L6 的數字應更正為 61.5。
2. **論文對初始化的描述前後不一致。** §5 寫均勻抽樣（TinyBERT 式），§5.2 寫「copying the bottom layers」。本儲存庫按 §5 實作；fork 原始程式碼的 bug 恰好讓 backbone 使用底部 6 層。論文實際採用哪一種無法從原文確定，這也會影響「修正 bug 1 是否更接近論文」的判斷（Phase 1 的初始 MSE 資料支援均勻抽樣）。
3. **L4 數字。** Table 1a（task+KD+attention）為 94.2，Table 1c（action tokens only，應為同一設定）為 93.9，論文中兩處不一致。

本檔案只做閱讀與整理：沒有執行 GPU 任務、訓練或評估。
