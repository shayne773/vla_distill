# OVX 雲端訓練操作手冊（第二份）：6 層 student 重新訓練（fp32 master weights）

版本：2026-11-06｜對應程式碼：`vla_distill-3d11c3e.tar.gz`

---

## 0. 這份文件是什麼

這份手冊在**已經執行過第一份手冊**（`runbook1/runbook.md`，即本資料夾的 `phase2-cloud-runbook.zh-TW.md`）的 OVX 上、沿用**同一個工作目錄**（預設 `/data/vla_work`），換上修正後的程式碼，重新訓練同樣的兩個 student。
這份文件可以單獨使用：照著步驟執行即可。它不安裝環境，需要的前提見本節最後的「需要的資源」。

**第一份手冊訓練出的 student 成功率低的主要原因（簡短版）：** openpi 的 PyTorch 訓練器在 `--pytorch-training-precision bfloat16` 下，把 master weights 和 AdamW 狀態也存成 bf16。
學習率（最高 2.5e-5）比大多數權重的 bf16 間距還小，所以每一步的更新都被四捨五入掉：**81.5% 的 transformer 權重在 3,000 步之後和初始值逐位元完全相同**。
也就是說，精度從 FP32 降到 BF16，讓大部分權重沒有被訓練到。這不是操作問題，是第一份手冊選用的訓練設定造成的。
新程式碼保留 bf16 運算（速度幾乎不變），但另外用一份 fp32 的 master weights 累積更新，每一步再寫回 bf16 權重。

要做的事：在一張 GPU 上依序跑兩次訓練（各 3,000 steps、batch 64，學習率和第一份手冊完全相同），再把結果帶出來。

| 順序 | 訓練 | 內容 | 預估時間（1 張 L40S-48Q） |
|---|---|---|---|
| 1 | `taskonly` | 6 層 student，只有 task loss，fp32 master weights | 約 4.0 小時 |
| 2 | `kd` | 同一個 student，task loss + teacher 蒸餾 loss，fp32 master weights | 約 6.5 小時 |

先跑 `taskonly`：它比較快，而且最直接驗證這個修正。時間以步驟 5 的實測為準。
若有兩張空的 GPU，兩個訓練可以同時跑（`P6P`），總時間約等於 `kd` 那一個。

### 0.1 和第一份手冊的差別

| 項目 | 第一份手冊（`vla_distill-966da5c.tar.gz`） | 本手冊（`vla_distill-3d11c3e.tar.gz`） |
|---|---|---|
| 程式碼 | `vla_distill-966da5c.tar.gz` | `vla_distill-3d11c3e.tar.gz`，直接解壓覆蓋 `$W/vla_distill`。只改 Python 檔，**不需要 `uv sync`**，Python 環境、資料集、teacher 都沿用 |
| 精度 | bf16 運算 + bf16 master weights + bf16 AdamW 狀態 | bf16 運算 + **fp32 master weights + fp32 AdamW 狀態** |
| batch、步數、學習率、seed | 64、3,000、warmup 300 → 2.5e-5 → cosine 到 2.5e-6、42 | 完全相同（這樣結果才能歸因於這個修正） |
| 凍結的參數（`FREEZE`） | 4 組（vision tower 等） | 再加 6 組從來收不到梯度的參數（見 4.1），可訓練參數 1,070.9M → 698.5M，學習結果不變 |
| 新增檢查 | 無 | 步驟 4：確認修正真的生效（不通過就不能開始訓練）；步驟 7：在 step 1500 再檢查一次 |
| GPU 記憶體 | peak reserved 約 38.9–39.4 GB | 預估約 44–45 GB（步驟 5 實測，上限 46 GB） |
| 工作目錄 | `$W/logs`、`$W/out` | 新的東西都放在 `$W/r2/`，帶出的檔案放 `$W/out2/`，不會和第一份手冊的產出混在一起 |

**需要的檔案：**

| 檔案 | 大小 | 用途 |
|---|---|---|
| `runbook2.md`（本文件） | 約 55 KB | 指令來源 |
| `vla_distill-3d11c3e.tar.gz` | 1.1 MB（1,141,294 bytes） | 程式碼壓縮檔（與本文件放在同一個資料夾）。sha256：`18b95943e25253b8b7c99b1b7360b561beea8404a1dc2a5f904965c0e52cc41a` |

**產出：**

1. 步驟 8 產生的 `$W/out2/` 整個資料夾（要帶出的結果）：4 個 student 權重（每個 4.11 GB，共約 16.5 GB）、log、`runs.txt`、`speed.txt`、`gate.txt`、`SHA256SUMS`。
2. 任何一步失敗時：`bash $W/r2/diag2.sh` 產生的診斷檔（幾十 KB），用來排查問題。

**需要的資源：** 已經執行過第一份手冊的機器與它的工作目錄 `/data/vla_work`（至少完成第一份手冊的步驟 0–3：環境檔、檢查腳本、Python 環境、資料集與轉好的 teacher）。磁碟需要約 70 GB 可用空間（兩張 GPU 同時跑需要約 90 GB），步驟 2 會清出空間並檢查。不需要 sudo。

---

## 1. 步驟總覽

| 步驟 | 代號 | 做什麼 | 時間 |
|---|---|---|---|
| 0 | `P0` | 唯讀檢查現有工作目錄（GPU、磁碟、第一份手冊留下的檔案） | 約 1 分鐘 |
| 1 | `P1` | 換上新程式碼、建立新的小腳本、用第一份手冊建立的 `check_env.sh`／`check_data.sh` 重新驗證環境 | 3–10 分鐘 |
| 2 | `P2`（必要時 `P2B`） | 清出磁碟空間 | 幾秒–2 分鐘 |
| 3 | `P3` | 建立 student 初始權重（給步驟 4、7 比對用） | 2–4 分鐘 |
| 4 | `P4` | **修正生效檢查**：20-step smoke run + 權重變化檢查 | 約 3–5 分鐘 |
| 5 | `P5`（不通過時 `P5F`） | batch 64 的 50-step 測速與記憶體檢查（`taskonly`、`kd` 各一次） | 約 20 分鐘 |
| 6 | `P6` 或 `P6P` | 兩個正式訓練（tmux，可續跑） | 約 10.5 小時（兩張 GPU：約 6.5 小時） |
| 7 | `P7` | 執行中檢查，並在 step 1500 自動再檢查一次修正是否生效 | 幾秒 |
| 8 | `P8` | 打包、checksum、帶出 | 2–5 分鐘 |

第一份手冊的 `P2A`／`P3A`（安裝環境、下載資料）這裡**不需要再跑**。只有在步驟 1 的 `check_env.sh` 或 `check_data.sh` 失敗時，才照附錄 R 修復。

---

## 2. 開始之前：定義 `step` 指令（每次登入做一次）

和第一份手冊一樣，每個指令區塊都有代號。請把這份 `runbook2.md` 放到雲端機器上（例如 `~/runbook2.md`），然後在終端機貼上：

```bash
export RB=~/runbook2.md
step(){ tr -d '\r' < "$RB" | sed -n "/^# >>> $1\$/,/^# <<< $1\$/p" > "${TMPDIR:-/tmp}/vla_$1.sh" && bash "${TMPDIR:-/tmp}/vla_$1.sh"; }
export W=/data/vla_work
echo "step helper ready (RB=$RB W=$W)"
```

注意 `RB` 指向 `runbook2.md`（不是第一份手冊的 `runbook.md`）。若存放的檔名或路徑不同，請把 `RB=` 後面改成實際的路徑，否則 `step` 會找不到任何區塊、什麼也不做。若終端機裡還有第一份手冊定義的 `step`，貼上這幾行就會覆蓋它。
`W` 是第一份手冊建立的工作目錄；若當時用的不是 `/data/vla_work`，請換成實際路徑。
請不要設定 `VLA_EXTRA`、`VLA_STEPS`、`DRY` 這三個環境變數，它們只供開發時在筆電上測試。
之後每一步只要輸入 `step P0`、`step P1` 這樣的一行。也可以直接複製 `# >>> 代號` 到 `# <<< 代號` 之間的內容貼到終端機，效果相同。

---

## 3. 步驟 0：檢查現有工作目錄（`P0`）

```bash
step P0
```

- 時間：約 1 分鐘（`du` 要掃過 Python 環境）。
- 只讀取資訊，唯一寫入的是 `$W/ovx_state.txt`。
- 成功：最後一行是 `state written to /data/vla_work/ovx_state.txt`。
- 請看輸出中的三件事：
  - `--- WORKSPACE` 下 `env.sh`、`check_env.sh`、`check_data.sh` 都是 `present`，`.venv` 也是 `present`。若有 `MISSING`，表示這個工作目錄沒有完成第一份手冊的步驟 0–3，先不要繼續（保留 `ovx_state.txt` 以便排查）。
  - `--- GPU`：要用的那張 GPU 的 `memory.used` 只有桌面程式的約 1,400 MiB。若有別人的訓練程式，請等它結束或換一張 GPU（之後用 `export GPU=<編號>` 指定）。
  - `--- PROCESSES`：沒有 `distill_pytorch.py` 在跑。
- `ovx_state.txt` 會在步驟 8 一起打包。

```bash
# >>> P0
: "${W:?run first: export W=/data/vla_work}"
OUT="$W/ovx_state.txt"
{
echo "=== ovx_state v2 $(date -u +%FT%TZ) host=$(hostname) W=$W"
echo "--- GPU"; nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv 2>&1
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv 2>&1 | head -20
echo "--- RAM"; free -g
echo "--- DISK"; df -hT "$W" /tmp /dev/shm 2>&1
echo "--- WORKSPACE"
for f in env.sh check_env.sh check_data.sh run_one.sh diag.sh vla_distill/.venv/bin/python vla_distill/checkpoints/pi05_libero_pytorch/model.safetensors; do
  if [ -e "$W/$f" ]; then echo "present: $f"; else echo "MISSING: $f"; fi
done
ls -la "$W" "$W/in" 2>&1
if grep -q "def fp32_master_weight_step" "$W/vla_distill/scripts/distill_pytorch.py" 2>/dev/null; then echo "code: already 3d11c3e or later"; else echo "code: runbook 1's code (966da5c), to be replaced in P1"; fi
echo "--- SIZES"; for d in "$W"/* "$W"/cache/* "$W"/cache/hf/* "$W"/vla_distill/.venv "$W"/vla_distill/checkpoints/* "$W"/vla_distill/checkpoints/*/*; do [ -e "$d" ] && du -shL "$d" 2>/dev/null; done
echo "--- LAST RUNS"; tail -n 8 "$W/logs/runs.txt" 2>/dev/null; cat "$W/logs/speed.txt" 2>/dev/null | head -3
echo "--- PROCESSES"; PR=$(pgrep -af "distill_pytorch|serve_policy|huggingface-cli" | cut -c1-200); echo "${PR:-none}"
(tmux ls 2>/dev/null || echo "no tmux sessions")
} 2>&1 | tee "$OUT"
echo "state written to $OUT"
# <<< P0
```

---

## 4. 步驟 1：換上新程式碼並重新驗證環境（`P1`）

先把 `vla_distill-3d11c3e.tar.gz`（與本文件放在同一個資料夾）放進 `$W/in/`，再執行：

```bash
step P1
```

- 時間：3–10 分鐘（大部分是 `check_data.sh` 載入資料集）。
- 這一步做的事：
  1. 驗證壓縮檔的 sha256，解壓**覆蓋** `$W/vla_distill`。`.venv/`、`checkpoints/`、資料集、快取都不會被動到。Python 環境是以 editable 方式安裝這個資料夾的，所以覆蓋之後新程式碼就生效，不需要 `uv sync`。
  2. 在 `$W/r2/` 建立 5 個新檔案：`env2.sh`（新的 `FREEZE` 清單）、`gate_check.py`（步驟 4、7 的檢查）、`run2.sh`（正式訓練）、`timed.sh`（步驟 5 測速）、`diag2.sh`（失敗時收集資訊）。第一份手冊的 `env.sh`、`run_one.sh` 等不會被修改。
  3. 確認 Python 載入的是新程式碼（`CODE_OK`）。
  4. 執行第一份手冊建立的 `check_env.sh`（`ENV_OK`）與 `check_data.sh`（`CHECK_DATA PASS`）。
- **成功：最後一行是 `P1 OK`**，上面有：
  - `CODE_OK fp32 master weights code is active`
  - `torch 2.7.1+cu128 cuda True`、`patched True`、`ENV_OK`
  - `TEACHER_OK`（或 `TEACHER_DIFFERENT_HASH`：第一份手冊在雲端自行轉換 teacher 時會出現，照樣可以繼續）、`parquet_files=1693 parquet_bytes=34938730091`、`DATA_OK episodes 1693 frames 273465 fps 10`、`CHECK_DATA PASS`
- 失敗時畫面會印出 `P1 FAIL: ...`：
  - `ENV_OK` 沒出現：照附錄 R 的 `R1` 修復 Python 環境，然後重新 `step P1`。
  - `CHECK_DATA FAIL`：照附錄 R 的 `R2` 修復資料，然後重新 `step P1`。
  - `CODE_FAIL` 或其他：`bash $W/r2/diag2.sh` 產生診斷檔以便排查，請不要繼續。

```bash
# >>> P1
: "${W:?run first: export W=/data/vla_work}"; source "$W/env.sh" && cd "$W" || exit 1
A="$W/in/vla_distill-3d11c3e.tar.gz"
echo "18b95943e25253b8b7c99b1b7360b561beea8404a1dc2a5f904965c0e52cc41a  $A" | sha256sum -c - || { echo "P1 FAIL: put vla_distill-3d11c3e.tar.gz into $W/in/"; exit 1; }
[ -x "$W/vla_distill/.venv/bin/python" ] || { echo "P1 FAIL: $W/vla_distill/.venv is missing (wrong W? see appendix R)"; exit 1; }
if pgrep -f "python[0-9.]* scripts/distill_pytorch.py" >/dev/null; then echo "P1 FAIL: a training process is still running"; exit 1; fi
tar --no-same-owner -xzf "$A" -C "$W" || { echo "P1 FAIL: untar"; exit 1; }
mkdir -p "$W/r2/logs" || exit 1

cat > "$W/r2/env2.sh" <<EOT
source "$W/env.sh"
EOT
cat >> "$W/r2/env2.sh" <<'EOT'
export R="$W/r2"
# Student parameters to freeze. The first four are runbook 1's list (vision tower, projector, token embedding,
# VLM lm_head). The other six never receive a gradient, because the action expert reads only each VLM layer's
# K/V: the expert's unused lm_head, the final VLM norm, and the last VLM layer's MLP, q/o projections and
# post-attention norm. "layers.5." is the LAST VLM layer of the 6-layer student only; another depth needs another index.
P=paligemma_with_expert; L=$P.paligemma.model.language_model
export FREEZE="$P.paligemma.model.vision_tower. $P.paligemma.model.multi_modal_projector. $L.embed_tokens. $P.paligemma.lm_head. $P.gemma_expert.lm_head. $L.norm. $L.layers.5.mlp. $L.layers.5.self_attn.q_proj. $L.layers.5.self_attn.o_proj. $L.layers.5.post_attention_layernorm."
unset P L
EOT

cat > "$W/r2/gate_check.py" <<'EOS'
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
EOS

cat > "$W/r2/run2.sh" <<'EOS'
#!/usr/bin/env bash
# usage: bash run2.sh taskonly|kd [gpu_index]
# VLA_RECIPE=mw (default: bf16 compute + fp32 master weights, batch 64, 3000 steps)
# VLA_RECIPE=fp32 (fallback, option 3: full fp32, batch 16, 12000 steps; only when step P5F said so)
source "$(dirname "$(readlink -f "$0")")/env2.sh"
case "${1:-}" in
  kd)       CFG=pi05_libero_l06 ;;
  taskonly) CFG=pi05_libero_l06_taskonly ;;
  *) echo "usage: bash run2.sh taskonly|kd [gpu_index]"; exit 2 ;;
esac
REC="${VLA_RECIPE:-mw}"
case "$REC" in
  mw)   STEPS="${VLA_STEPS:-3000}"; EXP="${1}_mw_b64_s3000"
        ARGS="--pytorch-training-precision bfloat16 --pytorch-fp32-master-weights --batch-size 64 --lr-schedule.warmup-steps 300 --lr-schedule.decay-steps 3000" ;;
  fp32) STEPS="${VLA_STEPS:-12000}"; EXP="${1}_fp32_b16_s12000"
        ARGS="--pytorch-training-precision float32 --batch-size 16 --lr-schedule.peak-lr 1.25e-5 --lr-schedule.warmup-steps 1200 --lr-schedule.decay-steps 12000 --lr-schedule.decay-lr 1.25e-6" ;;
  *) echo "VLA_RECIPE must be mw or fp32"; exit 2 ;;
esac
MID=$((STEPS / 2))
if [ -n "${2:-}" ]; then export CUDA_VISIBLE_DEVICES="$2"; fi
grep "p4_gate_smoke" "$R/gate.txt" 2>/dev/null | tail -n 1 | grep -q "^GATE PASS" || { echo "RUN $1 REFUSED: step P4 has not passed (no GATE PASS in $R/gate.txt)" | tee -a "$R/runs.txt"; exit 1; }
cd "$W/vla_distill" || exit 1
CK="checkpoints/$CFG/$EXP"
if [ -f "$CK/$STEPS/model.safetensors" ]; then echo "RUN $1 ALREADY DONE ($CK/$STEPS)" | tee -a "$R/runs.txt"; exit 0; fi
MODE=""; if ls -d "$CK"/[0-9]* >/dev/null 2>&1; then MODE="--resume"; fi
LOG="$R/logs/${1}_${REC}_$(date +%Y%m%d_%H%M%S).log"
echo "START $1 recipe=$REC mode=${MODE:-fresh} gpu=${CUDA_VISIBLE_DEVICES:-all} log=$LOG $(date -Is)" | tee -a "$R/runs.txt"
${DRY:+echo} .venv/bin/python scripts/distill_pytorch.py "$CFG" --exp-name "$EXP" $MODE $ARGS \
  --num-train-steps "$STEPS" --save-interval "$MID" --seed 42 --log-interval 10 --num-workers "${VLA_WORKERS:-4}" \
  --pytorch-freeze-prefixes $FREEZE ${VLA_EXTRA:-} >> "$LOG" 2>&1
RC=$?
if [ $RC -eq 0 ] && [ -f "$CK/$STEPS/model.safetensors" ]; then
  R0=PASS
  # The extra save one step before the end is never used, and the mid-run optimizer state is only needed to resume
  # before the final checkpoint exists. The final checkpoint keeps optimizer.pt + master_params.pt for --resume.
  rm -rf "${CK:?}/$((STEPS - 1))"; rm -f "$CK/$MID/optimizer.pt" "$CK/$MID/master_params.pt"
else R0="FAIL rc=$RC"; fi
echo "RUN $1 $R0 recipe=$REC $(date -Is) log=$LOG" | tee -a "$R/runs.txt"
[ "$R0" = PASS ]
EOS

cat > "$W/r2/timed.sh" <<'EOS'
#!/usr/bin/env bash
# usage: bash timed.sh mw|fp32  -> 50-step timing of taskonly and kd at the real batch size; appends to $R/speed.txt
source "$(dirname "$(readlink -f "$0")")/env2.sh"
REC="${1:-mw}"
case "$REC" in
  mw)   N=3000;  P_taskonly=4.7;  P_kd=7.7
        ARGS="--pytorch-training-precision bfloat16 --pytorch-fp32-master-weights --batch-size 64 --lr-schedule.warmup-steps 300 --lr-schedule.decay-steps 3000" ;;
  fp32) N=12000; P_taskonly=2.33; P_kd=3.74
        ARGS="--pytorch-training-precision float32 --batch-size 16 --lr-schedule.peak-lr 1.25e-5 --lr-schedule.warmup-steps 1200 --lr-schedule.decay-steps 12000 --lr-schedule.decay-lr 1.25e-6" ;;
  *) echo "usage: bash timed.sh mw|fp32"; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
cd "$W/vla_distill" || exit 1
V=PASS; EST=""
for RUN in taskonly kd; do
  if [ "$RUN" = kd ]; then CFG=pi05_libero_l06; PRED=$P_kd; else CFG=pi05_libero_l06_taskonly; PRED=$P_taskonly; fi
  LOG="$R/logs/timed50_${RUN}_${REC}.log"
  echo "timing $RUN ($REC), about $(awk -v p="$PRED" 'BEGIN { printf "%.0f", (50 * p + 150) / 60 }') min ..."
  .venv/bin/python scripts/distill_pytorch.py "$CFG" --exp-name "timed50_$REC" --overwrite $ARGS \
    --num-train-steps 50 --log-interval 10 --save-interval 100000 --num-workers "${VLA_WORKERS:-4}" \
    --pytorch-freeze-prefixes $FREEZE ${VLA_EXTRA:-} > "$LOG" 2>&1
  RC=$?
  rm -rf "checkpoints/$CFG/timed50_$REC"
  tr '\r' '\n' < "$LOG" | grep -E "Trainable params|fp32 master weights for|step=[0-9]+ loss|Saved checkpoint|Error|out of memory" | sed -E 's/^Training:.*\]([0-9]{2}:[0-9]{2}:[0-9]{2}\.)/\1/' | cut -c1-170 | tail -n 9
  L=$(tr '\r' '\n' < "$LOG" | awk -v rc="$RC" -v run="$RUN" -v rec="$REC" -v pred="$PRED" -v n="$N" '
    /after_backward/ { if (match($0, /peak_reserved: [0-9.]+/)) pr = substr($0, RSTART + 15, RLENGTH - 15) }
    /fp32 master weights for/ { mw = 1 }
    /step=[234]0 loss/ { if (match($0, /time=[0-9.]+/)) { t += substr($0, RSTART + 5, RLENGTH - 5); k += 10 } }
    /Saved checkpoint/ { saved = 1 }
    /out of memory/ { oom = 1 }
    END {
      if (rc != 0 || k == 0 || !saved) { printf "TIMED %s FAIL recipe=%s rc=%s steps_timed=%d saved=%d out_of_memory=%d\n", run, rec, rc, k, saved, oom; exit }
      s = t / k; v = "PASS"; why = ""
      if (rec == "mw" && !mw) { v = "FAIL"; why = why " no_master_weights_line" }
      if (pr + 0 > 46) { v = "FAIL"; why = why " peak_reserved>46GB" }
      if (s > pred * 1.5) { v = "FAIL"; why = why " >50%_slower" } else if (s > pred * 1.1 && v == "PASS") { v = "SLOW"; why = why " 10-50%_slower" }
      printf "TIMED %s %s recipe=%s s_per_step=%.2f predicted=%.2f ratio=%.2f peak_reserved_GB=%s master_weights_line=%s est_hours=%.1f%s\n", run, v, rec, s, pred, s / pred, pr, (mw ? "yes" : "no"), s * n / 3600, why
    }')
  echo "$L" | tee -a "$R/speed.txt"
  case "$L" in *" FAIL "*) V=FAIL ;; *" SLOW "*) [ "$V" = PASS ] && V=SLOW ;; esac
  EST="$EST est_hours_$RUN=$(echo "$L" | sed -n 's/.*est_hours=\([0-9.]*\).*/\1/p')"
done
echo "SPEED_GATE $V recipe=$REC$EST $(date -Is)" | tee -a "$R/speed.txt"
EOS

cat > "$W/r2/diag2.sh" <<'EOS'
#!/usr/bin/env bash
# writes one diagnostic text file when something fails
source "$(dirname "$(readlink -f "$0")")/env2.sh"
O="$R/diag_$(date +%Y%m%d_%H%M%S).txt"
{
echo "=== diag2 $(date -Is) host=$(hostname)"
nvidia-smi 2>&1 | head -30; free -g; df -hT "$W" /dev/shm 2>&1
echo "--- runs.txt"; tail -n 20 "$R/runs.txt" 2>/dev/null
echo "--- gate.txt"; tail -n 10 "$R/gate.txt" 2>/dev/null
echo "--- speed.txt"; tail -n 10 "$R/speed.txt" 2>/dev/null
for f in $(ls -t "$R"/logs/*.log 2>/dev/null | head -3); do
  echo "--- tail $f"; tr '\r' '\n' < "$f" | grep -v '^[[:space:]]*$' | tail -n 60 | cut -c1-400
done
echo "--- processes"; pgrep -af "distill_pytorch|uv sync|huggingface-cli" | cut -c1-200
echo "--- sizes"; du -sh "$W"/cache/* "$W"/vla_distill/checkpoints/*/* 2>/dev/null
} > "$O" 2>&1
echo "diagnostic file written: $O"
EOS

cd "$W/vla_distill" || exit 1
.venv/bin/python - <<'PY' || { echo "P1 FAIL: CODE_FAIL -> bash $W/r2/diag2.sh"; exit 1; }
import dataclasses, os
import openpi
from openpi.training import config as c
src = os.path.realpath(os.path.dirname(openpi.__file__))
want = os.path.realpath(os.path.join(os.environ["W"], "vla_distill", "src", "openpi"))
fields = {f.name for f in dataclasses.fields(c.TrainConfig)}
ok = src == want and "pytorch_fp32_master_weights" in fields
ok = ok and all(c.get_config(n).pytorch_fp32_master_weights for n in ("pi05_libero_l06", "pi05_libero_l06_taskonly"))
ok = ok and "def fp32_master_weight_step" in open("scripts/distill_pytorch.py").read()
print("openpi imported from", src)
print("CODE_OK fp32 master weights code is active" if ok else "CODE_FAIL")
raise SystemExit(0 if ok else 1)
PY
bash "$W/check_env.sh" || { echo "P1 FAIL: check_env.sh -> appendix R, block R1"; exit 1; }
bash "$W/check_data.sh" || { echo "P1 FAIL: check_data.sh -> appendix R, block R2"; exit 1; }
ls -la "$W"/r2/env2.sh "$W"/r2/gate_check.py "$W"/r2/run2.sh "$W"/r2/timed.sh "$W"/r2/diag2.sh && echo "P1 OK"
# <<< P1
```

---

## 5. 步驟 2：清出磁碟空間（`P2`，必要時 `P2B`）

新的 checkpoint 比第一份手冊的大：每個 checkpoint 資料夾約 12.5 GB（`model.safetensors` 4.11 GB + `optimizer.pt` 5.6 GB + `master_params.pt` 2.8 GB），訓練中最多同時有 3 個。
`/data` 總共約 147 GB，所以要先清掉第一份手冊留下、之後不會再用到的檔案。

```bash
step P2
```

- 時間：幾秒。
- `P2` 只刪除**確定不會再用到**的東西，並列出每一項的大小：
  - 第一份手冊的 smoke 與 50-step 測速 checkpoint（`smoke`、`timed50`）；
  - 第一份手冊兩個訓練的 `2999` 資料夾，以及它們的 `optimizer.pt`（只用於續跑第一份手冊的訓練，這裡不會用）；
  - Hugging Face 下載用的暫存 `$W/cache/hf/xet`；
  - 轉換 teacher 時下載的 JAX checkpoint（`.../pi05_libero/params`，teacher 已經轉好）。
- 第一份手冊的 4 個 student 權重（`model.safetensors`，若有）、資料集、Arrow 快取、teacher、Python 環境都**保留**。
- 成功：最後一行是 `DISK OK free=...GB need=70GB`。
- 若最後一行是 `DISK LOW ...`：請執行 `step P2B`。
- 若打算用兩張 GPU 同時跑（`P6P`），請改用 `VLA_NEED_GB=90 step P2`。

```bash
# >>> P2
: "${W:?run first: export W=/data/vla_work}"; source "$W/r2/env2.sh" && cd "$W/vla_distill" || exit 1
if pgrep -f "python[0-9.]* scripts/distill_pytorch.py" >/dev/null; then echo "P2 FAIL: a training process is running"; exit 1; fi
NEED="${VLA_NEED_GB:-70}"
free_gb(){ df -P -BG "$W" | awk 'NR == 2 { sub("G", "", $4); print $4 }'; }
echo "free before: $(free_gb) GB"
for p in checkpoints/pi05_libero_l06/smoke checkpoints/pi05_libero_l06/timed50 \
         checkpoints/pi05_libero_l06/kd_b64_s3000/2999 checkpoints/pi05_libero_l06_taskonly/taskonly_b64_s3000/2999 \
         checkpoints/pi05_libero_l06/kd_b64_s3000/*/optimizer.pt checkpoints/pi05_libero_l06_taskonly/taskonly_b64_s3000/*/optimizer.pt \
         "$W/cache/hf/xet" "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero/params"; do
  if [ -e "$p" ]; then echo "delete $(du -sh "$p" | cut -f1)  $p"; rm -rf "$p"; fi
done
F=$(free_gb)
if [ "$F" -ge "$NEED" ]; then echo "DISK OK free=${F}GB need=${NEED}GB"; else echo "DISK LOW free=${F}GB need=${NEED}GB -> step P2B"; fi
# <<< P2
```

### 5.1 `P2B`：空間仍不夠時

`P2B` 會刪除第一份手冊的 4 個 student 權重，以及第一份手冊 `P8` 打包出的 `$W/out/`。
區塊內列出的 4 個 sha256 是第一份手冊的訓練在 OVX 上產出、已經帶出保存的那 4 個權重；`P2B` 只刪除和它們完全相同的檔案，不相同的會保留（此時 `$W/out/` 也會保留）。
它也會刪除 `uv` 的下載快取 `$W/cache/uv`（只有附錄 R 重裝環境時才會用到，屆時會重新下載）。
**若還需要第一份手冊的權重（例如要做評估），請先不要執行 `P2B`，改用其他方式清出磁碟空間。**

```bash
step P2B
```

- 時間：1–2 分鐘（計算 16 GB 的 sha256）。
- 成功：最後一行是 `DISK OK ...`。仍然 `DISK LOW`：磁碟空間不足以訓練，請停下來，保留 `ovx_state.txt` 和畫面輸出以便排查。

```bash
# >>> P2B
: "${W:?run first: export W=/data/vla_work}"; source "$W/r2/env2.sh" && cd "$W/vla_distill" || exit 1
if pgrep -f "python[0-9.]* scripts/distill_pytorch.py" >/dev/null; then echo "P2B FAIL: a training process is running"; exit 1; fi
NEED="${VLA_NEED_GB:-70}"
free_gb(){ df -P -BG "$W" | awk 'NR == 2 { sub("G", "", $4); print $4 }'; }
OLD=0
while read -r h f; do
  if [ ! -f "$f" ]; then continue; fi
  if echo "$h  $f" | sha256sum -c - >/dev/null 2>&1; then echo "delete (identical to the exported copy) $f"; rm -f "$f"
  else echo "KEEP (differs from the exported copy) $f"; OLD=1; fi
done <<'EOT'
8a419dde174a8b08fb2da671b30a9e429a9dc1ff32d51ce517ba59c3169d0236 checkpoints/pi05_libero_l06/kd_b64_s3000/1500/model.safetensors
c3044759f864b14b8dd21cacf921aaf85216f421c8e7999ce9cec030b1885344 checkpoints/pi05_libero_l06/kd_b64_s3000/3000/model.safetensors
1812765e20c886ab6272fbf5e67e903e7de7848eaae0dae8d720558b6dd8043e checkpoints/pi05_libero_l06_taskonly/taskonly_b64_s3000/1500/model.safetensors
00d6679b499731f98d6196a95212fc6443165c3d95a59166f77b5d4178de9aec checkpoints/pi05_libero_l06_taskonly/taskonly_b64_s3000/3000/model.safetensors
EOT
if [ $OLD -eq 0 ]; then rm -rf "$W/out"; echo "deleted $W/out (runbook 1's export)"; else echo "kept $W/out because a file differed"; fi
if [ -d "$W/cache/uv" ]; then echo "delete $(du -sh "$W/cache/uv" | cut -f1)  $W/cache/uv"; rm -rf "$W/cache/uv"; fi
F=$(free_gb)
if [ "$F" -ge "$NEED" ]; then echo "DISK OK free=${F}GB need=${NEED}GB"; else echo "DISK LOW free=${F}GB need=${NEED}GB -> not enough disk to train; stop and keep ovx_state.txt and this output"; fi
# <<< P2B
```

---

## 6. 步驟 3：建立 student 初始權重（`P3`）

訓練程式會在開始時，從 teacher 的 18 層中挑 6 層（第 0、3、7、10、14、17 層）組成 student。
步驟 4 和 7 要比較「訓練後」和「訓練前」的權重，所以先把「訓練前」的 student 存成一個檔案（4.02 GB）。這只用 CPU。

```bash
step P3
```

- 時間：2–4 分鐘。
- 成功：最後一行是 `INIT_OK identical to the laptop init`，上面有 `layer mapping (student <- teacher): [0, 3, 7, 10, 14, 17]`。
- 若最後一行是 `INIT_OK_DIFFERENT_HASH ...`：不算失敗（表示這台機器上的 teacher 檔和筆電上的不是逐位元相同；步驟 1 的 `check_data.sh` 若印出 `TEACHER_DIFFERENT_HASH` 就會這樣），請記下那一行，然後繼續。
- 失敗（`P3 FAIL`）：`bash $W/r2/diag2.sh` 產生診斷檔以便排查。

```bash
# >>> P3
: "${W:?run first: export W=/data/vla_work}"; source "$W/r2/env2.sh" && cd "$W/vla_distill" || exit 1
I=checkpoints/pi05_libero_l06_init
if [ ! -f "$I/model.safetensors" ]; then
  .venv/bin/python scripts/phase1/make_student_init.py --teacher-dir checkpoints/pi05_libero_pytorch \
    --student-config pi05_libero_l06 --output-dir "$I" > "$R/logs/make_init.log" 2>&1 || { tail -n 25 "$R/logs/make_init.log"; echo "P3 FAIL -> bash $W/r2/diag2.sh"; exit 1; }
fi
grep -E "layer mapping|load_state_dict" "$R/logs/make_init.log" | cut -c1-200
H=$(sha256sum "$I/model.safetensors" | cut -d' ' -f1)
if [ "$H" = bdd4540dd2f64e7a90e5171f8a02a3c3fd5ed50c342a1ea245f35966f49ea984 ]; then echo "INIT_OK identical to the laptop init"
else echo "INIT_OK_DIFFERENT_HASH (note this line): $H"; fi
# <<< P3
```

---

## 7. 步驟 4：確認修正真的生效（`P4`）

**這一步沒有通過，就不能開始正式訓練**（`run2.sh` 會拒絕執行）。

做法：用 8 個 episodes、batch 2、固定學習率 2.5e-5 跑 20 個 step，存一個 checkpoint，然後用 `gate_check.py` 和步驟 3 的初始權重比較。
檢查的是 action expert 各層的權重矩陣中，數值 |w| ≥ 2⁻⁶ 的那些元素有多少比例改變了：

- 第一份手冊的 bf16 訓練方式：**0.0000**（一個都沒變；第一份手冊在 OVX 上訓練 3,000 步之後的 checkpoint 也是 0.0000）；
- 有 fp32 master weights：筆電上實測 **0.5520**；
- 門檻是 0.25。

兩個條件都要成立才算通過：

1. 訓練 log 裡有 `fp32 master weights for 698.5M trainable params` 這一行；
2. `gate_check.py` 印出 `-> PASS`。

```bash
step P4
```

- 時間：約 3–5 分鐘（含載入 teacher 與 student；結束時自動刪除這個約 25 GB 的暫存 checkpoint）。
- **通過條件：最後一行以 `GATE PASS p4_gate_smoke` 開頭**，例如
  `GATE PASS p4_gate_smoke step20 | fp32 master weights for 698.5M trainable params | changed fraction ...: 0.55.. (...) -> PASS | ...`
  這一行也會存到 `$W/r2/gate.txt`。
- 參考值：`Froze 1313.8M params`、`Trainable params: 698.5M`、`fp32 master weights for 698.5M trainable params`；changed fraction 約 0.5–0.6。
- 失敗（`GATE FAIL`）：**請不要繼續**，執行 `bash $W/r2/diag2.sh` 產生診斷檔以便排查。

```bash
# >>> P4
: "${W:?run first: export W=/data/vla_work}"; source "$W/r2/env2.sh" && cd "$W/vla_distill" || exit 1
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
I=checkpoints/pi05_libero_l06_init/model.safetensors
[ -f "$I" ] || { echo "P4 FAIL: run step P3 first"; exit 1; }
CK=checkpoints/pi05_libero_l06_taskonly/gate_smoke
.venv/bin/python scripts/distill_pytorch.py pi05_libero_l06_taskonly --exp-name gate_smoke --overwrite \
  --batch-size 2 --num-train-steps 20 --save-interval 20 --log-interval 1 --num-workers 2 \
  --pytorch-training-precision bfloat16 --pytorch-fp32-master-weights --seed 42 \
  --lr-schedule.warmup-steps 1 --lr-schedule.decay-steps 100000 --data.lerobot-episodes 0 1 2 3 4 5 6 7 \
  --pytorch-freeze-prefixes $FREEZE ${VLA_EXTRA:-} > "$R/logs/gate_smoke.log" 2>&1
RC=$?
tr '\r' '\n' < "$R/logs/gate_smoke.log" | grep -E "Froze|Trainable params|fp32 master weights for|step=(0|1|19) loss|Saved checkpoint|Error|out of memory" | sed -E 's/^Training:.*\]([0-9]{2}:[0-9]{2}:[0-9]{2}\.)/\1/' | cut -c1-170
MW=$(tr '\r' '\n' < "$R/logs/gate_smoke.log" | grep -o "fp32 master weights for [0-9.]*M trainable params" | head -n 1)
if [ $RC -eq 0 ] && [ -f "$CK/20/model.safetensors" ]; then G=$(.venv/bin/python -I "$R/gate_check.py" "$I" "$CK/20/model.safetensors" 2>&1 | tail -n 1)
else G="smoke run failed rc=$RC"; fi
if [ -n "$MW" ] && echo "$G" | grep -q -- "-> PASS$"; then V="GATE PASS"; else V="GATE FAIL"; fi
rm -rf "$CK"
echo "$V p4_gate_smoke step20 | ${MW:-NO fp32 master weights LINE} | $G | $(date -Is)" | tee -a "$R/gate.txt"
# <<< P4
```

---

## 8. 步驟 5：50-step 測速與記憶體檢查（`P5`）

用正式設定（完整資料集、batch 64、bf16 運算 + fp32 master weights、新的 `FREEZE`）先跑 `taskonly` 50 個 step，再跑 `kd` 50 個 step。
目的：確認 batch 64 加上 master weights 放得進這張 GPU，並量出實際的每步秒數。第一份手冊 `P5` 的 memory bandwidth 量測這裡不需要。

```bash
step P5
```

- 時間：約 20 分鐘（`taskonly` 約 7 分鐘、`kd` 約 10 分鐘，加上載入；每次結束時自動刪除約 12.5 GB 的暫存 checkpoint）。
- 會印出兩行 `TIMED ...`，最後一行是 `SPEED_GATE ...`，全部也會存到 `$W/r2/speed.txt`。
- 每一個訓練的判定：

| 判定 | 條件 |
|---|---|
| `PASS` | log 有 `fp32 master weights for` 那一行、`peak_reserved_GB` ≤ 46、`s_per_step` 不超過預估值的 1.1 倍（`taskonly` 4.7、`kd` 7.7，即 ≤ 5.17 與 ≤ 8.47） |
| `SLOW` | 同上，但 `s_per_step` 是預估值的 1.1–1.5 倍 |
| `FAIL` | 程式出錯、`out of memory`、`peak_reserved_GB` > 46、沒有 `fp32 master weights` 那一行，或比預估慢 1.5 倍以上 |

- 預期：`taskonly` 約 4.7–4.8 s/step、`kd` 約 7.7 s/step；`peak_reserved_GB` 兩者約 44–45。
- **通過條件：最後一行是 `SPEED_GATE PASS recipe=mw ...`**，例如
  `SPEED_GATE PASS recipe=mw est_hours_taskonly=4.0 est_hours_kd=6.4 ...`

### 8.1 結果不是 `PASS` 時怎麼做（請照這個順序處理）

1. **`SPEED_GATE SLOW`**：先用 `nvidia-smi` 確認這張 GPU 上沒有別人的程式。可以直接繼續步驟 6（`SLOW` 會記錄在 `speed.txt`，在步驟 8 一起打包）。
2. **任一行是 `FAIL`，而且原因是 `no_master_weights_line`**：程式碼沒有生效。請不要繼續，執行 `bash $W/r2/diag2.sh` 產生診斷檔以便排查。
3. **`FAIL` 的原因是 `>50%_slower`**：用 `nvidia-smi` 確認 GPU 沒有被別人佔用，再執行一次 `step P5`。仍然 `FAIL`：停在這裡，不要開始訓練；執行 `bash $W/r2/diag2.sh`，用 `$W/r2/speed.txt` 與 `diag2` 檔排查。
4. **`FAIL` 的原因是 `peak_reserved>46GB` 或 `out_of_memory=1`**（記憶體不夠）：
   1. 用 `nvidia-smi` 確認除了桌面程式（約 1,400 MiB）之外，這張 GPU 是空的。若有別的程式，等它結束後重新 `step P5`。
   2. GPU 是空的仍然失敗：改用備案（選項 3，全 fp32、batch 16、12,000 steps），先執行 `step P5F` 測速。**請不要自行改 batch size 或其他設定**，備案的設定都已經寫好。
   3. `P5F` 的最後一行是 `SPEED_GATE PASS recipe=fp32` 或 `SPEED_GATE SLOW recipe=fp32`：在步驟 6 之前先 `export VLA_RECIPE=fp32`，然後照常執行步驟 6–8（`runs.txt` 的 `recipe=fp32` 與輸出資料夾名稱會記錄用了備案）。備案在一張 GPU 上約需 20 小時（`taskonly` 約 7.8、`kd` 約 12.5 小時）。
   4. `P5F` 也 `FAIL`：停在這裡，不要開始訓練；執行 `bash $W/r2/diag2.sh`，用 `$W/r2/speed.txt` 與 `diag2` 檔排查。

```bash
# >>> P5
: "${W:?run first: export W=/data/vla_work}"; source "$W/r2/env2.sh" && cd "$W/vla_distill" || exit 1
if pgrep -f "python[0-9.]* scripts/distill_pytorch.py" >/dev/null; then echo "P5 FAIL: a training process is running"; exit 1; fi
grep "p4_gate_smoke" "$R/gate.txt" 2>/dev/null | tail -n 1 | grep -q "^GATE PASS" || { echo "P5 FAIL: step P4 has not passed"; exit 1; }
bash "$R/timed.sh" mw
# <<< P5
```

### 8.2 `P5F`：備案（選項 3）的測速

只有在 8.1 的第 4 點要求時才執行。設定：`--pytorch-training-precision float32 --batch-size 16 --num-train-steps 12000 --save-interval 6000 --lr-schedule.peak-lr 1.25e-5 --lr-schedule.warmup-steps 1200 --lr-schedule.decay-steps 12000 --lr-schedule.decay-lr 1.25e-6`，以及同樣的 `FREEZE`。
樣本數和主方案相同（12,000 × 16 = 3,000 × 64），學習率依 batch 變小 4 倍而除以 2，warmup 看過的樣本數也相同。
全 fp32 沒有 bf16 捨入的問題，所以不需要 master weights，log 也不會有那一行。

```bash
step P5F
```

- 時間：約 15 分鐘。預估 `taskonly` 2.33 s/step、`kd` 3.74 s/step，`peak_reserved_GB` 約 42。
- 判定方式同 `P5`，最後一行是 `SPEED_GATE ... recipe=fp32 ...`。

```bash
# >>> P5F
: "${W:?run first: export W=/data/vla_work}"; source "$W/r2/env2.sh" && cd "$W/vla_distill" || exit 1
if pgrep -f "python[0-9.]* scripts/distill_pytorch.py" >/dev/null; then echo "P5F FAIL: a training process is running"; exit 1; fi
grep "p4_gate_smoke" "$R/gate.txt" 2>/dev/null | tail -n 1 | grep -q "^GATE PASS" || { echo "P5F FAIL: step P4 has not passed"; exit 1; }
bash "$R/timed.sh" fp32
# <<< P5F
```

---

## 9. 步驟 6：正式訓練

兩個訓練都由 `$W/r2/run2.sh` 啟動，在 tmux（或 nohup）裡執行，終端機斷線不影響訓練。
和第一份手冊的 `run_one.sh` 一樣會自動判斷狀態：已完成就跳過；有 checkpoint 就從最新的 checkpoint 續跑（`--resume`，fp32 master weights 也會一起還原）；否則從頭開始。所以**任何中斷之後，重新執行同一個 `step` 指令即可**。

| 訓練 | config | 存放位置（`$W/vla_distill/checkpoints/` 之下） |
|---|---|---|
| `taskonly` | `pi05_libero_l06_taskonly` | `pi05_libero_l06_taskonly/taskonly_mw_b64_s3000/` |
| `kd` | `pi05_libero_l06` | `pi05_libero_l06/kd_mw_b64_s3000/` |

兩者的設定：`--pytorch-training-precision bfloat16 --pytorch-fp32-master-weights --batch-size 64 --num-train-steps 3000 --save-interval 1500 --lr-schedule.warmup-steps 300 --lr-schedule.decay-steps 3000 --seed 42`，其餘（peak 2.5e-5、最後 2.5e-6）是 config 預設值，和第一份手冊相同。
seed 42 和第一份手冊相同，所以資料順序也相同，兩份手冊的結果可以逐一配對比較。

每個訓練會存 `1500`、`2999`、`3000` 三個 checkpoint。訓練成功結束後，`run2.sh` 會自動刪除用不到的 `2999`，以及 `1500` 裡的 `optimizer.pt`、`master_params.pt`；`3000` 完整保留，以便之後延長訓練時續跑。

`P6` 和 `P6P` 會先確認步驟 4 是 `GATE PASS`、步驟 5（或 `P5F`）是 `PASS` 或 `SLOW`，否則拒絕啟動。

### 9.1 一張 GPU：依序跑（`P6`）

```bash
export GPU=0          # 換成要用的 GPU 編號
step P6
```

### 9.2 兩張空的 GPU：同時跑（`P6P`）

需要約 90 GB 磁碟空間（`VLA_NEED_GB=90 step P2` 確認過）。

```bash
export GPU=0 GPU2=1   # 兩張空 GPU 的編號；taskonly 用 GPU，kd 用 GPU2
step P6P
```

- 時間：依 `speed.txt` 的 `est_hours_taskonly` 與 `est_hours_kd`，預估約 4.0 與 6.4 小時，加上每次存 checkpoint 約 1 分鐘。
- 啟動成功：畫面印出 `started in tmux session ...` 或 `started with nohup ...`。約 3 分鐘後用 `step P7` 應能看到 `step=0 loss=...`。
- **完成條件：`$W/r2/runs.txt` 裡有 `RUN taskonly PASS` 和 `RUN kd PASS` 兩行。**
- 若某個訓練曾經中斷並續跑，`runs.txt` 會多一行 `START ... mode=--resume`。這不算失敗，但分析結果時要考慮這件事（續跑後的資料順序和不中斷時不同）；`runs.txt` 會在步驟 8 一起打包。
- 失敗（`RUN ... FAIL` 或 `REFUSED`）：`bash $W/r2/diag2.sh` 產生診斷檔以便排查。

```bash
# >>> P6
: "${W:?run first: export W=/data/vla_work}"; R="$W/r2"; REC="${VLA_RECIPE:-mw}"
if pgrep -f "python[0-9.]* scripts/distill_pytorch.py" >/dev/null || pgrep -f "^bash [^ ]*/run2\.sh (taskonly|kd)" >/dev/null || tmux ls -F "#S" 2>/dev/null | grep -qE "^vla2(_taskonly|_kd)?$"; then echo "training is already running (or its tmux session still exists); use: step P7"; exit 0; fi
grep "p4_gate_smoke" "$R/gate.txt" 2>/dev/null | tail -n 1 | grep -q "^GATE PASS" || { echo "P6 REFUSED: step P4 has not passed"; exit 1; }
grep "^SPEED_GATE .* recipe=$REC " "$R/speed.txt" 2>/dev/null | tail -n 1 | grep -q -E "^SPEED_GATE (PASS|SLOW) " || { echo "P6 REFUSED: no SPEED_GATE PASS/SLOW for recipe=$REC (step P5, or P5F for the fallback)"; exit 1; }
CMD="VLA_RECIPE=$REC bash $R/run2.sh taskonly ${GPU:-}; VLA_RECIPE=$REC bash $R/run2.sh kd ${GPU:-}"
if command -v tmux >/dev/null 2>&1; then tmux new-session -d -s vla2 "$CMD" && echo "started in tmux session 'vla2' (watch: tmux attach -t vla2 ; detach: Ctrl-b d)"
else nohup bash -c "$CMD" > "$R/logs/nohup.out" 2>&1 & echo "started with nohup pid=$!"; fi
# <<< P6
```

```bash
# >>> P6P
: "${W:?run first: export W=/data/vla_work}"; : "${GPU:?export GPU=<index>}"; : "${GPU2:?export GPU2=<index>}"; R="$W/r2"; REC="${VLA_RECIPE:-mw}"
if pgrep -f "python[0-9.]* scripts/distill_pytorch.py" >/dev/null || pgrep -f "^bash [^ ]*/run2\.sh (taskonly|kd)" >/dev/null || tmux ls -F "#S" 2>/dev/null | grep -qE "^vla2(_taskonly|_kd)?$"; then echo "training is already running (or its tmux session still exists); use: step P7"; exit 0; fi
grep "p4_gate_smoke" "$R/gate.txt" 2>/dev/null | tail -n 1 | grep -q "^GATE PASS" || { echo "P6P REFUSED: step P4 has not passed"; exit 1; }
grep "^SPEED_GATE .* recipe=$REC " "$R/speed.txt" 2>/dev/null | tail -n 1 | grep -q -E "^SPEED_GATE (PASS|SLOW) " || { echo "P6P REFUSED: no SPEED_GATE PASS/SLOW for recipe=$REC (step P5, or P5F for the fallback)"; exit 1; }
if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s vla2_taskonly "VLA_RECIPE=$REC bash $R/run2.sh taskonly $GPU" && tmux new-session -d -s vla2_kd "VLA_RECIPE=$REC bash $R/run2.sh kd $GPU2" && echo "started in tmux sessions 'vla2_taskonly' and 'vla2_kd'"
else
  VLA_RECIPE=$REC nohup bash "$R/run2.sh" taskonly "$GPU" > "$R/logs/nohup_taskonly.out" 2>&1 &
  VLA_RECIPE=$REC nohup bash "$R/run2.sh" kd "$GPU2" > "$R/logs/nohup_kd.out" 2>&1 &
  echo "started two runs with nohup"
fi
# <<< P6P
```

---

## 10. 步驟 7：執行中檢查（`P7`）

隨時可以執行，不影響訓練。建議啟動後 5 分鐘看一次，之後每 1–2 小時看一次，**並且在每個訓練過了 step 1500 之後至少看一次**。

```bash
step P7
```

正常的樣子：

- `runs.txt` 有 `START taskonly recipe=mw mode=fresh ...`。
- 每個 log 下面有一行 `fp32 master weights for 698.5M trainable params`（備案 `fp32` 沒有這一行，屬正常）。
- log 每 10 個 step 一行 `step=N loss=... lr=... grad_norm=... time=...s`，`time` 是最近 10 個 step 的秒數，應該穩定（`taskonly` 約 47 秒、`kd` 約 77 秒）。
- `ETA` 行顯示預估剩餘小時數。
- `nvidia-smi` 的 GPU 使用率大多在 90% 以上，記憶體約 45–48 GB（含桌面程式）。
- 到 step 1500 和 3000 時出現 `Saved checkpoint at step ...`，存檔期間訓練會暫停約 1–2 分鐘。
- 每個訓練的 `1500` checkpoint 出現後，`P7` 會自動跑一次 `gate_check.py`（約 30 秒，只做一次，結果存到 `gate.txt`），應該印出 `GATE PASS p7 ...`。參考值：筆電上用同樣的學習率排程（warmup 300、decay 3000）訓練到 step 1500，changed fraction 是 0.8972；第一份手冊在 OVX 上訓練的兩個 step 1500 checkpoint 都是 0.0000。

不正常、需要處理的樣子：

| 現象 | 處理 |
|---|---|
| `GATE FAIL p7 ...`，或 log 沒有 `fp32 master weights` 那一行（主方案） | 修正沒有生效，繼續跑只是浪費時間。請停止訓練：`tmux kill-session -t vla2`（兩張 GPU 時是 `tmux kill-session -t vla2_taskonly; tmux kill-session -t vla2_kd`；用 nohup 啟動時是 `pkill -f run2.sh; pkill -f scripts/distill_pytorch.py`），然後 `bash $W/r2/diag2.sh` 產生診斷檔 |
| `loss=nan` 或 `inf` | `bash $W/r2/diag2.sh` 產生診斷檔，不需要停止 |
| `time=` 突然變成原來的 2 倍以上並持續 | 看 `nvidia-smi` 是否有別人的程式佔用同一張 GPU；`bash $W/r2/diag2.sh` 產生診斷檔 |
| 10 分鐘以上沒有新的 `step=` 行，GPU 使用率 0% | `bash $W/r2/diag2.sh` 產生診斷檔 |
| `out of memory`、`Bus error`、`No space left on device` | `bash $W/r2/diag2.sh` 產生診斷檔。`No space left` 可以先執行 `step P2B` 再 `step P6` 續跑 |
| 機器重開機或程式消失 | 重新定義 `step`（第 2 節），再執行 `step P6`（或 `P6P`；用了備案的話先 `export VLA_RECIPE=fp32`），它會自動續跑 |

```bash
# >>> P7
: "${W:?run first: export W=/data/vla_work}"; source "$W/r2/env2.sh" && cd "$W/vla_distill" || exit 1
echo "--- runs.txt"; tail -n 6 "$R/runs.txt" 2>/dev/null || echo "(no runs yet)"
for f in $(ls -t "$R"/logs/taskonly_*.log "$R"/logs/kd_*.log 2>/dev/null | head -n 2); do
  echo "--- $f"
  M=$(tr '\r' '\n' < "$f" | grep -o "fp32 master weights for [0-9.]*M trainable params" | head -n 1)
  echo "${M:-(no 'fp32 master weights' line: expected only for VLA_RECIPE=fp32)}"
  tr '\r' '\n' < "$f" | grep -E "step=[0-9]+ loss|Saved checkpoint|Resumed|Error|out of memory" | sed -E 's/^Training:.*\]([0-9]{2}:[0-9]{2}:[0-9]{2}\.)/\1/' | tail -n 4 | cut -c1-150
  tr '\r' '\n' < "$f" | awk '/num_train_steps=/ { if (match($0, /num_train_steps=[0-9]+/)) n = substr($0, RSTART + 16, RLENGTH - 16) }
    /step=[0-9]+ loss/ { if (match($0, /step=[0-9]+ loss/)) st = substr($0, RSTART + 5, RLENGTH - 10); if (match($0, /time=[0-9.]+/)) tm = substr($0, RSTART + 5, RLENGTH - 5) }
    END { if (st > 0 && n > 0) printf "ETA %.1f h (step %d of %d, %.2f s/step)\n", (n - st) * tm / 10 / 3600, st, n, tm / 10 }'
done
I=checkpoints/pi05_libero_l06_init/model.safetensors
for c in checkpoints/pi05_libero_l06_taskonly/taskonly_mw_b64_s3000/1500 checkpoints/pi05_libero_l06/kd_mw_b64_s3000/1500; do
  [ -f "$c/model.safetensors" ] || continue
  T="p7 $(basename "$(dirname "$c")")/1500"
  if grep -q "$T " "$R/gate.txt" 2>/dev/null; then grep "$T " "$R/gate.txt" | tail -n 1; continue; fi
  G=$(.venv/bin/python -I "$R/gate_check.py" "$I" "$c/model.safetensors" 2>&1 | tail -n 1)
  if echo "$G" | grep -q -- "-> PASS$"; then V="GATE PASS"; else V="GATE FAIL"; fi
  echo "$V $T | $G | $(date -Is)" | tee -a "$R/gate.txt"
done
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv 2>&1
df -h "$W" | tail -n 1
ls -d "$W"/vla_distill/checkpoints/pi05_libero_l06*/*_mw_b64_s3000/[0-9]* "$W"/vla_distill/checkpoints/pi05_libero_l06*/*_fp32_b16_s12000/[0-9]* 2>/dev/null
# <<< P7
```

---

## 11. 步驟 8：打包帶出（`P8`）

### 11.1 打包

兩個訓練都顯示 `PASS` 之後執行：

```bash
step P8
```

- 時間：2–5 分鐘（計算約 16.5 GB 的 sha256）。
- 成功：最後一行是 `PACKAGE READY: .../out2 (N files)`，上面列出每個檔案的大小。
- 產出在 `$W/out2/`：

| 內容 | 大小 |
|---|---|
| `taskonly_mw_b64_s3000_step1500/`、`taskonly_mw_b64_s3000_step3000/` 的 `model.safetensors` | 各 4.11 GB |
| `kd_mw_b64_s3000_step1500/`、`kd_mw_b64_s3000_step3000/` 的 `model.safetensors` | 各 4.11 GB |
| 每個資料夾內的 `assets/`（normalization 統計） | 各約 2 KB |
| `*.log`、`runs.txt`、`speed.txt`、`gate.txt`、`ovx_state.txt` | 共幾 MB |
| `SHA256SUMS` | 幾 KB |
| 合計 | 約 16.5 GB |

`optimizer.pt`、`master_params.pt` 只用於續跑，**不需要帶出，請留在 OVX 上**（之後若要延長訓練會用到）。
若用了備案（`fp32`），資料夾名稱是 `*_fp32_b16_s12000_step6000`／`_step12000`，每個 `model.safetensors` 是 fp32，約 8.1 GB，合計約 32 GB。

```bash
# >>> P8
: "${W:?run first: export W=/data/vla_work}"; source "$W/r2/env2.sh" && cd "$W/vla_distill" || exit 1; O="$W/out2"; mkdir -p "$O"
for r in pi05_libero_l06_taskonly/taskonly_mw_b64_s3000 pi05_libero_l06/kd_mw_b64_s3000 pi05_libero_l06_taskonly/taskonly_fp32_b16_s12000 pi05_libero_l06/kd_fp32_b16_s12000; do
  [ -d "checkpoints/$r" ] || continue
  case "$r" in *_s12000) S="6000 12000" ;; *) S="1500 3000" ;; esac
  for s in $S; do
    src="checkpoints/$r/$s"; dst="$O/$(basename "$r")_step$s"
    if [ ! -f "$src/model.safetensors" ]; then echo "MISSING $src"; continue; fi
    mkdir -p "$dst" && { ln -f "$src/model.safetensors" "$dst/" 2>/dev/null || cp "$src/model.safetensors" "$dst/"; } && cp -r "$src/assets" "$dst/"
  done
done
cp "$R"/logs/*.log "$R"/*.txt "$W"/ovx_state.txt "$O"/ 2>/dev/null
cd "$O" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
find . -type f -printf '%s\t%p\n' | sort -k2 | awk -F'\t' '{ printf "%10.1f MB  %s\n", $1 / 1e6, $2; t += $1 } END { printf "%10.1f MB  TOTAL\n", t / 1e6 }'
echo "PACKAGE READY: $O ($(grep -c . SHA256SUMS) files)"
# <<< P8
```

### 11.2 帶出

請依院內規定把 `$W/out2/` 整個資料夾帶出（可以打包成一個 zip；檔名建議帶日期，例如 `pi05_libero_l06_mw_weights_<YYYYMMDD>.zip`）。
帶到目的地後，在該資料夾內執行 `sha256sum -c SHA256SUMS` 驗證每個檔案。

- 若只能先帶出小檔案：請先帶出 `*.log`、`runs.txt`、`speed.txt`、`gate.txt`、`SHA256SUMS`（共幾 MB），權重另外安排。
- 若可帶出的總量有限：優先順序是兩個 `step3000`（共 8.2 GB），其次是兩個 `step1500`。
- 這些權重衍生自 Gemma，受 Gemma 授權條款約束，僅供內部研究使用。

### 11.3 之後可以刪除的東西

| 路徑 | 大小 | 何時可以刪 |
|---|---|---|
| `$W/out2/` | 約 16.5 GB（若是 hard link 則不另佔空間） | 帶出的檔案在目的地通過 `sha256sum -c SHA256SUMS` 驗證之後 |
| `$W/vla_distill/checkpoints/pi05_libero_l06_init` | 4.0 GB | 兩個訓練都 `PASS`、`P7` 的兩個 `GATE PASS p7` 都出現之後 |
| 其他（Python 環境、資料集、teacher、`*_mw_b64_s3000/3000` 的 `optimizer.pt` 與 `master_params.pt`） | | **請保留**，下一次延長訓練（例如 10k steps）會直接沿用 |

---

## 12. 失敗時

任何一步失敗，請執行下面這一行，產生診斷檔以便排查：

```bash
bash $W/r2/diag2.sh
```

（若 `P1` 還沒建立 `$W/r2/diag2.sh`，請改用第一份手冊建立的 `bash $W/diag.sh`，並保存畫面上的錯誤訊息與 `ovx_state.txt`。）

常見錯誤對照：

| 訊息 | 可能原因 | 可以先試的事 |
|---|---|---|
| `P1 FAIL: put vla_distill-3d11c3e.tar.gz into ...` | 壓縮檔不在 `$W/in/`，或檔案在傳輸中損壞（sha256 不符） | 重新放一次檔案；仍失敗：用 `sha256sum` 和第 0 節列出的 sha256 比對，不同就要重新取得檔案 |
| `CODE_FAIL` | Python 載入的不是 `$W/vla_distill` 的程式碼 | `bash $W/r2/diag2.sh` 產生診斷檔 |
| `patched False` 或 `ENV_OK` 沒出現 | Python 環境被更動過 | 附錄 R 的 `R1` |
| `RUN ... REFUSED` | 步驟 4 沒有 `GATE PASS` | 回到 `step P4` |
| `P6 REFUSED: no SPEED_GATE ...` | 步驟 5 沒有跑，或結果是 `FAIL` | 照第 8.1 節處理 |
| `CUDA out of memory` | GPU 被別人佔用，或記憶體不足 | `nvidia-smi` 確認該 GPU 是空的；照第 8.1 節第 4 點，不要自行改 batch |
| `No space left on device` | 磁碟滿了 | `step P2`、`step P2B`，再重跑同一步 |
| `Bus error` | `/dev/shm` 太小（第一份手冊在 OVX 上的訓練沒有發生） | `export VLA_WORKERS=0` 後重跑，並記下用了這個設定 |
| `Revision v2.1 ... not found, using version v2.0`、`The dataset you requested ... is in 2.0 format` | 正常的警告 | 不用處理 |

---

## 附錄 R：只有在步驟 1 檢查失敗時才需要（修復環境或資料）

這兩個區塊和第一份手冊的 `P2A`、`P3A` 相同（第一份手冊的路徑 A，需要對外網路），只是程式碼換成新的壓縮檔。

### R1：重建 Python 環境（`ENV_OK` 沒出現時）

```bash
step R1
```

- 時間：3–10 分鐘（若 `$W/cache/uv` 被刪除過，需要重新下載約 5 GB）。
- 成功：最後一行是 `ENV_OK`。之後重新 `step P1`。
- 失敗：`bash $W/r2/diag2.sh`（或 `bash $W/diag.sh`）產生診斷檔，並一起查看 `$W/logs/setup_uv.log`。

```bash
# >>> R1
: "${W:?run first: export W=/data/vla_work}"; source "$W/env.sh" && cd "$W" || exit 1
A="$W/in/vla_distill-3d11c3e.tar.gz"
echo "18b95943e25253b8b7c99b1b7360b561beea8404a1dc2a5f904965c0e52cc41a  $A" | sha256sum -c - || { echo "R1 FAIL: put vla_distill-3d11c3e.tar.gz into $W/in/"; exit 1; }
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/0.12.19/install.sh | env UV_INSTALL_DIR="$W/bin" sh || { echo "R1 FAIL: uv install"; exit 1; }
tar --no-same-owner -xzf "$A" -C "$W" && cd "$W/vla_distill" || exit 1
uv sync --frozen --no-install-package evdev --no-install-package av > "$W/logs/setup_uv.log" 2>&1 || { tail -n 25 "$W/logs/setup_uv.log"; echo "R1 FAIL: uv sync"; exit 1; }
uv pip install --python .venv/bin/python --only-binary=:all: "av==15.1.0" >> "$W/logs/setup_uv.log" 2>&1 || { tail -n 25 "$W/logs/setup_uv.log"; echo "R1 FAIL: av"; exit 1; }
cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/ || exit 1
bash "$W/check_env.sh"
# <<< R1
```

### R2：補齊資料集、teacher 或 normalization 統計（`CHECK_DATA FAIL` 時）

已經存在的檔案不會重抓，所以只會補上缺少的部分。

```bash
step R2
```

- 時間：只缺小檔案時約 1 分鐘；若整個資料集都要重抓，約 30–60 分鐘。
- 成功：最後一行是 `CHECK_DATA PASS`。之後重新 `step P1`。
- 失敗：`bash $W/r2/diag2.sh`（或 `bash $W/diag.sh`）產生診斷檔，並一起查看 `$W/logs/data_download.log`。

```bash
# >>> R2
: "${W:?run first: export W=/data/vla_work}"; source "$W/env.sh" && cd "$W/vla_distill" || exit 1
T="$OPENPI_DATA_HOME/big_vision"; N="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero"
mkdir -p checkpoints "$T"
if [ -f checkpoints/pi05_libero_pytorch/model.safetensors ]; then echo "teacher already in place"
elif [ -f "$W/in/pi05_libero_pytorch/model.safetensors" ]; then cp -rL "$W/in/pi05_libero_pytorch" checkpoints/ || exit 1
else
  .venv/bin/python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero', gs={'token': 'anon'}))" || { echo "R2 FAIL: teacher download"; exit 1; }
  .venv/bin/python examples/convert_jax_model_to_pytorch.py --config_name pi05_libero \
    --checkpoint_dir "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero" \
    --output_path checkpoints/pi05_libero_pytorch --precision bfloat16 > "$W/logs/convert_teacher.log" 2>&1 || { tail -n 25 "$W/logs/convert_teacher.log"; echo "R2 FAIL: teacher conversion"; exit 1; }
  cp -r "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero/assets" checkpoints/pi05_libero_pytorch/
fi
mkdir -p "$N"
[ -f "$N/norm_stats.json" ] || curl -fsSL -o "$N/norm_stats.json" https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json || { echo "R2 FAIL: norm stats"; exit 1; }
[ -f "$T/paligemma_tokenizer.model" ] || curl -fsSL -o "$T/paligemma_tokenizer.model" https://storage.googleapis.com/big_vision/paligemma_tokenizer.model || { echo "R2 FAIL: tokenizer"; exit 1; }
.venv/bin/huggingface-cli download physical-intelligence/libero --repo-type dataset --revision v2.0 \
  --local-dir "$HF_HOME/lerobot/physical-intelligence/libero" --max-workers 8 > "$W/logs/data_download.log" 2>&1 || { tail -n 15 "$W/logs/data_download.log"; echo "R2 FAIL: dataset download"; exit 1; }
bash "$W/check_data.sh"
# <<< R2
```

---

## 附錄 B：記憶體與時間的估計依據

單位 GB = 10⁹ bytes。基準是第一份手冊的訓練在 OVX 的 L40S-48Q 上的實測（`speed.txt`、`timed50.log`、兩個訓練 log）：

| 項目 | 第一份手冊（bf16 master weights） | 本手冊（fp32 master weights，新 `FREEZE`） |
|---|---|---|
| 可訓練參數 | 1,070.9M（其中 702.7M 收得到梯度） | 698.5M（655.4M bf16 矩陣 + 43.1M fp32），收不到梯度的已凍結 |
| fp32 master weights | 無 | 698.5M × 4 B = 2.8 GB |
| AdamW 狀態 | 2.98 GB（bf16） | 5.6 GB（fp32） |
| 峰值（allocated → reserved），`kd` | 36.03 → 38.88 GB（實測） | 約 41.5 → 44.3 GB（預估） |
| 峰值 reserved，`taskonly` | 39.38 GB（實測） | 約 44.8 GB（預估） |
| 每步秒數 `kd`／`taskonly` | 7.66／4.69（實測） | 約 7.73／4.77（optimizer 多約 0.06 秒） |
| 3,000 steps 時間 `kd`／`taskonly` | 6.5／4.0 小時（實測，含啟動與存檔） | 約 6.5／4.0 小時 |
| 每個 checkpoint 資料夾 | 7.1 GB | 12.5 GB |

OVX 上的 L40S-48Q 顯示 49,152 MiB（51.5 GB），桌面程式佔用約 1.4 GB，扣掉 CUDA context 後約 49.5 GB 可用；步驟 5 的上限 46 GB 是為了在 3,000 步中保留約 3 GB 的餘裕。
