# OVX 雲端訓練操作手冊：π0.5 → 6 層 student 蒸餾（Phase 2）

版本：2026-10-04（步驟 5 加入 GPU memory bandwidth 量測）｜對應程式碼：`vla_distill` commit `966da5c`

---

## 0. 這份文件是什麼

院內 OVX 算力無法從院外連線，所以請您在院內代為執行下面的指令。
這份文件可以單獨使用：不需要看過 repo，也不需要即時詢問。

要做的事只有一件：在一張 GPU 上跑兩次訓練（各 3,000 steps、batch 64），再把結果帶出來。

| 訓練 | 內容 | 預估時間（1 張 48 GB L40S） |
|---|---|---|
| `kd` | 6 層 student，task loss + teacher 蒸餾 loss | 約 4.4–7.8 小時 |
| `taskonly` | 同一個 student，只有 task loss（對照組） | 約 2.6–4.7 小時 |

時間是估計值，步驟 5 的 50-step 測速會給出實際數字。
若有兩張空的 GPU，兩個訓練可以同時跑，總時間約等於較長的那一個。

**您會收到：**

| 檔案 | 大小 | 用途 |
|---|---|---|
| `runbook.md`（本文件） | 約 50 KB | 指令來源 |
| `vla_distill-966da5c.tar.gz` | 1.0 MB | 程式碼（repo 是 private，所以用壓縮檔，不需要任何帳號或金鑰） |
| 離線 bundle（只有在雲端沒有對外網路時才需要，見附錄 C） | 約 52.5 GB | Python 環境、資料集、teacher 權重 |

**請您回傳：**

1. 步驟 0 的 `ovx_probe.txt`（幾 KB，請在做其他步驟之前先寄回）。
2. 步驟 8 產生的 `$W/out/` 資料夾：4 個 student 權重（每個 4.11 GB，共 16.5 GB）、log、`SHA256SUMS`。
3. 任何一步失敗時：`bash $W/diag.sh` 產生的那個文字檔。

**需要的資源：** 1 張記憶體 ≥ 40 GB 的 NVIDIA GPU、主記憶體 ≥ 48 GB、可寫入的磁碟空間 ≥ 200 GB（最少約 150 GB）、Linux x86_64（glibc ≥ 2.28，例如 Ubuntu 20.04 以上或 RHEL 8 以上）、`git`、`curl`、`tar`。不需要 sudo，也不需要 Docker。

---

## 1. 步驟總覽

| 步驟 | 代號 | 做什麼 | 時間 |
|---|---|---|---|
| 0 | `P0`、`P0B` | 環境探測（唯讀）與速度量測 | 1–5 分鐘 |
| 1 | `P1` | 建立工作目錄、環境檔與 4 個小腳本 | 幾秒 |
| 2 | `P2A`（有網路）或 `P2B`（無網路） | 程式碼與 Python 環境 | 3–10 分鐘 |
| 3 | `P3A`（有網路）或 `P3B`（無網路） | 資料集與 teacher 權重 | 15–60 分鐘 |
| 4 | `P4` | 10-step smoke run | 約 3 分鐘 |
| 5 | `P5` | batch 64 的 50-step 測速（含 GPU memory bandwidth 量測） | 約 8–15 分鐘 |
| 6 | `P6` 或 `P6P` | 兩個正式訓練（tmux，可續跑） | 7–13 小時 |
| 7 | `P7` | 執行中檢查 | 幾秒 |
| 8 | `P8` | 打包、checksum、帶出 | 2–5 分鐘 |

---

## 2. 開始之前：定義 `step` 指令（每次登入做一次）

為了讓您少打字，本文件的每個指令區塊都有代號。
請先把 `runbook.md` 放到雲端機器上（例如 `~/runbook.md`），然後在終端機貼上這三行：

```bash
export RB=~/runbook.md
step(){ tr -d '\r' < "$RB" | sed -n "/^# >>> $1\$/,/^# <<< $1\$/p" > "${TMPDIR:-/tmp}/vla_$1.sh" && bash "${TMPDIR:-/tmp}/vla_$1.sh"; }
echo "step helper ready"
```

之後每一步只要輸入 `step P0`、`step P1` 這樣的一行。
如果無法把 `runbook.md` 放上去，也可以直接複製各區塊中 `# >>> 代號` 到 `# <<< 代號` 之間的內容貼到終端機，效果相同。

步驟 1 之後，每次重新登入還要再輸入一行（把路徑換成您選定的工作目錄）：

```bash
export W=/path/to/workdir
```

---

## 3. 步驟 0：環境探測

### 3.1 `P0`：唯讀探測

先 `cd` 到您打算用來放工作目錄的磁碟（空間最大的那個，例如 `/raid/<您的帳號>` 或 `/scratch/<您的帳號>`），再執行：

```bash
step P0
```

- 時間：約 1 分鐘（對外連線若被擋，每個網址最多等 15 秒）。
- 這個區塊只讀取資訊，唯一寫入的是目前目錄下的 `ovx_probe.txt`。proxy 設定中的帳號密碼會被遮蔽成 `***`。
- 成功：最後一行是 `probe written to .../ovx_probe.txt`。
- 請把 `ovx_probe.txt` 寄回。若某個指令不存在，輸出中會顯示錯誤訊息，不影響其他項目，照樣寄回即可。

```bash
# >>> P0
OUT="$PWD/ovx_probe.txt"
{
echo "=== ovx_probe v1 $(date -u +%FT%TZ) host=$(hostname) user=$(id -un) uid=$(id -u) pwd=$PWD"
echo "--- OS"; (. /etc/os-release 2>/dev/null; echo "$PRETTY_NAME"); uname -srm; ldd --version 2>&1 | head -1
echo "--- GPU"; nvidia-smi 2>&1 | head -45
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv 2>&1
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv 2>&1 | head -20
echo "--- CPU/RAM"; lscpu 2>/dev/null | grep -E "^(Model name|CPU\(s\)|Thread|Core|Socket|NUMA node\(s\))"; echo "nproc=$(nproc)"; free -g
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes /sys/fs/cgroup/cpu.max; do [ -r "$f" ] && echo "$f=$(cat "$f")"; done
echo "--- DISK"; df -hT . "$HOME" /tmp /dev/shm 2>&1; df -i . 2>&1 | tail -1
findmnt -T . -o TARGET,SOURCE,FSTYPE,OPTIONS 2>&1 | tail -1
for d in /raid /scratch /data /workspace /local /nvme /mnt; do [ -d "$d" ] && df -hT "$d" 2>/dev/null | tail -1; done
(quota -s 2>/dev/null | tail -3)
for d in . "$HOME" /tmp /dev/shm; do if [ -w "$d" ]; then echo "writable: $d"; else echo "NOT writable: $d"; fi; done
echo "ulimit_n=$(ulimit -n) ulimit_u=$(ulimit -u)"
echo "--- TOOLS"; for t in bash curl wget git tar gzip tmux screen nohup python3 rsync sha256sum split gcc docker podman apptainer singularity sbatch srun qsub bsub kubectl module; do printf "%s=%s\n" "$t" "$(command -v "$t" 2>/dev/null || echo -)"; done
echo "--- CONTAINER/SCHEDULER"; if [ -f /.dockerenv ]; then echo "dockerenv=yes"; else echo "dockerenv=no"; fi
head -3 /proc/1/cgroup 2>/dev/null; echo "pid1=$(cat /proc/1/comm 2>/dev/null)"; (systemd-detect-virt 2>/dev/null || true)
env | grep -E "^(SLURM_JOB_ID|SLURM_JOB_GPUS|PBS_JOBID|LSB_JOBID|KUBERNETES_SERVICE_HOST|NVIDIA_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES)=" || echo "no scheduler/container env vars"
echo "--- SUDO"; if sudo -n true 2>/dev/null; then echo "sudo=yes(passwordless)"; else echo "sudo=no-or-needs-password"; fi
echo "--- PROXY"; PX="$(env | grep -i -E "^(https?|all|no|ftp)_proxy=" | sed -E 's#//[^@/]*@#//***@#')"; echo "${PX:-no proxy variables}"
echo "--- INTERNET (http code, seconds; 000 = unreachable)"
for u in https://huggingface.co/api/datasets/physical-intelligence/libero \
         https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
         https://github.com https://pypi.org/simple/pip/ \
         https://files.pythonhosted.org/packages/86/f6/0b473dab52dfdea05f28f3578b1c56b6c796ce85e76951bab7c4e38d5a74/av-14.4.0.tar.gz \
         https://download.pytorch.org/whl/cu128/ https://astral.sh/uv/0.12.19/install.sh \
         https://github.com/astral-sh/python-build-standalone/releases/latest; do
  printf "%s " "$u"; curl -sS -L -m 15 -r 0-1023 -o /dev/null -w "%{http_code} %{time_total}s\n" "$u" 2>&1 | tail -1
done
} 2>&1 | tee "$OUT"
echo "probe written to $OUT"
# <<< P0
```

### 3.2 `P0B`：速度量測

```bash
step P0B
```

- 時間：1–3 分鐘。
- 這個區塊會在目前目錄寫入一個 1 GiB 暫存檔和 2,000 個小檔案，量完立刻刪除；另外下載約 240 MB 到 `/dev/null`（不落地）。結果附加到同一個 `ovx_probe.txt`。
- 若雲端沒有對外網路，最後兩行會顯示連線錯誤，這是預期的。

```bash
# >>> P0B
OUT="$PWD/ovx_probe.txt"
{
echo "=== ovx_probe throughput $(date -u +%FT%TZ) pwd=$PWD"
echo "--- DISK WRITE 1GiB (fdatasync)"; dd if=/dev/zero of=./.vla_probe.bin bs=1M count=1024 conv=fdatasync 2>&1 | tail -1
echo "--- DISK READ 1GiB (direct if supported)"; (dd if=./.vla_probe.bin of=/dev/null bs=1M iflag=direct 2>&1 || dd if=./.vla_probe.bin of=/dev/null bs=1M 2>&1) | tail -1
rm -f ./.vla_probe.bin
echo "--- SMALL FILES: create and delete 2000"; mkdir -p ./.vla_probe.d; s=$(date +%s%N)
for i in $(seq 2000); do echo x > "./.vla_probe.d/$i"; done; rm -rf ./.vla_probe.d
echo "small_files_ms=$(( ($(date +%s%N) - s) / 1000000 ))"
echo "--- NET DOWNLOAD"
curl -sS -m 180 -o /dev/null -r 0-209715199 -w 'GCS 200MB: %{size_download} bytes %{speed_download} B/s %{time_total}s http=%{http_code}\n' \
  https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/params/ocdbt.process_0/d/0eaaecefaa9720d30a32cc56e65fd345
curl -sS -L -m 180 -o /dev/null -w 'HF 37MB: %{size_download} bytes %{speed_download} B/s %{time_total}s http=%{http_code} host=%{url_effective}\n' \
  https://huggingface.co/datasets/physical-intelligence/libero/resolve/v2.0/data/chunk-000/episode_000001.parquet | cut -c1-170
} 2>&1 | tee -a "$OUT"
# <<< P0B
```

### 3.3 探測結果決定走哪一條路

| 探測項目 | 結果 | 走哪一條 |
|---|---|---|
| GPU 記憶體（`memory.total`） | ≥ 40,000 MiB（48 GB 的 L40S、L40、A40 通常顯示約 46,000 MiB） | 照本文件執行，batch 64，單張 GPU |
| | 24,000–40,000 MiB | 請停下來，把 `ovx_probe.txt` 寄回。batch 64 預估需要約 35–41 GB，放不下，需要我們決定替代設定 |
| 空閒 GPU 數量（`memory.used` 接近 0 的張數） | 1 張 | 步驟 6 用 `P6`（兩個訓練依序跑） |
| | ≥ 2 張 | 步驟 6 可用 `P6P`（兩個訓練各用一張，同時跑） |
| GPU 上已有別人的程式 | 是 | 選一張 `memory.used` 接近 0 的 GPU，之後用 `export GPU=<編號>` 指定 |
| `driver_version` | ≥ 525 | 可以。低於 525 請寄回探測結果 |
| INTERNET 各網址 | 全部是 `200` 或 `206` | **路徑 A**：`P2A`、`P3A` |
| | 任何一個是 `000`、`403`、`407` 或其他數字 | **路徑 B**：`P2B`、`P3B`，需要離線 bundle（附錄 C）。請把探測結果寄回，我們會準備 bundle |
| PROXY | 有設定 | 路徑 A 仍可用：`curl`、`uv`、Hugging Face 下載都會使用 `https_proxy` 環境變數 |
| 目前目錄的檔案系統（`df -hT` 的 Type） | `ext4`、`xfs`（本機磁碟） | 最理想 |
| | `nfs`、`lustre`、`gpfs`、`cifs`、`fuse.*`（網路磁碟） | 可以用，但環境安裝與程式啟動會變慢。若機器上有本機磁碟（常見於 `/raid`、`/scratch`、`/local`），請改用本機磁碟當工作目錄 |
| 可用空間（`df -hT .` 的 Avail） | ≥ 200 GB | 可以 |
| | 150–200 GB | 可以，但 `P5` 之後請依第 11.3 節刪除暫存 |
| | < 150 GB | 請寄回探測結果 |
| `/dev/shm` 大小 | ≥ 2 GB | 可以 |
| | < 2 GB（Docker 預設只有 64 MB） | 見附錄 A.2 |
| `HOME` | `NOT writable` | 不影響，`P1` 會把 `HOME` 轉到工作目錄內 |
| `glibc`（`ldd` 那一行） | ≥ 2.28 | 可以。低於 2.28 請寄回探測結果 |
| `git`、`curl`、`tar` | 有 | 可以。缺任何一個請寄回探測結果 |
| `tmux` | 有 | `P6` 用 tmux。沒有的話 `P6` 會自動改用 `nohup` |
| `sbatch`/`srun`/`qsub` 存在，或有 `SLURM_JOB_ID` | 是 | 這台機器用排程器，見附錄 A.1 |
| `dockerenv=yes` 或 `KUBERNETES_SERVICE_HOST` | 是 | 您已經在容器內，照常執行，但請確認工作目錄是持久化的 volume，並看附錄 A.2 |
| `sudo` | 有或沒有 | 都可以，本文件不需要 sudo |

---

## 4. 步驟 1：工作目錄與環境檔（`P1`）

`scripts/phase1/env.sh` 是筆電（WSL）專用的，雲端不要用它。雲端只需要下面這個環境檔，它把所有快取都指到工作目錄內，不依賴家目錄，也不需要編譯器。

```bash
export W=/path/to/workdir     # 換成您選定的目錄（還不存在也可以）
step P1
```

- 時間：幾秒。
- 成功：最後一行是 `P1 OK`，並列出 5 個檔案。
- 之後請把收到的 `vla_distill-966da5c.tar.gz`（以及離線 bundle，若有）放進 `$W/in/`。

```bash
# >>> P1
: "${W:?run first: export W=/path/to/workdir}"
mkdir -p "$W"/bin "$W"/cache "$W"/logs "$W"/out "$W"/in "$W"/home || exit 1
if [ -f ovx_probe.txt ] && [ ! -f "$W/ovx_probe.txt" ]; then cp ovx_probe.txt "$W/"; fi
cd "$W" || exit 1
cat > "$W/env.sh" <<EOT
export W="$W"
EOT
cat >> "$W/env.sh" <<'EOT'
export PATH="$W/bin:$PATH"
export UV_CACHE_DIR="$W/cache/uv" UV_PYTHON_INSTALL_DIR="$W/cache/uv-python" UV_LINK_MODE=copy UV_NO_MODIFY_PATH=1
export UV_PYTHON_PREFERENCE=only-managed GIT_LFS_SKIP_SMUDGE=1
export HF_HOME="$W/cache/hf" OPENPI_DATA_HOME="$W/cache/openpi" XDG_CACHE_HOME="$W/cache/xdg"
export JAX_PLATFORMS=cpu WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FREEZE="paligemma_with_expert.paligemma.model.vision_tower. paligemma_with_expert.paligemma.model.multi_modal_projector. paligemma_with_expert.paligemma.model.language_model.embed_tokens. paligemma_with_expert.paligemma.lm_head."
if [ ! -w "$HOME" ]; then export HOME="$W/home"; fi
if [ -f "$W/offline" ]; then export UV_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1; fi
EOT

cat > "$W/check_env.sh" <<'EOS'
#!/usr/bin/env bash
source "$(dirname "$(readlink -f "$0")")/env.sh"; cd "$W/vla_distill" || exit 1
.venv/bin/python - <<'PY'
import torch, transformers, jax, lerobot, av
from transformers.models.siglip import check
ok = check.check_whether_transformers_replace_is_installed_correctly()
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(" gpu", i, p.name, round(p.total_memory / 1e9, 1), "GB")
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
y = (x @ x).float().abs().mean().item()
print("transformers", transformers.__version__, "patched", ok, "| jax", jax.__version__, jax.devices(), "| av", av.__version__)
assert ok and y > 0
print("ENV_OK")
PY
EOS

cat > "$W/check_data.sh" <<'EOS'
#!/usr/bin/env bash
source "$(dirname "$(readlink -f "$0")")/env.sh"; cd "$W/vla_distill" || exit 1
N="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero"
D="$HF_HOME/lerobot/physical-intelligence/libero"
T="checkpoints/pi05_libero_pytorch"
FAIL=0
echo "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6  $OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model" | sha256sum -c - || FAIL=1
echo "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84  $N/norm_stats.json" | sha256sum -c - || FAIL=1
if echo "d9ef6bb72672509355ed19ebf08909e04e9af0ec4918a5d5ce9fa7c25aaf7ea4  $T/model.safetensors" | sha256sum -c -; then echo "TEACHER_OK identical to the evaluated teacher"
elif [ -f "$T/model.safetensors" ]; then echo "TEACHER_DIFFERENT_HASH (report this line): $(sha256sum "$T/model.safetensors")"
else echo "TEACHER_MISSING"; FAIL=1; fi
echo "parquet_files=$(find -L "$D/data" -name '*.parquet' | wc -l) parquet_bytes=$(find -L "$D/data" -name '*.parquet' -printf '%s\n' | awk '{s+=$1} END {print s+0}')"
.venv/bin/python - <<'PY' || FAIL=1
import os
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
eps = [int(x) for x in os.environ.get("VLA_EPISODES", "").split()] or None
ds = LeRobotDataset("physical-intelligence/libero", episodes=eps)
print("DATA_OK episodes", ds.num_episodes, "frames", ds.num_frames, "fps", ds.fps)
PY
if [ $FAIL -eq 0 ]; then echo "CHECK_DATA PASS"; else echo "CHECK_DATA FAIL -> bash $W/diag.sh"; fi
exit $FAIL
EOS

cat > "$W/run_one.sh" <<'EOS'
#!/usr/bin/env bash
# usage: bash run_one.sh kd|taskonly [gpu_index]
source "$(dirname "$(readlink -f "$0")")/env.sh"
case "${1:-}" in
  kd)       CFG=pi05_libero_l06;          EXP=kd_b64_s3000 ;;
  taskonly) CFG=pi05_libero_l06_taskonly; EXP=taskonly_b64_s3000 ;;
  *) echo "usage: bash run_one.sh kd|taskonly [gpu_index]"; exit 2 ;;
esac
if [ -n "${2:-}" ]; then export CUDA_VISIBLE_DEVICES="$2"; fi
STEPS="${VLA_STEPS:-3000}"
cd "$W/vla_distill" || exit 1
CK="checkpoints/$CFG/$EXP"
if [ -f "$CK/$STEPS/model.safetensors" ]; then echo "RUN $1 ALREADY DONE ($CK/$STEPS)" | tee -a "$W/logs/runs.txt"; exit 0; fi
MODE=""; if ls -d "$CK"/[0-9]* >/dev/null 2>&1; then MODE="--resume"; fi
LOG="$W/logs/${1}_$(date +%Y%m%d_%H%M%S).log"
echo "START $1 mode=${MODE:-fresh} gpu=${CUDA_VISIBLE_DEVICES:-all} log=$LOG $(date -Is)" | tee -a "$W/logs/runs.txt"
${DRY:+echo} .venv/bin/python scripts/distill_pytorch.py "$CFG" --exp-name "$EXP" $MODE \
  --batch-size 64 --num-train-steps "$STEPS" --save-interval 1500 --log-interval 10 --num-workers "${VLA_WORKERS:-4}" \
  --pytorch-training-precision bfloat16 --lr-schedule.warmup-steps 300 --lr-schedule.decay-steps 3000 \
  --pytorch-freeze-prefixes $FREEZE ${VLA_EXTRA:-} >> "$LOG" 2>&1
RC=$?
if [ $RC -eq 0 ] && [ -f "$CK/$STEPS/model.safetensors" ]; then R=PASS; else R="FAIL rc=$RC"; fi
echo "RUN $1 $R $(date -Is) log=$LOG" | tee -a "$W/logs/runs.txt"
[ "$R" = PASS ]
EOS

cat > "$W/diag.sh" <<'EOS'
#!/usr/bin/env bash
# writes one text file to send back when something fails
source "$(dirname "$(readlink -f "$0")")/env.sh"
O="$W/out/diag_$(date +%Y%m%d_%H%M%S).txt"
{
echo "=== diag $(date -Is) host=$(hostname)"
nvidia-smi 2>&1 | head -30; free -g; df -hT "$W" /dev/shm 2>&1
echo "--- runs.txt"; tail -n 20 "$W/logs/runs.txt" 2>/dev/null
for f in $(ls -t "$W"/logs/*.log 2>/dev/null | head -3); do
  echo "--- tail $f"; tr '\r' '\n' < "$f" | grep -v '^[[:space:]]*$' | tail -n 60 | cut -c1-400
done
echo "--- processes"; pgrep -af "distill_pytorch|uv sync|huggingface-cli" | cut -c1-200
echo "--- sizes"; du -sh "$W"/cache/* "$W"/vla_distill/checkpoints/*/* 2>/dev/null
} > "$O" 2>&1
echo "send this file back: $O"
EOS
ls -la "$W"/env.sh "$W"/check_env.sh "$W"/check_data.sh "$W"/run_one.sh "$W"/diag.sh && echo "P1 OK"
# <<< P1
```

---

## 5. 步驟 2：程式碼與 Python 環境

先確認 `vla_distill-966da5c.tar.gz` 已放在 `$W/in/`。兩條路徑擇一。

安裝時刻意跳過兩個只有原始碼、需要編譯器的套件：`evdev`（鍵盤輸入，訓練用不到）與 `av` 14.4.0（改裝有預編譯 wheel 的 15.1.0）。所以雲端不需要 gcc。

### 5.1 路徑 A：有對外網路（`P2A`）

```bash
step P2A
```

- 時間：3–10 分鐘（下載約 5 GB 的 Python 套件）。
- 成功：最後一行是 `ENV_OK`，上面會印出 `torch 2.7.1+cu128 cuda True`、每張 GPU 的名稱與記憶體、`patched True`。
- 失敗：畫面會印出 `P2A FAIL` 和 log 尾端。請執行 `bash $W/diag.sh` 並寄回產生的檔案。

```bash
# >>> P2A
: "${W:?run first: export W=/path/to/workdir}"; source "$W/env.sh" && cd "$W" || exit 1
A="$W/in/vla_distill-966da5c.tar.gz"
echo "608237d052783be3f59bfe730af0d67576f3442e6576d37b7889959c2de948a0  $A" | sha256sum -c - || { echo "P2A FAIL: put vla_distill-966da5c.tar.gz into $W/in/"; exit 1; }
curl -LsSf https://astral.sh/uv/0.12.19/install.sh | env UV_INSTALL_DIR="$W/bin" sh || { echo "P2A FAIL: uv install"; exit 1; }
tar --no-same-owner -xzf "$A" -C "$W" && cd "$W/vla_distill" || exit 1
uv sync --frozen --no-install-package evdev --no-install-package av > "$W/logs/setup_uv.log" 2>&1 || { tail -n 25 "$W/logs/setup_uv.log"; echo "P2A FAIL: uv sync"; exit 1; }
uv pip install --python .venv/bin/python --only-binary=:all: "av==15.1.0" >> "$W/logs/setup_uv.log" 2>&1 || { tail -n 25 "$W/logs/setup_uv.log"; echo "P2A FAIL: av"; exit 1; }
cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/ || exit 1
bash "$W/check_env.sh"
# <<< P2A
```

### 5.2 路徑 B：沒有對外網路（`P2B`）

先把離線 bundle 的所有檔案放進 `$W/in/`（或放在別處，然後 `export B=<bundle 目錄>`）。

```bash
step P2B
```

- 時間：2–15 分鐘（checksum 約 52 GB，加上解開 10.3 GB 的環境快取）。
- 成功與失敗的判讀同 `P2A`。

```bash
# >>> P2B
: "${W:?run first: export W=/path/to/workdir}"; B="${B:-$W/in}"; touch "$W/offline"; source "$W/env.sh" && cd "$W" || exit 1
(cd "$B" && sha256sum -c SHA256SUMS.bundle | grep -v ': OK$'; test "${PIPESTATUS[0]}" -eq 0) || { echo "P2B FAIL: bundle checksum"; exit 1; }
tar --no-same-owner -xf "$B/vla_env_cache.tar" -C "$W" || { echo "P2B FAIL: untar env cache"; exit 1; }
tar --no-same-owner -xzf "$B/vla_distill-966da5c.tar.gz" -C "$W" && cd "$W/vla_distill" || exit 1
uv sync --frozen --no-install-package evdev --no-install-package av > "$W/logs/setup_uv.log" 2>&1 || { tail -n 25 "$W/logs/setup_uv.log"; echo "P2B FAIL: uv sync"; exit 1; }
uv pip install --python .venv/bin/python --only-binary=:all: "av==15.1.0" >> "$W/logs/setup_uv.log" 2>&1 || { tail -n 25 "$W/logs/setup_uv.log"; echo "P2B FAIL: av"; exit 1; }
cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/ || exit 1
bash "$W/check_env.sh"
# <<< P2B
```

---

## 6. 步驟 3：資料集與 teacher 權重

訓練資料是 Hugging Face 上的公開資料集 `physical-intelligence/libero`（revision `v2.0`）：1,693 個 episodes、273,465 個 frames、34.94 GB。
第一次載入時，程式會在 `$W/cache/hf/datasets` 再寫一份同樣大小的 Arrow 快取，所以資料集共佔約 70 GB。

teacher 是 Physical Intelligence 官方的 π0.5 LIBERO checkpoint，轉成 PyTorch 後是一個 7.23 GB 的檔案。

### 6.1 路徑 A：有對外網路（`P3A`）

teacher 有兩種來源，`P3A` 會自動判斷：

- 若您收到我們提供的 `pi05_libero_pytorch/` 資料夾，請放進 `$W/in/`。`P3A` 會直接使用它。這是首選，因為它和我們在筆電上評估過的 teacher 是同一個檔案。
- 若沒有，`P3A` 會從公開的 Google Cloud Storage 下載官方 JAX checkpoint（12.4 GB）並在雲端轉換。轉換需要約 30 GB 主記憶體，約 3–10 分鐘。

```bash
step P3A
```

- 時間：資料集下載 34.94 GB，在 20 MB/s 下約 30 分鐘；之後建立 Arrow 快取約 3–10 分鐘。
- 成功：最後一行是 `CHECK_DATA PASS`，上面有 `DATA_OK episodes 1693 frames 273465 fps 10`、`parquet_files=1693 parquet_bytes=34938730091`，以及 `TEACHER_OK`。
- 若出現 `TEACHER_DIFFERENT_HASH`（只可能發生在雲端自行轉換時）：不算失敗，請把那一行寄回，然後繼續。
- 下載中斷：重新執行 `step P3A` 即可，已下載的檔案不會重抓。
- 失敗：`bash $W/diag.sh`，並附上 `$W/logs/data_download.log`。

```bash
# >>> P3A
: "${W:?run first: export W=/path/to/workdir}"; source "$W/env.sh" && cd "$W/vla_distill" || exit 1
T="$OPENPI_DATA_HOME/big_vision"; N="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero"
mkdir -p checkpoints "$T"
if [ -f checkpoints/pi05_libero_pytorch/model.safetensors ]; then echo "teacher already in place"
elif [ -f "$W/in/pi05_libero_pytorch/model.safetensors" ]; then cp -rL "$W/in/pi05_libero_pytorch" checkpoints/ || exit 1
else
  .venv/bin/python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero', gs={'token': 'anon'}))" || { echo "P3A FAIL: teacher download"; exit 1; }
  .venv/bin/python examples/convert_jax_model_to_pytorch.py --config_name pi05_libero \
    --checkpoint_dir "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero" \
    --output_path checkpoints/pi05_libero_pytorch --precision bfloat16 > "$W/logs/convert_teacher.log" 2>&1 || { tail -n 25 "$W/logs/convert_teacher.log"; echo "P3A FAIL: teacher conversion"; exit 1; }
  cp -r "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero/assets" checkpoints/pi05_libero_pytorch/
fi
mkdir -p "$N"
[ -f "$N/norm_stats.json" ] || curl -fsSL -o "$N/norm_stats.json" https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json || { echo "P3A FAIL: norm stats"; exit 1; }
[ -f "$T/paligemma_tokenizer.model" ] || curl -fsSL -o "$T/paligemma_tokenizer.model" https://storage.googleapis.com/big_vision/paligemma_tokenizer.model || { echo "P3A FAIL: tokenizer"; exit 1; }
.venv/bin/huggingface-cli download physical-intelligence/libero --repo-type dataset --revision v2.0 \
  --local-dir "$HF_HOME/lerobot/physical-intelligence/libero" --max-workers 8 > "$W/logs/data_download.log" 2>&1 || { tail -n 15 "$W/logs/data_download.log"; echo "P3A FAIL: dataset download"; exit 1; }
bash "$W/check_data.sh"
# <<< P3A
```

### 6.2 路徑 B：沒有對外網路（`P3B`）

```bash
step P3B
```

- 時間：3–20 分鐘（複製 teacher 7.2 GB；資料集若和工作目錄在同一個檔案系統就只建立連結，否則複製 35 GB），加上建立 Arrow 快取約 3–10 分鐘。
- 成功與失敗的判讀同 `P3A`。
- 注意：若資料集是用連結的方式接上，訓練結束前請不要刪除或卸載 bundle 目錄。

```bash
# >>> P3B
: "${W:?run first: export W=/path/to/workdir}"; B="${B:-$W/in}"; source "$W/env.sh" && cd "$W/vla_distill" || exit 1
L="$HF_HOME/lerobot/physical-intelligence/libero"
mkdir -p "$OPENPI_DATA_HOME" "$(dirname "$L")" checkpoints
tar --no-same-owner -xf "$B/openpi_cache.tar" -C "$OPENPI_DATA_HOME" || { echo "P3B FAIL: openpi cache"; exit 1; }
if [ ! -e "$L/meta/info.json" ]; then
  if [ "$(stat -c %d "$B/libero_lerobot")" = "$(stat -c %d "$W")" ]; then ln -sfn "$(readlink -f "$B/libero_lerobot")" "$L"; else cp -r "$B/libero_lerobot" "$L"; fi || { echo "P3B FAIL: dataset"; exit 1; }
fi
[ -f checkpoints/pi05_libero_pytorch/model.safetensors ] || cp -rL "$B/pi05_libero_pytorch" checkpoints/ || { echo "P3B FAIL: teacher"; exit 1; }
bash "$W/check_data.sh"
# <<< P3B
```

---

## 7. 步驟 4：smoke run（`P4`）

用 8 個 episodes、batch 2 跑 10 個訓練 step，確認整條流程（資料載入、teacher 載入、student 初始化、forward、backward、存 checkpoint）能動。

若要指定 GPU，先 `export GPU=<編號>`（預設用第 0 張）。

```bash
step P4
```

- 時間：約 3 分鐘。
- **通過條件：最後一行是 `SMOKE PASS`。**
- 參考值：`step=0 loss=1.1697`、`step=9 loss=0.9379`；模型建立後 GPU 記憶體 `allocated: 11.59GB`。不同 GPU 的 loss 在小數第二、三位可能略有不同，屬正常。loss 出現 `nan` 或 `inf` 則不正常。
- 失敗（`SMOKE FAIL`）：`bash $W/diag.sh` 並寄回。請不要繼續後面的步驟。

```bash
# >>> P4
: "${W:?run first: export W=/path/to/workdir}"; source "$W/env.sh" && cd "$W/vla_distill" || exit 1
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
SMOKE_EXTRA="paligemma_with_expert.paligemma.model.language_model.layers. paligemma_with_expert.paligemma.model.language_model.norm. paligemma_with_expert.gemma_expert.lm_head."
.venv/bin/python scripts/distill_pytorch.py pi05_libero_l06 --exp-name smoke --overwrite \
  --batch-size 2 --num-train-steps 10 --log-interval 1 --save-interval 100000 --num-workers 2 \
  --pytorch-training-precision bfloat16 --data.lerobot-episodes 0 1 2 3 4 5 6 7 \
  --pytorch-freeze-prefixes $FREEZE $SMOKE_EXTRA > "$W/logs/smoke.log" 2>&1
RC=$?
tr '\r' '\n' < "$W/logs/smoke.log" | grep -E "step=[0-9]+ loss|Saved checkpoint|after_model_creation|Error|out of memory" | sed -E 's/^Training:.*\]([0-9]{2}:[0-9]{2}:[0-9]{2}\.)/\1/' | cut -c1-170
if [ $RC -eq 0 ] && [ -f checkpoints/pi05_libero_l06/smoke/9/model.safetensors ]; then echo "SMOKE PASS"; else echo "SMOKE FAIL rc=$RC -> bash $W/diag.sh"; fi
# <<< P4
```

---

## 8. 步驟 5：50-step 測速（`P5`）

用正式設定（完整資料集、batch 64、bf16、vision encoder 與 token embedding 凍結，其餘可訓練）跑 50 個 step。目的有三個：確認 batch 64 放得進這張 GPU、量出每個 step 的實際秒數、量出寫一個 checkpoint 要多久。
開始之前會先花約 25 秒量測這張 GPU 本身的 memory bandwidth（copy／read／write，GB/s）與 bf16 GEMM 速率（TFLOPS），訓練期間每 0.5 秒記錄一次 `nvidia-smi` 的 `utilization.memory`，用來判斷 memory bandwidth 在這張卡上佔每個 step 多少時間。這些量測只寫入 `$W/logs`，不影響通過條件。

```bash
step P5
```

- 時間：約 8–15 分鐘（含開頭約 25 秒的 GPU 量測）。
- **通過條件：輸出中有一行以 `TIMED PASS` 開頭**，例如
  `TIMED PASS s_per_step=6.20 peak_reserved_GB=38.5 est_hours_kd=5.2 est_hours_taskonly=3.1`。
  這一行也會存到 `$W/logs/speed.txt`，請連同 `ovx_probe.txt` 寄回（可以不等我們回覆，直接繼續步驟 6）。
- `TIMED PASS` 之後還會印出 4 行 `PROBE`（GPU 名稱、`copy_GBps`、`read_GBps`、`write_GBps`、`gemm_bf16_TFLOPS`）與 2 行 `MEMBW`（訓練期間的 `utilization.memory`，以及 `traffic_share_of_measured`：memory traffic 佔實測 step 時間的比例）。它們也會附加到 `speed.txt`，一併寄回即可。這幾行只是量測，數字高低都不算失敗；若顯示 `util=NA(...)` 或 `MEMBW train_samples=0` 也請照常繼續。
- L40S 的參考值：`copy_GBps` 約 690–860、`traffic_share_of_measured` 約 55–70%。
- 預期：`peak_reserved_GB` 約 37–41（預估 allocated 35.4 GB）。48 GB 的 GPU 應有約 7–11 GB 餘裕。
- 若 `peak_reserved_GB` 超過 45，或出現 `out of memory`、`TIMED FAIL`：**請不要自行改 batch size**，執行 `bash $W/diag.sh` 並寄回，等我們回覆。改 batch 會改變實驗設定，需要我們決定。
- 若 `est_hours_kd` 超過 12：也請先寄回 `speed.txt` 再決定是否開始。

```bash
# >>> P5
: "${W:?run first: export W=/path/to/workdir}"; source "$W/env.sh" && cd "$W/vla_distill" || exit 1
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
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
.venv/bin/python scripts/distill_pytorch.py pi05_libero_l06 --exp-name timed50 --overwrite \
  --batch-size 64 --num-train-steps 50 --log-interval 10 --save-interval 100000 --num-workers "${VLA_WORKERS:-4}" \
  --pytorch-training-precision bfloat16 --lr-schedule.warmup-steps 300 --lr-schedule.decay-steps 3000 \
  --pytorch-freeze-prefixes $FREEZE > "$W/logs/timed50.log" 2>&1
RC=$?
kill "$SMI" 2>/dev/null
tr '\r' '\n' < "$W/logs/timed50.log" | grep -E "step=[0-9]+ loss|Saved checkpoint|after_backward|Error|out of memory" | sed -E 's/^Training:.*\]([0-9]{2}:[0-9]{2}:[0-9]{2}\.)/\1/' | cut -c1-200 | tail -n 14
tr '\r' '\n' < "$W/logs/timed50.log" | awk -v rc="$RC" '
  /after_backward/ { if (match($0, /peak_reserved: [0-9.]+/)) pr = substr($0, RSTART + 15, RLENGTH - 15) }
  /step=[234]0 loss/ { if (match($0, /time=[0-9.]+/)) { t += substr($0, RSTART + 5, RLENGTH - 5); n += 10 } }
  /Saved checkpoint/ { saved = 1 }
  END { if (rc == 0 && n > 0 && saved) { s = t / n; printf "TIMED PASS s_per_step=%.2f peak_reserved_GB=%s est_hours_kd=%.1f est_hours_taskonly=%.1f\n", s, pr, s * 3000 / 3600, s * 3000 / 3600 * 0.6 }
        else printf "TIMED FAIL rc=%s steps_timed=%d saved=%d\n", rc, n, saved }' | tee "$W/logs/speed.txt"
{ grep PROBE "$W/logs/membw_probe.txt"
  awk -F', *' '$1 >= 50 { n++; g += $1; m += $2; if ($2 > mx) mx = $2 } END { if (n) printf "MEMBW train_samples=%d util_gpu_mean=%.0f util_mem_mean=%.0f util_mem_max=%d\n", n, g / n, m / n, mx; else print "MEMBW train_samples=0" }' "$W/logs/timed50_gpu.csv"
  awk -v kd=2723 -v fl=739 '/copy_GBps/ { split($2, a, "="); c = a[2] } /gemm_bf16_TFLOPS/ { split($2, a, "="); f = a[2] } /TIMED PASS/ { split($3, a, "="); s = a[2] } END { if (c > 0 && f > 0 && s > 0) printf "MEMBW traffic_s=%.2f compute_s=%.2f measured_s=%.2f traffic_share_of_measured=%.0f%%\n", kd / c, fl / f, s, 100 * kd / c / s }' "$W/logs/membw_probe.txt" "$W/logs/speed.txt"
} | tee -a "$W/logs/speed.txt"
# <<< P5
```

---

## 9. 步驟 6：正式訓練

兩個訓練都由 `$W/run_one.sh` 啟動。它在 tmux（或 nohup）裡執行，所以終端機斷線不影響訓練。
它也會自動判斷狀態：已完成就跳過；有 checkpoint 就從最新的 checkpoint 續跑（`--resume`）；否則從頭開始。所以**任何中斷之後，重新執行同一個 `step` 指令即可**。

每個訓練會留下三個 checkpoint 目錄：`1500`、`2999`、`3000`。`2999` 是程式在倒數第二步多存的一份，不需要帶出。

### 9.1 一張 GPU：依序跑（`P6`）

```bash
export GPU=0          # 換成要用的 GPU 編號；若由排程器分配 GPU，請不要設定這個變數
step P6
```

### 9.2 兩張空的 GPU：同時跑（`P6P`）

```bash
export GPU=0 GPU2=1   # 兩張空 GPU 的編號
step P6P
```

- 時間：依 `speed.txt` 的 `est_hours_kd` 與 `est_hours_taskonly`。預估 `kd` 4.4–7.8 小時、`taskonly` 2.6–4.7 小時。
- 啟動成功：畫面印出 `started in tmux session ...` 或 `started with nohup ...`。約 2–3 分鐘後用 `step P7` 應能看到 `step=0 loss=...`。
- **完成條件：`$W/logs/runs.txt` 裡有 `RUN kd PASS` 和 `RUN taskonly PASS` 兩行。**
- 若某個訓練曾經中斷並續跑，`runs.txt` 會多一行 `START ... mode=--resume`。這不算失敗，但請在回信中提一下（續跑的資料順序和不中斷時不同，我們分析時需要知道）。
- 失敗（`RUN ... FAIL`）：`bash $W/diag.sh` 並寄回。

```bash
# >>> P6
: "${W:?run first: export W=/path/to/workdir}"
if pgrep -f "python[0-9.]* scripts/distill_pytorch.py" >/dev/null; then echo "a training process is already running; use: step P7"; exit 0; fi
CMD="bash $W/run_one.sh kd ${GPU:-}; bash $W/run_one.sh taskonly ${GPU:-}"
if command -v tmux >/dev/null 2>&1; then tmux new-session -d -s vla "$CMD" && echo "started in tmux session 'vla' (watch: tmux attach -t vla ; detach: Ctrl-b d)"
else nohup bash -c "$CMD" > "$W/logs/nohup.out" 2>&1 & echo "started with nohup pid=$!"; fi
# <<< P6
```

```bash
# >>> P6P
: "${W:?run first: export W=/path/to/workdir}"; : "${GPU:?export GPU=<index>}"; : "${GPU2:?export GPU2=<index>}"
if pgrep -f "python[0-9.]* scripts/distill_pytorch.py" >/dev/null; then echo "a training process is already running; use: step P7"; exit 0; fi
if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s vla_kd "bash $W/run_one.sh kd $GPU" && tmux new-session -d -s vla_taskonly "bash $W/run_one.sh taskonly $GPU2" && echo "started in tmux sessions 'vla_kd' and 'vla_taskonly'"
else
  nohup bash "$W/run_one.sh" kd "$GPU" > "$W/logs/nohup_kd.out" 2>&1 &
  nohup bash "$W/run_one.sh" taskonly "$GPU2" > "$W/logs/nohup_taskonly.out" 2>&1 &
  echo "started two runs with nohup"
fi
# <<< P6P
```

---

## 10. 步驟 7：執行中檢查（`P7`）

隨時可以執行，不影響訓練。建議啟動後 5 分鐘看一次，之後每 1–2 小時看一次。

```bash
step P7
```

正常的樣子：

- `runs.txt` 有 `START kd mode=fresh ...`。
- log 每 10 個 step 一行 `step=N loss=... lr=... grad_norm=... time=...s`，`time` 是最近 10 個 step 的秒數，應該穩定。
- `ETA` 行顯示預估剩餘小時數。
- `nvidia-smi` 的 GPU 使用率大多在 90% 以上，記憶體約 37–43 GB。
- 到 step 1500 和 3000 時出現 `Saved checkpoint at step ...`，存檔期間訓練會暫停約 0.5–2 分鐘。

不正常、需要回報的樣子：

| 現象 | 處理 |
|---|---|
| `loss=nan` 或 `inf` | `bash $W/diag.sh` 並寄回，不需要停止（它不會自己好，但 log 有用） |
| `time=` 突然變成原來的 2 倍以上並持續 | 看 `nvidia-smi` 是否有別人的程式佔用同一張 GPU；`bash $W/diag.sh` 並寄回 |
| 10 分鐘以上沒有新的 `step=` 行，GPU 使用率 0% | `bash $W/diag.sh` 並寄回 |
| `out of memory`、`Bus error`、`No space left on device` | `bash $W/diag.sh` 並寄回。`Bus error` 通常是 `/dev/shm` 太小，見附錄 A.2 |
| 機器重開機或程式消失 | 重新 `export W=...`、定義 `step`，再執行 `step P6`（或 `P6P`），它會自動續跑 |

```bash
# >>> P7
: "${W:?run first: export W=/path/to/workdir}"
echo "--- runs.txt"; tail -n 6 "$W/logs/runs.txt" 2>/dev/null || echo "(no runs yet)"
for f in $(ls -t "$W"/logs/kd_*.log "$W"/logs/taskonly_*.log 2>/dev/null | head -2); do
  echo "--- $f"
  tr '\r' '\n' < "$f" | grep -E "step=[0-9]+ loss|Saved checkpoint|Resumed|Error|out of memory" | sed -E 's/^Training:.*\]([0-9]{2}:[0-9]{2}:[0-9]{2}\.)/\1/' | tail -n 4 | cut -c1-150
  tr '\r' '\n' < "$f" | grep -E "step=[0-9]+ loss" | tail -n 1 | awk '{ if (match($0, /step=[0-9]+ loss/)) st = substr($0, RSTART + 5, RLENGTH - 10); if (match($0, /time=[0-9.]+/)) tm = substr($0, RSTART + 5, RLENGTH - 5); if (st > 0) printf "ETA %.1f h (step %d of 3000, %.2f s/step)\n", (3000 - st) * tm / 10 / 3600, st, tm / 10 }'
done
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv 2>&1
df -h "$W" | tail -1
ls -d "$W"/vla_distill/checkpoints/pi05_libero_l06*/*/[0-9]* 2>/dev/null
# <<< P7
```

---

## 11. 步驟 8：打包帶出（`P8`）

### 11.1 打包

兩個訓練都顯示 `PASS` 之後執行：

```bash
step P8
```

- 時間：2–5 分鐘（計算 16.5 GB 的 sha256）。
- 成功：最後一行是 `PACKAGE READY: .../out (N files)`，上面列出每個檔案的大小。
- 產出在 `$W/out/`：

| 內容 | 大小 |
|---|---|
| `kd_b64_s3000_step1500/model.safetensors`、`kd_b64_s3000_step3000/model.safetensors` | 各 4.11 GB |
| `taskonly_b64_s3000_step1500/model.safetensors`、`taskonly_b64_s3000_step3000/model.safetensors` | 各 4.11 GB |
| 每個資料夾內的 `assets/`（normalization 統計） | 各約 2 KB |
| `*.log`、`runs.txt`、`speed.txt`、`ovx_probe.txt` | 共幾 MB |
| `SHA256SUMS` | 幾 KB |
| 合計 | 約 16.5 GB |

`optimizer.pt`（每個約 3 GB）只用於續跑，不需要帶出。

```bash
# >>> P8
: "${W:?run first: export W=/path/to/workdir}"; cd "$W/vla_distill" || exit 1; O="$W/out"; mkdir -p "$O"
for r in pi05_libero_l06/kd_b64_s3000 pi05_libero_l06_taskonly/taskonly_b64_s3000; do for s in 1500 3000; do
  src="checkpoints/$r/$s"; dst="$O/$(basename "$r")_step$s"
  if [ ! -f "$src/model.safetensors" ]; then echo "MISSING $src"; continue; fi
  mkdir -p "$dst" && { ln -f "$src/model.safetensors" "$dst/" 2>/dev/null || cp "$src/model.safetensors" "$dst/"; } && cp -r "$src/assets" "$dst/"
done; done
cp "$W"/logs/*.log "$W"/logs/*.txt "$W"/ovx_probe.txt "$O"/ 2>/dev/null
cd "$O" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
find . -type f -printf '%s\t%p\n' | sort -k2 | awk -F'\t' '{ printf "%10.1f MB  %s\n", $1 / 1e6, $2; t += $1 } END { printf "%10.1f MB  TOTAL\n", t / 1e6 }'
echo "PACKAGE READY: $O ($(grep -c . SHA256SUMS) files)"
# <<< P8
```

### 11.2 帶出

請依院內規定把 `$W/out/` 整個資料夾帶出（外接硬碟、院內允許的傳輸服務等，方式由您決定）。我們收到後會用 `sha256sum -c SHA256SUMS` 驗證每個檔案。

- 若傳輸管道有單檔大小上限，可以先切檔（每塊 1.9 GB），我們這邊再合併：

```bash
cd "$W/out" && for f in */model.safetensors; do split -b 1900M -d "$f" "$f.part_"; done
# 合併（我們這邊做）：cat model.safetensors.part_* > model.safetensors
```

- 若只能帶出小檔案：請至少先寄回 `*.log`、`runs.txt`、`speed.txt`、`SHA256SUMS`（共幾 MB），權重另外安排。
- 若可帶出的總量有限：優先順序是兩個 `step3000`（共 8.2 GB），其次是兩個 `step1500`。
- 這些權重衍生自 Gemma，受 Gemma 授權條款約束，僅供內部研究使用。

### 11.3 可以刪除的東西

| 路徑 | 大小 | 何時可以刪 |
|---|---|---|
| `$W/vla_distill/checkpoints/pi05_libero_l06/timed50`、`.../smoke` | 約 7.1 GB、4.9 GB | `P5` 通過後 |
| `$W/cache/uv` | 約 9.6 GB | `P2A`/`P2B` 通過後（重裝環境時會需要重新下載） |
| `$W/cache/hf/xet` | 最多約 10 GB | `P3A` 通過後 |
| `$W/cache/openpi/openpi-assets/checkpoints/pi05_libero/params` | 12.4 GB（只有在雲端自行轉換 teacher 時才有） | `P3A` 通過後 |
| 各訓練的 `2999` 目錄 | 各約 7.1 GB | 該訓練 `PASS` 之後 |
| 整個 `$W` | 約 180 GB | 我們確認收到並驗證 checksum 之後 |

---

## 12. 失敗時

任何一步失敗，請執行下面這一行，並把它印出的那個檔案寄回：

```bash
bash $W/diag.sh
```

（若 `P1` 還沒通過，`diag.sh` 還不存在，請直接把畫面上的錯誤訊息和 `ovx_probe.txt` 寄回。）

常見錯誤對照：

| 訊息 | 可能原因 | 可以先試的事 |
|---|---|---|
| `curl: (6) Could not resolve host`、`(7) Failed to connect`、`(28) timed out` | 沒有對外網路或被防火牆擋 | 改走路徑 B |
| `error: Failed to build av` 或 `evdev` | 沒有照本文件的指令安裝（少了 `--no-install-package`） | 重新執行 `step P2A` |
| `ModuleNotFoundError: No module named 'pytest'` | 安裝時加了 `--no-dev` | 重新執行 `step P2A`（本文件的指令沒有 `--no-dev`） |
| `patched False` 或 `ENV_OK` 沒出現 | `transformers_replace` 沒有複製進去 | 重新執行 `step P2A` 或 `P2B` |
| `cuda False` | driver 太舊、容器沒有掛 GPU、或 `CUDA_VISIBLE_DEVICES` 設錯 | 寄回 `ovx_probe.txt` 與 `diag` 檔 |
| `CUDA out of memory` | GPU 被別人佔用，或記憶體不足 | `nvidia-smi` 確認該 GPU 是空的；不要自行改 batch |
| `Bus error`、`unable to write to file </torch_...>` | `/dev/shm` 太小 | 附錄 A.2 |
| `No space left on device`、`Disk quota exceeded` | 磁碟或配額滿了 | 依第 11.3 節刪除暫存後重跑同一步 |
| `PermissionError` 指向 `~/.cache` 或 `/home/...` | 沒有先 `source $W/env.sh`（各區塊都會自動做；手動下指令時要記得） | 用 `step` 重跑該步 |
| `Revision v2.1 ... not found, using version v2.0` | 正常的警告，不是錯誤 | 不用處理 |
| `The dataset you requested ... is in 2.0 format` | 正常的警告，不是錯誤 | 不用處理 |

---

## 附錄 A：排程器與容器

### A.1 機器使用排程器（Slurm 等）

這一節的指令未經實測，請依院內的實際規定調整 partition、帳號等參數。

- 步驟 0–3 不需要 GPU（`check_env.sh` 的 GPU 檢查除外），可以在登入節點做；若登入節點沒有 GPU，`ENV_OK` 那一步請用互動式工作執行，例如 `srun --gres=gpu:1 --pty bash`，進去後重新 `export W=...` 再 `bash $W/check_env.sh`。
- 步驟 4、5：在互動式 GPU 工作中執行 `step P4`、`step P5`。此時**不要設定 `GPU` 變數**：`P4`/`P5` 預設會使用排程器分給您的第 0 張可見 GPU。
- 步驟 6：不要用 `P6`，改用排程器送出（同樣不指定 GPU 編號）：

```bash
sbatch --job-name=vla_kd       --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00 --output="$W/logs/slurm_%j.out" --wrap "bash $W/run_one.sh kd"
sbatch --job-name=vla_taskonly --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00 --output="$W/logs/slurm_%j.out" --wrap "bash $W/run_one.sh taskonly"
```

- 若工作有時間上限而被中止，重新送出同一行即可，`run_one.sh` 會自動續跑。

### A.2 在容器內執行，或 `/dev/shm` 太小

這一節同樣未經實測。

- 資料載入的 worker 透過 `/dev/shm` 把每個 batch（約 116 MB）傳給主程式，4 個 worker 最多同時佔用約 1 GB。Docker 預設的 `/dev/shm` 只有 64 MB，會造成 `Bus error`。
- 首選：請用 `--shm-size=8g`（或 `--ipc=host`）重新啟動容器，並確認有 `--gpus all`（或對應的 Kubernetes 設定），且工作目錄掛在持久化的 volume 上。
- 無法重啟容器時的替代做法：不使用 worker。執行步驟 5、6 之前先 `export VLA_WORKERS=0`。資料載入改在主程式進行，每個 step 會多花約 2–3 秒。請在回信中註明用了這個設定。

---

## 附錄 B：為什麼 48 GB 的 GPU 可以跑 batch 64

NVIDIA OVX 伺服器的 GPU 是 48 GB 等級（目前規格是 4 或 8 張 L40S；較早的世代是 L40、A40，也都是 48 GB），不是 80 GB。原先的計畫以為需要 80 GB，那是保守的估計。重新量測並計算後的需求如下（單位 GB = 10⁹ bytes，bf16，gradient checkpointing 開啟）：

| 項目 | 記憶體 | 依據 |
|---|---|---|
| teacher 權重 | 7.47 | 量測 |
| student 權重 | 4.11 | 量測 |
| AdamW 狀態 | 2.98 | 量測（只有 7.03 億個參數真正有梯度） |
| student 訓練時的暫存（含梯度），batch 64 | 20.84 | 量測 batch 1–24，每個樣本 0.305 GB，線性外推 |
| **峰值合計（allocated）** | **35.4** | |
| 加上記憶體配置器的保留（5–15%） | 約 37–41 | |
| 48 GB GPU 的可用記憶體 | 約 48 | `nvidia-smi` 通常顯示約 46,000 MiB；實際數字以步驟 0 的探測結果為準 |

這是外推值，不是在 48 GB GPU 上的實測。步驟 5 的 `peak_reserved_GB` 就是用來確認這個估計的。

`scripts/distill_pytorch.py` 不支援 gradient accumulation，所以「batch 32 累積兩次」不是現成的選項；若 batch 64 放不下，需要我們決定替代方案，請不要自行更改。

多張 GPU：程式有多 GPU（DDP）的程式路徑，但尚未驗證，所以本文件不使用它。有兩張空 GPU 時，建議的用法是 `P6P`：兩個訓練各用一張，各自仍是單 GPU 訓練。

---

## 附錄 C：離線 bundle 的內容（路徑 B）

bundle 由我們在院外準備，內容如下。`P2B` 會先用 `SHA256SUMS.bundle` 驗證全部檔案。

| 檔案或資料夾 | 大小 | 內容 |
|---|---|---|
| `vla_distill-966da5c.tar.gz` | 1.0 MB | 程式碼 |
| `vla_env_cache.tar` | 10.28 GB | `uv` 執行檔、Python 3.11、全部 Python 套件的快取 |
| `openpi_cache.tar` | 4.3 MB | tokenizer 與 normalization 統計 |
| `pi05_libero_pytorch/` | 7.23 GB | teacher 權重（`model.safetensors`、`config.json`、`assets/`） |
| `libero_lerobot/` | 34.94 GB | 資料集（`meta/` 與 `data/`，1,693 個 parquet 檔） |
| `SHA256SUMS.bundle` | 約 200 KB | 上列所有檔案的 sha256 |
| 合計 | 約 52.5 GB | |

固定不變、可以單獨核對的 checksum：

```text
608237d052783be3f59bfe730af0d67576f3442e6576d37b7889959c2de948a0  vla_distill-966da5c.tar.gz
d9ef6bb72672509355ed19ebf08909e04e9af0ec4918a5d5ce9fa7c25aaf7ea4  pi05_libero_pytorch/model.safetensors
5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a  pi05_libero_pytorch/config.json
b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84  norm_stats.json
8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6  paligemma_tokenizer.model
```

`vla_env_cache.tar` 是在 Ubuntu 24.04（x86_64）上製作的，內含的都是通用的 Linux wheel，目標機器需要 glibc ≥ 2.28。
