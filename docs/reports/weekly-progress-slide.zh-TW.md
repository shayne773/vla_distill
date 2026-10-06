# VLA Distillation 本週進度（Pi0.5 → 6-layer student，Shallow-π）

---

## Slide 1：本週進度總覽

- 完成 OVX 雲端訓練 runbook（約 10 個指令），已寄給 mentor
- Mentor 確認雲端環境：GPU 為 L40S（48 GB）、可連外網下載、磁碟空間 ≥ 200 GB
- 修正原計畫：不需要 80 GB GPU；batch 64 預估 peak 約 35.4 GB（外推值，由 50-step 測速確認）
- 訓練內容：KD run + task-only 對照組，各 3,000 steps，單張 L40S 預估合計約 7–13 小時
- 下一步：下週四在 OVX 實際訓練 → 權重帶回 → 在筆電上做 LIBERO evaluation
- 待解決：16.5 GB 的 student 權重如何帶出院外

---

## Slide 2：Memory bandwidth

- 問題：模型每次運算在 GPU memory 搬移的資料量，對上 GPU 的 memory bandwidth，是否成為瓶頸
- Inference（筆電 RTX 5080 Laptop，實測）：每個 query teacher 搬 34.6 GB、student 搬 15.0 GB（teacher 的 43%）
- Denoising loop（10 steps）是 bandwidth-bound：每步重讀 action expert 的 weights，約佔 latency 的 17–23%
- SigLIP + VLM 的 prefix 是 compute-bound，佔其餘大部分 latency
- 模型預測與實測相符：teacher 144.7 vs 145.7 ms；student 59.3 vs 58.6–62.1 ms
- Training（L40S，依原廠規格推估）：KD step 搬 2,723 GB，memory traffic 約佔每步 55–60%；只影響訓練時間，不影響結果
- Read vs write：inference 約 3.3 : 1、training 約 1.2 : 1；實測 write 不比 read 慢，比例不改變結論

---

## Slide 3：ViT（SigLIP）

- 位置：Pi0.5 內建的 SigLIP So400m/14 vision encoder，不是獨立的模型；每個 sample 編碼 3 張 image
- Teacher 與 student 的 SigLIP 完全相同且 frozen → 不影響 teacher vs student 的比較
- 對訓練不是問題：沒有 backward、沒有 optimizer state，佔 KD step memory traffic 的 10.9%
- 限制壓縮效果：distillation 只縮小 transformer layers（佔 teacher parameters 的 66.5%），SigLIP（11.5%）與 token embedding（14.6%）等不變 → layers −67%，但 parameters 只 −44%、latency −60%
- 佔比上升：parameters 11.5% → 20.6%；inference memory traffic 14% → 32%
- 第三張 image 是全零的 placeholder，仍被完整編碼，佔 SigLIP 運算的三分之一

---

## Slide 4：訓練資料

- Dataset：Hugging Face `physical-intelligence/libero`（CC-BY-4.0）
- 規模：1,693 episodes、273,465 frames、40 tasks（LIBERO-Spatial / Object / Goal / 10）；兩個 256×256 camera、10 fps；下載 34.9 GB
- 來源：LIBERO 的人類 teleoperation demonstrations → OpenVLA 重新 render 並過濾（2,000 → 1,693 episodes）→ Physical Intelligence 轉成 LeRobot 格式
- 與 teacher（`pi05_libero`）fine-tune 時用的是同一份資料
- KD 與 task-only 使用相同資料、相同 seed → paired comparison
- 沒有 held-out task：evaluation 是同樣 40 個 tasks 的 simulator rollout（LIBERO 的標準做法）

---

## Slide 5：用 synthetic data 訓練 student 的可行性與效果

- 做法：讓 teacher 在 LIBERO simulator 中執行 task，把 rollouts 錄下來作為額外的訓練資料
- 可行性：技術上可行，但目前沒有 rollout → LeRobot 格式的記錄工具，需要先開發
- 效果有限：KD 的 teacher target 是每個 batch 線上計算，rollouts 只增加新的 states，不增加新的 teacher 指導
- 會模糊對照組：task-only student 若用 teacher 產生的資料訓練，等於對 teacher 做 behaviour cloning，KD 與 task-only 的差異變得不清楚
- 可能污染 evaluation：產生資料時必須避開所有用於測試的 initial states
- 結論：本次實驗不建議使用
- 未來用途：若 student 因偏離 demonstration 軌跡而失敗，on-policy rollouts 可補足人類 demonstration 沒有涵蓋的狀態

---

## Slide 6：未來取得更多資料的可能性

- LIBERO-90（`jesbu1/libero_90_lerobot`）：3,959 episodes、69.5 GB，格式相容；但 teacher 沒有在這些 tasks 上 fine-tune，evaluation 也不測 → 目前沒有明顯效益
- LIBERO-Plus（14,347 episodes）：是規劃中的 robustness evaluation，拿來訓練會污染測試
- `lerobot/libero`（video 版，1.9 GB）：同一批 episodes，但格式不相容（LeRobot v3.0）
- DROID、ALOHA 等其他 dataset：不同機器人，與 LIBERO 上的比較無關
- Teacher-generated rollouts：需要先開發記錄工具，適合作為後續改善 student 的手段
- 建議：本次實驗只用 `physical-intelligence/libero` 以維持 paired comparison；是否擴充資料等 student 訓練結果出來後再評估
