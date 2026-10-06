# π0.5 蒸餾實驗：VLA 壓縮／蒸餾現況調查、評估方法與實驗規劃

- 日期：2026-09-28
- 文獻範圍：截至 2026 年 9 月的 arXiv 論文、標準與法規公開資訊
- 實測資料：Phase 1（筆電 RTX 5080 Laptop 16 GB），見 https://github.com/shayne773/vla_distill/pull/1（branch `fm/vla-phase1`）
- 數字標示原則：每個文獻數字都附來源（arXiv 編號或網址）。取自二手摘要、未能逐字核對原文的內容標為「未核實」。查不到的項目直接寫「未找到」，不以推測補齊。

---

## 摘要

### 主要結論

| # | 結論 | 依據 |
|---|---|---|
| 1 | **方法維持 Shallow-π。** 在這次調查範圍內，它仍是唯一針對 π0／π0.5 做層數蒸餾、並公開程式碼的方法（icsl-Jeon/openpi，Apache-2.0；student 權重未找到公開版本）。論文數字：π0.5 由 18 層降到 6 層（VLM backbone 與 action expert 同步），LIBERO 平均 96 → 95，FLOPs 3.39T → 1.30T，H100 CUDA time 25.5 → 11.3 ms。**沒有找到在 π0.5 上明顯更好、又有程式碼的蒸餾方法，因此計畫不需要改方向。** | 2601.20262 Table 2 |
| 2 | **有兩個可以疊加的方向，列為選用延伸。** 一是去噪步數蒸餾 SnapFlow：π0.5 由 10 步降到 1 步，E2E 274 → 83 ms（A800），LIBERO 97.75 → 98.75，但未找到公開程式碼。二是訓練後量化 QuantVLA：π0.5 W4A8 LIBERO 97.1 → 97.6，程式碼為 Apache-2.0。兩者都和層數蒸餾正交。 | 2604.05656；2602.20309 |
| 3 | **「能蒸餾到多小」在 LIBERO 上看不到下限。** 0.2B、從頭訓練的 TurboVLA 就有 97.7%；0.63B 的 VLA-Adapter 有 97.3%。這反映的是 LIBERO 已經飽和，不代表壓縮沒有損失。退化只在較難或有擾動的設定中出現：π0.5-L6 在 LIBERO-Long 由 93 降到 90；π0 在 LIBERO-Plus 整體只有 53.6%；LIBERO-PRO 在泛化設定下成功率降到 0.0%。**目前沒有任何公開研究在擾動 benchmark 上評估蒸餾後的 π0.5。** | 2607.27205；2509.09372；2601.20262；2510.13626；2510.03827 |
| 4 | **產品化／認證：目前不存在專門針對 learned VLA policy 的認證制度。** 最接近的是 ISO/IEC TS 22440（功能安全與 AI 系統），仍在 CD 階段。歐盟 Machinery Regulation 2023/1230 自 2027-01-20 起適用：以 machine learning 執行安全功能的安全元件或機械，必須經第三方符合性評估。依 AI Omnibus（Regulation (EU) 2026/1744），機械類 AI 的技術要求改由 Machinery Regulation 的 delegated acts 納入，期限 2028-08-02。實務做法是讓 learned policy 不承擔安全功能，由 safety-rated controller（例如 PL d Cat 3）加上 ISO 10218-2:2025 的 PFL／SSM 包覆；runtime monitor 仍屬研究階段。 | 第 5 節 |
| 5 | **要讓 before/after 比較能支撐產品化討論，只看 LIBERO 平均成功率不夠。** 需要：paired 設計、事先定義的 non-inferiority margin、足夠的樣本數（400 個 paired episodes 只能可靠偵測約 5 pt 的下降，1,000 個約 3 pt）、擾動 benchmark、目標硬體上的延遲分佈、動作層級指標，以及失敗模式分析。 | 第 4 節檢定力表 |
| 6 | **Phase 1 已完成。** Teacher 96.0%（Wilson 95% CI 93.6–97.5），重現了論文與 openpi 的公布值。未訓練的 6 層 student 為 0.0%。每個 action chunk 的延遲 145.7 → 58.6 ms；推論記憶體 6.97 → 3.84 GiB；推論用參數 3,353M → 1,749M。另外修正了 fork 蒸餾程式的 3 個錯誤，並確認 fork 缺少論文的 attention-distillation loss。 | `results/phase1/SUMMARY.md` |

### 計畫一覽

| 項目 | 內容 |
|---|---|
| 比較模型 | teacher（18 層）、KD student（6 層）、task-only 對照（6 層，相同算力）、未訓練 student（已完成） |
| 訓練 | 3k steps × batch 64，bf16，1.5k 與 3k 各存一個 checkpoint。KD 與對照除 `loss_weight_teacher` 外設定完全相同。先使用 fork 已釋出的兩個 loss（task + teacher velocity） |
| 主要評估 | LIBERO 4 suites × 10 tasks × 25 trials，每個模型 1,000 個 paired episodes。以 exact McNemar 檢定與 paired 95% CI 判讀；主要指標另加動作層級 MSE（相對 teacher） |
| 建議加做 | LIBERO-Plus 分層子集（每模型約 700 episodes）；第二個訓練 seed |
| 運算需求 | 1 張 80 GB GPU（H100 優先），約 **4–5 H100 GPU-hours**（A100 80GB 約 8–10 小時）；磁碟 200 GB；RAM 64 GB；約 70 GB 公開資料下載 |
| 時程 | 取得算力後約 2 週：雲端 1 天，其餘在筆電上評估並撰寫報告 |

### 需要確認的事項

1. 部門雲端的 GPU 型號與記憶體（是否有 80 GB 卡）、可用時數，以及是否允許從 Hugging Face 與 GCS 下載約 70 GB 公開資料。
2. 是否有目標部署硬體（例如 Jetson Orin／Thor 或特定工控機）。若有，延遲應在該硬體上量測（第 4.3 節）。
3. 第 7.2 節的選用延伸要做哪些。我的建議是「第二個 seed」加「LIBERO-Plus 子集」優先。

---

## 1. 範圍與做法

- 目的：為 π0.5 蒸餾實驗建立可比較、可重現的 before/after 評估，並說明它和產品化／認證要求之間的距離。學生模型的絕對表現不是重點，重點是和 baseline 的對照。
- 調查方式：以 github.com/guanweifan/awesome-efficient-vla（2026-09 版，收錄 350 篇以上）與兩篇綜述（2510.17111、2510.24795）為起點，逐篇讀 arXiv 摘要或 HTML 全文取數字，再以 GitHub API 確認程式碼與授權。
- 標準與法規：ISO／IEC 官網對自動擷取回應 403，因此標準狀態引用 ISO 專案鏡像頁（iss.rs）、標準機構與法律事務所的公開說明，逐項註明來源。

## 2. VLA 壓縮／蒸餾方法調查

表中「前 → 後」均為原論文自報數字，各論文的評估條件（trials 數、硬體、計時方式）不同，不能跨列直接比較。

### 2.1 深度（層數）蒸餾與剪枝

| 方法（來源） | Teacher → Student | 壓縮 | 加速（硬體） | Benchmark 前 → 後 | 實機證據 | 程式碼／權重（授權） |
|---|---|---|---|---|---|---|
| **Shallow-π**（2601.20262） | π0.5 18+18 層 → 6+6 層（VLM + action expert） | FLOPs 3.39T → 1.30T（−62%） | CUDA time 25.5 → 11.3 ms（H100）；Jetson Thor E2E 130 → 78 ms | LIBERO 平均 96 → 95（L6）、97（L9）；Long 93 → 90（L6）。π0：95 → 94（L6）。Loss 消融（π0.5，L6）：task only 93.0／+KD 93.9／+attention 94.6；L4 為 94.2 | ALOHA（Jetson Orin）4 項任務、RB-Y1 人形（Jetson Thor）3 項，每項 5–20 次；student ≥ teacher（teacher 延遲較長）。例：Recycle 12/20 → 17/20 | 程式碼：icsl-Jeon/openpi（Apache-2.0，最後提交 2026-01-29，**不含 attention loss**）。權重：未找到 |
| CLP（2606.20246） | π0 18 層 → 12 層（3.5B → 2.7B）；GR00T-N1.5 VLM 12 → 7、DiT 16 → 8 | 參數 −23%（π0） | 211 → 152 ms（RTX 4070，1.39×） | LIBERO π0 94.6 → 93.9；GR00T 93.9 → 93.0 | 4 種機器人、10 項任務（GR00T 73.5 → 75.9%） | 專案頁 clpvla.github.io；程式碼狀態未核實 |
| Drop-Then-Recovery（2606.27755） | OpenVLA-OFT，移除一半語言模型 blocks 後微調 | 語言模型 blocks −50% | 未核實 | LIBERO 95.0 → 98.3；只保留 2 個 blocks 仍與 baseline 相當 | 未見 | s1ghhh/VLADrop（MIT） |
| VLM2VLA 冗餘分析（2606.31382） | OpenVLA、π0.5，免訓練多模組剪枝 | 參數 −12~30% | 未核實 | 約保留原表現的 90%（未做恢復微調） | 未見 | 未找到 |
| ActDistill（2511.18082） | 任一 VLA → 自衍生輕量 student（動態路由） | 運算量 −50% 以上 | 最高 1.67× | 「相當或更好」（摘要未列具體 teacher，未核實） | 未核實 | 未找到 |

### 2.2 去噪步數蒸餾（flow／diffusion action head）

| 方法（來源） | 對象 | 步數 | 加速（硬體） | Benchmark 前 → 後 | 實機 | 程式碼 |
|---|---|---|---|---|---|---|
| **SnapFlow**（2604.05656） | π0.5（另測 SmolVLA） | 10 NFE → 1 NFE（progressive self-distillation，無外部 teacher） | 去噪 9.6×；E2E 274 → 83 ms（3.3×，A800-80G）；SmolVLA E2E 3.56× | LIBERO 97.75 → 98.75（**每 task 只有 10 episodes，共 400**） | 無（論文自述需要實機驗證） | 未找到公開程式碼。訓練約 12 h、1 張 GPU（VLM 凍結，30k steps，batch 4） |
| One-step VLA「Let It Be Simple」（2606.05737） | 未於摘要列出 | 1 步 | 未核實 | LIBERO-Long 95.6% | 未核實 | 未找到 |
| ReactVLA（2606.14255） | 新模型（improved MeanFlow，1 至數步） | 1 至數步 | 「比主流 VLA 快 4×」；實機延遲 < 38.6 ms | LIBERO、RoboIMI；自述優於 SmolVLA、π0 | 有 | 專案頁（程式碼未核實） |
| Temporal Redundancy（2607.12287） | VLA（2 步 schedule + 增量 token 更新） | 2 步 | > 2× | 最高 98%（LIBERO、RoboTwin） | 有 | 「將釋出」 |

補充：CrossVLA（2605.21854）量測到 π0.5 的 `sample_actions` 延遲中去噪迴圈占 78.6%。該研究也指出，prefix K/V cache 的加速上限約 21%，且 chunk／token 層級的 cache 策略使成功率降到 0–80%。這說明免訓練的 cache 方法未必安全。

### 2.3 跨架構蒸餾到小型 student

| 方法（來源） | Teacher → Student | 壓縮比 | 速度（硬體） | Benchmark 前 → 後（每格 200 trials） | 實機 | 程式碼 |
|---|---|---|---|---|---|---|
| VLA-AD（2605.16241） | OpenVLA-7B → 158M（Long-CLIP + LoRA + MLP head） | 44× | 12.5 Hz（RTX 4090，3.28×） | object 62.0 → 62.5；spatial 79.0 → 79.5；goal 81.0 → 79.5（只測 3 個 suite） | 無 | 未找到 |
| 同上 | π0.5 → 158M | 約 25×（以 4B 推算） | 13.2 Hz（3.45×） | object 90.0 → 96.5；spatial 87.0 → 93.0；goal 94.5 → 94.0。**Teacher 數字明顯低於官方 π0.5 LIBERO（約 96–98）**，「student 勝過 teacher」須保守解讀 | 無 | 未找到 |
| CoTinyVLA（2607.25487） | 35B 推理模型（CoT 蒸餾）→ 0.9B（Qwen3.5-0.8B） | — | 推論記憶體 2.25 GiB | LIBERO-Plus 四個擾動 suite 均優於最強的 7B baseline | 未核實 | BrainJellyPie/CoTinyVLA（未標示授權） |

### 2.4 量化與低位元 VLA

| 方法（來源） | 對象 | 位元 | 記憶體／速度 | LIBERO 前 → 後 | 實機 | 程式碼（授權） |
|---|---|---|---|---|---|---|
| **QuantVLA**（2602.20309，CVPR'26） | π0.5、GR00T N1.5、OpenVLA | W4A8（π0.5）；W4A4 | π0.5 量化部分 4.27 → 1.28 GB（A100） | π0.5 97.1 → 97.6（W4A8）、95.3（W4A4）；GR00T 86.5 → 88.0 | 未見 | AIoT-MLSys-Lab/QuantVLA（Apache-2.0） |
| HoloQ-VLA（2605.28803；awesome list 稱 Ω-QVLA） | π0.5、GR00T-N1.5 | W4A4（含 action head） | 靜態記憶體 −74.2% | π0.5 97.1 → 98.0；GR00T 87.0 → 87.8 | 有（摘要述及） | 未找到 |
| ActQuant（2605.24011） | π0.5、OpenVLA-OFT | ≤ 3 bpw；2.5 bpw | backbone 14.3 → 2.7 GB | π0.5 相對保留 94.8%；OpenVLA-OFT 95.0%（2.5 bpw：90.1%） | UR3 實機成功率維持 | OmniModel.cpp（未核實） |
| BitVLA（2506.07530） | 原生 1-bit（BitNet b1.58 2B4T） | 權重三值 {−1, 0, 1} | 記憶體 11×、E2E 4.4× | 與 OpenVLA-OFT 相當 | 有 | ustcwhy/BitVLA（MIT）；HF `lxsy/bitvla-bf16` |

### 2.5 Token pruning、caching 與 runtime 最佳化

| 方法（來源） | 對象 | 做法 | 加速（硬體） | 前 → 後 | 實機 | 程式碼（授權） |
|---|---|---|---|---|---|---|
| EfficientVLA（2506.10100） | CogACT | 免訓練：層剪枝 + 視覺 token 篩選 + diffusion 特徵 cache | 1.93×；FLOPs 降至 28.9% | SIMPLER −0.6 pt | 未見 | 未核實 |
| VLA-Cache（2502.02175，NeurIPS 2025） | OpenVLA、OpenVLA-OFT、CogACT | 相鄰影格 token 的 KV 重用 | OpenVLA 51.91 → 31.83 ms（RTX 4090） | OpenVLA 84.4 → 83.8；OFT 96.8 → 97.4 | Kinova Jaco2 82.1 → 84.6% | siyuhsu/vla-cache（Apache-2.0） |
| FlashVLA（2505.21200） | OpenVLA | token 剪枝 + 動作重用 | 82.7 → 52.9 ms（H100）；FLOPs −55.7% | 74.4 → 73.7 | 未見 | 未找到（z-lab/flashvla 是另一個同名專案） |
| LightVLA（2509.12594） | 未於摘要列出 | 可微分 token 剪枝 | FLOPs −59.1%；延遲 −38.2% | +2.6 pt；Shallow-π 表中列為平均 97、22.0 ms | 未核實 | 專案頁 |
| Realtime-VLA（2510.26742） | π0 級模型，**權重不變** | CUDA graphs、kernel fusion、串流 | 2 視角：naive PyTorch 106.5 ms／openpi JAX 53.7 ms → 27.3 ms（RTX 4090） | 數學等價，不影響準確度 | 接住掉落筆的任務 100% | Dexmal/realtime-vla（MIT） |

### 2.6 小型原生 VLA（非蒸餾，作為大小參考點）

| 模型（來源） | 參數 | LIBERO 平均 | 其他 | 程式碼（授權） |
|---|---|---|---|---|
| SmolVLA（2506.01844） | 450M | 87.3（同表 π0 為 86.0） | SO-100 實機 78.3% | huggingface/lerobot（Apache-2.0） |
| VLA-Adapter（2509.09372） | 約 627M（0.5B + 97M） | 97.3（Pro 版 98.5） | CALVIN ABC→D 4.42；219.2 Hz（H100） | OpenHelix-Team/VLA-Adapter（MIT），HF 有權重 |
| Evo-1（2511.04555） | 0.77B | 94.8 | 實機 78% | MINT-SJTU/Evo-1（MIT） |
| TurboVLA（2607.27205） | 0.2B | 97.7 | 31.2 ms、0.9 GB VRAM（RTX 4090） | H-EmbodVis/TurboVLA（Apache-2.0） |

### 2.7 世界模型表徵蒸餾（world-model folding）

| 方法（來源） | Teacher → Student | 壓縮 | 加速（硬體） | Benchmark | 實機 | 程式碼／權重 |
|---|---|---|---|---|---|---|
| Enfold（2607.26657v3） | 影片世界模型 Cosmos-Predict 2.5（2B）→ 由當前影像與指令推論的預測表徵（DINOv3 ViT-H+/16 encoder）＋ flow-matching action head | 無：整體約 3B，與 π0.5（推論用 3,353M）同級。省下的是推論時不再執行影片生成 | 每個 action chunk 134 ms（Flash 版 49 ms），相對 Fast-WAM 493 ms 為 3.7×／10.1×（硬體未核實） | LIBERO 97.8（Flash 97.5；Fast-WAM 97.6）；RoboTwin 2.0 平均 91.77；消融：只用 action 監督 94.9、多層 generator states 監督 97.8 | AgileX 雙臂 4 項任務：分佈內 89.7%（Fast-WAM 77.8%）、分佈外 76.6%（70.0%）；論文自述分佈偏移下落後 π0.5 6.6 pt | github.com/zwl666666/enfold（有程式碼；未見權重釋出說明） |

- **做法：** 訓練時，讓 encoder 的表徵去預測世界模型在處理真實未來影格時的多層內部狀態（G2R），再用這個表徵回頭作為世界模型生成的條件（R2G）。Action head 只接收 stop-gradient 後的表徵，任務 loss 不直接改變 encoder。部署時只執行 encoder 與 action head，不生成影片。
- **訓練與推論流程（逐步說明）：**
  1. 訓練資料是錄好的機器人資料，因此每個樣本都有：當前影像與指令、真實的未來影片片段、真實動作。
  2. 把真實未來片段輸入世界模型（teacher-forced），記錄它在多個層的內部 hidden states，作為監督目標。
  3. Encoder 只看當前影像與指令，輸出表徵 z；再由一個訓練用的小 head（Fω）從 z 預測上一步的 hidden states（G2R loss，更新 encoder 與 Fω）。目標是世界模型「看著真實未來時的內部狀態」，不是未來影格的像素；論文消融中改成預測像素只有 96.0%（多層 hidden states 為 97.8%）。
  4. 世界模型以 stop-gradient 後的 z 為條件生成未來（R2G loss，只更新世界模型），讓 z 保持對預測未來有用。
  5. Action head 以 stop-gradient 後的 z 預測動作（task loss，只更新 action head）。動作 loss 不會回傳到 encoder，encoder 只由世界模型的監督塑形。
  6. 推論時世界模型與 Fω 都不執行：當前影像與指令 → encoder → z → action head → 動作。
- **性質：** 這是「特徵／表徵層級的蒸餾」（teacher 為世界模型，student 為 encoder），把世界模型對未來的知識摺疊進 VLA 的表徵中；它不是把既有 policy 壓縮成較小 policy 的模型壓縮。
- **與 π0.5 蒸餾的關係：不適用。** Teacher 是影片世界模型而不是 policy，π0.5 的能力沒有被轉移，得到的是另一種架構的新模型，而不是較小的 π0.5。模型大小與 π0.5 相當，加速是相對於需要生成影片的 world-action model，而不是相對於 π0.5 這類直接輸出動作的 policy。訓練還需要執行 2B 的影片模型來產生監督目標，成本明顯高於本計畫。它無法回答本實驗的核心問題：同一個 π0.5 縮小後損失多少。
- **可參考之處：** 以「未來預測」作為額外監督訊號來強化小型 student 的表徵，是可能的研究延伸，但未在 π0.5 上驗證，屬於新研究題目，不列入本計畫。

### 2.8 對本實驗的意涵

- **方法選擇不變。** π0／π0.5 的層數蒸餾中，只有 Shallow-π 提供程式碼。CLP 使用剪枝加微調、不做蒸餾，只剪到 12 層。VLM2VLA 是免訓練剪枝，約只保留 90% 的表現。
- **最值得納入的延伸有兩個。** (a) 量化：免訓練、有 Apache-2.0 程式碼，只需筆電。(b) 步數蒸餾：論文報告的加速最大，但需要自行實作。兩者都放在選用項目（第 7.2 節）。
- **Cache 類方法有降低成功率的反例**（2605.21854），產品化時必須個別驗證，不宜預設為無損。

## 3. 蒸餾可以多小，而不失動作正確性？

### 3.1 各方法家族的證據

| 家族 | 規模／算力 vs 保留成功率（LIBERO，除非另註） | 證據強度與缺口 |
|---|---|---|
| 層數蒸餾（π0.5） | 18 → 9 層：96 → 97；18 → 6 層（FLOPs −62%，CUDA time 2.3×）：96 → 95，Long 93 → 90；18 → 4 層：94.2（消融）（2601.20262） | 單一論文、單一 benchmark；未報告 trials 數與 CI；未測擾動 benchmark；實機每項 5–20 次 |
| 剪枝＋微調（不蒸餾） | π0 18 → 12 層（參數 −23%，1.39×）：94.6 → 93.9（2606.20246） | 壓縮幅度小於層數蒸餾 |
| 免訓練剪枝 | 參數 −12~30%：約保留 90%（2606.31382） | 沒有恢復訓練時損失明顯 |
| 步數蒸餾 | 10 → 1 NFE（E2E 3.3×）：97.75 → 98.75（2604.05656） | n = 400；無實機；無程式碼 |
| 量化 | W4A8：97.1 → 97.6；W4A4：95.3 或 98.0（兩篇論文結果不同）；≤ 3 bpw：相對保留 94.8%（2602.20309、2605.28803、2605.24011） | 同為 π0.5 W4A4，兩種方法相差 2.7 pt，顯示方法細節影響大 |
| 跨架構小 student | 158M（比 OpenVLA-7B 小 44×）：與 teacher 相當，但 teacher 本身偏弱（2605.16241） | 只測 3 個 suite；teacher 未達官方水準 |
| 原生小模型 | 0.2B–0.77B：94.8–97.7，不需要 teacher（2607.27205、2509.09372、2511.04555） | 顯示 LIBERO 無法區分大小模型 |

### 3.2 為什麼 LIBERO 飽和會掩蓋退化

1. **天花板效應。** Teacher 在 96–98% 時，只剩 2–4 pt 的下降空間。以 400 個 episodes 計，單一模型的 Wilson CI 為 ±2.0 pt（第 4.2 節）。1–2 pt 的差異在常見評估規模下無法與雜訊區分。
2. **訓練與測試分佈相同。** LIBERO 的評估沿用訓練資料的同一組任務與場景，初始狀態來自固定清單。LIBERO-PRO 指出模型傾向「背下動作序列與環境配置」：在泛化設定（換物件、改指令、改初始狀態或環境）下，成功率由 90% 以上降到 0.0%（2510.03827）。LIBERO-Plus 在 7 個擾動維度下，π0 整體只有 53.6%、OpenVLA-OFT 為 69.6%（2510.13626）。**同一個 LIBERO 分數，在擾動下可以有很大差異。**
3. **二元成功率不反映動作品質。** 勉強成功、動作抖動、步數變長、夾爪反覆開合，都記為成功。Phase 1 的動作層級指標在 LIBERO 分數之外提供了連續訊號（第 6 節）。
4. **退化集中在長時程任務。** Shallow-π 的平均只降 1 pt，但降幅集中在 Long suite（π0.5 93 → 90；π0 92 → 85）。只看平均會漏掉。
5. **評估規模小且未配對。** 各論文多用每 task 10–50 次、沒有報告 CI 或配對檢定（SnapFlow 每 task 10 次；Shallow-π 未報告次數）。

### 3.3 證據薄弱之處

- 沒有公開研究在 LIBERO-Plus／LIBERO-PRO／SimplerEnv 上評估**蒸餾後的 π0.5**。CoTinyVLA 用了 LIBERO-Plus，但它是 CoT 蒸餾，不是 π0.5 policy 蒸餾。
- 實機證據每項任務只有 5–20 次，且 teacher 在嵌入式平台上受延遲拖累。Shallow-π 的實機結果比較的是「部署後的整體系統」，不是「動作正確性」本身。
- 多種壓縮的疊加（層數、步數、量化）只有理論推算（SnapFlow Table 4 推估 E2E 5–6×），沒有實測。
- 缺乏與安全相關的指標，例如接觸力、碰撞與工作空間越界。
- 除了 Phase 1 重現 teacher 96.0% 之外，沒有找到獨立重現 Shallow-π 的結果。

**小結：** 就 π0.5 而言，證據最充分的點是 6 層：深度約為 1/3，FLOPs −62%，以 30k steps 加 KD 訓練後，LIBERO 損失 ≤ 1–2 pt，損失集中在長時程任務。更小的設定（L4、跨架構 158M、1-bit）要嘛只有單一消融數字，要嘛以較弱的 teacher 為比較基準。至於「在擾動與實機條件下是否仍無損」，目前沒有答案，這正是本實驗可以補上的部分。

## 4. 蒸餾前後的 benchmark 比較方法

### 4.1 LIBERO 以外的 benchmark

| Benchmark（來源） | 內容 | 現況／授權 | 對本實驗的適用性 |
|---|---|---|---|
| LIBERO（4 suites） | 40 個模擬任務 | 已飽和 | 主要比較，但須搭配下列項目 |
| **LIBERO-Plus**（2510.13626） | 10,030 個任務，7 個擾動維度（物件配置、相機視角、機器人初始狀態、語言、光線、背景、感測器雜訊） | sylvestf/LIBERO-plus（GitHub 未偵測到 LICENSE 檔），2026-01 仍更新 | **建議**：與 LIBERO 同 API、同訓練資料，可直接評估 π0.5-LIBERO checkpoint（整合須先驗證） |
| LIBERO-PRO（2510.03827） | 物件、初始狀態、指令、環境 4 種擾動 | Zxy-MLlab/LIBERO-PRO（MIT），2026-05 修訂 | LIBERO-Plus 的備案 |
| SimplerEnv（2405.05941） | Google Robot／WidowX，與實機結果具相關性的模擬評估 | MIT | 需要以 Bridge／Fractal 訓練的模型，**不適用** π0.5-LIBERO |
| RoboCasa365（2603.04356，ICLR 2026） | 365 項廚房任務、2,500 個場景 | 支援 π0、GR00T | 需先微調 teacher，成本高 |
| CALVIN（arXiv 2112.03227） | 長時程語言指令串接（ABC→D） | mees/calvin（MIT） | 需先微調 teacher |
| RoboTwin 2.0（2506.18088） | 50 項雙臂任務，5 個維度的 domain randomization | MIT | 需先微調 teacher |
| 實機協定 | RoboArena（2506.18123）：跨機構雙盲 pairwise；TRI LBM（2507.05331）：盲測隨機試驗；最佳實務見 2409.09491 | — | 有目標機器人時採用：雙盲、隨機順序、同初始條件 |

### 4.2 配對設計與統計

- **配對。** 每個模型使用相同的 task、初始狀態、env seed，以及每次 query 的 flow-matching noise seed（Phase 1 已實作）。比較 teacher 與 student 在同一個 episode 上的結果，只有不一致的 episodes 會影響檢定。
- **區間與檢定。** 單一模型用 Wilson 95% CI；兩模型比較用 exact McNemar 檢定，並報告成功率差的 paired 95% CI。
- **Non-inferiority。** 事先定義 margin δ，當 (teacher − student) 的 95% 信賴上界 < δ 時判定不劣於 teacher。研究用途建議 δ = 3–5 pt。產品化時 δ 應由應用端的風險評估決定，不應由模型端決定。
- **序貫檢定。** 資源有限時可採用 STEP（2503.10966，試驗數最多省 32%），或 anytime-valid 方法（2603.13616，可用部分完成度等非二元指標，最多省 70%）。

**表 4-1　單一模型的 Wilson 95% CI（觀察值 96%）**

| episodes | 100 | 400 | 1,000 | 2,000 |
|---|---|---|---|---|
| 95% CI | 90.2–98.4（±4.1） | 93.6–97.5（±2.0） | 94.6–97.0（±1.2） | 95.0–96.8（±0.9） |

**表 4-2　偵測成功率下降所需的 episodes（基準 96%，α = 0.05 雙尾，power 80%）**

| 下降幅度 | 非配對（每模型） | 配對 McNemar，p01 = 0 | p01 = 1% | p01 = 2% |
|---|---|---|---|---|
| 1 pt | 6,745 | 783 | 2,353 | 3,923 |
| 2 pt | 1,863 | 391 | 783 | 1,175 |
| 3 pt | 906 | 260 | 434 | 609 |
| 5 pt | 381 | 155 | 218 | 281 |
| 10 pt | 128 | 77 | 92 | 108 |

p01 是「student 成功、teacher 失敗」的 episode 比例，代表兩者失敗集合不重疊的程度。配對設計在 p01 小時最省樣本。

**表 4-3　固定 paired episodes 數時的檢定力（exact McNemar，每格模擬 4,000 次）**

| paired episodes | p01 | 下降 1 pt | 2 pt | 3 pt | 5 pt |
|---|---|---|---|---|---|
| 400（Phase 1 規模） | 1% | 13% | 44% | 74% | 97% |
| 400 | 2% | 10% | 28% | 55% | 90% |
| **1,000（每 task 25 次）** | 1% | 36% | 87% | 99% | 100% |
| 1,000 | 2% | 25% | 70% | 95% | 100% |
| 2,000（每 task 50 次） | 1% | 70% | 99% | 100% | 100% |
| 2,000 | 2% | 46% | 96% | 100% | 100% |

解讀：
- 400 個 paired episodes 只能可靠偵測約 5 pt 的下降。
- 1,000 個可偵測 3 pt。
- 2 pt 需要約 2,000 個。
- 1 pt 的差異（Shallow-π 消融中 KD 與 task-only 的差距約 0.9 pt）在實際可行的模擬預算內無法可靠偵測。因此需要更敏感的動作層級指標（第 4.4 節）。

計算方法見附錄 A。

### 4.3 延遲與控制頻率

同一個 π0.5 在不同論文中的延遲：
- 25.5 ms：H100，只計 CUDA time（2601.20262）
- 274 ms：A800，E2E（2604.05656）
- 145.7 ms：本實驗筆電，E2E chunk，含 torch.compile

π0 則為 27.3 ms（RTX 4090 最佳化後）與 106.5 ms（naive PyTorch）（2510.26742）。**跨論文的毫秒數不可比較。** 建議的量測協定（Phase 1 已採用前四項）：

1. 在目標硬體上以 batch 1 量測；先暖機 20 次，再計時 200 次；每次呼叫後做 CUDA synchronize。
2. 報告 p50／p95（產品化另加 p99 與最大值），並分開報告「模型呼叫」與含前處理的「E2E」。
3. 固定精度（bf16）、去噪步數（10）、action chunk 長度（10），以及 replan 間隔（每 5 步）。
4. 以 teacher／student 的延遲**比值**跨硬體比較。
5. 同步執行時，每個 replan 週期 = 推論延遲 + 執行 k 個動作的時間，因此延遲會直接降低有效的反應頻率。延遲本身也會影響成功率：Shallow-π 實機中 teacher 因延遲較長而表現較差（2601.20262 Table 3，30 Hz 控制迴路、每次推論執行 7 個動作）。

### 4.4 動作層級指標

| 指標 | 用途 | Phase 1 |
|---|---|---|
| Action chunk MSE（相對 teacher，held-out 影格） | 對小差異最敏感的連續指標 | 已有（457 frames，bootstrap CI） |
| Teacher 自身噪聲底線（換 noise seed） | 把 MSE 標準化為「幾倍於 teacher 自身變異」 | 0.0086 |
| 首個動作 MSE | 閉迴路中實際執行的部分 | 已有 |
| 夾爪開合方向一致率 | 離散決策的正確性 | 已有 |
| 各維度最大偏差（L∞）、平滑度（jerk）、夾爪抖動次數 | 產品化關心的最壞情況與動作品質 | 建議新增 |
| 成功 episode 的步數 | 效率退化 | 已有（mean steps） |

### 4.5 失敗模式分析

- 列出 teacher 與 student 結果不一致的 episodes 與各 task 的差異表。Phase 1 的 teacher 失敗集中在 11 個 task，其中 libero_10「put both moka pots on the stove」為 6/10。
- 將失敗分類為：未抓到、抓錯物件（語言對應錯誤）、中途掉落、停滯或逾時、碰撞。對不一致的 episodes 錄影並人工標註。
- 擾動 benchmark 依擾動維度拆開報告。例如 LIBERO-Plus 指出，模型對相機視角與機器人初始狀態特別敏感，成功率可由 95% 降到 30% 以下（2510.13626）。

## 5. 產品化與認證

### 5.1 相關標準與法規（2026-09 現況）

| 標準／法規（範圍） | 現況 | 與 learned policy 的關係 | 來源 |
|---|---|---|---|
| ISO 10218-1:2025／-2:2025（工業機器人／應用與工作單元） | 2025-02 發布；EN 版於 2026-09-07 刊登 OJ，成為 Machinery Directive 的調和標準（未以 EUR-Lex 核實） | 依安全功能指定預設 PL；新增 Class 1／2 分級；協作要求（PFL、SSM、hand guiding）併入 -2。安全功能須由 safety-rated 部件實現 | ibf-solutions |
| ISO/TS 15066:2016（協作） | 主要內容已併入 ISO 10218-2:2025 | 接觸力與壓力上限，是限制 policy 物理後果的主要依據 | ibf-solutions |
| ISO 13482（個人照護 → 服務機器人） | 2014 版有效；ISO/FDIS 13482「Safety requirements for service robots」於 2026-09-15 進入 FDIS 投票（stage 50.20） | 依公開說明，修訂版涵蓋「軟體與 AI 對安全的影響」（未取得全文，未核實） | iss.rs（專案 83498）；Intertek |
| ISO 25785-1（動態穩定行動機器人，含人形） | ISO/CD，stage 30.60（2026-07-08） | 人形斷電即跌倒，fail-passive 假設不成立（2608.02809） | iss.rs（專案 91469） |
| IEC 61508（通用功能安全，SIL） | 第 2 版（2010）有效；第 3 版在 CDV，預估 2027 年初（未核實） | 以傳統軟硬體為前提 | 61508.org |
| ISO 13849-1:2023（機械控制部件，PL） | 第 4 版，2023-04 | 含軟體要求；受認證的是安全控制器，不是 policy | iso.org |
| UL 3300:2024（SCIEE 服務機器人） | 第 1 版，2024-05-14 | 排除工業用途；以產品安全為主 | ANSI webstore |
| ISO/IEC TR 5469:2024（功能安全與 AI） | 已發布的**技術報告**（資訊性，不能據以認證） | 依 AI 技術類別與使用層級描述風險；涵蓋「以非 AI 安全功能保護 AI 控制的設備」 | iso.org |
| **ISO/IEC TS 22440-1/-2/-3**（功能安全與 AI：要求、指引、範例） | CD，stage 30.60（2026-04-11），尚未發布 | **第一份針對 AI 用於安全相關系統訂定要求的國際文件** | iss.rs（專案 89535） |
| **EU Machinery Regulation 2023/1230** | 2027-01-20 起適用；業界延後 AI 條款的請求已於 2026-03 撤回 | Annex I Part A 第 5、6 點：以 machine learning 執行安全功能、具「完全或部分自我演化行為」的安全元件或機械，**須經 Notified Body 第三方評估**。Annex III 1.2.1 要求宣告控制系統的預期行為、限制與自主程度（二手摘要，EUR-Lex 原文未能擷取） | nemko；sopx；ibf-solutions |
| **EU AI Act 2024/1689 + AI Omnibus（Regulation (EU) 2026/1744）** | Omnibus 2026-07-27 生效；Annex III 高風險義務延至 2027-12-02，Annex I 產品延至 2028-08-02 | 機械移至 Annex I Section B：機械類 AI 的技術要求改經 Machinery Regulation 的 delegated acts 納入（期限 2028-08-02），在此之前可援用 AI Act 的調和標準 | EC；Eurogip；Gibson Dunn |
| ISO/IEC 42001:2023（AI 管理系統） | 2023-12 發布，可認證 | 認證的是**組織的 AI 管理流程**，不是模型行為 | ANAB |
| ISO/PAS 8800:2024（道路車輛 AI 安全，參考） | 2024-12 | 延伸 ISO 26262／21448，是較成熟的 AI 安全論證範本 | UL Solutions |
| 台灣《人工智慧基本法》 | 2025-12-23 三讀；2026-01-14 公布施行 | 框架法；未查到其下針對機器人或 learned policy 的認證制度 | 數位部；理律 |

### 5.2 Learned VLA policy 的認證現況

- **目前沒有任何專門針對 learned VLA policy（或一般 learned robot policy）的認證制度或可認證標準。** 我查到的都是：通用機器人安全標準（以傳統控制為前提）、AI 管理系統（認證組織，不認證模型）、仍在草案階段的功能安全 AI 規範（ISO/IEC TS 22440），以及法規層級的要求（Machinery Regulation 的第三方評估）。
- 從法規面來看，**如果 learned policy 被用來執行安全功能**，在歐盟就會落入 Machinery Regulation Annex I Part A 的第三方評估。但我找不到目前已可用來證明這類元件符合要求的調和標準。這是實務上把 policy 排除在安全功能之外的主要原因。
- 已知與未知：
  - 已知：法規時程、標準狀態，以及「safety-rated 外層」的做法可行（下表）。
  - 未知：TS 22440 定稿後的具體要求；Machinery Regulation delegated acts 的內容；ISO 13482 修訂版中 AI 條款的實際內容（未取得全文）；模型更新（例如換成蒸餾版）在法規上是否構成需要重新評估的變更（未查到明確規定，未核實）。

### 5.3 業界目前的補足方式

| 層級 | 做法 | 成熟度 | 來源 |
|---|---|---|---|
| 安全功能與 policy 分離 | Learned policy 只負責「任務表現」。速度、力、工作空間、保護性停止由 safety-rated controller 執行，例如 UR e-Series 有 17 項安全功能，經 TÜV NORD 依 EN ISO 13849-1 認證為 Cat.3 PL d，並符合 ISO 10218-1 | 可認證，已普遍採用 | universal-robots.com（TÜV NORD 證書） |
| 協作限制 | PFL（力與壓力上限）、SSM（速度與距離監控），依 ISO 10218-2:2025（原 ISO/TS 15066） | 可認證 | ibf-solutions |
| 失敗偵測 monitor | SAFE：從 VLA 內部特徵預測失敗，以 conformal prediction 設定門檻（OpenVLA、π0、π0-FAST） | 研究階段 | 2506.09937 |
| 安全濾波器 | 以 control barrier function 的 QP 在控制頻率上過濾 VLA 動作（2606.09749 attention-guided；2609.22462 全身安全） | 研究階段，未認證 | 2606.09749；2609.22462 |
| 驗證器的取捨 | 約 150 種 policy 驗證方法中，沒有「便宜又可信」的檢查器（"no free checker"） | 綜述 | 2609.09250 |
| 組織流程 | ISO/IEC 42001 AIMS：資料、訓練、評估與變更管理的可稽核流程 | 可認證（組織層級） | ANAB |
| 人形機器人 | 斷電會使雙足機器人跌倒，傳統「斷電即安全」的論證不成立。作者未宣稱端到端達到 PL e／SIL 3，而是把不可認證的部分侷限在平衡控制介面 | 研究階段 | 2608.02809 |

### 5.4 對本實驗的意涵

- 在「safety-rated 外層包覆 policy」的架構下，蒸餾後的 student 不承擔安全功能。產品化要證明的是**任務表現不劣於原模型**，以及**行為仍落在外層限制之內**，而不是「模型本身通過認證」。
- 本實驗能提供的「產品化等級」證據：
  1. 事先定義 margin 的 paired non-inferiority 檢定。
  2. 擾動 benchmark 下的表現。
  3. 目標硬體上的延遲分佈（p99、最大值）。
  4. 相對 teacher 的動作偏差上界（L∞）與平滑度。
  5. 失敗分類。
  6. 可重現性：固定所有 seed，並記錄設定與 checkpoint 雜湊。
- 本實驗無法提供的：任何形式的認證，以及實機安全指標（接觸力、碰撞）。後者需要實機與 safety-rated 外層才能量測。

## 6. Phase 1 結果：計畫的證據基礎

來源：`fm/vla-phase1` 的 `results/phase1/SUMMARY.md`、`README.md`（實驗紀錄第 58–201 行）、`results/phase1/libero/*/summary.json`、`results/phase1/bench/*.json`。

| 項目 | Teacher `teacher_l18`（π0.5 LIBERO，18+18 層） | 未訓練 student `student_l06_init`（6+6 層） |
|---|---|---|
| LIBERO 成功率（400 個 paired episodes） | **96.0%**（93.6–97.5） | **0.0%**（0.0–1.0） |
| spatial／object／goal／libero_10 | 98／97／96／93 | 0／0／0／0（全部跑到步數上限） |
| Exact McNemar | 只有 teacher 成功 384、只有 student 成功 0，p = 5.1e-116 | |
| 延遲 p50／p95（每個 action chunk，筆電） | 145.7／155.5 ms | 58.6／61.0 ms（2.5×） |
| 推論 GPU 記憶體峰值（allocated） | 6.97 GiB | 3.84 GiB |
| 參數（總計／推論用） | 3,617M／3,353M | 2,012M／1,749M |
| Transformer 層參數（VLM／action expert） | 1,982M／430M | 661M／147M |
| 動作 MSE vs teacher（457 個 held-out frames） | 0（teacher 自身噪聲底線 0.0086） | 0.181（0.167–0.195），為底線的 21 倍 |
| 夾爪方向一致率 | 100%（換 seed：98.6%） | 71.1% |

- **重現性：** teacher 的 96.0% 與論文 Table 2（96）和 openpi 公布值（96.85，每 task 50 次）一致，確認 PyTorch 路徑與評估流程正確。
- **只減層、不訓練，policy 完全失效**（96% → 0%）。student 的能力必須全部來自訓練，這也是 Phase 2 要量測的。
- **總參數只降 44%，延遲卻降 60%。** 原因是 SigLIP（415M）與 token embedding（527M，查表，幾乎不耗運算）都沒有縮減。
- **修正了 fork 的三個錯誤：**
  1. Backbone 的層映射沒有生效：`startswith` 的第二個參數誤傳成 bool，導致 VLM 保留第 0–5 層，而不是 [0, 3, 7, 10, 14, 17]。修正後初始 MSE 由 0.266 降到 0.181，夾爪一致率由 49% 升到 71%。
  2. Teacher 與 student 看到的影像增強不同。
  3. `loss_weight_*` 設定沒有被套用。修正後 `loss_weight_teacher=0` 即為 task-only 對照。
- **Fork 缺少論文的 attention-distillation loss**：只有 task loss 與 teacher velocity loss。
- **訓練可行性（筆電）：** VLM 可訓練時（1.07B trainable），batch 2 在第一次 AdamW 更新時 OOM（峰值 13.5 GB）。VLM 凍結時（147M trainable），batch 8 可以正常訓練（2.3 s/step，峰值 15.0 GB），存出的 checkpoint 也能重新載入並推論。**因此完整的 Phase 2 需要雲端 GPU。**

## 7. 實驗規劃與需求

### 7.1 目標與假設

目標：在相同算力下，量化「蒸餾（KD）」相對於「只用任務 loss 訓練同一個淺層模型」的效果，並以產品化角度可用的方式，報告 student 相對 teacher 的損失。

| 假設 | 內容 | 主要判讀指標 |
|---|---|---|
| H1 | 3k steps 的 KD student 可以恢復大部分 teacher 表現（目標 ≥ 90%；論文以 30k steps 達到 95） | LIBERO 成功率與 paired CI |
| H2 | 相同算力下，KD student 的動作比 task-only 更接近 teacher | **動作 MSE vs teacher（主要）**；成功率差（次要，1 pt 級差異可能偵測不到） |
| H3 | 訓練不改變延遲與記憶體：student 延遲維持約 2.5× 優勢 | bench 的 p50／p95 |
| H4 | 在擾動下，student 的退化大於 teacher（壓縮損及泛化） | LIBERO-Plus 子集各維度的 paired 差異 |

主要與次要指標、non-inferiority margin（研究用 δ = 5 pt）都在訓練前定案，避免事後挑選。

### 7.2 比較模型

| ID | 內容 | 訓練 | 必要性 | 額外成本 |
|---|---|---|---|---|
| `teacher_l18` | π0.5 LIBERO 官方 checkpoint | — | 必要 | 已有 400 episodes，補 600 episodes（筆電約 1.5 h） |
| `student_l06_init` | 層映射初始化，未訓練 | — | 必要（已完成） | 0 |
| `student_l06_kd` | task + teacher velocity loss | 3k steps × 64 | 必要 | 約 1.7–2.2 H100-h |
| `student_l06_taskonly` | 同上，`loss_weight_teacher=0` | 3k steps × 64 | 必要 | 約 1.1–1.4 H100-h |
| 上兩者的 1.5k checkpoint | 學習曲線的第二個點 | — | 必要 | 0（僅評估時間） |
| O1 第二個訓練 seed（KD + 對照） | 估計訓練隨機性，避免把 seed 差異誤判為 KD 效果 | 同上 | **選用，建議** | 約 +2.8–3.6 H100-h |
| O2 實作 attention-distillation loss | 補上論文的第三個 loss | 3k steps | 選用 | 開發 1–2 天，另約 +1.7–2.2 H100-h |
| O3 其他深度（L9 或 L4） | 大小與表現曲線 | 3k steps | 選用 | 每個深度約 +1.5–3 H100-h（依每樣本 FLOPs 推估） |
| O4 SnapFlow 1-NFE（套在 L6 student） | 步數蒸餾疊加層數蒸餾 | 自行實作（未找到程式碼） | 選用 | 開發 2–4 天；論文約 12 GPU-h（A800，VLM 凍結）。筆電能否執行未驗證 |
| O5 W4A8 量化（QuantVLA） | 量化疊加蒸餾 | 免訓練 | 選用 | 雲端 0 h；在筆電整合 1–2 天（與 openpi PyTorch 路徑的相容性未驗證） |
| O6 論文長度 30k steps | 接近論文設定 | 30k × 64 | 選用 | 約 +28–36 H100-h（主計畫的 8 倍） |

### 7.3 訓練設定

- 依 Phase 1 README（第 178–201 行）：bf16、gradient checkpointing，student 除 vision encoder 與 token embedding 外全部可訓練，batch 64，3,000 steps（約 0.7 epoch；1 epoch = 4,273 steps），peak LR 2.5e-5，warmup 300，decay 3,000，每 1,500 步存一次 checkpoint。KD 與對照使用相同的 seed 與資料順序。
- **缺少的 attention-distillation loss：建議先用已釋出的兩個 loss 跑主計畫**，O2 留待之後。理由：
  1. 主比較是 KD 對 task-only，缺少的 loss 對兩組影響相同，比較仍然公平。
  2. 論文消融中 attention loss 只貢獻約 +0.7 pt（L6：93.9 → 94.6）。依表 4-3，這個幅度在 2,000 個 episodes 以內都無法可靠偵測。
  3. 論文 Table 1c 顯示，把 attention loss 套到全部 token 會使訓練不穩定（L9／L6／L4 成功率 0.0／61.5／0.0），實作細節敏感，先做會增加主計畫失敗的風險。
  4. 評估流程完全共用，之後補做 O2 不需要重做其他部分。

### 7.4 評估協定

| 項目 | 設定 | 理由 |
|---|---|---|
| 主要評估 | LIBERO 4 suites × 10 tasks × **25 trials**（初始狀態 0–24），每模型 1,000 個 paired episodes；其餘設定同 Phase 1（env seed 7、每次 query 固定 noise seed、每 5 步 replan、10 個去噪步） | 依表 4-3，可以 ≥ 95% 的 power 偵測 3 pt 下降。400 個只能偵測約 5 pt，2,000 個的評估時間則加倍 |
| 統計 | 各模型的 Wilson CI；teacher 對各 student、KD 對 task-only 的 exact McNemar 與 paired 95% CI；non-inferiority（δ = 5 pt） | 第 4.2 節 |
| 擾動（建議） | LIBERO-Plus 分層子集：7 個維度 × 約 100 episodes，約 700 episodes，只評 teacher、KD@3k、task-only@3k，同樣配對 | 回答 H4；補上第 3.3 節的證據缺口 |
| 延遲與記憶體 | `bench_model.py`（與 Phase 1 相同）；有目標硬體時在該硬體上加測 | 第 4.3 節 |
| 動作層級 | `action_agreement.py`，457 個 held-out frames；新增 L∞ 與 jerk | 第 4.4 節 |
| 評估環境 | 所有模型都在同一台筆電上評估；雲端只負責訓練 | 避免硬體與數值差異混入比較 |

筆電評估時間（依 Phase 1 實測推估）：接近 teacher 水準的模型每 1,000 episodes 約 2.4 h；較弱的模型因 episodes 跑到步數上限，約 4.3 h。主要評估合計約 11–19 h，LIBERO-Plus 子集約 5–9 h，都可以在夜間執行。

### 7.5 指標總表

| 類別 | 指標 |
|---|---|
| 閉迴路表現 | 各 suite 與整體成功率（Wilson CI）、paired 差異（McNemar、95% CI）、成功 episode 的步數 |
| 強健性 | LIBERO-Plus 各擾動維度的成功率與 paired 差異 |
| 動作正確性 | chunk／首個動作 MSE vs teacher（以 teacher 噪聲底線標準化）、夾爪方向一致率、L∞ 偏差、jerk |
| 效率 | 延遲 p50／p95／p99、推論記憶體、推論用參數、FLOPs（推估） |
| 失敗分析 | 不一致 episode 清單、失敗分類、代表性錄影 |
| 訓練 | loss 曲線、實際 s/step 與 GPU-hours（據以修正成本估計） |

### 7.6 運算需求（可直接對照部門雲端）

| 資源 | 需求 | 依據 |
|---|---|---|
| GPU | **1 張 80 GB**（H100 80GB 優先；A100 80GB 可行）。48 GB 卡只能用 batch 32，需要 gradient accumulation，fork 是否支援尚未確認 | 估計峰值 45–55 GB：teacher 7.5 GB + student 權重、梯度與 AdamW 約 16.4 GB + 啟動值 20–30 GB（README 第 185–189 行） |
| GPU 時數（主計畫） | H100：KD 1.7–2.2 h + 對照 1.1–1.4 h + 環境建置約 1 h ≈ **4–5 h**。A100：約 **8–10 h** | 每個 batch-64 step 約 0.77 PFLOP；H100 約 2–2.6 s/step，A100 約 5–6.5 s/step（README 第 179–184 行） |
| GPU 時數（選用） | O1 +2.8–3.6；O2 +1.7–2.2；O3 每個深度 +1.5–3；O6 +28–36（H100-h） | 第 7.2 節 |
| 主記憶體 | ≥ 64 GB（checkpoint 轉換峰值 28.8 GB） | README 第 163 行 |
| CPU | 8–16 vCPU（資料載入） | 建議值 |
| 磁碟 | ≥ 200 GB：LeRobot LIBERO 54 GB、teacher（JAX + PyTorch）、4 個 student checkpoint（每個權重 4.1 GB + optimizer 約 8 GB，約 50 GB）、Python 環境 | README 第 186、194 行；`distill_smoke.json` |
| 網路 | 下載約 70 GB（Hugging Face `physical-intelligence/libero`、`gs://openpi-assets` 公開 bucket）；只需回傳 4 個 × 4.1 GB 的權重 | — |
| 軟體 | Linux x86_64、CUDA 12.x driver、Python 3.11、uv；不需要 sudo 或 Docker | Phase 1 環境 |
| 帳號 | 不需要額外帳號（資料與 checkpoint 均公開） | — |

### 7.7 時程（T0 = 取得雲端 GPU 當天）

| 時間 | 工作 | 地點 |
|---|---|---|
| 本週 | 本報告；確認算力與目標硬體；先在筆電準備 25-trial 評估、L∞／jerk 指標，以及 LIBERO-Plus 的相容性測試 | 筆電 |
| T0 | 建置環境、下載資料、轉換 teacher；先跑 50 steps 測速後修正成本估計；接著依序跑 KD 與對照，各 3k steps | 雲端（約 4–5 H100-h） |
| T0+1 ~ T0+5 天 | LIBERO 1,000-episode 評估（4 個 student checkpoint，加上 teacher 補測 600 episodes）、bench、動作層級指標 | 筆電 |
| T0+6 ~ T0+8 天 | LIBERO-Plus 子集、失敗分析 | 筆電 |
| T0+9 ~ T0+10 天 | 比較報告 | — |
| 之後（選用） | O1／O2／O3／O5 依結果與剩餘算力決定 | 雲端／筆電 |

### 7.8 風險與對策

| 風險 | 影響 | 對策 |
|---|---|---|
| batch 64 在 80 GB 上 OOM | 無法依設定訓練 | gradient checkpointing 已開啟；備案為凍結更多層，或改 batch 32 加 gradient accumulation（先確認 fork 支援） |
| 實際速度比估計慢 1 倍 | GPU 時數約 10 H100-h | T0 先以 50 steps 測速，再決定是否縮短 steps |
| 3k steps 不足，student 遠低於 teacher | H1 不成立 | 主要問題（KD 對 task-only）仍可回答；1.5k 與 3k 兩點可看出趨勢；視結果延長訓練 |
| KD 與 task-only 差距只有 1 pt 級 | 成功率無法分辨 | 以動作 MSE 為 H2 的主要指標；如實報告 CI（這與論文 0.9 pt 的消融結果一致） |
| 訓練 seed 的變異大於 KD 效果 | 結論不可靠 | O1 第二個 seed |
| LIBERO-Plus 整合不順 | 缺少強健性證據 | 本週先做相容性測試；備案改用 LIBERO-PRO（MIT） |
| 筆電軟體渲染使評估變慢（約 18 env steps/s） | 時程延後 | 4 個 worker 在夜間執行；必要時在雲端 CPU 上跑模擬 |
| 授權 | student 權重受 Gemma 條款約束 | 權重僅內部使用；程式碼為 Apache-2.0 |

### 7.9 交付物

1. KD 與 task-only 的 student checkpoints（1.5k、3k），僅供內部使用。
2. 所有模型的原始結果 JSON 與自動產生的 `SUMMARY.md`：成功率（含 CI 與 McNemar）、延遲、記憶體、參數、動作層級指標、LIBERO-Plus 各維度結果。
3. 比較報告：驗證 H1–H4 的結果、失敗分類、代表性錄影，並依第 5.4 節說明其作為產品化證據的範圍與限制。
4. 可重現的腳本與設定（repo 內），以及實際的 GPU 時數與成本，供後續規劃參考。

## 8. 參考文獻

各論文以 arXiv 編號引用（https://arxiv.org/abs/<編號>）；程式碼連結見第 2 節各表。

- **蒸餾與壓縮方法：** 2601.20262 Shallow-π（github.com/icsl-Jeon/openpi；icsl-jeon.github.io/shallow-pi）；2604.05656 SnapFlow；2605.16241 VLA-AD；2605.21854 CrossVLA；2606.20246 CLP；2606.27755 Drop-Then-Recovery；2606.31382 VLM2VLA parameter redundancy；2511.18082 ActDistill；2606.05737 One-step VLA；2606.14255 ReactVLA；2607.12287 Reducing Temporal Redundancy；2607.25487 CoTinyVLA；2602.20309 QuantVLA；2605.28803 HoloQ-VLA；2605.24011 ActQuant；2506.07530 BitVLA；2506.10100 EfficientVLA；2502.02175 VLA-Cache；2505.21200 FlashVLA；2509.12594 LightVLA；2510.26742 Realtime-VLA；2607.26657 Enfold（github.com/zwl666666/enfold；zwl666666.github.io/enfold）。
- **小型原生 VLA：** 2506.01844 SmolVLA；2509.09372 VLA-Adapter；2511.04555 Evo-1；2607.27205 TurboVLA。
- **綜述：** 2510.17111；2510.24795；github.com/guanweifan/awesome-efficient-vla。
- **Benchmark 與評估方法：** 2510.13626 LIBERO-Plus；2510.03827 LIBERO-PRO；2405.05941 SimplerEnv；2603.04356 RoboCasa365；2112.03227 CALVIN；2506.18088 RoboTwin 2.0；2506.18123 RoboArena；2507.05331 TRI LBM；2409.09491 Robot Learning as an Empirical Science；2503.10966 STEP；2603.13616 Beyond Binary Success。
- **安全與監控：** 2506.09937 SAFE；2606.09749 Attention-Guided Safety Filter；2609.22462 VLPSA；2604.23775 VLA Safety survey；2608.02809 人形機器人 fail-passive gap；2609.09250 No Free Checker。
- **標準與法規：**
  - ISO 10218:2025：iso.org/standard/73933.html；ibf-solutions.com/en/seminars-and-news/news/new-standards-for-industrial-robots-en-iso-10218-1-and-2
  - ISO/FDIS 13482：iss.rs/en/project/show/iso:proj:83498；intertek.com/resources/webinars/2026/industrial-service-robots/
  - ISO/CD 25785-1：iss.rs/en/project/show/iso:proj:91469
  - ISO/IEC TS 22440-1：iss.rs/en/project/show/iso:proj:89535
  - ISO/IEC TR 5469：iso.org/standard/81283.html
  - ISO 13849-1:2023：iso.org/standard/73481.html
  - IEC 61508 ed.3：61508.org/knowledge/standards-development/
  - UL 3300：webstore.ansi.org/standards/ul/ansiul33002024
  - ISO/IEC 42001：blog.ansi.org/anab/iso-iec-42001-ai-management-systems/
  - ISO/PAS 8800：ul.com/resources/isopas-88002024-road-vehicles-safety-and-artificial-intelligence
  - Machinery Regulation：digital.nemko.com/regulations/eu-machinery-regulation；sopx.io/insights/eu-machinery-regulation-2023-1230/；ibf-solutions.com/en/news-and-knowledge/technical-papers-and-news-on-ce-marking/postponement-of-mr-requirements-for-cybersecurity-and-ai
  - AI Omnibus：digital-strategy.ec.europa.eu/en/news/ai-omnibus-enters-force；eurogip.fr/en/machinery-regulation-2023-1230-what-the-digital-omnibus-on-ai-changes/；gibsondunn.com/eu-ai-act-omnibus-agreement-postponed-high-risk-deadlines-and-other-key-changes/
  - UR e-Series TÜV 證書：universal-robots.com/media/1802431/ur-e-series_iso13849-1_14097610-fusi.pdf
  - 《人工智慧基本法》：moda.gov.tw/press/press-releases/18316；leeandli.com/TW/NewslettersDetail/7566.htm

## 附錄 A：檢定力計算

- Wilson 區間：標準公式，z = 1.96。
- 非配對兩比例：n = [z_{α/2}·√(2p̄(1−p̄)) + z_β·√(p0(1−p0) + p1(1−p1))]² / Δ²，其中 p0 = 0.96，α = 0.05，power 80%。
- 配對 McNemar（Connor 1987）：n = [z_{α/2}·√p_d + z_β·√(p_d − Δ²)]² / Δ²，其中 p_d = p10 + p01，Δ = p10 − p01。
- 固定 n 的檢定力：依 (p10, p01, 1 − p10 − p01) 做多項分佈抽樣，每格 4,000 次，以 exact binomial（McNemar）p < 0.05 判定拒絕。
- 執行方式：`uv run --with scipy --with numpy python power.py`（scipy 與 numpy 由 uv 臨時安裝），輸出即為表 4-1 至 4-3。

## 附錄 B：本報告使用的資料與位置

| 內容 | 位置 |
|---|---|
| Phase 1 實測結果 | `git fetch origin fm/vla-phase1` 後的 `results/phase1/SUMMARY.md`；`results/phase1/libero/teacher_l18/summary.json`（各 task 成功次數，失敗集中的 11 個 task 由此取得）；`results/phase1/bench/teacher_l18.json`（延遲、參數） |
| Phase 2 成本估計與訓練設定 | `README.md` 第 165–201 行（實驗紀錄「Training readiness」與「Phase-2 resource estimate」） |
| Fork 錯誤修正 | `README.md` 第 118–135 行；`src/openpi/models_pytorch/layer_mapping.py` |
| Phase 1 PR | https://github.com/shayne773/vla_distill/pull/1 |
| 文獻數字 | 各論文的 arXiv 摘要頁或 HTML 全文（第 8 節） |
| 程式碼授權 | GitHub API `repos/<owner>/<repo>` 的 `license.spdx_id`（2026-09-27 查詢） |
