# Report: "World-Action Models for Robot Learning and Control: A Survey" (arXiv:2609.16074v1)

**Paper.** Zuxing Lu\*, Hongjia Zhai\*, Guanzhi Wang, Huajian Zeng, Jiaqi Yang, Jingyu Liu, Lei Cheng, Yuantai Zhang, Yuheng Qiu, Zezhou Cheng, Ivan Laptev, Danfei Xu, Benjamin Riviere, Giuseppe Loianno, Eric Xing, Xingxing Zuo† (\*equal contribution, †corresponding author). Ten of the sixteen authors are at MBZUAI; the others are at Caltech, Amazon FAR, University of Virginia, Georgia Tech, New York University and UC Berkeley. arXiv:2609.16074v1 [cs.RO], 13 Sep 2026, a v1 preprint. The PDF has 19 pages: body on pp. 1–15, 268 references on pp. 15–19. Project page: https://rcl-robotics.github.io/Awesome-World-Action-Models.

**How to read this report.** Section 1 stands alone as a three-minute summary. Sections 2–6 and 8 report what the paper says, and each statement carries its location in the paper (§ = section, Eq., Tab., Fig.). Section 7 is assessment and, with the last paragraph of section 1, is the only place containing judgments that are not the paper's. Bracketed numbers such as [23] are the paper's own reference numbers from the PDF reference list. The tables in 4.2, 5.2 and 8 are look-up material and can be skimmed. Section 9 records how the paper was read and what could not be verified.

---

## 1. Summary

**What the paper is.** A robotics-oriented survey of World-Action Models (WAMs): models that couple prediction of future world states with generation of executable robot actions (Abstract, §I). It is a qualitative, organizing survey. It reports no benchmark results for the methods it covers and runs no experiments.

**The argument.** Vision-Language-Action (VLA) policies are "primarily reactive": they map the current observation, a short history and an instruction to the next action without modeling how alternative actions would change the future, which hurts on long-horizon, delayed-effect, occluded and contact-rich tasks (§I). World models do model consequences, but on their own they are not enough for control, because the dynamics model, planner and policy are usually trained or deployed separately, so "visually plausible futures are not necessarily actionable" and explicit planning is expensive and sensitive to model bias (§I). WAMs are presented as the combination: one model, or one coupled pipeline, that predicts the future and produces the action (§I, §II-A).

**What it organizes.**

| Part | Content | Where |
|---|---|---|
| Definition | Formal definitions of world model, VLA and WAM; four foundational threads; a four-component generative view | §II, Eq. 1–15, Fig. 2 |
| Taxonomy | Five axes: model components, transition modeling, architecture structure, training pipeline, data modality and scaling | §III, Fig. 3–7 |
| Applications | Manipulation, navigation, autonomous driving; 66 methods in one summary table | §IV, Tab. I |
| Evaluation | 20 metric definitions; 48 datasets, simulators and benchmarks | §V, Tab. II–III |
| Open problems | Six challenge areas | §VI |

**Main takeaways.**

1. **Two ways to couple prediction and action.** *Joint prediction* generates future observations and actions together (Eq. 14, 16). *Inverse dynamics* (IDM) first generates a future visual plan and then infers the actions that realize it (Eq. 15, 17). The paper states that neither is universally superior (§III-D1).
2. **Two ways to wire the network.** *End-to-end* systems use one backbone for everything; *dual systems* separate a video expert from an action expert. Crossing this with takeaway 1 gives a 2×2 grid of designs (§III-C, Fig. 5).
3. **The field has shifted from learning dynamics to aligning pretrained video models with control.** Recent systems start from large video generators (Wan, Cosmos), and the central problem becomes "aligning pretrained generative dynamics with embodiment-specific control" (§III-A3, §III-B1).
4. **Data forms a pyramid.** Internet video (large, no actions) gives world-dynamics priors; egocentric human demonstrations sit in the middle; robot trajectories (small, precise) give action grounding (§III-E2, Fig. 7).
5. **Prediction plays a different role in each domain.** In manipulation it shapes action representations or serves as an imagined training environment; in navigation it is a look-ahead module for a separate planner; in driving it is also used for scenario synthesis (§IV).
6. **Evaluation does not yet show whether prediction helps control.** Task success rarely isolates whether success comes from better prediction, stronger imitation priors or task-specific policy learning, and video-quality metrics are "insufficient for action grounding". Evaluations should report both predictive quality and downstream control utility (§V-A).
7. **Six open challenges:** action alignment, world–action factorization, spatial and multi-view consistency, long-horizon memory, neural simulators for closed-loop policy learning, and inference latency (§VI).

**Assessment in brief (detail in section 7).** The paper's formal vocabulary and its 2×2 design grid are useful, and its coverage runs to August 2026. Its weaknesses are that it presents no quantitative evidence for its central premise that predicting the future improves control; that its definition of a WAM is stretched until it covers plain world models, reinforcement learning in imagination, and models that skip future prediction at inference time; that its main summary table uses different categories from its own taxonomy and omits DreamZero, one of the two works the definition is built on; and that it contains several incorrect or duplicated references.

---

## 2. What a World-Action Model is, and where the paper draws the boundary

### 2.1 Formal definitions (§II-A)

The setting is a partially observable Markov decision process with observations **o**, actions **a**, a latent state space, a latent action space and a language instruction ℓ (§II-A).

| Model class | Definition in the paper | Equation |
|---|---|---|
| World model (WM) | Predicts the next latent state from the current latent state and action: ẑ_{t+1} = f_WM(z_t, a_t) | Eq. 1 |
| VLA | Predicts an action chunk from observation history and instruction, "without explicitly modeling future state trajectories": â_{t:t+H} = f_VLA(o_{<t}, ℓ) | Eq. 2 |
| **WAM** | Jointly predicts the future observation trajectory and the action chunk from observation history and instruction, "explicitly modeling how actions affect future states": ô_{t:t+H}, â_{t:t+H} = f_WAM(o_{<t}, ℓ) | Eq. 3 |

The introduction gives a broader working definition: models that couple predictive world representations with executable action generation, "either by jointly predicting future states and actions or by predicting future states and then inferring actions through an inverse-dynamics interface" (§I).

### 2.2 The four-component view (§II-C)

Following UWM [19], the paper decomposes a WAM's generative process into four conditional distributions and says WAMs usually optimize several at once (§II-C):

| Component | Distribution | Equation |
|---|---|---|
| Policy | p(a_{t:t+H} \| o_{<t}, ℓ) | Eq. 10 |
| Forward dynamics | p(o_{t:t+H} \| o_{<t}, a_{t:t+H}) | Eq. 11 |
| Inverse dynamics | p(a_{t:t+H} \| o_{<t}, o_{t:t+H}) | Eq. 12 |
| Visual planning | p(o_{t:t+H} \| o_{<t}, ℓ) | Eq. 13 |

### 2.3 Boundary against neighbouring model classes

| Neighbour | How the paper characterizes it | What separates a WAM, per the paper | Where |
|---|---|---|---|
| Conventional world models | Predictive models of environment dynamics; the action is an input (Eq. 1). Useful for imagination, planning, counterfactual evaluation, data-efficient learning | A predictive world model alone "is not sufficient for effective robot control"; dynamics model, planner and policy are separate, which creates a mismatch between predictive fidelity and control utility. A WAM outputs the action as well | §I, §II-A |
| Model-based RL | Uses a dynamics model for planning and policy optimization. PlaNet's recurrent state-space model (Eq. 4) is the reference design; later models are "explicit" with a decoder (Dreamer, DreamerV2, DreamerV3) or "implicit" with latent regularization (TD-MPC, TD-MPC2, DINO-WM, LeWM) (Eq. 5) | WAMs "increasingly incorporate multimodal foundation-model conditioning, large-scale offline robot data, and amortized action generation". Explicit planning over learned dynamics is described as expensive, sensitive to model bias and hard to combine with high-frequency control | §I, §II-B1 |
| Video generation, including action-conditioned generation | Generates future observations from text, image or video history (Eq. 6). "Better viewed as visual predictive planners rather than explicit dynamics models" | Supplies only the visual-planning component (Eq. 13). A WAM adds the action, jointly or through inverse dynamics | §II-B2, §II-C |
| Latent action pretraining | Infers latent actions from consecutive frames of action-free video and uses them to model latent transitions (Eq. 7) | Treated as a foundation and a component (the latent action decoder), not as a competitor | §II-B3, §III-A4 |
| Reactive VLA policies | Map observation and instruction directly to actions, by autoregressive token prediction (RT-2, OpenVLA; Eq. 8) or flow matching (π0, π0.5, GR00T N1; Eq. 9) | WAMs "introduce predictive structure for reasoning about action consequences" | §I, §II-B4 |

The paper presents WAMs as unifying these threads "within a conditional generative framework that jointly models future observations and the actions responsible for them" (§II-C). Fig. 1 tells the same story as a timeline in five stages: latent world models and generative foundations (before 2023), foundation VLAs (2023), generalist VLAs and generative world models (2024), video-action models and flow VLAs (2025), and unified WAMs (2026: Cosmos Policy, Fast-WAM, DreamZero, LingBot-VA, GigaWorld-Policy).

---

## 3. The taxonomy (§III)

§III organizes WAMs on five axes. The seven taxonomy headings named in the abstract map onto them as follows: representations and action interfaces are the model components (3.1), followed by transition modeling (3.2), architectures (3.3), training pipelines (3.4), and data modalities and scaling strategies (3.5).

### 3.1 Model components: representations and action interfaces (§III-A, Fig. 3)

| Module | Option | Examples the paper names | Property the paper states |
|---|---|---|---|
| Language encoder | Global text encoder | CLIP [80] | Open-vocabulary grounding from image–text contrastive learning |
| | Dense sequence encoder | T5 [81] | Token-level representations for detailed conditioning |
| | Autoregressive LLM | GPT [82], LLaMA [83] | World knowledge, reasoning, long-horizon instruction following |
| Visual encoder | 2D VAE | Latent diffusion [4], Stable Video Diffusion [84] | Efficient spatial compression, limited temporal consistency |
| | Video VAE | Wan [76], CogVideoX [85] | Joint spatial and temporal encoding; helps future video prediction |
| | Representation encoder | DINO family [86]–[88], CLIP [80], SigLIP [89], V-JEPA 2.1 [90] | Semantic and geometric features; a representation space, not a pixel generator |
| | 3D and multi-view encoder | Point clouds [91], occupancy grids [59], bird's-eye-view features [92] | Explicit geometry for spatial reasoning |
| | Vision-language model | Qwen-VL [93], Prismatic [94], PaliGemma [95], as reused by OpenVLA and π0 | Visual features aligned with language |
| Backbone | Unified transformer trained from scratch | GR-1 [13] | Learns dynamics from embodied and video data with no pretrained temporal prior |
| | Pretrained video foundation model | DreamZero [23] on Wan [76] or Cosmos [77] | Shifts the problem "from learning dynamics to aligning visual predictions with control" |
| Action decoder | Action tokenizer | Vector quantization or bins [96], RT-2 [6], OpenVLA [11] | LLM-compatible and scalable across embodiments; quantization loses precision |
| | Latent action model | LAPA [78], AdaWorld [53], ViPRA [97]; DreamDojo [98] | Decouples actions from specific hardware; learns from action-free video |
| | Diffusion or flow-matching head | Diffusion Policy [8], π0 [16], GR00T N1 [79] | Continuous, multimodal actions; higher inference cost |

Fig. 3 draws the same decomposition as a pipeline with an explicit "world prediction" output (imagined observation, latent rollout, scene evolution) feeding the action decoder, and an optional closed-loop feedback path. The figure also lists BERT among language encoders; the text does not.

### 3.2 Transition modeling (§III-B, Fig. 4)

| Paradigm | Formulation | Methods and how the paper describes them |
|---|---|---|
| Joint prediction | p(a_{t+1}, o_{t+1} \| o_t, ℓ) (Eq. 16). Acts as "both a direct policy and an auxiliary learning signal" | GR-1 [13], GR-2 [38]: autoregressive co-prediction of actions and future visual representations. VPP [20]: text-guided video prediction with a diffusion policy head. UWM [19]: unified video and action diffusion. WorldVLA [39]: shared autoregressive token space for text, images and actions. Motus [99]: structured joint attention with separate schedulers for video and action. ViPRA [97], Cosmos Policy [21]: strong pretrained visual priors |
| Inverse dynamics, explicit | p_plan(o_{t+1} \| o_t, ℓ) · p_idm(a_{t+1} \| o_t, o_{t+1}) (Eq. 17), "plan-then-act" | PathDreamer [2] (navigation; limited planning horizon). Seer [60]: action-token attention restricted to current and predicted future states. DreamZero [23]: chunk-level autoregressive video generation, with "notable inference overhead" |
| Inverse dynamics, implicit | Trained jointly, but actions can be decoded directly at inference | LingBot-VA [24]: Mixture of Transformers [100] with asynchronous inference. Fast-WAM [22], GigaWorld-Policy [25]: decoupled attention for visual generation and action generation |
| Goal-conditioned | Predicts toward a target future state, not only the next observation | Act2Goal [102]: goal-conditioned IDM. π0.7 [63]: "can be viewed as a goal-conditioned WAM" that predicts actions toward a goal image |

### 3.3 Architecture structures (§III-C, Fig. 5)

Fig. 5 crosses structure with transition paradigm. The figure shows the four cells schematically and does not place methods in them.

| | Joint prediction | Inverse dynamics |
|---|---|---|
| **End-to-end** (one backbone) | (a) Unified backbone emits visual and action tokens together | (b) Autoregressive backbone emits visual tokens, then action tokens |
| **Dual system** (video expert + action expert) | (c) Experts joined by Mixture-of-Transformers shared attention | (d) Video expert feeds action expert through cross-attention |

| Structure | Methods the paper places here | Notes from the paper |
|---|---|---|
| End-to-end | GR-1, GR-2, Seer; PAD [103] and UWM (diffusion transformers that jointly denoise image and action tokens); WorldVLA, DreamZero, Cosmos Policy, ViPRA (pretrained DiT and video backbones); DWAM [105] | Main difficulty is aligning heterogeneous token spaces. Cosmos Policy turns actions, future states and values into latent-frame-like representations. "Under fixed robot-data budgets, richer vision-language and video representations generally improve action generalization and robustness"; DWAM reports better robustness and adaptation efficiency at higher inference cost (§III-C1) |
| Dual system, shared-attention routing | LingBot-VA, Fast-WAM | Modality-specific projections fused in shared self-attention (§III-C2) |
| Dual system, cross-attention transfer | mimic-video [106]; VidMan [67], VPP [20]; ImageWAM [107] (image-editing backbone in place of video generation, transfer through a key-value cache). π0, π0.5 and GR00T are cited as examples of this pattern "in VLAs" | More modular and controllable at the interface, at the cost of design complexity at the module boundary. Helix [108] is cited for a fast–slow split that reduces VLM compute on humanoids (§III-C2) |

### 3.4 Training pipeline (§III-D, Fig. 6)

| Stage | Step | Content and named works |
|---|---|---|
| Pre-training | Spatiotemporal representation | Pretrained video backbones: Seedance [110], Wan [76], Cosmos [77], Sora [15]. Multi-view pre-training improves geometric consistency (Cosmos Policy [21] for manipulation, [111] for driving). JEPA-WAM [112] and LaWAM [113] predict in learned feature space instead of pixels |
| | Action representation | Explicit: OpenVLA [11] discrete tokens; FAST [109] discrete cosine transform; OpenVLA-OFT [114] continuous actions with bidirectional attention. Latent: LAPA [78], AdaWorld [53], ViPRA [97]; VLA-JEPA [115] combines latent actions with a flow-matching head |
| | Video–action alignment | UWM's four routes: forward dynamics, inverse dynamics, action generation, video generation. DreamZero: video generation + inverse dynamics. Fast-WAM and GigaWorld-Policy: video + action generation. Cosmos Policy: action generation + video generation + forward dynamics. "Neither joint prediction nor IDM is universally superior" [22] |
| Post-training | Policy fine-tuning | LoRA [116], adapters, residual action heads. The image-prediction objective (Eq. 13) improves generalization to unseen tasks [23]; the action-grounding objective (Eq. 12) supports human-to-robot and robot-to-robot transfer [115]. DreamZero "achieves strong cross-embodiment transfer with only about 30 minutes of real-robot data" |
| | Data augmentation | Label-preserving trajectory rewriting (GigaBrain-0 [62], GigaWorld-0 [117], Cosmos-Transfer [118]). Action-grounded trajectory synthesis, where inverse-dynamics or latent action models recover pseudo-actions for generated video (DreamGen [119], WorldArena [120]) |
| | Reinforcement learning | Three routes: simulation RL (sim-to-real gap); real-world RL (SERL [127], RL-token [126]; interaction and safety cost); world-model RL (World4RL [66], World-VLA-Loop [64], WAM-OPD [128]; less physical interaction, risk of model bias). Typical pipeline: action-conditioned world model, reward predictor [129], imitation-initialized policy, refinement on imagined trajectories [130]. Integrating jointly predicted observations, values and actions for scalable WAM post-training "remains an open problem" |

### 3.5 Data modality and scaling (§III-E, Fig. 7)

| Modality | Role in WAM design, per the paper (§III-E1) |
|---|---|
| Language | Task-level conditioning; WAMs typically encode it separately and fuse it by cross-attention |
| Multi-view video | Better spatial understanding; generating consistent multi-view predictions remains hard [131] |
| 3D geometric structure | Explicit geometric constraints [132], [133]; occupancy and geometric prediction is promising [59], [134], [135]; 4DGS-WAM [136] and 4D-WAM [137] are early general 3D world prediction |
| Contact and audio | Tactile [138], [139] and audio signals reveal interaction events vision misses; either sensed directly [141], [142] or predicted from vision [143] |
| Proprioception | Direct physical state, helpful under occlusion; Cosmos Policy also uses predicted actions as state to reduce out-of-distribution issues |

| Data tier (Fig. 7b) | Scale and precision | What it supplies | Works named (§III-E2) |
|---|---|---|---|
| Internet video | Massive, broad | World dynamics prior | GR-1, GR-2 (visual pre-training); LAPA, DreamDojo (latent actions from unlabeled video); Wan, Cosmos |
| Egocentric demonstrations | Medium, weak labels | Motion and intention | UMI [147]; EgoDex [144], Ego4D [145]; ActiveGlasses [148]; EgoScale [149]; Dyna-2 [150], which "scales first-person human manipulation videos to one million hours" |
| Embodied trajectories | Scarce, precise | Action grounding | Open X-Embodiment [146], DROID [151], BridgeData [152]; LBM [153] |

---

## 4. Applications (§IV, Tab. I)

### 4.1 Role of prediction by domain

| Domain | Why it matters, per the paper | Groups the paper identifies | Closing question the paper poses |
|---|---|---|---|
| Manipulation (§IV-A) | "The most common testbed", since a useful rollout must model object states, contact dynamics, occlusions and task progress | Video-action and unified policies; structured state models (geometry, physics, touch, audio); imagination-based policy optimization | Whether predicted futures improve executable control in contact-rich settings, under distribution shift and over long horizons |
| Navigation (§IV-B) | Less contact-sensitive, more dependent on long-horizon spatial reasoning under partial observability | Look-ahead planning, where the world model supplies candidate futures and a separate planner selects actions; latent planning and world-model RL | Value is measured by waypoint selection, long-horizon consistency and robustness, and these models "need not generate explicit images" |
| Driving (§IV-C) | Real-time, safety and multi-agent constraints; prediction is also used for scenario synthesis | Controllable video generation; occupancy forecasting; unified perception-prediction-planning; world-model RL | Connecting visual realism, traffic consistency, safety and closed-loop driving performance |

### 4.2 Table I, condensed

Tab. I lists 66 methods with a category, a backbone and an evaluation note. Categories as the table caption defines them: **VAM** video-action models; **Uni** unified WAMs (joint observation and action generation); **WM-Plan** world-model-guided planning (the world model predicts, an external planner generates the action); **RL** imagination-based RL; **3D** 3D or geometry-based. Each entry below reads *method (backbone; evaluation or note)*.

| Domain | Category | Methods |
|---|---|---|
| Manipulation (39) | VAM (12) | GR-1 (GPT; LIBERO, CALVIN), GR-2 (GPT, 38M; LIBERO, CALVIN), VPP (video pred.; dexterous), Seer (GPT+mask; 100K demos), VidMan (video diffusion + adapter; Bridge), Cosmos Policy (Cosmos-P2; bimanual), ViPRA (flow-match; 100–200 demos), Fast-WAM (video pred.; LIBERO, RoboTwin), Act2Goal (goal-cond.; RoboTwin 2.0), VLA-JEPA (JEPA+flow; LIBERO), LAWM (latent action; LIBERO), HiLAM (hierarchical latent; long-horizon) |
| | Uni (7) | PAD (DiT; LIBERO), UWM (4-mode DiT; LIBERO), GigaWorld-Policy (action-centered; DROID), WorldVLA (Chameleon AR; LIBERO), Motus (MoT; RoboTwin), LingBot-VA (causal-diffusion MoT; AgiBot), World2Act (LLM+skill; RoboCasa) |
| | RL (12) | EVA (video WM; RoboTwin 2.0), RISE (latent; contact-rich), VLA-MBPO (video WM; LIBERO-Plus), GPC (rollout WM; Push-T), WMPO (pixel WM; LIBERO), RoboHorizon (LLM+WM; sparse reward), World-VLA-Loop (feedback; ManiSkill), IWS (video WM; 10-min @15FPS), World4RL (diffusion WM; frozen env), DreamPlan (video WM; VLA finetune), VAMPO (GRPO/denoise; LIBERO, CALVIN), PRWM (contrastive RL; DROID) |
| | 3D (8) | ManiGaussian++ (Gaussian; bimanual), ParticleFormer (particles; deformable), GWM (VAE+diffusion; MetaWorld), PointWorld (point cloud; 2M+ trajectories), PIN-WM (physics-informed; non-prehensile), VT-WM (visuo-tactile; contact), Cloth-WM (DreamerV2; cloth), Audio-WM (acoustic; liquid pour) |
| Navigation (9) | WM-Plan (5) | Pathdreamer (panoramic generation; R2R), DreamWalker (latent + MCTS; R2R, VLN-CE), NWM (multi-domain; Habitat, Ego4D), perceptive FDM (GRU; ANYmal quadruped), MonoDream (monocular; R2R-CE, RxR-CE) |
| | RL (3) | X-MOBILITY (latent WM environment; mobile robot), RWM (GRU; A1 quadruped), WMP (RSSM; Isaac) |
| | VAM (1) | InternVLA-N1 (dual-system latent; R2R-CE) |
| Driving (18) | WM-Plan (8) | DriveDreamer (video generation; nuScenes), GAIA-1 (text-video; large-scale), DriveDreamer-2 (multi-camera generation; nuScenes), GAIA-2 (multi-camera, scaled; OpenDV), DrivingWorld (long-horizon AR; nuScenes), DrivingGen (policy-conditioned evaluation; nuScenes), OccWorld (voxel tokens; nuScenes-Occ), Drive-OccWorld (multi-step occupancy; nuScenes-Occ) |
| | Uni (8) | ADriver-I (AR image+action; nuScenes), DrivingGPT (large AR video; NAVSIM), Doe-1 (autoregressive; nuScenes), OccLLaMA (occupancy+language; nuScenes-QA), ProphetDWM (long-horizon; NAVSIM, nuPlan), VaViM/VaVAM (video AR + action AR; OpenDV), DriveLaW (language-action; nuScenes), AutoMoT (asynchronous MoT; Bench2Drive) |
| | RL (2) | DreamerAD (Dreamer-AD; nuScenes, nuPlan), World4Drive (virtual-environment RL; NAVSIM, nuPlan) |

Category totals across the table: VAM 13, Uni 15, WM-Plan 13, RL 17, 3D 8 (tally made for this report from Tab. I).

The text adds a few distinctions the table does not carry. Among manipulation policies, WorldVLA, Motus and LingBot-VA generate observations and actions jointly at inference; World2Act and Act2Goal use predicted future states or goals as intermediate control targets; VLA-JEPA, LAWM and HiLAM "mainly use predictive objectives to improve latent representations, avoiding costly online video rollout during deployment" (§IV-A1). In driving, DriveVLA-W0 [189] studies how generated data affects downstream policies (§IV-C1); it is not in Tab. I.

---

## 5. Datasets, benchmarks, metrics and protocols (§V)

### 5.1 Metric families and what the paper says each can and cannot show (§V-A)

| Family | Metrics named | What it shows | Limit the paper states |
|---|---|---|---|
| Task-level | Success rate, route completion, driving score, episode return, chain length | Closest to the final control objective | Rarely isolates whether success comes from better prediction, stronger imitation priors, or task-specific policy learning |
| Trajectory | ADE/FDE, L2 displacement, NE, SPL, nDTW, ATE, RPE | Path and pose consistency; useful for planning-heavy WAMs | Low trajectory error does not guarantee safety under feedback |
| Generation | FID, FVD, LPIPS, PSNR, SSIM | Visual fidelity; necessary for video-based WAMs | Insufficient for action grounding |
| Semantic, language, occupancy | mIoU, mAP, NDS, BLEU, CIDEr, ROUGE | Quality of structured outputs | None stated |
| Deployment | Latency, collision, comfort, infractions | Deployment constraints | None stated |

The paper's protocol recommendation is a single sentence: WAM evaluations are most informative when they report both predictive quality and downstream control utility (§V-A). For driving it adds that "a driving WAM should … not be judged by video realism alone" and that open-loop recorded datasets and closed-loop simulators serve different purposes (§V-D).

Tab. II gives formulas for 20 metrics: SR, SPL, NE, nDTW (task completion and navigation); ADE, FDE, ATE, RPE (trajectory and localization); FID, FVD, PSNR, SSIM, LPIPS (generation quality); BLEU, ROUGE-L, CIDEr, METEOR (language); DS, PDMS, HNS (driving and RL aggregates). Two details it specifies: Driving Score averages route-level products of route completion and infraction score, not the product of separate means; and PDMS for sample *i* is NC·DAC·(5·TTC + 5·EP + 2·C)/12, averaged over samples (Tab. II).

### 5.2 Table III: datasets, simulators and benchmarks

Tab. III lists 48 resources in five blocks. Each entry below reads *name (reported scale)*. The caption warns that scales keep the units of their primary sources and are not directly comparable; "N/A" marks an open-ended simulator and "—" unavailable scale information. Type (real, simulated, mixed) is given per group.

| Block | Entries | Metrics the table associates with the block | How the paper says to read it |
|---|---|---|---|
| Robot demonstrations (§V-B) | Real: DROID (76k trajectories), BridgeData V2 (60,096 trajectories), AgiBot-World (1M+ trajectories). Mixed: Open X-Embodiment (1M+ trajectories) | SR, FVD, FID, LPIPS, accuracy | Paired robot data supplies the action grounding needed for control; used for pre-training and adaptation, not as standalone benchmarks |
| Human video (§V-B) | Real: Something-Something v2 (220k clips), Ego4D (3.67k hours), EgoDex (829 hours / 194 tasks), Kinetics-700 (700 classes), HowTo100M (136M clips) | Downstream SR; ATE and RPE for Ego4D | Improves predictive priors; contributes only after latent action learning, inverse-dynamics alignment or robot fine-tuning |
| Manipulation suites (§V-C) | Sim: LIBERO (130 tasks), CALVIN (34 tasks), MetaWorld (50 tasks), RoboCasa (100 tasks / 120 scenes), MimicGen (18 tasks / 50k+ demos), RLBench (100+ tasks), ManiSkill 1/2/3 (20+ tasks), SimplerEnv (—), Push-T (1 task). Mixed: RoboTwin / 2.0 (50+ tasks), RoboDojo (60 tasks) | SR throughout; average chain length (CALVIN); normalized or average score; accuracy; IoU (Push-T); FVD, LPIPS, PSNR, SSIM, FID where video is evaluated | "A spectrum from policy evaluation to prediction-quality evaluation", not interchangeable benchmarks |
| Driving (§V-D) | Real: nuScenes (1k scenes), nuPlan (1.3k hours), OpenScene (120+ hours), Waymo Open Motion (103k segments), OpenDV-2K (2,059 hours), BDD100K (100k videos), KITTI (6 hours), Argoverse (327.8k sequences), ZOD (1.5k scenes). Sim: Bench2Drive (220 routes / 12 towns), CARLA (N/A). Mixed: NAVSIM v1/v2 (—) | Video: FID, FVD, LPIPS. Structured prediction: ADE, FDE, L2, mIoU, mAP, accuracy, miss rate. Planning and safety: DS, RC, IS, PDMS, EPDMS, NC, TTC, comfort, collision | Recorded datasets support open-loop prediction; NAVSIM, CARLA and Bench2Drive support closed-loop planning. The three metric groups "should be combined" |
| Navigation (§V-E) | Real: Matterport3D (90 buildings), RECON (—), TartanDrive (—), SCAND (—). Sim: R2R / R2R-CE (7.2k paths), RxR / RxR-CE (16.5k paths), Habitat (N/A), ScaleVLN (4.9M trajectories) | SR, SPL, NE, nDTW, oracle success; ATE and RPE when prediction or mapping is involved | Evaluation "should capture more than successful arrival". §V-E also names CoPeD [246], which is not in the table |
| Foundational RL (§V-F) | Sim: DMControl (28 tasks), Atari ALE (55 games), Atari 100k (26 games / 100k steps), Procgen (16 games), Craftax (1 game), DMLab (—), BSuite (23 tasks). Real: FMB (—) | Return, reward, normalized score, HNS, IQM; SR and cycle time (FMB) | "Not substitutes for embodied robot evaluation"; a methodological check that a predictive model supports planning, value learning or policy improvement. FMB is a bridge case with real-world success and cycle time |

---

## 6. Open challenges and future directions (§VI)

| § | Challenge | Problem as the paper states it | Directions and works named |
|---|---|---|---|
| VI-A | Decoupling motion intent and action alignment | WAMs separate task-level motion reasoning ("what to happen") from low-level control ("how to act"), under an imbalance between visual and action data | Stage-wise alignment: pretrain video foundation models [76], [77] on large visual data, then adapt with embodied data; IDM-style WAMs (DreamZero, LingBot-VA, GigaWorld-Policy) suit this. Preserve pretrained priors with parameter freezing, action-alignment layers, residual action heads, and stronger anti-forgetting mechanisms |
| VI-B | World–action factorization | World prediction and action selection are "related but fundamentally different problems"; the trajectory distribution factorizes into an agent term p_π(a\|s) and a "universe" term p_μ(s′\|s, a) (Eq. 18). Joint optimization "may encourage a monolithic WAM to generate overly optimistic states that are easily exploited by its policy" | Train the world model and the agent model separately on their own objectives [49], [248]. Use the separate world model as a simulator for offline RL, runtime planning and model predictive control. Open: controlling model bias and coordinating the two |
| VI-C | Spatial and multi-view consistency | Video backbones struggle with geometric constraints, object permanence and cross-view consistency, which degrades state estimation, planning and inverse-dynamics decoding | Explicit 3D prediction: point clouds [157], occupancy fields, 3D Gaussians [250]; the difficulty is integrating them with scalable generative backbones. Egocentric observation with fixed inter-camera extrinsics [251] |
| VI-D | Long-horizon memory | Agents must retain task progress, past interactions and occluded information; longer observation windows are costly and redundant | Model-integrated memory: MEM [252] (video short-term + text long-term), EventVLA [253] (stores task-critical visual events), MemoryVAM [254] (episodic memory in video prediction and action decoding); these need architecture-specific training. External memory: Harness VLA [255], a frozen VLA with an agentic planner that keeps execution traces, success rules and failure models |
| VI-E | Neural simulators and closed-loop policy learning | Imitation learning has limited closed-loop correction, depends on high-quality expert demonstrations and cannot improve from failure | Neural simulators as predictive environments for policy evaluation and improvement [257], with accumulated prediction error as the main risk [258]. Rewards from tasks (Dreamer, TD-MPC), goals (LeWM, DINO-WM) or learned reward models (π0.6 [124], GigaBrain-0.5 [125]); long-horizon credit assignment [259]. Self-supervised RL [260], [261] "may replace the IDM-based action expert in dual systems"; reward-free and offline learning at scale remain open |
| VI-F | Inference latency and efficiency | Multi-step sampling, rollout generation and action decoding delay feedback. "Efficiency is therefore a control requirement, not only an implementation detail" | Few-step generation: consistency models [263], mean flow [264], rectified flow [5], SANTS [262], provided they preserve action-conditioned transitions and temporal consistency. Deployment: quantization, pruning, operator fusion, batching, memory-layout optimization [265], distillation [266], [267], KV-cache reuse [23]; also shorter rollout horizons, fewer imagined trajectories, less frequent replanning [268] |

The conclusion summarizes the requirements as: predictions must be grounded in robot actions, preserve spatial and temporal consistency, expose uncertainty, support closed-loop improvement and satisfy real-time constraints, and benchmarks must measure task success, safety, robustness and embodiment transfer (§VII).

---

## 7. Assessment (this section is the report's judgment, not the paper's content)

### 7.1 What the survey does well

- **A compact formal vocabulary.** Eq. 1–3 separate world model, VLA and WAM by what is input and what is output. The four conditionals (Eq. 10–13) let any method be described by which terms it trains and which it uses at inference. §III-D1 does this for four methods, and it is the most informative comparison in the paper.
- **A discriminating design grid.** The 2×2 of Fig. 5 separates designs that are often lumped together, and the "implicit IDM" label (§III-B2) picks out models that train with prediction but can skip it at inference.
- **Currency and breadth.** Of 268 references, 99 carry an arXiv identifier and 52 of those are 2026 identifiers, running to August 2026. Coverage spans three domains, 66 methods and 48 evaluation resources.
- **A sound position on evaluation.** §V consistently separates "looks right" from "controls well" and open-loop from closed-loop, and says plainly that task success does not isolate the contribution of prediction.

### 7.2 What it leaves out or treats thinly

- **No quantitative synthesis.** The body contains no success rates, no control frequencies or latencies, and no table comparing methods on a shared benchmark. The only method-level figures are DreamZero's "about 30 minutes" of robot data (§III-D2), Dyna-2's one million hours (§III-E2) and a few notes in Tab. I. A reader cannot learn from this survey how much any WAM gains over a matched reactive VLA.
- **The motivating question is posed but not answered.** The survey says evaluation rarely isolates whether prediction helps (§V-A) but proposes no protocol to isolate it, such as matched ablations with and without the prediction objective. Three cited works have titles that ask exactly this: "Do world action models need test-time future imagination?" [22], "Do world action models generalize better than VLAs? A robustness study" [105], and "Do world action models really need video generation, or just image editing?" [107]. The survey gives each a clause and reports none of their comparisons. Those papers were not read for this report.
- **Table I does not use the paper's own taxonomy.** §III builds a joint/IDM by end-to-end/dual grid, but Tab. I classifies by VAM, Uni, WM-Plan, RL and 3D, and no table places methods on the §III axes. Labels also disagree with the text: Cosmos Policy is a joint-prediction, end-to-end model in §III-B1 and §III-C1 but "VAM", not "Uni", in Tab. I; Fast-WAM and GigaWorld-Policy are treated as a pair in §III-B2 and §III-D1 but labelled "VAM" and "Uni"; the "3D" group includes Cloth-WM (DreamerV2 backbone), Audio-WM and VT-WM, which §IV-A2 itself calls multimodal, not geometric.
- **Central methods are missing from the unified table.** DreamZero is cited for the definition of a WAM (Eq. 3) and is the most-cited work in the body (14 citations), yet it does not appear in Tab. I or anywhere in §IV. Also absent from Tab. I: π0.7, DWAM, ImageWAM, mimic-video, JEPA-WAM, LaWAM and DreamDojo, all discussed in §III.
- **Announced content that is not delivered.** The introduction and conclusion announce "generalist embodied robotics" as a fourth application area (§I, §VII); §IV has three. The contributions list "embodiment transfer, and safe deployment" among open challenges (§I); §VI has no subsection on either. The conclusion says predictions must "expose uncertainty" (§VII); uncertainty is not discussed in §III–§VI. §VI opens with "four major bottlenecks" and then presents six.
- **Safety, failure modes and protocols.** Safety is mentioned in passing (driving constraints and metrics, the cost of real-world RL) and has no treatment of its own. What a policy should do when its predicted future is wrong gets one clause (§VI-E1). §V-A is titled "Metric Families and Evaluation Protocols" but contains metric definitions and one recommendation sentence, and several metrics used in Tab. III (mIoU, mAP, EPDMS, IQM, miss rate) are not defined in Tab. II.
- **Navigation is thin.** Nine methods, three of which are legged-locomotion or platform-dynamics world models described with no language instruction (§IV-B), which sits outside the paper's own Eq. 3.

### 7.3 Where the framing is contestable

- **The definition is elastic.** Eq. 3 defines a WAM as one model that outputs future observations and actions from history and language. The introduction widens this to predict-then-infer pipelines, and the conclusion widens it again to using predictions "for action selection, policy learning, or inverse-dynamics decoding" (§VII). Tab. I then includes 13 "WM-Plan" entries, where by the caption an external planner generates the action (among them the generators GAIA-1 and DriveDreamer), and 17 entries for RL in imagination. In total 38 of 66 entries fall outside the two categories the caption describes as video-action or unified models. On this usage "WAM" comes close to "any embodied system that contains a predictive model", which is the world-model and model-based RL literature that §II set out to distinguish it from.
- **The line against "reactive VLAs" is about training, not behavior.** The paper counts as WAMs models that can drop prediction at inference (Fast-WAM, GigaWorld-Policy, §III-B2) or use it only to shape representations (VLA-JEPA, LAWM, HiLAM, §IV-A1). At deployment these map observations to actions as a VLA does. Conversely, §III-C2 uses π0, π0.5 and GR00T, which §II-B4 classifies as VLAs, as representative dual-system examples. The distinction that survives is whether a prediction loss was used in training. The paper never says so, and this weakens its opening claim that WAMs differ from VLAs by "reasoning about action consequences".
- **§VI-B argues against the paper's own subject.** It recommends training world model and agent separately and warns that joint optimization may yield over-optimistic states that the policy exploits. That is an argument against the joint model of Eq. 3 and against the "Unified WAM" endpoint of Fig. 1, yet it is offered as a future direction and never reconciled with §II–§III. Its support is two position papers, [49] and [248], whose first author is a co-author of this survey, plus an unquantified "promising results".
- **"VLAs are reactive" is asserted, not shown.** The claim carries the motivation (§I), but no evidence is given that the named failure cases come from the absence of prediction and not from data or capacity.
- **Evidence quality.** Some prominent claims rest on non-archival sources: Dyna-2's "advantage of WAMs over VLAs" is a company research report [150], Helix is a company web page [108], and many 2026 citations are preprints.

### 7.4 Reference and production errors found

These were checked against the paper's own reference list; none required reading the cited works.

| Location | Issue |
|---|---|
| [86], cited for the DINO visual encoder (§III-A2) | The entry is "DINO: DETR with improved denoising anchor boxes for end-to-end object detection", an object detector that shares the name |
| [249], cited for "occupancy fields" (§VI-C1) | The entry is "Advances and applications of occupancy models", *Methods in Ecology and Evolution*, a statistical ecology paper |
| [241], cited for DMLab (Tab. III) | The entry is "The visual task adaptation benchmark" |
| [54] and [172] | The same title, "Robotic world model: A neural network simulator for robust policy optimization in robotics", listed twice with different venues |
| [59] and [174] | OccWorld and DriveDreamer are both given as ECCV 2024, pp. 55–72 |
| Fig. 1 against the reference list | UWM is "RSS 2025" in the figure and arXiv-only in [19]; Cosmos Policy is "ICLR 2026" in the figure and arXiv-only in [21]; DreamZero is "arXiv 2026" in the figure and "ICLR Workshop, 2026" in [23] |
| Tab. III | Ego4D scale is 3.67k hours while the cited title [145] reads "3,000 hours"; the Something-Something row is labelled "SSv2 V2" |
| §I and PDF header | One sentence in the second paragraph of §I is printed twice; the running header is the template default, "JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021" |
| arXiv HTML version | No reference list is rendered and every citation appears as a raw BibTeX key; the PDF is the usable version |

### 7.5 Overall

The paper is a useful map and vocabulary for a fast-moving area and a good index of 2025–2026 work. It should be read as a catalogue with a proposed taxonomy, not as evidence that world-action modeling outperforms reactive policies: it contains no such evidence, its category boundary is loose enough to take in most of the earlier world-model literature, and its own §VI-B questions the unified design it is organized around.

---

## 8. Reading list: the works the paper treats as most central

Selection uses three signals from the paper itself: works cited to define a WAM (§I, Eq. 3), works on the Fig. 1 timeline, and citation count in the body (counted from the HTML, tables included). Identifiers are exactly as the paper's reference list gives them. Where the list gives a venue and no arXiv id, the venue is shown and no id has been added.

| Work | Ref. | Identifier as given in the paper | Cited | Role in the paper |
|---|---|---|---|---|
| **Unified WAMs and video-action models** | | | | |
| DreamZero, "World action models are zero-shot policies" | [23] | ICLR Workshop, 2026 (no arXiv id given) | 14 | Defines WAM (Eq. 3); explicit IDM on a video backbone |
| Cosmos Policy | [21] | arXiv:2601.16163 | 13 | Joint prediction on a pretrained video model |
| UWM, "Unified world models" | [19] | arXiv:2504.02792 | 11 | Defines WAM (Eq. 3); four-component view |
| GR-2 | [38] | arXiv:2410.06158 | 9 | Autoregressive video-action co-prediction |
| GR-1 | [13] | ICLR, 2024 | 8 | Same; from-scratch backbone example |
| VPP, "Video prediction policy" | [20] | ICML, 2025 | 8 | Video prediction with a diffusion policy head |
| VidMan | [67] | NeurIPS, 2024 | 8 | Dual-system example |
| ViPRA | [97] | NeurIPS Workshop, 2025 | 8 | Latent actions aligned to robot control |
| Fast-WAM | [22] | arXiv:2603.16666 | 6 | Implicit IDM; "neither paradigm is universally superior" |
| LingBot-VA, "Causal world modeling for robot control" | [24] | arXiv:2601.21998 | 6 | Mixture-of-Transformers dual system |
| WorldVLA | [39] | arXiv:2506.21539 | 6 | Shared autoregressive token space |
| Seer, "Predictive inverse dynamics models are scalable learners for robotic manipulation" | [60] | ICLR, 2025 | 5 | Structured inverse-dynamics interface |
| GigaWorld-Policy | [25] | arXiv:2603.17240 | 4 | Action-centered implicit IDM |
| DWAM, "Do world action models generalize better than VLAs? A robustness study" | [105] | arXiv:2603.22078 | 3 | Robustness evidence (§III-C1) |
| π0.7 | [63] | arXiv:2604.15483 | 2 | Cited in §I for WAMs; goal-conditioned WAM |
| **Foundations** | | | | |
| π0 | [16] | RSS, 2025 | 10 | Flow-matching VLA; starred in Fig. 1 |
| OpenVLA | [11] | CoRL, 2024 | 9 | Autoregressive VLA |
| Wan | [76] | arXiv:2503.20314 | 7 | Pretrained video backbone |
| LAPA, "Latent action pretraining from videos" | [78] | ICLR, 2025 | 7 | Latent action thread |
| Diffusion Policy | [8] | IJRR, vol. 44, 2025 | 6 | Continuous action head |
| Cosmos, "World simulation with video foundation models for physical AI" | [77] | arXiv:2511.00062 | 5 | Pretrained video backbone |
| TD-MPC | [1] | ICML, 2022 | 5 | Implicit latent world model |
| Dreamer, "Dream to control" | [50] | ICLR, 2020 | 4 | Latent-imagination RL |
| DreamerV3 | [18] | Nature, vol. 640, 2025 | 4 | Same lineage |
| "World models" | [29] | arXiv:1803.10122 | 2 | Origin of the term |
| **World-model RL, domain anchors, position papers** | | | | |
| World4RL | [66] | arXiv:2509.19080 | 5 | RL refinement in a diffusion world model |
| World-VLA-Loop | [64] | arXiv:2602.06508 | 5 | Closed loop between world model and VLA |
| Pathdreamer | [2] | ICCV, 2021 | 5 | Navigation look-ahead |
| OccWorld | [59] | ECCV, 2024 | 5 | Occupancy forecasting for driving |
| GAIA-1 | [68] | arXiv:2309.17080 | 3 | Generative driving world model |
| "Critique of world model" | [49] | arXiv:2507.05169 | 3 | Basis of §VI-B |
| "Critique of agent model" | [248] | arXiv:2606.23991 | 2 | Basis of §VI-B |
| "Towards generalist embodied AI: A survey on world models for VLA agents" | [30] | TechRxiv, 2026 | 1 | The one reference with "survey" in its title |

The paper also maintains a method list at its project page (Abstract). That page was not consulted for this report.

---

## 9. Method, evidence and limits

**What was done.** The whole paper was read: all seven sections, three tables, seven figures with captions, and the reference list. No cited work was read, and no experiment was performed.

**Sources retrieved (5 Oct 2026).**

| Command | Result |
|---|---|
| `curl -sSL https://arxiv.org/html/2609.16074v1` | HTTP 200, 452,842 bytes |
| `curl -sSL https://arxiv.org/pdf/2609.16074v1` | HTTP 200, 4,531,144 bytes, 19 pages |
| `curl` of the six figure images the HTML references (`wam_background`, `wam_component`, `wam_paradigms`, `wam_models`, `wam_pipeline`, `wam_data`, all `.png`) | HTTP 200 each |

**Completeness checks.**

- The HTML was converted to text with a standard-library HTML parser (93,830 characters). Headings I through VII and all subsections through VI-F2 are present.
- The HTML References section is empty and citations render as BibTeX keys, so the PDF text was extracted with `pypdf` (121,405 characters). It contains the same body and a reference list numbered [1]–[268].
- BibTeX keys were matched to PDF reference numbers by aligning the word before each citation in the two versions, then spot-checked by searching the PDF text (for example "LingBot-VA [24]", "DINO [86]–[88]", "occupancy fields [249]", "DMLab [241]").
- Figures 2–7 were viewed as images. Figure 1 is assembled from fragments in the HTML, so PDF page 1 was rendered with `pypdfium2` and viewed.
- Counts in this report were computed from the extracted text: 66 entries in Tab. I (VAM 13, Uni 15, WM-Plan 13, RL 17, 3D 8), 48 in Tab. III, 20 metrics in Tab. II, 268 references of which 99 have an arXiv id and 52 a 2026 id, and per-work citation counts from the HTML body.

**Not verified.**

- Dataset scales in Tab. III, the notes in Tab. I and venue labels were transcribed from the paper and not checked against primary sources. The inconsistencies in 7.4 are internal to the paper; which side of each is correct was not determined.
- The paper's statements about individual methods (for example DreamZero's 30 minutes of data, DWAM's robustness result, Dyna-2's one million hours) were not checked against those works.
- The project page was not opened.
- Citation counts depend on text extraction and are indicative of emphasis, not exact bibliometrics.
- The reference errors in 7.4 can be confirmed from the PDF reference list. The points in 7.2 and 7.3 are arguments, and a reader may weigh them differently.

**Recommendation.** Use the paper as a taxonomy and a reading index, in the PDF version. For any decision that depends on whether world-action modeling improves control over a reactive policy, go to the primary comparisons the survey cites but does not summarize, starting with [22], [105] and [107], and to the definitional works [19] and [23]. Check dataset scales, venue labels and the references listed in 7.4 against primary sources before reusing them.
