# task1-3 优化建议（QEVD-FIT 阶段）

## 0. 原则

- Fit3D 到位前，本阶段只用 QEVD-FIT 内部 GT 做失败模式分析和方案筛选，不把结果当作 AIFit 论文指标复现。
- 先补独立标注和评估口径，再改算法。避免继续在未标注的失败段上调阈值，也避免在 7 个已标注段上同时调参和验收。
- AIFit、峰值基线和后续 hybrid 必须作为三个独立方法保留，使用同一份 GT 比较；不要把 hybrid 的收益算到 AIFit 上。
- 所有新增门控都要输出拒因和中间证据，不能只留下最终 rep 列表。

## 0.1 V0 最终快照（2026-10-25，已收口）

- 权威目录：`VLS/V0/results/segmentation_final/`；主结果文档：`VLS/V0/docs/task1-3_results.md`。
- 默认配置：`angle_scoring=legacy`、`p2a_tail_filter=guarded`、平滑 `7/2`。
- 最终指标：AIFit 365、Peak 341；5 段盲标 IoU=0.30 `66/66/0/0`、micro F1 `1.0`、raw IoU `0.6851`；IoU=0.50 F1 `0.8333`。
- P2-A 目标：`0218/air_jump_rope=39`，`0173/air_jump_rope=24`；walking-lunge stride `12/12`。
- 最终快照包含 `manifest.json`、CSV/Markdown 主表与消融表、6 张总图、21 张非零段定性图、golden 回归报告和 SHA-256 checksum。
- 冻结规则：不再修改 `segmentation/`、`segmentation_p0c/`、`segmentation_p0d1/`、`segmentation_p0d2/`、`segmentation_p1d/`、`segmentation_p1b/`、`segmentation_p2a/`、`segmentation_p2a_robustness/`；新实验另建目录。

当前未完成项只剩外部验证：Fit3D 官方逐 rep GT、真正独立的第二标注者、P2-A 新尾部样本、`9/2` 独立留出验证，以及 `mountain_climbers` / `toe_touchers` 口径复核。P1-C 在当前 count FP 已降到 0 后不启动。

## 1. 优先级总览

| 优先级 | 优化项 | 目的 |
|---|---|---|
| P0-A | 对高风险段做独立/盲标 GT | 消除循环标注，覆盖真正需要修的失败模式 |
| P0-B | 扩充评估指标与匹配口径 | 防止宽松 IoU 和宏平均掩盖过分割 |
| P0-C | 动作窗口拆成 active bouts | 避免休息和组间过渡被 AIFit 强行铺成 rep |
| P0-D | 整段 landmark 相位 + 残差边界抖动 | 先修 Stage A 整段平移，再修 Stage B 局部漂移 |
| P1-A | 修主角度评分并做多角度晚期融合 | 解决自动角度选择不稳定 |
| P1-B | 补齐 `rep_unit`，条件增加双侧相位检测 | 先明确 rep 口径；只有 GT 确认半周期错误时再做融合 |
| P1-C | 条件增加 AIFit-first peak support hybrid | peak 只做支持/反证，不用峰值决定候选数量 |
| P1-D | 使用鲁棒模板和可解释置信度 | 解释剩余 duplicate/transition，不直接用单一阈值门控 |
| P2-A | 收尾 `air_jump_rope` transition FP（已完成） | 盲标已确认 39 个 GT；默认 `guarded`，新 golden 为 365/341 |
| P2-B | 复核 `mountain_climbers` | 检查动作窗口、主角度和左右交替信号 |

P0-C 后的实际顺序：P0-C 收口检查 → P0-D1/D2 → P1-D → P1-B1 → P1-A → P1-C（条件）→ P2。

## 2. P0：先保证 GT 和评估可信

### P0-A：高风险段独立标注

现有 7 段内部 GT 用于第一轮筛查，但 `0161/lunge_jumps` 和 `0218/walking_lunges` 保留了 AIFit 草稿边界，AIFit IoU=1 存在循环性。下一步至少补标：

- `0161/pushup`：休息/恢复段假 rep。
- `0173/boxing_squat_punches`：末尾不完整深蹲。
- `0161/lunge_jumps`：独立重标，取消草稿边界复用。
- `0218/walking_lunges`：独立重标，并明确 `rep_unit`。
- 追加 `0218/air_jump_rope`、`0162/mountain_climbers` 等未标注低置信段。

标注时优先隐藏 AIFit 红线和峰值蓝线，只看原始抽帧或视频。无法增加第二标注者时，至少隔天做一次独立二标；建议二标比例不低于 20%，记录 start/end 帧差异和 rep 数差异。

验收标准：

- 高风险段不再直接沿用草稿边界。
- 每个动作明确记录 `rep_unit` 和边界相位。
- 报告 `pending`、`delete`、`add` 之外，还能按失败类型统计 false positive。

**执行状态（2026-10-25）**：盲标主体已完成。新 GT 保存在 `VLS/V0/results/segmentation/annotation_blind/`，5 段共 66 个 rep 进入评估；`0162/mountain_climbers` 因相位和姿态质量不稳定整段排除。已抽查 22.7% rep 做同一标注者的原始帧重读，但独立第二标注者仍为 `pending`，不能把当前一致率写成 inter-annotator agreement。结果与复现命令见 `VLS/V0/docs/task1-3_p0a_results.md`。

### P0-B：评估口径

当前匹配默认 `IoU >= 0.10`，过于宽松，不足以单独支撑 F1 结论。建议：

- 同时报告 `IoU` 阈值 0.10、0.30、0.50，0.30/0.50 作为主要质量指标。
- 保留 raw IoU 和 phase-aligned IoU，两者分开解释；phase-aligned IoU 不能替代原始边界精度。
- 报告过分割率、欠分割率、带符号的 start/end 边界误差、p95 边界误差。
- 将 false positive 分为 `rest`、`transition`、`incomplete_rep`、`duplicate`，分别统计。
- 同时给 macro 和按 rep 数加权的 micro 指标，避免短视频段和小段被过度放大。
- 参数选择与最终验收按视频或动作分组，不在同一批段上调参后再原样报告。

**执行状态（2026-10-25）**：已完成。`blind_eval.csv` 同时覆盖 IoU 0.10/0.30/0.50；新增 `blind_eval_by_rep_unit.csv`、`blind_segmentation_errors.csv` 和完整 signed/p95 边界误差。split/merge 主口径为“交集至少占较短区间 50%”，避免短周期边界平移误报。详细结果见 `VLS/V0/docs/task1-3_p0b_results.md`。

## 3. P0：修停顿/过渡和边界相位

### P0-C：先拆 active bouts，再跑 AIFit

当前 AIFit 没有“零 rep”状态。动作窗口里混入休息或过渡后，边界仍会被铺满，单靠 rep 内低方差惩罚无法可靠删除错误 rep。并且现有假 rep 可能时长正常、局部角度变化也不小，单纯方差阈值抓不到。

建议实现：

- 用约 0.5 s 滚动窗口计算主角度角速度、多角度活动度、静止平台和低可见度比例。
- 先检测动作窗口内部的 rest/transition 区间，把窗口拆成若干 active bout。
- 每个 active bout 内单独运行 AIFit；bout 之间输出 `non_rep` 区间，不生成 rep。
- 原局部幅度过滤 `local_range < 0.25 * global_range` 继续保留，但只作为辅助证据；不要再用同类方差阈值重复过滤。
- 如果某个窗口无法稳定拆出 active bout，整段标记低置信，不强行输出 rep。

验收标准：

- `0161/pushup` 的休息/恢复假 rep 被删除，同时不丢有效 rep。
- `0173/boxing_squat_punches` 末尾不完整深蹲被归入 `non_rep`。
- `0153/squat` 半跪过渡被归入 `transition`。
- `run.json` 记录 active bout、拒绝区间、拒因和所用证据。

**执行状态（2026-10-25）**：已完成规则版 V0。`segment_reps.py` 新增 0.5 s 多角度活动度、显著低活动区间、逐 bout AIFit、末尾 incomplete 检测，以及 `run.json`/`non_rep_intervals.csv`/`rejected_reps.csv` 证据输出。三段验收结果为 `0153/squat` 9→8、`0161/pushup` 7→5、`0173/boxing_squat_punches` 11→10。P0-C 盲标评估在 IoU=0.30 下由 AIFit F1 `0.956` 提升到 `0.977`，FP `5→2`；剩余为 boxing 的 1 个 duplicate 和 air jump rope 的 1 个 transition。峰值基线保持独立，不加 active-bout 门控。详细结果见 `VLS/V0/docs/task1-3_p0c_results.md`。参数在包含验收段的样本上确定，尚未做视频级留出，因此该提升只作为 QEVD-FIT 内部诊断，不是 Fit3D 官方指标。

**P0-C 收口检查（进入 P0-D 前）**：

- 冻结 P0-C golden regression：5 段盲标 GT、3 段指定验收、IoU=0.30/0.50 指标和 rep 数作为后续改动的最低回归集。
- 抽帧复核未盲标的 rep 数变化：`0153/jumping_jack` 28→30、`0161/butt_kickers` 19→17 有周期/原始帧支持；`0218/toe_touchers` 原 4 个候选完全落在 fit extension、不与 active core 相交，收口后为 0。该段口径仍未解决，继续排除，不作为 P0-D/P1 调参真值。
- 增加 synthetic 回归：周期正弦、周期 + 休息平台、半周期结尾、变周期和低可见度缺帧；任何 P0-D/P1 改动都必须同时跑 synthetic 和 golden regression。
- P0-C 参数是在包含验收段的样本上确定且未做视频级留出，后续不得继续用同一小样本反复调活动度阈值后再宣称泛化结果。

**执行状态（2026-10-25）**：已完成。冻结配置为 `VLS/V0/configs/p0c_golden.json`；统一入口 `VLS/V0/scripts/check_p0c_regression.py` 在 `exact` 模式下 synthetic `7/7`、22 段计数、5 段盲标和 active-core 不变量均为零差异。P0-C 收口时 AIFit 全量为 366 个 rep，三段验收与 IoU=0.30/0.50 指标不变；P2-A 后最终为 365。详细记录见 `VLS/V0/docs/task1-3_p0c_closure.md`。

### P0-D：边界相位一致性

P0-C 后，IoU=0.30 的 AIFit F1 已达 `0.977`，只剩 2 个 count-level FP；但 IoU=0.50 F1 仅 `0.677`，平均边界误差约 `9.0` 帧、p95 约 `20` 帧。当前主要损失已经转为边界相位，因此 P0-D 应保持最高优先级。

当前盲标误差包含两类，不能用同一个门控同时处理：

- **整段相位平移**：`0161/lunge_jumps` 约 -18 帧且起点标准差仅 1.1 帧；`0173/boxing_squat_punches` 约 -11 帧；`0218/walking_lunges` 约 -9 帧。这类问题主要来自 Stage A 的绝对相位选择，相位方差惩罚无法修复。
- **bout 内边界抖动**：`0161/pushup` 起点平均偏移 +24 帧、标准差 27.2 帧；boxing 起止标准差约 12 帧。这类问题才适合约束 Stage B 的逐边界移动。

`0218/air_jump_rope` 的相位平移接近 0、边界误差较小，主要问题仍是 split/merge 和尾部 transition，不作为 P0-D 的主验收段。

建议拆成 D1、D2 两个独立步骤，不要一次叠加多种强约束：

**P0-D1：landmark 约束的整段相位选择**

- 优先使用盲标元数据中的 `phase_landmark` / `phase_polarity`，在 Stage A 选择靠近同极性峰/谷的起始相位，而不是只取相关分数最大相位。
- `pushup`、`boxing_squat_punches`、`air_jump_rope` 使用 peak/standing-extension；`lunge_jumps` 使用 trough；`walking_lunges` 明确统一到同一侧落脚相位。
- 如果局部极值不清晰或多个候选相位分数接近，输出 `phase_ambiguity` / `low_phase_confidence`，不要强行吸附。
- D1 目标是先消除整段平移；验收时必须同时报告 `phase_shift_frames`、raw IoU 和 signed start/end，不能只看 phase-aligned IoU。

**P0-D2：残差边界抖动**

- 仅在 D1 完成后仍有明显 bout 内抖动时，再加入边界相位方差惩罚或 ±10% 周期内吸附。
- 优先处理 `pushup` 起点和 boxing 起止；不要对 air jump rope 的短周期边界强行使用长周期门控。
- D1/D2 分开提交和评估，避免无法判断收益来自绝对相位还是局部约束。

验收标准：

- D1 后 `lunge_jumps` 的绝对 `phase_shift_frames` 从 19 降到 ≤5，`boxing_squat_punches` 从 26 降到 ≤8，`walking_lunges` 从 8 降到 ≤5。
- D2 后 `pushup` 起点误差标准差从 27.2 降到 <10，boxing 起止标准差从约 12 降到 <8；未达到时保留原边界并标记低相位置信度。
- 5 段盲标的 rep 数和 IoU=0.30 F1 不下降；raw micro IoU 不低于 P0-C 的 `0.585`，IoU=0.50 F1 从 `0.677` 开始提升。
- `0173/good_morning` 降为第一轮 GT 的辅助检查，不再作为 P0-D 主验收段；P0-C 后它已无整段 phase shift，且不在 5 段盲标集内。
- “假 rep 不落在恢复段”保留为 P0-C 验收，不属于 P0-D 的核心指标。

**执行状态（2026-10-25）**：P0-D1 已实现并完成回归。`lunge_jumps` shift `19 -> 1`、`walking_lunges` shift `8 -> 5`，达到目标；`boxing_squat_punches` 因相位极值覆盖率不足按设计回退，shift 仍为 `26`，未达到 `<=8`。22 段 rep 数不变，盲标 IoU=0.30 F1 保持 `0.977`，raw micro IoU `0.585 -> 0.629`，IoU=0.50 F1 `0.677 -> 0.722`。P0-C exact/guard 与 D1 frozen regression 均通过。详见 `VLS/V0/docs/task1-3_p0d1_results.md`。

**P0-D2 执行状态（2026-10-25）**：已实现并冻结回归，验收为 **partial**。Stage B 后新增动作级 `plateau_entry` 吸附，只对 `pushup` 和 `boxing_squat_punches` 启用；`--no-phase-snap` 可精确复现 D1。boxing start/end SD `12.73 / 12.18 -> 3.42 / 4.77`，phase shift `26 -> 3`，达到 `<8`；pushup start SD `27.22 -> 16.33`，未达到 `<10`，剩余误差来自两个 bout 的 fit-window 外首边界。5 段 rep 数不变，IoU=0.30 TP/FP/FN `65/2/1 -> 66/1/0`、F1 `0.977 -> 0.992`，raw micro IoU `0.629 -> 0.685`，IoU=0.50 F1 `0.722 -> 0.827`。22 段仍为 366 rep；D2 frozen、D1 exact、P0-C exact/guard 均通过。详见 `VLS/V0/docs/task1-3_p0d2_results.md`。

**P1-D 执行状态（2026-10-25）**：已实现并完成回归，验收为 **pass_review_only / partial**。新增 IRLS/medoid/trimmed 模板和 `visibility`、`shape_similarity`、`phase_consistency`、`motion_evidence` 四项证据；legacy `confidence` 保留，低置信 rep 只输出 `review_priority` / `reject_reason`，自动删除关闭。IoU=0.30 的唯一 transition FP（`0218/air_jump_rope` `1663-1679`）的 `motion_evidence` 排该段倒数第 2、低于 TP p05；相对 legacy confidence，ROC AUC `+0.121`、AP `+0.00192`。在 P0-C recall `0.98485` 下 motion 的 precision 可从 `0.98507` 到 `1.0`，但会损失当前 66 个 TP 中的 1 个，因此不启用门控。模板替换与 D2 边界指标完全一致；LOSO 因所有 FP 集中在单一视频段而不可识别，已如实记录。详见 `VLS/V0/docs/task1-3_p1d_results.md`。

## 4. P1：主角度、rep 口径与融合

P0-C 后的 P1 调整原则：

- P0-D1 完成前不启动 P1-A/P1-C，避免主角度或 hybrid 的变化掩盖相位问题。
- P1-D 提前到 P0-D 之后；当前剩余 FP 主要是 mid-bout duplicate，单靠活动度门控无法解决。
- P1-B 的元数据可以并行补；`0218/walking_lunges` 已有 `stride` 盲标且未发现半周期错误，`0162/walking_lunges` 仍无盲标 GT，左右相位融合需等两段口径都确认后再决定。
- P1-C 降为条件实验；若 P0-D/P1-D 后只剩 1–2 个 count FP，不引入 hybrid 的额外复杂度。

### P1-A：修主角度评分

当前 `choose_auto_angle()` 已经计算了 `periodicity`，但最终只按动态范围和有效覆盖率排序，周期性没有参与选择。先修这个逻辑，再做复杂融合：

- 有效覆盖率和最小动态范围作为硬门槛。
- 评分至少包含周期性、主峰显著度、周期内相位一致性和边界稳定性。
- 设置动作相关周期范围；不要所有动作共用 0.4–5.0 s。
- 多角度融合采用晚期融合：分别估计周期和相位，选择周期一致性最高的角度簇，不直接平均原始角度。
- 用 action-level 留出验证选择权重；V0 先做规则版，不直接引入 SiTrEx AU 学习模型。

验收口径需要调整：固定角度不是 GT，不能把“auto 与 fixed 更一致”或“top-1 更稳定”当成正确性。当前 P0-C 消融中，`lunge_jumps` auto 6 vs fixed 12（盲标 GT 6），`air_jump_rope` auto 40 vs fixed 5（盲标 GT 39），说明现有自动选择在已标注段反而更接近正确计数。

验收标准：

- 5 段盲标的 rep 数、IoU=0.30 F1 和 raw IoU 不下降；`lunge_jumps`、`air_jump_rope`、`walking_lunges` 保持当前正确计数。
- 输出候选角度的周期一致性、相位一致性、动态范围和歧义分数；分数接近时标记 `angle_ambiguous`，不强行切换 top-1。
- 换角度消融必须同时报告 rep 数和 GT IoU，不能只报告角度一致性；只有在 action-level 留出集上同时改善计数和边界时才替换当前评分。
- `0162/mountain_climbers` 在整段排除状态解除前不作为 P1-A 验收样本。

**P1-A 执行状态（2026-10-25）**：已实现并通过专项回归，但验收为 **pass_experiment / not_enabled**。新增动作相关周期范围、周期性、主峰显著度、周期内相位一致性、边界稳定性和周期簇晚期融合；全量输出 136 条候选证据。当前 19/22 段为 `angle_ambiguous` 并回退 legacy，3 段由簇选择；5 段盲标、全量 rep 数和边界与 P1-D 完全一致，但没有计数或边界收益。按“只有同时改善计数和边界才替换”的口径，默认仍为 `legacy`，`evidence` 保留为候选审计模式。详见 `VLS/V0/docs/task1-3_p1a_p2a_results.md`。

### P1-B：明确 `rep_unit` 和左右交替

`rep_unit` 元数据继续保留，但原计划中的左右相位融合需要降级并拆成两步。P0-B 已按 `rep_unit` 分组，盲标 `0218/walking_lunges` 以 `stride` 评估时 AIFit 为 6/6、F1 1.0，目前没有证据表明单侧角度造成了半周期漏计。

**P1-B1：补齐元数据（可立即并行）**

- 为 22 个动作段补齐 `rep_unit`，至少区分：
  - `side`：单侧一次算一个 rep。
  - `stride`：左右各一步算一个 rep。
  - `cycle`：一个完整动作周期算一个 rep。

- 明确 `phase_landmark` 和边界极性，评估表继续按 `rep_unit` 分组。
- 优先补盲标 `0162/walking_lunges`，并与已有 `0218/walking_lunges` 的 `stride` GT 对齐口径；`0162` 当前 AIFit 6 vs peak 10，未确认前不能判断哪边正确，也不能据此启动左右融合。

**执行状态（2026-10-25）**：已完成。22 段元数据写入 `VLS/V0/configs/rep_metadata.json`，`segment_reps.py` 已向后兼容输出 `rep_unit/phase_landmark/phase_polarity`。`0162/walking_lunges` 原始帧盲标为 12 次交替迈步、6 个完整 `stride`，AIFit 6/6、IoU=0.30 F1 `1.0`；Peak 10 reps 为 6 TP + 4 FP，F1 `0.75`。`0218/walking_lunges` 保持 6/6；两段合并 AIFit 12/12。未发现单侧角度半周期错误，**P1-B2 不启动**。P2-A 前全量 rep 数为 AIFit 366、Peak 341；P2-A 后最终为 365/341，相对 P1-D 边界差异仅目标尾部。结果与验证命令见 `VLS/V0/docs/task1-3_p1b1_results.md`。

**P1-B2：左右相位融合（条件执行）**

- 只有盲标确认单侧角度确实把半周期算成一个 rep，或漏掉另一侧周期时，才联合左右膝/髋角、触地事件和不对称度。
- 只用单侧角度或左右角差过零点仍可能把半周期误算为完整周期。
- 验收必须按 `stride`/`cycle` 分组，并证明融合后计数和边界至少一项改善、另一项不退化。

**执行状态（2026-10-25）**：不启动。P1-B1 的两段 walking lunges 在 `stride` 口径下 AIFit 均为 6/6，没有半周期误计或另一侧漏计证据；Peak 的 10 reps 是单步事件过分割，不能作为启动融合的理由。

### P1-C：peak + AIFit hybrid

P1-C 降级为条件实验。当前剩余 AIFit count FP 只有 boxing 的 1 个 duplicate 和 air jump rope 的 1 个 transition；峰值基线本身 F1 `0.744`、air jump rope 28/39、`0218/toe_touchers` 为 0。让 peak 决定候选数量风险大于收益。

如果启动，只做 **AIFit-first** 版本：

- AIFit 负责 rep 数和主边界；peak 只提供支持/反证，不重新生成整段候选。
- 每个 AIFit rep 计算 peak support：是否包含一个高显著度主峰/谷、是否包含多个主峰、峰显著度相对局部运动幅度的比例。
- peak support 不足时保留 AIFit rep 并输出 `peak_support=low`，不直接用 peak 计数替换。
- 只有 peak 与 AIFit 都明确支持删除，且不造成 air jump rope recall 下降时，才允许删除 duplicate。
- 保留 AIFit、peak、hybrid 三行结果；hybrid 不能替代现有基线。

验收标准：

- 5 段盲标无新增 FN，IoU=0.30 F1 不低于 P0-D/P1-D 版本。
- air jump rope 保持 39 个 TP，不因 peak 支持弱而删除真实短周期 rep。
- boxing duplicate 被删除时不能丢失对应 GT rep；其余段边界误差不显著增加。
- `0218/toe_touchers` 等峰值基线为 0 的段必须自动回退 AIFit，不能因融合改变 rep 数。
- 若 P0-D/P1-D 后只剩 1–2 个 count FP，取消 P1-C，不进入 V0 主流程。

### P1-D：鲁棒模板和置信度

**执行结果（2026-10-25）**：已完成 review-only 版本。默认 IRLS 模板；四项证据已写入 `reps.csv`/`run.json`；`motion_evidence` 解释 Air Jump Rope 尾部 transition FP，但固定 recall 下无法在不损失当前 D2 TP 的情况下自动删除，故 `confidence_auto_delete=false`。PR、固定 recall、模板消融和 LOSO 可识别性结果见 `VLS/V0/docs/task1-3_p1d_results.md`。

P1-D 提前到 P0-D 之后执行。当前单一 `confidence = visibility × similarity` 不能作为删除门控：P0-C 的两个 FP confidence 为 `0.282`、`0.355`，但 TP 的 25 分位也有 `0.592`；若使用 `<0.4` 阈值，会删除约 10 个有效 rep。当前剩余错误需要更细的可解释证据，而不是调高单一阈值。

建议：

- 先用 medoid、trimmed mean 或迭代重加权生成模板，再计算 rep 相似度。
- 置信度至少拆成 `visibility`、`shape_similarity`、`phase_consistency`、`motion_evidence` 四项。
- `phase_consistency` 使用 P0-D 后的 landmark 偏移和边界相位方差，不使用 phase-aligned IoU。
- 用 action-level / leave-one-segment-out 校准，报告固定 recall 下的 precision、PR 曲线和每项特征的消融，不在同一批盲标段上拟合后再原样报告。
- 低置信 rep 不直接删除，先输出 `review_priority`、分项置信度和 `reject_reason` 供人工复核；自动删除只在至少两个独立证据同时支持且不损失盲标 TP 时启用。

验收标准：

- 5 段盲标上至少一个分项能解释剩余 duplicate/transition FP，并报告其相对单一 confidence 的增益。
- 在 recall 不低于 P0-C 的前提下比较 precision；如果只能通过损失 TP 才能提高 precision，则不启用自动门控。
- 模板替换不能让 raw IoU、IoU=0.50 F1 或边界 p95 退化；收益必须来自边界或错误排序，而不是只让 confidence 分布更“好看”。

## 5. P2：特定动作

### `0218/air_jump_rope`

盲标已确认该段为 39 个完整 rep（`cycle`），P0-C AIFit 当前为 40，峰值基线为 28。剩余问题不是“40 vs 28 哪边正确”，而是 AIFit 尾部 `1663-1679` transition 未删除，以及短周期边界存在 split/merge；峰值基线存在明显漏检。

AIFit 中位周期约 17 帧（0.57 s），直接加 0.8 s 最小 rep 时长可能误删真实 rep。后续调整：

- 使用动作相关的 `min_duration_ratio × τ*`，不要全局固定秒数。
- 优先增加身体垂直位移、髋中点高度或踝部运动作为辅助信号；肘角可能主要反映手臂摆动。
- 报告不同 cadence 下的 rep 数变化，检查是否把左右脚或绳子周期重复计数。
- 只针对尾部 transition 和 split/merge 做收尾；保持 39 个完整 rep 的 recall，不把峰值 28 作为目标计数。

**P2-A 执行状态（2026-10-25，已收口）**：仅对 `air_jump_rope` 的最后一个 rep 启用“尾部位置 + 至少两项独立证据”门控；`0218` 的 `1663-1679` 同时满足相位 apex 缺失、低 shape similarity 和低 motion evidence，被删除并记录拒因。`0173/air_jump_rope` 保持 24 rep，其余 21 段逐边界不变；5 段盲标 IoU=0.30 从 `67/66/1/0` 改善为 `66/66/0/0`，F1 从 `0.99248` 到 `1.0`。默认值已切为 `--p2a-tail-filter guarded --smooth-window 7 --smooth-polyorder 2`，新基线为 `results/segmentation_p2a/`（AIFit 365、Peak 341），由 `configs/p2a_golden.json` 和 `check_p2a_regression.py` 约束；旧阶段回归显式传 `--p2a-tail-filter off`。

平滑消融显示 `7/2` 是唯一在不改变任何 segment-method 边界时保持主指标的配置；`11/2` 把 `0218` 压到 12 rep，`9/2` 虽略降 p95 但改变 33 个边界，仅作事后候选。角度噪声在 `1–2°` 内接近基线，`5°` 起明显退化，`10°` 不可用；nose 相位噪声 `0.002–0.02 m` 下 P2-A 目标段均保持，但 `0.005 m` 单个 trial 已使 `0173` 增加 1 rep。详细的配置、seed、CSV/JSON 和限制见 `VLS/V0/docs/task1-3_p1a_p2a_results.md` 与 `results/segmentation_p2a_robustness/`。

### `0162/mountain_climbers`

先复核动作窗口是否包含过渡，再检查主角度。建议同时使用双侧髋/膝角度、左右相位差和躯干稳定性；在定义 `side` 或 `stride` 前，12 vs 2 不能用于判断算法优劣。若窗口本身不干净，整段标记 `exclude`，不要继续调分割参数。

## 6. 工程和回归测试

- 增加 synthetic 测试：周期正弦、周期 + 休息平台、半周期结尾、变周期、低可见度缺帧。
- 建立 golden regression：P0-A 5 段盲标 GT、P0-C 三段指定验收和 22 段 rep 数快照；第一轮 7 段内部 GT 仅作辅助。任何算法改动都输出 rep 数、FP/FN、raw IoU、phase shift、IoU=0.50 F1、边界 SD 和拒绝区间对比。
- P0-C 图已增加 non-rep、active bout 和被拒绝 rep；后续补 activity / rolling speed 子图，便于判断 P0-D/P1 改动是否改变了活动门控。
- 保持输出接口兼容：`{video_id, reps: [{rep_id, start_frame, end_frame, confidence}]}`，新增字段只做向后兼容扩展。

## 7. 建议执行顺序

1. **P0-C 收口（已完成）**：golden regression、synthetic tests 和三段抽帧复核已完成；后续改动统一运行 `check_p0c_regression.py --mode guard`。
2. **P0-D1（已完成，boxing 未达标）**：用 `phase_landmark` / `phase_polarity` 约束 Stage A。lunge、walking lunges 已达标；boxing 因相位极值覆盖不足回退，留给 D2/P1-B 继续分析。
3. **P0-D2（已完成，pushup 首边界 partial）**：boxing 已达标；pushup 除两个 fit-window 截断首边界外均已收敛。后续若继续追 `<10`，必须先设计 fit-window 外相位恢复并重新验证不会生成恢复段 rep，不能直接扩大吸附半径。
4. **P1-D**：做鲁棒模板、四项可解释置信度和 leave-one-segment-out 校准；先排序/复核，不自动删除 TP。
5. **P1-B1（已完成）**：22 段 `rep_unit` 和相位元数据已补齐；`0162/walking_lunges` 已确认 6 个 stride，与 `0218` 合并 AIFit 12/12，未触发 P1-B2。
6. **P1-A**：在 P0-D 后修角度评分，以 GT 计数/IoU 和非回归为验收，不以固定角度一致性或 top-1 稳定为目标。
7. **P1-C**：只有剩余 count FP 仍大于 2 且 P0-D/P1-D 无法解释时才做 AIFit-first hybrid；否则取消。
8. **P2-A（已完成收口）**：默认 `guarded`，新 golden 固定 365/341 和 22 段计数；平滑与噪声实验已完成。后续仅在 Fit3D、独立二标或新增尾部样本到位后复验，不继续在当前盲标上调阈值。
9. 最后再考虑学习式 AU 或更复杂的动态规划分割，避免 V0 过早增加不可解释参数。
