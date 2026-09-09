# V1-R 执行结果

- decision: `blocked_seed0_clean_dev`
- status: `completed_seed0_clean_dev_failed`
- latest completed stage: `V1-R.2K.seed0`
- selected label contract: `delta_pose_float32_identity`
- frozen B1-2K-20 seed-0 clean-dev success rate: `3/40 = 0.075`.
- scientific claim status: 量化后标签、点观测和 chunking 合同已通过；策略基线未建立，不宣称策略学习增益或真实机器人迁移。
- training authorization: `B1-2K-20` seed 0 已完成；confirm、其他 seed、V2 和 V3 仍未授权。

## V1-R.2K seed-0 clean-dev

- CUDA training completed with return code `0`, 300000 effective steps, primary checkpoint `300000.pt`.
- Primary checkpoint SHA-256: `e827f24d228c295c3f5f09d0d051692d573f73d7b41a181eefe4f84d590f7589`.
- Layout successes: `0/10`, `2/10`, `0/10`, `1/10`; total `3/40`.
- Current initial-state matches: `40/40`; historical compatibility hash matches: `35/40`.
- Simulator exceptions: `0`; action decode errors: `0`.
- Failure stages: `no_approach=9`, `no_grasp=22`, `post_grasp_drop=6`.
- Gate: failed; required at least `20/40`.
- Full upload-safe report: `experiments/v1r/reports/v1r_2k_seed0_clean_dev.md`.

The next stage is `diagnose_pilot_coverage_or_learnability`. Do not start
confirm rollouts, remaining seeds, V2, or V3 from this result.

## V1-R.2K.3-F point-observation freeze

- 修复 Point Bridge 包装器定位：不再因外层 `__getattr__` 转发而把状态和固定点模板写到错误层。
- 20/20 episode 的机器人点使用上一条夹爪命令；每个 episode 只采样一次 float32 对象坐标系模板，后续逐帧仅做刚体传播。
- 动态验证通过：机器人夹爪点 `20/20`、固定对象模板 `20/20`、对象点刚体传播 `20/20`、点值有限 `20/20`、真实 40 步 chunk `80/80`，显式 populated mask 已启用。
- 修正后的四个 PKL、manifest、verification、episode 有序映射和 `0005`/`0006` 补丁身份记录在 `experiments/v1r/manifests/b1_2k_dataset_artifact_index.csv`；旧 PKL 标记为 `superseded_point_observation_mismatch`。
- 数据范围严格限定为四布局各 5 条的 `balanced_20_episode_seed0_pilot`，未穷尽候选全集。门槛通过后仅授权一个 `B1-2K-20` seed 0；不直接授权 confirm、V2 或 V3。

## V1-R.2K.3 B1-2K dataset freeze

- 新增 `0006-mimiclabs-delta-pose-chunk-contract.patch`：delta 模式末端只补零位移并保持最后夹爪命令，时间聚合使用显式 populated mask，合法全零 delta 不再被误判为未填充。
- 使用 `/home/xushijie/vico-point/third_party/pointbridge` 的正式运行环境生成四个布局 PKL；20 个 episode 全部来自已接受量化源 artifact 的 `states_before` 和 `float32_labels`，未调用旧 PKL 生成入口。
- 原生 `BCDataset(action_chunking=true, num_queries=40)` 读取检查：4/4 文件、20/20 episode、20/20 标签精确一致、20/20 上一条夹爪命令一致；末端 padding 为 `zero_motion_hold_last_gripper`。
- 该阶段生成的旧点 PKL 后续发现机器人夹爪点和对象模板不满足部署一致性，现由 V1-R.2K.3-F 修正版取代；动作块修复和历史结果仍保留。

## V1-R.2K.2 seed-0 readiness

- 新增 `0005-mimiclabs-delta-pose-float32-identity-training.patch`，采集、回放、`BCDataset` 与 `BCAgent` 共同调用 `vico_point.action_contracts`；delta 标签保持 float32 identity，部署输出 float64 控制命令，不再使用动作 min-max。
- 现有 20 条工件经过真实 `BCDataset` 和 PyTorch `DataLoader` 后标签与 batch 均为 `20/20` 精确一致，共享 decoder 命令 `20/20` 精确一致，agent 部署后处理检查通过。
- 本轮不训练网络。现有工件还没有完整 Point Bridge 点输入训练 PKL，也没有冻结 B1-2K 全成功数据清单，因此 seed 0 继续未授权。

## V1-R.2K quantized-at-source data

- 采集严格使用 `raw_action_float64 -> float32 label -> float64 controller command`，不使用数据集 min-max、额外缩放、裁剪、平滑或第二次量化。
- 按布局内 demo 编号升序执行候选；每次尝试只在回合开始恢复初态，后续连续执行，不在中途恢复状态；41 次尝试全部保留。
- 四布局分别为 `22/5`、`9/5`、`5/5`、`5/5`（attempted/accepted），共 20 条接收演示。旧 V1-R.2J-N 的四条诊断结果未被覆盖；新采集中的 `layout_1/demo_18` 是独立的量化源采集结果。
- 保存的 float32 标签经共享解码器回放，20/20 通过；控制命令的值、顺序和长度完全一致，采集与回放状态序列逐步完全一致，最大绝对误差为 `0.0`。
- 该门槛冻结数值合同，但只授权另行决定是否重建训练数据并启动 seed 0；不授权 B0/B1、confirm、V2 或 V3。
- legacy V1 结果未被覆盖，confirm 未用于覆盖 dev 失败。

## V1-R.2J executable action-contract reconstruction

- 固定使用 V1-R.2I 的同一 20 条正式连续成功演示；没有重新采集、没有加载中间状态、没有策略训练或推理。
- 绝对合同基础回放：S0-old `3/20`、S0-transition `3/20`、S1-PB `16/20`、S1-world `17/20`；统一成功后尾段诊断分别为 `11/20`、`9/20`、`18/20`、`18/20`。四个绝对合同均未达到严格 `20/20`。
- 新增 7 维 `delta_pose` 接口并通过官方 Point Bridge `delta_pose`/delta OSC 运行时验证。初态匹配、归一化往返、夹爪符号保持和官方 delta runtime 均为 `20/20`，但任务执行回放为 `17/20`。
- delta 失败轨迹为布局 1 `demo_18`、布局 3 `demo_10`、布局 4 `demo_2`。因此 delta 仅登记为未验证的明确备用接口，不能冻结为训练标签合同。
- V1-R.2J 结论：实验完成，绝对动作合同与 delta 备用合同均未通过严格 `20/20`；`selected_label_contract: null`，B0/B1、seed 0、confirm、V2、V3 均未授权。

## V1-R.2J-F numeric path diagnostic

- 在四条冻结演示上执行 24 次短回放，交叉比较采集环境/Point Bridge `delta_pose` 入口与 raw `float64`、raw `float32`、shared min-max -> `float32` -> inverse `float64` 三种动作值路径。
- 结果为 A/C `4/4`、E/F `3/4`、B/D `1/4`。对应环境入口的底层 `step()` 动作数组、调用数量和任务结果一致，Point Bridge 入口不是直接触发因素。
- `demo_18` 在直接 `float32` 下已经失败；min-max -> `float32` -> inverse 又额外使 `demo_10` 和 `demo_2` 失败。失败阶段均为 `contact_without_grasp`。
- 结论：问题已收窄为数值动作表示/精度敏感性，但还没有通过 `20/20` 的训练数值合同；`selected_label_contract: null`，所有训练和下游方法实验继续未授权。
- 证据：`experiments/v1r/reports/delta_pose_path_diagnostic_2j_f.json`、`delta_pose_path_diagnostic_2j_f.md`、`v1r_2j_f_delta_pose_path_diagnostic.yaml`。

## V1-R.2J-N numeric contract candidate

- 仅在 Point Bridge 入口测试 `raw.astype(float32).astype(float64)`，不做 min-max；四条定点轨迹结果为 `3/4`，失败仍是布局 1 `demo_18` 的 `contact_without_grasp`。
- 候选与直接 float32 Point Bridge 路径的底层动作、控制目标和物理轨迹逐值一致，因此控制器入口 dtype 不是修复。按固定规则没有运行全 20 条。
- 本机 robosuite `1.4.1` 的 `math.isclose` 分支存在，但四条数据中没有旋转零值分支决策变化，该假设在当前样本上不成立。
- 下一阶段为 V1-R.2K quantized-at-source 独立数据版本；旧 20 条合同失败结果不覆盖，训练及 V2/V3 继续未授权。

## V1-R.2J evidence hashes

- absolute runtime result: see `experiments/v1r/manifests/executable_absolute_label_contract_artifact_index.csv`
- delta runtime result: see `experiments/v1r/reports/delta_pose_contract.json` and the same artifact index

## V1-R.2F audits

- initial_state_pairing_audit: `passed`
- runner_parity_audit: `passed`
- cpu_cuda_parity_audit: `blocked_missing_results` (CUDA visible; no paired CPU/CUDA result)
- expert_replay_audit: `failed`
- per_layout_failure_analysis: `complete`

## 首个差异定位

限定诊断已经找到首个可解释差异：动作 0 的控制目标重建正确，但执行一个 `0.05 s` 控制周期后 EEF 状态先发生偏差；delta OSC 随后把该偏差带入动作 1 及之后的目标。MuJoCo 3.1.1 中布局 1/2/3 的首步位置误差为 `0.221022/0.405555/0.0603514 mm`，布局 1/2 还出现更大的早期目标跟踪距离和翻译动作饱和。该证据支持先修复动作执行/环境契约，再考虑训练；不授权重新训练，也不宣称四布局通过。

## V1-R.2G absolute-pose contract

- 正式运行时：robosuite `1.4.1`、MuJoCo `3.3.5`。
- 原生动作链：`BCDataset`、`act_subsample=1`、原生四元数/6D 转换及训练/部署归一化往返。
- 官方环境：`point_bridge.suite.mimiclabs.make()`、`OSC_POSE`、`control_delta=False`、20 Hz；不运行策略网络。
- 初态与执行：20/20 初态严格匹配，3,832 个执行步均有目标、实际 EEF、误差和边界记录，模拟器异常为 0；训练归一化标签为 `float32`，部署反归一化动作为 `float64`。
- 结果：布局 1/2/3/4 分别为 `1/5`、`3/5`、`4/5`、`5/5`，整体 `13/20`；7 条失败均为 `no_grasp`。
- 决策：情况 B，`pointbridge_absolute_pose_contract_gate=failed`，B0/B1 训练、V2 和 V3 均未授权；随后进入 V1-R.2H 定点实验，不重新训练。

## V1-R.2H pose-target / gripper-alignment factorial

- 固定条件：继续使用 V1-R.2G 的同一 20 条轨迹、同一初态、robosuite `1.4.1`、MuJoCo `3.3.5`、`OSC_POSE`、`control_delta=False` 和 20 Hz，仅改变 P0/P1 位姿来源与 G0/G1 夹爪索引。
- 完整性：A 基线逐轨迹复现 13/20；80/80 初态严格恢复，模拟器异常为 0，环境 XML、机器人基座、旋转转换、夹爪符号、轨迹长度和成功判定全部保持不变。
- 结果：A/B/C/D 分别为 `13/20`、`9/20`、`13/20`、`11/20`。B 修复 A 失败 `0/7` 且回退 4 条成功，C 修复 `2/7` 且回退 2 条，D 修复 `1/7` 且回退 3 条。
- 抓取证据：80/80 记录首次接触和闭合前后 5 步窗口；只有布局 3 `demo_6` 在 C/D 下出现至少 5 步稳定抓取。G0 闭合偏移恒为 `-1`，G1 恒为 `0`，但修正夹爪单帧时序没有改善总体结果。
- 决策：`result_4_no_group_reaches_20_of_20`，`pose_gripper_factorial_gate=failed`。B0/B1 训练、confirm、V2 和 V3 均未授权；停止使用当前逐帧保存状态生成的 PKL 作为正式训练数据。
- 下一阶段：在相同正式运行时中顺序执行或重新生成成功演示，只保留当前环境里真实连续成功的轨迹，再从这些轨迹构造 Point Bridge 训练标签。

## V1-R.2I sequential success demonstrations

- 独立入口：不调用旧 PKL 生成器；每个候选仅恢复一次初态，后续全部使用连续 `env.step()`。41/41 初态严格恢复，中途状态恢复为 0，模拟器异常为 0。
- 采集结果：布局 1/2/3/4 各 5 条连续任务成功演示，共 20 条；分别执行 22/9/5/5 个候选达到目标。
- 实际命令回放：四布局均为 `5/5`，合计 `20/20`，通过新数据自身可复现门槛。
- 标签回放：S0 为 `3/20`，布局 1/2/3/4 分别 `1/0/1/1`；S1 为 `16/20`，分别 `4/5/3/4`。两者均未达到严格 `20/20`，不选择训练标签合同。
- 边界：20 条均达到任务成功，但连续五步稳定抓取诊断为 0/20；该诊断不作为准入过滤。20 条仅是合同验证样本，不宣称训练覆盖充分。
- 决策：V1-R.2I 实验完成，`sequential_success_demo_capture_gate` 和 `actual_command_replay_gate` 通过，S0/S1 标签门槛失败。B0/B1 训练、confirm、V2、V3 继续未授权；下一步只在同一 20 条数据上诊断并重建可执行绝对标签合同。

## V1-R.2I evidence hashes

- local capture manifest: `87b4ac57bd02c8ab77ef81e5e22cfafb777f3682b7c4c8c0092083c15ac1786f`
- local replay verification: `8227453583f7a424da61fc8a4cdb98a058dfd5af61a73090ec5928b6eed94679`

## V1-R.2H evidence hashes

- `pointbridge_pose_gripper_factorial.json`: `ca642dcbd7e7fc9d2f8c1f993f7100a1b4d553709922af49bad82e8d43c68460`
- `pointbridge_pose_gripper_factorial.csv`: `da1f4931c975ca8826a72510a91f71be9a5a9d3f8b99c66cde568573160be3cb`
- `pointbridge_pose_gripper_factorial.md`: `ea7ca975dc1bafef4df17868f5f50fd102309b2cdc71b6653693e58922e7de44`
- `v1r_2h_pose_gripper_factorial.yaml`: `f93fb7ba8a5a007dd5f7a7890448fcfd544895391e265339a6f0326804b6b039`
