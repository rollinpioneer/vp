# V1-R 执行结果

- decision: `blocked_pose_gripper_factorial`
- latest completed stage: `V1-R.2H`
- clean baseline success rate: `0.05`
- scientific claim status: 未授权；未通过或不可执行的门槛保持 blocked/failed。
- legacy V1 结果未被覆盖，confirm 未用于覆盖 dev 失败。

## V1-R.2F audits

- initial_state_pairing_audit: `passed`
- runner_parity_audit: `passed`
- cpu_cuda_parity_audit: `blocked_unavailable_cuda`
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

## V1-R.2H evidence hashes

- `pointbridge_pose_gripper_factorial.json`: `ca642dcbd7e7fc9d2f8c1f993f7100a1b4d553709922af49bad82e8d43c68460`
- `pointbridge_pose_gripper_factorial.csv`: `da1f4931c975ca8826a72510a91f71be9a5a9d3f8b99c66cde568573160be3cb`
- `pointbridge_pose_gripper_factorial.md`: `ea7ca975dc1bafef4df17868f5f50fd102309b2cdc71b6653693e58922e7de44`
- `v1r_2h_pose_gripper_factorial.yaml`: `f93fb7ba8a5a007dd5f7a7890448fcfd544895391e265339a6f0326804b6b039`
