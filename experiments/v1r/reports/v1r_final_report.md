# V1-R 执行结果

- decision: `blocked_clean_baseline`
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
- 决策：情况 B，`pointbridge_absolute_pose_contract_gate=failed`，B0/B1 训练、V2 和 V3 均未授权。下一步只修复绝对动作/环境契约，不重新训练。
