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
