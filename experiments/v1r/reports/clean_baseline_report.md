# V1-R.2 Clean Baseline

状态：`blocked_clean_baseline`。

2026-09-07 的真实 Point Bridge seed-0 B0 开发集评测完成 40/40 条：成功 `2/40 = 0.05`，`simulator_exception=0`，`action_decode_error=0`，初始状态哈希配对检查通过。checkpoint SHA-256 为 `46c8619a39e816e0d82a627e6b68c725b5e63ee3c57afb47c3897eab2510c7c7`。

确认集三 seed 评测没有被用来覆盖开发集失败，因此当前没有声称 confirm gate 通过。后续动作限定为修复数据/动作接口，或切换到预先登记的支持任务；不得把旧 V1 结果当作新 clean baseline。

详细机器可读记录：`clean_baseline_dev_seed0.json`、`clean_baseline_gate.json`。
