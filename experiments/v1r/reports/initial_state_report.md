# V1-R.2F Initial State Audit

状态：`passed`。clean 状态：140；runner parity 状态：10。

dev/confirm 唯一状态：`{'dev': 40, 'confirm': 100}`；每个 clean scenario 的 training seed 0/1/2 引用同一状态路径和 SHA-256。

旧 V1 的 10 个 seed-only 场景历史 hash 可重现数：`0/10`，因此后续不再把 simulator seed 当作完整初态。

运行时对每个状态重复恢复 2 次，并严格比较 `H_expected == H_actual`。
