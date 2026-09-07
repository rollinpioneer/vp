# V1-R.2 Frozen Clean Dev

- 状态：`complete`；科学门槛：`blocked_clean_baseline`
- seed：0；场景：40；成功：2；成功率：0.05
- 初态严格匹配：40/40
- 模拟器异常：0；动作解码异常：0
- confirm：未运行

| layout | scenarios | successes | success rate |
|---:|---:|---:|---:|
| 1 | 10 | 0 | 0.000 |
| 2 | 10 | 0 | 0.000 |
| 3 | 10 | 0 | 0.000 |
| 4 | 10 | 2 | 0.200 |

失败阶段：`no_approach=11`、`no_grasp=24`、`post_grasp_drop=3`。

原始 rollout CSV SHA-256：`d86e6615801c498a32eb9e50881c5edb43f45eec0b5d171ecd1855ce15981cdd`。

结果低于预注册的 0.50 dev 阈值，且专家回放布局 1/2 未通过，因此没有运行 confirm 或启动 B0/B1 重训。
