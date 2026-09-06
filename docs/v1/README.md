# V1 交付说明

## 四条件

| 条件 | 遮挡 | 新增约束上下文 |
|---|---:|---:|
| E00 | 否 | 否 |
| E10 | 是 | 否 |
| E01 | 否 | 是 |
| E11 | 是 | 是 |

`E10/E11` 的遮挡会同时改变观测可用性；`E01/E11` 的约束因素只表示新增的关键避障约束，不把桌面和支撑面误算为“无场景”。

## 已完成

- 两个开发任务：`bowl_on_plate`、`mug_on_plate`。
- 三个训练 seed：`0/1/2`。
- 每任务每条件默认 50 个固定场景。
- 场景先按 `source_episode_group_id` 划分，再派生条件；同一原始 episode group 不跨 train/validation/test。
- 三个开发基线名称已冻结：任务点、预算匹配全场景点、高预算全场景点。
- 输出配对 CSV、按方法/条件汇总 JSON 和诊断报告。

## 运行

```bash
python scripts/run_v1.py
```

结果写入 `outputs/v1/`，场景注册表写入 `manifests/scenario_registry.csv`。当前 evaluator 是确定性的合成诊断环境，只用于验证实验 plumbing；报告中的 `scientific_status` 明确标为 `not_a_pointbridge_result`。

## 进入真实 V1 前必须完成

1. 获取并验证锁定的 Point Bridge 源码、许可证和依赖环境。
2. 用官方 `bowl_on_plate` 跑通训练—评测闭环，再接入 `mug_on_plate`。
3. 用 RGB-D/仿真可见深度同时生成颜色、深度和点遮挡，禁止向策略读取隐藏对象真值。
4. 为 E01/E11 每个新布局提供可行示范，不能继续使用会穿过障碍的旧动作标签。
5. 用真实上游回合替换合成 evaluator，并保留输入等级、控制周期、动作解码和异常计数。
