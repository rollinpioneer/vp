# Point Bridge 输入边界（V1-R）

本文档把上游已有能力和 ViCo-Point 的研究变量分开，避免把输入变化误写成新方法。

## 上游已有能力

- MimicLabs 默认配置包含机器人点、`obj_of_interest` 任务相关物体点，以及 `use_proprio` 本体状态。
- `suite.use_full_scene_pcd` 是 Point Bridge 已有的完整可见场景深度点云模式，默认点数配置为 512。
- `suite.use_vlm_points=false` 时，仿真点由当前物体位姿和预采样物体点更新；这不是自然视觉前端，也不会自然产生“上一帧旧点”。
- `use_vlm_points=true` 时，上游提供 `segment_depth` 与 `point_tracking` 两条视觉点路径。
- 分割深度路径在掩码缺失或采样失败时可能沿用前一帧二维点/深度；V1-R 必须把这类 fallback 单独记录为 `previous_fallback`。
- Point Bridge 的 CoTracker 包装路径调用了 `tracks, _ = tracker(...)`，因此当前上游包装没有保留返回的可见性输出；V1-R 适配层只在本项目代码中保留该字段，不修改上游 checkout。

## V1-R 的输入模式

| `point_mode` | 含义 | 输入等级 |
|---|---|---|
| `gt_mesh_points` | 仿真当前物体点，仅用于 O/P 诊断 | O/P |
| `visible_depth_gt_role` | 当前可见深度点，仿真角色标签 | P |
| `segment_depth` | 分割掩码加深度投影 | V |
| `point_tracking` | 在线点跟踪、跨相机三角化或融合 | V |
| `full_scene_depth` | 完整可见场景深度点云 | P/V，取决于前端 |

V1-R 不允许把 O/P 的仿真点结果写成真实视觉 V 级证据。`last_reliable_hold` 是自定义因果适配器的来源标签，不是 CoTracker 预测结果；`group_centroid_fill` 只是 Point Bridge 固定张量兼容填充，也不能称作当前观测。

## 不允许的归因

Point Bridge 已经有本体状态和完整场景点云选项。因此，“加入机器人状态”“首次加入环境点”“输入更多点后成功率上升”都不能单独作为 ViCo-Point 创新证据。V1-R 的问题必须通过真实视觉审计、固定预算和任务点别名场景证明。
