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
- E00/E10 使用同一个 simulator seed、初始状态键、点身份键和 scenario ID；`manifests/v1_e00_e10_scenarios.csv` 已冻结 150 对、300 行。
- 三个开发基线名称已冻结：任务点、预算匹配全场景点、高预算全场景点。
- 输出配对 CSV、按方法/条件汇总 JSON 和诊断报告。
- 三个正式 seed 均使用 `300000.pt` 完成固定 50 对 E00/E10 场景和 oracle 诊断上界，共 150 对场景。
- 正式聚合结果：E00 `45/150 = 0.30`，E10 `43/150 = 0.2867`，oracle `44/150 = 0.2933`。

## 运行

```bash
python scripts/run_v1.py
```

真实 Point Bridge checkpoint 使用：

```bash
python scripts/evaluate_pointbridge_paired.py /path/to/300000.pt \
  --training-seed 0 --scenario-count 50
```

结果写入 `outputs/v1/`，场景注册表写入 `manifests/scenario_registry.csv`。当前 evaluator 是确定性的合成诊断环境，只用于验证实验 plumbing；报告中的 `scientific_status` 明确标为 `not_a_pointbridge_result`。

上述命令用于合成诊断基线；正式 checkpoint 评测使用 `scripts/evaluate_pointbridge_paired.py`，结果文件默认写入 `outputs/v1/pointbridge_seed{0,1,2}/`。正式评测的结果文件不上传，文件名、用途、大小和 SHA-256 见 `manifests/not_uploaded_files.csv`。

## 真实可见性实现

- RGB 和 metric depth 使用同一预冻结图像平面遮挡区域。
- GT 对象点投影到当前相机并与遮挡后的 depth 做一致性判定。
- 正式 E10 分支只使用当前可见点或过去可靠位置保持；审计器拒绝 oracle 来源。
- `E10_ORACLE_HIDDEN_GT` 使用相同 RGB-D 与动作路径，只把隐藏任务点替换为真值，并明确标记为诊断上界。
- V2 代码骨架记录 mask、遮挡持续时间、误差、identity switch 和重现恢复时间；V2 正式实验未获授权、未完成。

## 真实 runtime gate

- 记录状态恢复：`bowl_on_plate_1/demo_0` 连续捕获 40 步，256 个稳定对象点，`recorded_state_restored=true`。
- E00 平均可见比例：`0.71376953125`。
- E10 平均可见比例：`0.53505859375`；平均因果保持比例：`0.227734375`。
- 非 oracle 分支隐藏真值读取次数：`0`；oracle 仅标记为诊断上界。
- Point Bridge 最短训练门槛：早期两个单 episode PKL 的 3 个真实梯度更新和 3 个本地 checkpoint 均通过。
- 四布局训练数据：每个官方分片审计 300 条，成功数为 256、44、114、286；正式训练采用每布局 44 条的平衡子集。
- 四布局 1000-step seed-0 计时门槛返回 0，用时 68.14 秒。
- 三个正式 seed 均完成 `300010` steps，训练进程返回 0，并生成 `100000.pt`、`200000.pt`、`300000.pt`。
- 配对策略评测已完成：三种子各 50 对 E00/E10，并附带同路径 oracle 诊断上界。
- 全部配对不变量通过：同一初态、同一点身份、遮挡前动作一致，且非 oracle 隐藏真值读取为 0。

小型结果与未上传文件校验值见 `manifests/runtime_gates.yaml`。该门槛只证明运行链路可用，不证明策略成功率。

## 结论边界

V1 正式评测已完成并记录，但当前结果只描述固定配对场景上的成功率，不宣称方法收益、显著性或泛化结论。`E10_ORACLE_HIDDEN_GT` 仅作为诊断上界，不能作为非 oracle 方法结果。
