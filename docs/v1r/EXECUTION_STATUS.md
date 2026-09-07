# V1-R 执行状态（2026-09-07）

## 已完成

- V1-R.0：冻结 legacy 快照、锁定分支和 Point Bridge commit。
- V1-R.1：统一 `PointEvidence` 来源、时间年龄、Oracle 边界和 CoTracker/segment-depth 适配器。
- 清单：visibility factorial、clean dev/confirm、all-success episode index。
- 自动化入口：clean baseline、感知审计、阶段识别、目标 mask schedule、频率/相机汇总、上下文输入审计、V1-R 统一决策。
- 完整初态：保存 140 个 clean dev/confirm 状态和 10 个 runner parity 状态；150/150 在 CPU 上重复恢复两次并严格匹配，三 training seed 引用相同状态文件和 SHA-256。
- runner parity：旧 runner 5 个历史成功、5 个历史失败场景均用完整冻结状态复跑；legacy、新 runner、官方 Point Bridge 入口 10/10 对齐，前 20 步动作最大绝对差为 0.0，三者成功率均为 0.10。
- CPU/CUDA：当前环境没有可用 CUDA 设备，审计状态为 `blocked_unavailable_cuda`；不宣称 CPU/CUDA parity 通过，正式部署设备协议冻结为 CPU。
- 专家回放：修复 `reset_to()` 后 Panda 夹爪增量缓存及 OSC 参考关节的审计隔离；重新检查现有 20 条轨迹后，初态仍 20/20 精确恢复，布局 1/2 各 0/5，布局 3/4 各 5/5，整体仍 10/20。10 条失败记录从第 1 步开始偏离，说明缓存污染不是布局 1/2 的充分根因。
- 目标诊断严格限定为布局 1 `demo_0`、布局 2 `demo_0`、布局 3 `demo_3` 的前 20 步；delta OSC、时间对齐和目标重建均得到支持。绝对目标仅保留为跨运行时诊断对照，不改变数据动作契约，也未调整坐标或动作倍率。
- 冻结 clean dev：seed-0 40/40 rollout 完成，成功率 0.05；布局 1/2/3 各 0/10，布局 4 为 2/10。模拟器异常和动作解码异常均为 0，40/40 初态严格匹配。
- 失败阶段：`no_approach=11`、`no_grasp=24`、`post_grasp_drop=3`；38 个失败回合超时。
- 基础回归：缓存隔离测试加入后完整测试 `25/25` 通过；上传大小检查通过（159 个待跟踪文件均不超过 10 MiB），`git diff --check` 和 Python 语法检查通过。

## 诚实门槛状态

当前 `clean_baseline_gate` 为 `blocked_clean_baseline`：真实冻结 dev 已完成，但 seed-0 仅为 `0.05`，低于预注册的 `0.50` clean 门槛；专家回放也未通过布局 1/2。因此没有运行三 seed confirm，也没有重新训练 B0/B1。V1-R.3/R.4/R.5/R.6 没有被旧 V1 输出替代；缺少真实输入时继续保持 `blocked` 或 `unresolved`。

因此当前决策为：

```yaml
decision: blocked_clean_baseline
v1r_runtime_repairs: passed
initial_state_pairing_audit: passed
runner_parity_audit: passed
cpu_cuda_parity_audit: blocked_unavailable_cuda
deployment_device: cpu
expert_replay_audit: failed
clean_baseline_scientific_gate: blocked
v2_formal_experiment_authorized: false
v3_formal_experiment_authorized: false
```

根因不是新 runner，也不是回放审计的终态缓存污染：三条执行路径在同一完整状态上完全一致，且缓存隔离复核后布局 1/2 仍失败。当前需要先修复布局 1/2 的专家动作/环境契约，再重新执行 clean dev；布局 3 的专家回放成功但策略为 0/10，说明训练数据覆盖或策略泛化仍需单独处理。旧 V1 的约 30% E00/E10 结果仅保留在 `experiments/v1r/legacy_snapshot/`，没有写入新门槛结果。

## 输入与输出约定

完整逐场景证据位于 `experiments/v1r/reports/`。实际 `.npz` 状态和原始 rollout CSV 保持在 gitignored 的 `outputs/v1r/`；精确文件名、用途和 SHA-256 分别记录在状态索引及 `manifests/not_uploaded_files.csv`。

重新执行 gate 时，把独立 clean rollout CSV 和三项审计传给：

```bash
PYTHONPATH=src python experiments/v1r/scripts/evaluate_clean_baseline.py \
  --manifest experiments/v1r/manifests/clean_baseline_dev.csv \
  --checkpoint /path/to/frozen_seed0.pt \
  --rollouts /path/to/frozen_clean_dev.csv \
  --training-seed 0 \
  --runner-parity experiments/v1r/reports/runner_parity_report.json \
  --cpu-cuda-parity experiments/v1r/reports/cpu_cuda_parity.json \
  --expert-replay experiments/v1r/reports/expert_replay_report.json \
  --output experiments/v1r/reports/clean_baseline_gate.json
```

其他脚本同样只接受离线记录，不读取未来帧、不把 Oracle 点写成非 Oracle 输入，也不上传 checkpoint、HDF5、PKL、视频、状态 `.npz` 或 RGB-D 二进制文件。Point Bridge 上游保持 gitignored；MuJoCo 2.3 的 mesh scale/offset/path 兼容性通过 `patches/pointbridge/0002*` 和 `0003*` 在本地应用，保存 XML 中不受 MuJoCo 2.3 支持的 `texture colorspace` 和 `light type` 也会被兼容迁移移除。
