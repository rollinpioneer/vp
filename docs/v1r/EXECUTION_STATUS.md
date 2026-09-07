# V1-R 执行状态（2026-09-07）

## 已完成

- V1-R.0：冻结 legacy 快照、锁定分支和 Point Bridge commit。
- V1-R.1：统一 `PointEvidence` 来源、时间年龄、Oracle 边界和 CoTracker/segment-depth 适配器。
- 清单：visibility factorial、clean dev/confirm、all-success episode index。
- 自动化入口：clean baseline、感知审计、阶段识别、目标 mask schedule、频率/相机汇总、上下文输入审计、V1-R 统一决策。
- 基础回归：18 个单元测试通过，上传大小检查通过，`git diff --check` 通过。

## 诚实门槛状态

当前 `clean_baseline_gate` 为 `blocked`，因为独立确认集的真实 Point Bridge rollout CSV 尚未提供。V1-R.3/R.4/R.5/R.6 也没有被旧 V1 输出替代；缺少 RGB-D/GT 感知审计、reference mask schedule 和 LEFT_BLOCK/RIGHT_OPEN 成对场景时，脚本会写入 `blocked` 或 `unresolved`。

因此当前决策为：

```yaml
decision: blocked_clean_baseline
v2_formal_experiment_authorized: false
v3_formal_experiment_authorized: false
```

旧 V1 的约 30% E00/E10 结果仅保留在 `experiments/v1r/legacy_snapshot/`，没有被写入新门槛结果。

## 输入与输出约定

真实运行时，把独立 clean rollout CSV 传给：

```bash
PYTHONPATH=src python experiments/v1r/scripts/evaluate_clean_baseline.py \
  --manifest experiments/v1r/manifests/clean_baseline_confirm.csv \
  --checkpoint /path/to/frozen_seed0.pt \
  --rollouts /path/to/clean_seed0_rollouts.csv \
  --training-seed 0 \
  --output experiments/v1r/reports/clean_seed0_gate.json
```

其他脚本同样只接受离线记录，不读取未来帧、不把 Oracle 点写成非 Oracle 输入，也不上传 checkpoint、HDF5、PKL、视频或 RGB-D 二进制文件。
