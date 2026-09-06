# V0 交付说明

## 已完成

1. 独立 Python 包和测试入口，代码位于 `src/vico_point`。
2. 固定并在本地获取 Point Bridge 提交 `5d567a62d62b5a97c5960d45024e065349680cda`；第三方源码被顶层 `.gitignore` 排除。
3. 冻结输入等级 `P` 的开发接口、64 点预算（机器人 9、任务 16、上下文 39）、8 帧历史和 7 维动作协议。
4. 明确 `observed / predicted / unknown` 状态、点身份、时间戳、置信度与相机到基座标定字段。
5. `scripts/run_v0.py` 通过接口检查并写入 `experiments/v0/v0_handoff.yaml`。

6. 四个官方 `bowl_on_plate` HDF5 均按预期字节数下载，可由 `h5py` 打开，SHA-256 已记录在未上传清单。
7. 专用 Python 3.11 venv 已安装 MuJoCo、robosuite、MimicLabs 等运行依赖。
8. 保存模型 XML 可在回放前仅补齐不可见、无碰撞的 `reg_bbox` 与 extent-site metadata；40-step 记录状态恢复已通过。
9. 使用两个有效 PKL 分片完成真实 GPU 3-step Point Bridge 训练，生成并校验 `snapshot/0.pt`、`1.pt`、`2.pt`。

## 当前状态

`pointbridge_real_short_training_gate_passed_formal_training_pending`。RoboCasa Objaverse 资产、记录状态恢复、真实 RGB-D 捕获和短训练 checkpoint 链路均已通过。当前 PKL smoke 只含两个有效 episode，不构成正常场景学习结果；正式 300010-step 三种子训练和 checkpoint 策略评测尚未完成。

## V0 验收命令

```bash
python scripts/run_v0.py
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/run_pointbridge_short_gate.py --steps 3
```

## 不包含的内容

旧 `dm` 的 DataMIL、旧 checkpoint、Top-20 和缺少完整模拟器快照的 200 条状态轨迹均未迁移。它们不能作为当前视觉策略的数据或基线。
