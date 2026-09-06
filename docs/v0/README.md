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
9. 完成四个官方分片各 300 条轨迹的保存状态成功审计，通过数依次为 256、44、114、286。
10. 以共同可用的保守规模生成四布局各 44 条平衡成功轨迹；四个 PKL 均通过 episode 数、点张量形状和 SHA-256 校验。
11. 使用四布局 PKL 完成真实 GPU 1000-step 计时门槛，训练返回 0，用时 68.14 秒。

## 当前状态

`balanced_44_per_layout_pointbridge_dataset_ready_formal_training_pending`。RoboCasa Objaverse 资产、记录状态恢复、真实 RGB-D 捕获、四布局平衡训练集和 1000-step 训练链路均已通过。布局 2 在 300 条官方轨迹中仅有 44 条满足保守的保存终态/末动作成功判定，因此没有伪称每布局 100 条。正式 300010-step 三种子训练和 checkpoint 策略评测尚未完成。

## V0 验收命令

```bash
python scripts/run_v0.py
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/run_pointbridge_short_gate.py --steps 3
python scripts/train_pointbridge_three_seeds.py --steps 1000
```

## 不包含的内容

旧 `dm` 的 DataMIL、旧 checkpoint、Top-20 和缺少完整模拟器快照的 200 条状态轨迹均未迁移。它们不能作为当前视觉策略的数据或基线。
