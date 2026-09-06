# V0 交付说明

## 已完成

1. 独立 Python 包和测试入口，代码位于 `src/vico_point`。
2. 固定并在本地获取 Point Bridge 提交 `5d567a62d62b5a97c5960d45024e065349680cda`；第三方源码被顶层 `.gitignore` 排除。
3. 冻结输入等级 `P` 的开发接口、64 点预算（机器人 9、任务 16、上下文 39）、8 帧历史和 7 维动作协议。
4. 明确 `observed / predicted / unknown` 状态、点身份、时间戳、置信度与相机到基座标定字段。
5. `scripts/run_v0.py` 通过接口检查并写入 `experiments/v0/v0_handoff.yaml`。

6. 四个官方 `bowl_on_plate` HDF5 均按预期字节数下载，可由 `h5py` 打开，SHA-256 已记录在未上传清单。
7. 专用 Python 3.11 venv 已安装 MuJoCo、robosuite、MimicLabs 等运行依赖。

## 当前状态

`upstream_assets_verified_runtime_gate_in_progress`。真实 EGL 环境已经进入 MimicLabs 任务构造，当前正在补齐上游要求的 RoboCasa Objaverse 对象资产。正式 300010-step 三种子训练和 checkpoint 评测尚未完成。

## V0 验收命令

```bash
python scripts/run_v0.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 不包含的内容

旧 `dm` 的 DataMIL、旧 checkpoint、Top-20 和缺少完整模拟器快照的 200 条状态轨迹均未迁移。它们不能作为当前视觉策略的数据或基线。
