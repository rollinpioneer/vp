# V0 交付说明

## 已完成

1. 独立 Python 包和测试入口，代码位于 `src/vico_point`。
2. 固定 Point Bridge 提交 `5d567a62d62b5a97c5960d45024e065349680cda` 的适配边界，未复制第三方源码。
3. 冻结输入等级 `P` 的开发接口、64 点预算（机器人 9、任务 16、上下文 39）、8 帧历史和 7 维动作协议。
4. 明确 `observed / predicted / unknown` 状态、点身份、时间戳、置信度与相机到基座标定字段。
5. `scripts/run_v0.py` 通过接口检查并写入 `experiments/v0/v0_handoff.yaml`。

## 当前状态

`interface_ready_upstream_not_included`。压缩包没有 Point Bridge 源码、数据、环境或权重，因此不能把本地 smoke check 宣称为上游复现。下一步是在实验机获取锁定提交，保留其许可证，然后只接入 `PointBridgeAdapter`。

## V0 验收命令

```bash
python scripts/run_v0.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 不包含的内容

旧 `dm` 的 DataMIL、旧 checkpoint、Top-20 和缺少完整模拟器快照的 200 条状态轨迹均未迁移。它们不能作为当前视觉策略的数据或基线。
