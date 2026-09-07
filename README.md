# ViCo-Point：视觉稀疏点策略研究启动目录

这是一个独立的新项目目录。当前已锁定并在本地核验 Point Bridge、MimicLabs 和四个官方 `bowl_on_plate` 数据分片，实现了保存模型 XML 兼容迁移、真实回放 RGB-D/可见点遮挡、因果保持、oracle 隔离和 3×50 配对场景清单。真实 3-step 训练与 checkpoint 门槛、三种子正式策略训练和最短 V1 E00/E10 配对评测均已完成；完整计划中的 `mug_on_plate`、E01/E11 和三基线矩阵不在本次完成声明内。

主文档位于 `docs/视觉稀疏点策略_分阶段实验计划.md` 和同名 Word 文件。
V0/V1 已完成最短可复现实验路径。V1-R 修复与验证门槛代码已加入，但由于独立 clean rollout、真实感知审计和目标 mask 输入尚未全部提供，当前 V2/V3 正式实验仍未授权。

## 项目关系

- 上游工程底座：`NVlabs/pointbridge`，本次核对提交见 `configs/upstream_lock.yaml`。
- 概念与后续基线：`siddhanthaldar/Point-Policy`。
- 旧资产出处：`rollinpioneer/dm`，只读；不导入 DataMIL 包、不迁移旧模型与 Top-20。
- 上游代码和数据仅保存在本地 `third_party/pointbridge/`，整个目录被 Git 忽略；GitHub 只保存锁定提交、文件名、用途、大小和校验值。

## 当前实现

- `src/vico_point/core`：点帧、输入等级、预算、动作协议和运行元数据契约。
- `src/vico_point/perception`：因果可见点帧构造；缺失观测保留 mask，不用零坐标填充。
- `src/vico_point/belief`：任务点的最小因果保持状态，供 V1/V2 过渡。
- `src/vico_point/envs/scenarios.py`：E00/E10/E01/E11 场景注册，并按 source episode group 划分。
- `src/vico_point/evaluation`：按场景配对的成功率汇总。
- `scripts/run_v0.py`：运行接口检查并生成 `experiments/v0/v0_handoff.yaml`。
- `scripts/run_v1.py`：运行确定性的四条件 smoke experiment，生成场景清单和诊断结果。
- `experiments/v1/capture_pointbridge_replay.py`：从锁定 MimicLabs 回放捕获真实 RGB-D、标定与稳定 GT 点身份，并兼容迁移旧保存模型所缺的非物理 metadata。
- `experiments/v1/real_visibility_eval.py`：在同一捕获上配对执行 E00、E10 和仅诊断用隐藏真值上界。
- `manifests/v1_e00_e10_scenarios.csv`：3 个训练 seed、每个 seed 50 对固定 E00/E10 场景。
- `scripts/audit_mimiclabs_success.py`：按保存终态审计官方 HDF5 的成功轨迹。
- `scripts/generate_pointbridge_pkls.py`：应用锁定补丁并并行生成已审计的训练 PKL。
- `scripts/train_pointbridge_three_seeds.py`：以每布局 44 条平衡成功轨迹启动三 seed 正式训练。
- `scripts/evaluate_pointbridge_paired.py`：对真实 checkpoint 运行固定初态的 E00/E10/oracle 配对策略评测。
- `manifests/runtime_gates.yaml`：记录真实回放、可见性、数据生成和训练门槛的小型证据与本地大文件校验值。
- `experiments/v1r/scripts/`：V1-R clean baseline、感知审计、目标阶段/mask、频率/相机和统一决策入口。
- `docs/v1r/EXECUTION_STATUS.md`：V1-R 当前门槛状态和真实输入要求。

运行方式不需要安装第三方 Python 依赖：

```bash
python scripts/run_v0.py
python scripts/run_v1.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

`scripts/run_v1.py` 的合成结果只证明场景/配对/指标链路可运行。真实回放与观测门槛由 `experiments/v1/capture_pointbridge_replay.py` 和 `experiments/v1/real_visibility_eval.py` 提供；最短路径的三种子 Point Bridge checkpoint 和固定 E00/E10 配对策略评测已完成，结果边界见 `docs/v1/README.md`。

## GitHub 上传边界

仓库保留源代码、配置、文档、清单和测试的原始目录结构。数据、checkpoint、视频、缓存、压缩包和生成结果不进入 Git；具体文件名、用途与排除原因记录在 `manifests/not_uploaded_files.csv`。上传前运行：

```bash
python scripts/check_upload_size.py
```

详细规则见 `docs/UPLOAD_POLICY.md`。
Deploy key 的精确填写内容见 `docs/DEPLOY_KEY.md`。

## 初始化独立本地 Git 仓库

将整个目录放在旧仓库之外，建议与 `dm` 平级。然后运行：

```bash
bash scripts/init_new_repo.sh
```

脚本只初始化当前新目录，不下载依赖、不创建远端仓库、不提交或上传文件。
如需设置远端，由用户另行确定仓库名称和地址。

## 资产与状态

`manifests/asset_reuse.csv` 是待判断的复用清单，不表示资产已经迁移。
`configs/plan_defaults.yaml` 记录建议实验参数，不是可直接传给上游训练器的配置。
`experiments/v0` 至 `experiments/v6` 以及 `src/vico_point` 子目录按阶段保留。V0/V1 产物和边界说明见 `docs/v0/`、`docs/v1/`。

优先以 Point Bridge 首个支持任务复现，不因旧轨迹缺失或旧 U0 未完成而等待。
