# V1-R 执行状态（2026-09-10）

当前研究锚点已前移到 `V1-R.2K.seed0.checkpoint_diagnostic`：量化源采集、严格 20/20 回放、Point Bridge 训练动作路径回环、点观测部署一致性和真实 40 步 chunking 均已通过。冻结合同为
`delta_pose_float32_identity`；唯一授权的 `B1-2K-20` seed 0 已在 CUDA 上完成训练，但 frozen clean-dev 仅 `3/40`，未建立策略基线。legacy B0/B1、confirm、其他 seed、V2 和 V3 仍未授权。

## 已完成

- V1-R.0：冻结 legacy 快照、锁定分支和 Point Bridge commit。
- V1-R.1：统一 `PointEvidence` 来源、时间年龄、Oracle 边界和 CoTracker/segment-depth 适配器。
- 清单：visibility factorial、clean dev/confirm、all-success episode index。
- 自动化入口：clean baseline、感知审计、阶段识别、目标 mask schedule、频率/相机汇总、上下文输入审计、V1-R 统一决策。
- 完整初态：保存 140 个 clean dev/confirm 状态和 10 个 runner parity 状态；150/150 在 CPU 上重复恢复两次并严格匹配，三 training seed 引用相同状态文件和 SHA-256。
- runner parity：旧 runner 5 个历史成功、5 个历史失败场景均用完整冻结状态复跑；legacy、新 runner、官方 Point Bridge 入口 10/10 对齐，前 20 步动作最大绝对差为 0.0，三者成功率均为 0.10。
- CPU/CUDA：当前提权环境可见 8 张 NVIDIA A100，PyTorch CUDA `2.8.0+cu126` 可用；尚未提供成对 CPU/CUDA 结果，因此 parity 状态为 `blocked_missing_results`，不宣称 parity 通过。正式 seed-0 训练和 clean-dev 使用 CUDA。
- 专家回放：修复 `reset_to()` 后 Panda 夹爪增量缓存及 OSC 参考关节的审计隔离；重新检查现有 20 条轨迹后，初态仍 20/20 精确恢复，布局 1/2 各 0/5，布局 3/4 各 5/5，整体仍 10/20。10 条失败记录从第 1 步开始偏离，说明缓存污染不是布局 1/2 的充分根因。
- 目标诊断严格限定为布局 1 `demo_0`、布局 2 `demo_0`、布局 3 `demo_3` 的前 20 步；delta OSC、时间对齐和目标重建均得到支持。首个可解释差异定位在 `state[0] + action[0] -> state[1]`：动作 0 的实际控制目标与保存目标一致，但执行后 EEF 已出现偏差；从动作 1 开始，delta OSC 将该偏差递推进下一控制目标。MuJoCo 3.1.1 中三条轨迹的动作 1 目标递推残差均不超过 `0.000017 mm`。绝对目标仅保留为跨运行时诊断对照，不改变数据动作契约，也未调整坐标或动作倍率。
- V1-R.2G Point Bridge 绝对位姿契约：直接实例化原生 `BCDataset`，使用 `act_subsample=1`、`eef_states[1:]`、`gripper_states[1:]`、末动作重复、SciPy 四元数转换和 Point Bridge 原生 6D 旋转；标签再经过 `BCDataset.preprocess['actions']` 与 `PB.act` pose 分支反归一化。没有运行策略网络或语言网络。
- V1-R.2G 正式回放运行于 robosuite `1.4.1`、MuJoCo `3.3.5`，由 `point_bridge.suite.mimiclabs.make()` 创建 `OSC_POSE`、`control_delta=False`、20 Hz、10 维绝对动作环境。四布局各 5 条，20/20 精确恢复 PKL 初态，标签对齐 20/20，无模拟器异常；布局 1/2/3/4 分别通过 1/5、3/5、4/5、5/5，整体 13/20。
- 绝对位姿回放共记录 3,832 个执行步。每步均包含绝对目标、实际 EEF 位姿、平移/旋转跟踪误差、控制边界状态、抓取和任务成功。归一化标签按训练路径转为 `float32`，部署反归一化动作保持 `float64`。7 条失败均为 `no_grasp`；绝对 OSC 未配置平移或旋转 goal limits，边界命中均为 0。门槛按预注册决策落入情况 B，不能重新训练。
- V1-R.2H 在同一 20 条轨迹和同一初态上完成 P0/P1 位姿目标与 G0/G1 夹爪时序的 2x2 定点实验。A/B/C/D 分别为 `13/20`、`9/20`、`13/20`、`11/20`；A 逐轨迹精确复现 V1-R.2G，80/80 初态严格恢复，模拟器异常为 0，全部固定项检查通过。
- B 未修复 A 的 7 条失败且回退 4 条成功；C 修复 2/7 并回退 2 条；D 修复 1/7 并回退 3 条。四组均未达到直接专家标签的严格 `20/20` 门槛，因此结果为 `result_4_no_group_reaches_20_of_20`，不能把问题归结为简单的位姿来源或夹爪单帧索引修复。
- 抓取邻域证据覆盖全部 80 次运行：记录原始/实际闭合时刻、闭合距离、夹爪关节、首次接触、稳定抓取、最长抓取持续、闭合空抓、闭合前后各 5 步 EEF 位置误差和最终成功。只有布局 3 `demo_6` 在 C/D 下出现至少 5 步稳定抓取；G0 的闭合偏移统一为 `-1` 步，G1 统一为 `0` 步。
- V1-R.2I 新增独立顺序采集入口，不调用 `scripts/generate_pointbridge_pkls.py`，也不应用逐帧保存状态补丁。每个候选只在回合开始恢复一次完整初态，后续全部状态由正式环境连续 `env.step()` 产生；41/41 候选初态严格匹配，中途状态恢复为 0，模拟器异常为 0。
- 四布局各获得 5 条连续任务成功演示。布局 1/2/3/4 分别执行 22/9/5/5 个候选后达到 5/5；20 条工件分别保存实际 7 维控制命令、执行后模拟器状态、实测 EEF 位姿、同一步夹爪命令和控制器绝对目标。
- 新演示的实际命令回放为 `20/20`，说明这批数据自身可以从完整初态连续复现。训练标签回放分开执行：S0 原始 Point Bridge 下一位姿/下一夹爪标签为 `3/20`，S1 同一步控制器绝对目标/夹爪标签为 `16/20`；两者均未达到严格 `20/20`，因此未选择动作标签合同，也未重建训练 PKL 或启动 seed 0。
- 20 条任务成功演示中，连续五步 `_check_grasp=True` 的诊断为 0/20；该条件按协议保持非门槛，不能覆盖任务成功和实际命令回放结论。这 20 条只用于动作合同验证，不代表未来训练数据覆盖充分。
- 冻结 clean dev：唯一授权的 B1-2K-20 seed-0 训练已完成 300000 步，主 checkpoint 为 `300000.pt`；40/40 rollout 完成，成功率 `3/40=0.075`，布局 1/2/3/4 分别为 `0/10`、`2/10`、`0/10`、`1/10`。当前状态包初态匹配 `40/40`，模拟器异常和动作解码异常均为 0；历史兼容 hash 为 `35/40`，仅作诊断，不覆盖当前初态匹配结论。
- 失败阶段：`no_approach=9`、`no_grasp=22`、`post_grasp_drop=6`。clean-dev 门槛要求至少 `20/40`，本轮未通过。
- checkpoint 学习曲线诊断：使用训练前已冻结的中间 checkpoint，在相同 40 条 CUDA dev 初态上只读评估。`100000.pt=2/40`（布局 `0/10,1/10,0/10,1/10`），`200000.pt=6/40`（布局 `1/10,2/10,0/10,3/10`）；正式冻结的 `300000.pt=3/40`。结果显示同一 seed 的中期局部改善未在最终 checkpoint 保持，不改变正式 checkpoint 选择，也不授权 confirm、其他 seed、V2 或 V3。
- seed-0 结果报告：`experiments/v1r/reports/v1r_2k_seed0_clean_dev.md` 和 `.yaml`。训练 checkpoint、rollout CSV、PKL、状态包、XML、TensorBoard 文件均保持本地，不上传 Git。
- V1-R.2J 在同一 20 条 V1-R.2I 工件上完成绝对动作合同重建和 7 维 `delta_pose` 备用接口验证。绝对合同基础结果为 S0-old `3/20`、S0-transition `3/20`、S1-PB `16/20`、S1-world `17/20`；统一尾段诊断为 `11/20`、`9/20`、`18/20`、`18/20`，均未通过 `20/20`。
- `delta_pose` 的初态恢复、共享 min-max 归一化往返、夹爪符号保持和官方 delta runtime 均为 `20/20`，但正式任务执行回放为 `17/20`；失败为布局 1 `demo_18`、布局 3 `demo_10`、布局 4 `demo_2`。该接口保留为未验证备用 patch，不得作为训练合同。
- V1-R.2J-F 在四条冻结演示上完成 24 次数值路径诊断：A/C 原始 `float64` 均为 `4/4`，E/F 原始 `float32` 均为 `3/4`，B/D 共享 min-max -> `float32` -> inverse 均为 `1/4`。采集环境与 Point Bridge `delta_pose` 入口在对应数值条件下底层动作数组和结果一致，执行入口不是直接触发因素。
- 该诊断进一步收窄了原因：`demo_18` 对直接 `float32` 已敏感；min-max -> `float32` -> inverse 在此样本集上又使 `demo_10` 和 `demo_2` 失败。所有失败均为 `contact_without_grasp`。这证明数值动作路径存在精度敏感性，但尚未形成可用于训练的数值合同。
- V1-R.2J-N 仅在 Point Bridge 入口测试 `raw -> float32 label -> float64 controller` 候选：布局 1 `demo_18` 失败，其余三条成功，结果 `3/4`，未达到扩展到 20 条的固定前置条件。该路径与 2J-F 的直接 float32 路径在底层动作、控制目标、EEF、碗轨迹及成败上逐值一致，单独统一控制器输入 dtype 不能修复问题。
- 本机 robosuite `1.4.1` 确实包含 `math.isclose` 朝向更新分支，但四条轨迹中 raw/direct-float32/min-max 三种路径均无全零旋转步骤，分支决策变化为 0；“零变非零触发朝向更新”在本轮样本上被否定。下一阶段为独立版本 V1-R.2K，从采集第一步即执行固定 float32 标签到 float64 控制器解码路径，旧 20 条失败结果保持不变。
- V1-R.2K 量化源采集已完成：沿冻结的布局内数字 demo 顺序共尝试 41 次，接收 20 条，布局 1/2/3/4 分别为 5/5、5/5、5/5、5/5；所有尝试均保留，旧 V1-R.2J-N 结果未覆盖。
- V1-R.2K 严格回放已通过 `20/20`：保存的 float32 标签经共享解码器得到的 float64 控制命令在值、顺序和长度上完全一致，采集与回放状态序列逐步一致，最大绝对误差为 `0.0`。该结果只证明量化后的数据合同可执行，不证明策略可学习或真实机器人迁移。
- V1-R.2K.2 新增 Point Bridge `0005` 补丁，采集、回放、`BCDataset` 与 `BCAgent` 共同调用 `vico_point.action_contracts`，delta 标签不再做 min-max。20/20 数据集标签和 DataLoader batch、20/20 解码命令均逐元素一致，agent 后处理检查通过；由于完整点输入训练 PKL 和 B1-2K 清单尚未冻结，seed 0 仍未授权。
- V1-R.2K.3-F 修正 Point Bridge 包装器定位、上一条夹爪命令对应的机器人点和每 episode 固定对象模板；正式验证通过点观测四项 `20/20`、真实 40 步 chunk `80/80`。二进制身份和训练配置已冻结，随后按授权完成 `B1-2K-20` seed 0 训练。
- 基础回归：完整测试 `70/70` 通过；236 个已跟踪文件均不超过 10 MiB；JSON/YAML 解析、Python 语法检查和 `git diff --check` 通过。

## 当前门槛状态（截至 V1-R.2K.seed0.checkpoint_diagnostic）

V1-R.2I 的顺序采集与实际命令回放已经通过，但 `s0_label_replay_gate` 和 `s1_label_replay_gate` 均为 `failed`。V1-R.2J 又验证了 7 维 `delta_pose` 的归一化、夹爪符号和官方运行时均可执行，但任务回放只有 `17/20`，仍未达到严格门槛。真实冻结 clean dev 的 seed-0 仍仅为 `0.05`，低于预注册的 `0.50` clean 门槛；因此没有运行三 seed confirm，也没有重新训练 B0/B1。V1-R.3/R.4/R.5/R.6 没有被旧 V1 输出替代；缺少真实输入时继续保持 `blocked` 或 `unresolved`。

当前决策为：

```yaml
decision: blocked_seed0_clean_dev
status: completed_seed0_clean_dev_failed
latest_completed_stage: V1-R.2K.seed0.checkpoint_diagnostic
v1r_raw_delta_replay_diagnosis: localized_not_fully_causal
runner_parity: passed
initial_state_restoration: passed
pointbridge_absolute_pose_contract_gate: failed
pose_gripper_factorial_gate: failed
sequential_success_demo_capture_gate: passed
actual_command_replay_gate: passed
s0_label_replay_gate: failed
s1_label_replay_gate: failed
absolute_label_contract_reconstruction_gate: failed
delta_pose_normalization_gate: passed
delta_pose_gripper_sign_gate: passed
delta_pose_official_runtime_gate: passed
delta_pose_execution_replay_gate: failed
delta_numeric_candidate_gate: failed
selected_label_contract: delta_pose_float32_identity
clean_baseline_gate: failed
b0_b1_training_authorized: false
confirm_rollouts_authorized: false
v2_formal_experiment_authorized: false
v3_formal_experiment_authorized: false
seed0_training_started: true
seed0_training_completed: true
seed0_clean_dev_gate: failed
next_stage: diagnose_pilot_coverage_or_learnability
formal_runtime:
  robosuite: 1.4.1
  mujoco: 3.3.5
```

2J/2J-F/2J-N 的完整逐步运行结果保留在本地 gitignored `outputs/v1r/`；仓库只提交精简 JSON/Markdown/YAML 和文件索引，不上传状态、HDF5、PKL、XML 或运行时遥测大文件。

## 当前门槛状态（V1-R.2K.seed0.checkpoint_diagnostic）

```yaml
decision: blocked_seed0_clean_dev
status: completed_seed0_clean_dev_failed
latest_completed_stage: V1-R.2K.seed0.checkpoint_diagnostic
selected_label_contract: delta_pose_float32_identity
quantized_at_source_capture_gate: passed
strict_replay_gate: 20/20
point_observation_parity_gate: passed_20_of_20
chunked_dataloader_gate: passed_80_of_80
training_authorized: true_for_b1_2k_20_seed0_only
seed0_training_authorized: true
authorized_variant: B1-2K-20
authorized_seeds: [0]
seed0_training_started: true
seed0_training_completed: true
seed0_clean_dev_gate: failed
seed0_clean_dev_successes: 3
seed0_clean_dev_rollouts: 40
b0_b1_training_authorized: false
confirm_rollouts_authorized: false
v2_formal_experiment_authorized: false
v3_formal_experiment_authorized: false
training_action_path_gate: passed
training_dataset_manifest_gate: passed_4_corrected_pkls_20_episodes
next_stage: diagnose_pilot_coverage_or_learnability
```

V1-R.2K 冻结了可执行的量化源动作合同：`raw float64 -> float32 label -> float64 controller command`，2K.2 确认该合同已进入 Point Bridge 数据读取和 agent 后处理路径，2K.3-F 又修正并验证了夹爪相关机器人点、固定对象模板和真实 chunking。
这些结果仍不证明策略可学习或真实机器人迁移。唯一授权的 seed 0 已完成训练，但 frozen clean-dev 仅 `3/40`，低于 `20/40` 门槛。下一步是 pilot coverage / learnability 诊断；confirm、其他 seed、V2 和 V3 不随 seed-0 授权自动开放。

上述历史结果说明，根因不是新 runner，也不是回放审计的终态缓存污染：三条执行路径在同一完整状态上完全一致，且缓存隔离复核后布局 1/2 仍失败。V1-R.2G 证明原生绝对标签链只有 13/20；V1-R.2H 又排除了仅替换保存控制目标或修正夹爪一帧时序即可达到 20/20 的解释。V1-R.2I 进一步证明，正式运行时连续成功轨迹及其实际 delta 命令可以 20/20 复现；V1-R.2J/2J-F/2J-N 将阻塞收窄到数值动作路径，并排除了执行入口、控制器输入 dtype 和当前样本中的旋转零值分支。V1-R.2K 随后用量化源采集建立了新的可执行数据合同，但仍未授权策略训练。旧 V1 的约 30% E00/E10 结果仅保留在 `experiments/v1r/legacy_snapshot/`，没有写入新门槛结果。

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

V1-R.2G 的正式命令为：

```bash
/home/xushijie/.conda/envs/mimicgen/bin/python -m pip install \
  --target /tmp/v1r_mujoco335 --no-deps mujoco==3.3.5

env PYTHONPATH=/tmp/v1r_mujoco335 MUJOCO_GL=egl \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/xushijie/.conda/envs/mimicgen/bin/python \
  experiments/v1r/scripts/validate_pointbridge_absolute_pose.py \
  --output experiments/v1r/reports/pointbridge_absolute_pose_contract.json \
  --report experiments/v1r/reports/pointbridge_absolute_pose_contract.md \
  --gate-output experiments/v1r/reports/v1r_2g_pointbridge_absolute_pose_contract.yaml
```

V1-R.2H 的正式命令为：

```bash
env PYTHONPATH=/tmp/v1r_mujoco335 MUJOCO_GL=egl \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/xushijie/.conda/envs/mimicgen/bin/python \
  experiments/v1r/scripts/run_pointbridge_pose_gripper_factorial.py \
  --output experiments/v1r/reports/pointbridge_pose_gripper_factorial.json \
  --csv-output experiments/v1r/reports/pointbridge_pose_gripper_factorial.csv \
  --report experiments/v1r/reports/pointbridge_pose_gripper_factorial.md \
  --gate-output experiments/v1r/reports/v1r_2h_pose_gripper_factorial.yaml
```

该命令因科学门槛失败返回退出码 `2`；这表示实验已完成但四组均未通过，不是运行异常。

V1-R.2I 的正式命令为：

```bash
env PYTHONPATH=/tmp/v1r_mujoco335 MUJOCO_GL=egl \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/xushijie/.conda/envs/mimicgen/bin/python -u \
  experiments/v1r/scripts/capture_sequential_success_demos.py \
  --target-per-layout 5 \
  --candidate-manifest outputs/v0_success_audit_shard1.json \
  --candidate-manifest outputs/v0_success_audit_shard2.json \
  --candidate-manifest outputs/v0_success_audit_shard3.json \
  --candidate-manifest outputs/v0_success_audit_shard4.json

env PYTHONPATH=/tmp/v1r_mujoco335 MUJOCO_GL=egl \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/xushijie/.conda/envs/mimicgen/bin/python -u \
  experiments/v1r/scripts/verify_sequential_success_demos.py \
  --manifest outputs/v1r/sequential_success_demos_2i/manifest.json \
  --output outputs/v1r/sequential_success_demos_2i/verification.json
```

采集和实际命令回放成功；S0/S1 标签门槛失败是科学结论，不是运行异常。精简报告位于 `experiments/v1r/reports/sequential_success_demos.*`，逐文件未上传索引位于 `experiments/v1r/manifests/sequential_success_demo_artifact_index.csv`。
