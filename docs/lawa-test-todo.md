# LAWA 测试 TODO

LAWA = Latest Weight Averaging。它把当前 checkpoint 与回溯过去 N 个 checkpoint 的权重做滑动平均。它不是普通 reshard。

本清单只覆盖 LAWA。目标：证明 LAWA 结果数值正确、结构覆盖完整、格式互通、端到端可用、IO pattern 不失真。

状态含义：
- 已完成：仓库现有测试已覆盖，附证据文件。
- 部分完成：主路径已覆盖，但范围不足。
- 未开始：当前无行为测试。

执行约束：
- 所有测试在远端 ssh digest 执行。本地不跑，子任务不占本机资源。
- 统一使用 uv。依赖遵循 `uv.lock`。
- torch 用例需设置 `GDTENSOR_TORCH_TESTS=1`。

## 1. 数值正确性

| 编号 | 动作 | 状态 | 验收条件 |
|---|---|---|---|
| N1 | 朴素平均与独立 oracle 对照 | 已完成 | k=1/2/3/5，各 rank 本地值、文件解码值与独立平均逐元素一致。见 `tests/torch/test_lawa.py` |
| N2 | Welford 增量平均与朴素平均等价 | 已完成 | 同一 k 下 Welford 与朴素结果逐元素一致。见 `tests/torch/test_lawa.py` |
| N3 | 四轴拓扑下独立 numpy 平均对照 | 已完成 | 9 种 (PP,DP,EP,TP) 配置，含换拓扑，逐元素一致且每元素恰好存一次。见 `tests/torch/test_lawa_axes.py` |
| N4 | 加权滑动平均 | 已完成 | torch LAWA 加权（朴素 + Welford 增量）与独立加权平均一致；权重自动归一化，负/零和/非有限权重拒绝。见 `tests/torch/test_lawa.py`、`tests/support/lawa.py` |
| N5 | 合并 checkpoint 数量达到 10 | 已完成 | torch LAWA k=10（DP×TP 4 进程）与四轴 EP×TP k=10 通过；k=1 作为单版本对照。见 `tests/torch/test_lawa.py`、`tests/torch/test_lawa_axes.py` |
| N6 | 训练精度 dtype 覆盖 | 已完成 | BF16 权重在 F32 累积器下误差受控；直接 BF16 累积误差严格更大（>2×）被锁定；输出 dtype 保持 BF16。见 `tests/torch/test_lawa_dtype.py` |
| N7 | 大张量（跨 64MiB chunk）LAWA | 已完成 | 88MiB（>64MiB chunk）单张量 4 rank，平均后逐元素一致且产物文件确实跨 chunk。见 `tests/checkpoint/test_large_tensor_lawa.py` |

## 2. 结构与并行覆盖

| 编号 | 动作 | 状态 | 验收条件 |
|---|---|---|---|
| S1 | Shard 与 Replicate 混合并行 | 已完成 | DP/TP/DP+TP、正序与逆序 mesh、2–32 进程通过。见 `tests/torch/test_lawa.py` |
| S2 | PP/EP 子 mesh 与 expert owner | 已完成 | PP、EP、PP×EP、EP×TP、PP×DP×EP×TP 通过；非成员 rank 持空分片。见 `tests/torch/test_lawa_axes.py` |
| S3 | LAWA 前后换拓扑（reshard 后平均） | 已完成 | TP→DP×TP、PP×DP×TP→PP×TP、PP×EP×TP→PP×DP×EP 三例通过。见 `tests/torch/test_lawa_axes.py` |
| S4 | 大 world size 与真实多机 | 未开始 | 现最大 32 个真实进程，全部单机 gloo。验收：给出明确目标 world size（≥64）下单机通过结果；跨 node 场景在有资源后补 NCCL/多机 gloo 验证 |

## 3. 格式互通

| 编号 | 动作 | 状态 | 验收条件 |
|---|---|---|---|
| F1 | safetensors 与 sim distcp 两格式 LAWA | 已完成 | 两种格式下平均结果一致。见 `tests/torch/test_lawa.py` |
| F2 | 对接真实 `torch.distributed.checkpoint` | 已完成 | gloo 2-rank 真实 collective DCP 保存 k 版本、调用方侧 LAWA 平均、再写回真实 DCP，父进程 no_dist 读回与密集 oracle 一致。见 `tests/torch/test_real_dcp_lawa.py` |
| F3 | HF 索引目录（safetensors + index.json）平均 | 已完成 | 多分片 HF 目录按 weight_map 路由读取做 LAWA，输出仍为合法 HF 目录：索引覆盖全部 tensor、引用分片存在且持有对应 tensor、total_size 与负载字节一致。见 `tests/checkpoint/test_lawa_hf_index.py` |

## 4. 端到端

| 编号 | 动作 | 状态 | 验收条件 |
|---|---|---|---|
| E1 | DeepSeek V3 真实 ckpt save/load/merge 全链路 | 未开始 | 验收：真实 DeepSeek V3 checkpoint 完成 k 版本 LAWA，输出通过 CRC 与逐张量数值校验 |
| E2 | 合并过程故障原子性 | 已完成 | 真实 safetensors 会话注入 fsync 故障：所有 rank 返回 IO 错误、故障分片不发布、输入目录逐字节不变、清除故障后重跑通过 verify_crc。注意快照协议“结果未知保留 payload”，不断言同级分片被删。见 `tests/sim/test_lawa_failure_atomicity.py` |
| E3 | 输入 checkpoint 只读保证 | 已完成 | torch LAWA 多进程路径在平均后对全部输入目录做 sha256 逐字节断言。见 `tests/torch/test_lawa.py` |

## 5. IO pattern 与性能

| 编号 | 动作 | 状态 | 验收条件 |
|---|---|---|---|
| I1 | LAWA IO 计数与读取范围基准 | 未开始 | 验收：输出模拟 trace 与参考 trace 的 D1（调用次数）、D2（读取范围）偏差，数据面目标 D1=0、D2=0 |
| I2 | LAWA 读取顺序与节奏基准 | 未开始 | 验收：D3（顺序倒置率）维持现有 0.027% 量级；D4 出 θ=0.1/1/10ms 曲线，不用单点结论 |
| I3 | 容量与算力代价报告 | 未开始 | 每次 LAWA 实验报告：多版本读字节、跨 rank 通信字节（reshard 部分）、存储读放大、额外耗时；区分 prefill 与 decode、容量（KV）与算力（FLOPs） |
| I4 | k 与 world size 扫描 | 未开始 | 验收：k=1/2/3/5/10 与目标 world size 网格下的完成时间、峰值内存、读放大；报告随 k 的线性度 |

## 6. 负对照（防自欺）

| 编号 | 动作 | 状态 | 验收条件 |
|---|---|---|---|
| B1 | 源 checkpoint 损坏检测 | 已完成 | 等长 payload 交换被 CRC 与读回值同时检出。见 `tests/checkpoint/test_checkpoint_merge_crc.py`、`test_merge_crc_axes.py` |
| B2 | rank 分片错位检测 | 已完成 | 交换同一 tensor 两个 rank 分片必须检出。见 `tests/checkpoint/test_serial_parallel_crc.py` |
| B3 | LAWA 数值路径负对照 | 已完成 | 漏加版本、错除数 k±1、非法权重（负/零和/空/NaN/Inf）均被独立断言拒绝；等权退化为均值有正对照。见 `tests/torch/test_lawa_negative.py` |

## 当前最小可信结论

已证明（2026-10-26，digest 远端 88 项 LAWA 相关用例全通过）：
- 数值：朴素 / Welford / 加权平均、k=1/2/3/5/10、BF16-in-F32 累积，均与独立 oracle 一致。
- 结构：DP/TP/PP/EP 全部组合、混合并行、换拓扑，最多 32 个真实 gloo 进程。
- 格式：safetensors、sim distcp、真实 `torch.distributed.checkpoint`、多分片 HF 索引目录。
- 规模：88MiB 跨 64MiB chunk 大张量。
- 韧性：输入只读；发布故障原子不发布且可重跑；损坏与错位负对照。

未证明：S4 大 world size（≥64）与真实多机 NCCL、E1 DeepSeek V3 真实容量端到端、第 5 节 IO/性能 trace 指标（I1–I4）。因此仍不能宣称 LAWA 已在生产规模验证。
