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
| N4 | 加权滑动平均 | 部分完成 | checkpoint 合并层已覆盖等权与加权（见 `tests/checkpoint/test_checkpoint_merge_crc.py`）；torch LAWA 路径只测等权。验收：torch LAWA 加权结果与独立加权平均一致，权重和不归一时显式报错或归一化 |
| N5 | 合并 checkpoint 数量达到 10 | 未开始 | 现有 LAWA torch 用例 k≤5，四轴 k≤3。验收：k=10 结果正确；累加中途与保存结果均与 oracle 一致；覆盖至少一种四轴配置 |
| N6 | 训练精度 dtype 覆盖 | 未开始 | torch LAWA 现用 F32/F64；F16/BF16 仅在桥接测试出现。验收：BF16 权重累加在 F32 累积器下与参考实现误差在阈值内；明确直接在 BF16 累加的误差并决定是否禁止 |
| N7 | 大张量（跨 64MiB chunk）LAWA | 未开始 | 现有张量最多 64 个小张量。验收：单张量跨多个 chunk 时平均正确；多 rank 写同一大文件无区间重叠 |

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
| F2 | 对接真实 `torch.distributed.checkpoint` | 部分完成 | 真实 DCP save/load 已验证（见 `tests/torch/test_real_dcp_distributed.py`、`test_real_dcp_sharded.py`、`test_real_dcp_col_sharded.py`），但 LAWA 平均未跑在真实 DCP 文件上。验收：用真实 DCP 多 rank checkpoint 做 k 版本平均，结果与 oracle 一致 |
| F3 | HF 索引目录（safetensors + index.json）平均 | 未开始 | 验收：多分片 HF 目录作为 LAWA 输入，平均后输出仍为合法 HF 目录，`model.safetensors.index.json` 与实际分片一致 |

## 4. 端到端

| 编号 | 动作 | 状态 | 验收条件 |
|---|---|---|---|
| E1 | DeepSeek V3 真实 ckpt save/load/merge 全链路 | 未开始 | 验收：真实 DeepSeek V3 checkpoint 完成 k 版本 LAWA，输出通过 CRC 与逐张量数值校验 |
| E2 | 合并过程故障原子性 | 未开始 | 验收：某 rank 在 load、累加、save 阶段失败时，已提交快照不被部分覆盖；重跑可恢复；prepared 残留不发布 |
| E3 | 输入 checkpoint 只读保证 | 部分完成 | checkpoint 合并负例已验证读源文件不变。torch LAWA 路径需显式断言。验收：平均后 k 个输入目录逐字节不变 |

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
| B3 | LAWA 数值路径负对照 | 未开始 | 验收：故意漏加一个版本、用错除数 k、朴素/Welford 只走一条路径时，N1–N3 断言必须失败；变异项接入 `tests/mutation_check.py` 或独立脚本 |

## 当前最小可信结论

已证明：单机 CPU/gloo、小张量、k≤5（四轴 k≤3）、safetensors 与 sim distcp、朴素与 Welford、混合并行与换拓扑下 LAWA 数值正确。

未证明：k=10、BF16、大张量、真实 DCP LAWA、真实模型容量、多机、故障恢复、IO 性能。因此不能宣称 LAWA 已在生产规模验证。
