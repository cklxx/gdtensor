# gdtensor Feature Map

本文按“功能 → 入口 → 实现 → 证据”定位代码。设计细节见 `docs/DTensorsView-design.md`，测试状态见 `docs/lawa-test-todo.md`。本文不替代这两份文档。

## 1. 系统边界

`gdtensor` 是 view、布局、CopyPlan、range IO 和 `torch.DTensor` 桥的核心。它不决定 PP/DP/TP 拓扑、文件布局，也不执行 LAWA（`README.md:16`）。核心不序列化 payload、不提供 reduce API、不实现 barrier；这些职责留给调用方和实现层（`docs/DTensorsView-design.md:412`）。

`gdtensor_conn` 是 connector/test harness，不属于 release core API。它扮演训练框架、布局构造器和多 rank runner（`gdtensor_conn/__init__.py:1`）。生产框架应在自己的 `begin_load` / `begin_save` 中注入 backend（`README.md:84`）。

| 功能 | 调用入口 | 实现位置 | 职责 |
|---|---|---|---|
| Compute view | `gdtensor.builder.ComputeDTensorsView` | `gdtensor/internal/views.py:25` | 保存 tensor id、逻辑表和 compute accessor。 |
| Storage view | `gdtensor.builder.StorageDTensorsView` | `gdtensor/internal/views.py:57` | 在逻辑表和 accessor 外增加物理布局、`begin_save`、`begin_load`。 |
| Cache view | `gdtensor.builder.CacheDTensorsView` | `gdtensor/internal/views.py:86` | 描述节点侧缓存；无文件布局和 session。 |
| 布局构造 | connector 业务构造器 | `gdtensor_conn/models.py:306` | 生成 single、per-rank、rank-multi、per-stage、HF 等布局；核心只消费布局数据。 |
| CopyPlan | 内部 plan API | `gdtensor/internal/plan.py:252` | save 选择 compute→storage range；load 选择 storage→compute range（`gdtensor/internal/plan.py:274`）。 |
| IO 执行 | `gdtensor.io.rank_save` / `rank_load` | `gdtensor/internal/api.py:159`、`gdtensor/internal/api.py:184` | 调用 plan、校验 plan、打开 session、逐 range 复制并关闭 session。 |
| Range accessor | `DTensorDataAccessor` | `gdtensor/access.py:136` | 以 callback 读写原始字节，不绑定文件格式。 |
| 文件 accessor | `StorageDataAccessor` | `gdtensor_conn/access.py:29` | 将 payload range 映射到 session 文件偏移，再执行 pread/pwrite（`gdtensor_conn/access.py:81`）。 |
| 格式与发布 | `make_storage_view` | `gdtensor_conn/formats.py:465` | 选择 safetensors 或 DCP session，并提供 storage view。 |
| torch 桥 | 顶层五个桥函数 | `gdtensor/internal/torch_bridge.py:33` | 在 compute view、`GenericDTensor` 与 `torch.DTensor` 之间转换。 |
| LAWA | 调用方流程 | `docs/DTensorsView-design.md:421` | 选择 k 个版本、平均和输出策略均由调用方负责；无 LAWA 专用 storage API。 |

### View 到文件的定位链

```text
ComputeDTensorsView / StorageDTensorsView
  → DTensorTable + mesh：确定每个 rank 拥有的全局元素
  → PhysicalLayoutTable：确定 storage slot 所属文件
  → CopyPlan：生成 (source range, destination range)
  → session + accessor：payload offset 转为实际 file offset
  → FileIO / SdkFileIO / DCP backend：执行具体读写和发布
```

storage range 使用不含 header 的 `contiguous_payload_offset`；session/format 再转换为文件偏移（`docs/DTensorsView-design.md:189`）。plan 是纯函数，不打开 session；执行器不重新选择 range（`docs/DTensorsView-design.md:179`）。save 对 replicated 数据选择唯一写者，load 填满当前 rank 拥有的全部元素（`gdtensor/internal/plan.py:1`）。

## 2. LAWA 六步流程

以下是 `tests/torch/test_lawa.py` 的实际路径。设计中的朴素平均和 Welford 示例分别位于 `docs/DTensorsView-design.md:427` 和 `docs/DTensorsView-design.md:471`。

| 步骤 | 动作 | 代码证据 |
|---|---|---|
| 1. 选择输入和输出 | 调用方确定 k 个 checkpoint；core 不决定“最近 k 个”。输出使用独立 storage view。 | `docs/DTensorsView-design.md:423`；测试按 `input{index}` 和 `{method}{k}` 分目录（`tests/torch/test_lawa.py:188`）。 |
| 2. 准备 compute view | accumulator 初始化为零；每个输入版本使用 NaN 填充的 current view，避免漏读后得到假阳性。 | `tests/torch/test_lawa.py:184` |
| 3. 读取并换切分 | 每版调用 `rank_load(rank, current, storage)`，按目标 compute mesh 填入本地分片。 | `tests/torch/test_lawa.py:198`；load 执行链见 `gdtensor/internal/api.py:193`。 |
| 4. 桥接 tensor | 遍历 tensor id，取得 `GenericDTensor`，再用 `get_d_tensor` 取底层 `torch.DTensor`。 | `tests/torch/test_lawa.py:168`；桥入口见 `gdtensor/internal/torch_bridge.py:39`。 |
| 5. 平均并写回 accumulator | 朴素法累加后除以 k；Welford 路径每版执行 `previous + (source - previous) / step`。用 `reset_d_tensor` 写回。 | `tests/torch/test_lawa.py:173`、`tests/torch/test_lawa.py:210` |
| 6. 写回并复验 | `rank_save` 将 accumulator 写到新目录；随后重新 `rank_load`，逐 rank 对照独立均值。 | `tests/torch/test_lawa.py:212` |

`reset_d_tensor` 不是任意替换：shape、dtype、placements 和 mesh 必须保持一致（`gdtensor/internal/torch_bridge.py:102`）。四轴测试把“源拓扑保存 → 目标拓扑加载并平均 → 第三布局保存 → 重载和文件解码”串成同一路径（`tests/torch/test_lawa_axes.py:222`）。

加权平均使用同一六步流程。权重先归一化，空、非有限、负数和非正和会被拒绝（`tests/support/lawa.py:4`）；加权 Welford 按累计权重计算本版步长（`tests/torch/test_lawa.py:241`）。

## 3. 输入只读、IO 和原子发布

### 输入只读

load 只从 storage session 读取并写 compute view，结束后调用 `finish`（`gdtensor/internal/api.py:184`）。LAWA 多进程测试在平均前记录所有输入目录的 sha256，在平均和输出后逐目录比较（`tests/torch/test_lawa.py:417`）。故障注入测试也逐字节比较 k 个输入目录（`tests/sim/test_lawa_failure_atomicity.py:114`）。

只读结论适用于这些已测路径。公开 IO 本身不提供自动 checksum 校验；CRC 和独立 arange 检查属于测试 oracle（`tests/README.md:166`）。

### safetensors 发布

1. 各 rank 写 `.tmp` staging 文件。
2. 文件 leader 执行 sync 和原子 publish（`gdtensor_conn/formats.py:278`）。
3. 全部 shard 发布并完成 rank 投票后，最低 leader 可写 HF index，最后发布 `.gdtensor_committed`（`gdtensor_conn/formats.py:284`）。
4. load 要求 marker 存在且文件集合与布局完全一致（`gdtensor_conn/formats.py:379`）。

失败原子性指“目录不可见”，不是“回滚所有 shard”。commit 结果 unknown 时保留 payload，abort 不删除最终路径（`gdtensor_conn/formats.py:223`）。E2 测试明确允许同级 shard 已改名，但要求无 marker、故障 shard 未发布、load 拒绝、输入不变，清障后可重跑（`tests/sim/test_lawa_failure_atomicity.py:105`）。同一 root 的并发 save 不受支持；避免新旧 shard 混合时应使用新 root（`README.md:77`）。

### DCP 路径

`make_storage_view` 根据 `layout.container_format` 选择格式；该值为 `"distcp"` 时分派到 `DcpSaveSession` / `DcpLoadSession`（`gdtensor_conn/formats.py:465`、`gdtensor_conn/formats.py:508`）。connector 先按 CopyPlan 暂存 slot，rank 0 组装完整 tensor 后调用真实 `torch.distributed.checkpoint.save`（`gdtensor_conn/dcp_backend.py:181`、`gdtensor_conn/dcp_backend.py:477`）。load 要求 `.metadata` 存在，并可从真实 DCP archive 解析 payload 偏移（`gdtensor_conn/dcp_backend.py:523`）。

另有 F2 测试直接通过 2-rank collective DCP 保存 k 版、由调用方平均、再写回真实 DCP（`tests/torch/test_real_dcp_lawa.py:40`）。该测试证明真实 DCP 数值链路，不等于生产规模 connector 验证。

## 4. LAWA 历史证据与未验证边界

历史清单记录：2026-10-26，digest 远端 **88 项 LAWA 相关用例全部通过**（`docs/lawa-test-todo.md:71`）。这是仓库保存的历史结果，不是本次工作重新执行的结果。本次仅完成引用校验；提交钩子启动的测试已停止，未将其作为验证证据。

| 范围 | 历史证据 | 定位 |
|---|---|---|
| 数值 | 朴素/Welford/加权，k=1/2/3/5/10，独立 oracle。 | `docs/lawa-test-todo.md:17`；主路径 `tests/torch/test_lawa.py:47`。 |
| 结构 | PP/DP/EP/TP、子 mesh、换拓扑，最多 32 个真实单机 Gloo 进程。 | `docs/lawa-test-todo.md:29`；九种四轴配置 `tests/torch/test_lawa_axes.py:41`。 |
| dtype 与规模 | BF16 输入用 F32 累积；88 MiB 单 tensor 跨 64 MiB chunk。 | `tests/torch/test_lawa_dtype.py:1`；`tests/checkpoint/test_large_tensor_lawa.py:33`。 |
| 格式 | safetensors、sim distcp、真实 DCP、多分片 HF 索引目录。 | `docs/lawa-test-todo.md:38`；真实 DCP `tests/torch/test_real_dcp_lawa.py:1`。 |
| 韧性 | 输入只读、故障不发布且可重跑、数值和分片负对照。 | `docs/lawa-test-todo.md:46`、`docs/lawa-test-todo.md:63`。 |

| 编号 | 当前状态 | 未完成的生产证据 |
|---|---|---|
| S4 | 未开始 | `world_size >= 64` 的真实多机 NCCL 未验证。现有最大值为 32 个真实进程，且均为单机 Gloo（`docs/lawa-test-todo.md:36`）。 |
| E1 | 未开始 | 真实 DeepSeek V3 checkpoint 的 k 版 LAWA 容量、CRC 和逐 tensor 数值全链路未验证（`docs/lawa-test-todo.md:50`）。88 MiB 用例不能替代该容量验证。 |
| I1 | 未开始 | LAWA IO 调用次数 D1 和读取范围 D2 尚无基准（`docs/lawa-test-todo.md:58`）。 |
| I2 | 未开始 | 读取顺序 D3 和不同 θ 下的节奏 D4 尚无基准（`docs/lawa-test-todo.md:59`）。 |
| I3 | 未开始 | 多版本读字节、通信字节、读放大、耗时及容量/算力代价尚未测量（`docs/lawa-test-todo.md:60`）。 |
| I4 | 未开始 | k 与 world size 网格下的时间、峰值内存和读放大尚未扫描（`docs/lawa-test-todo.md:61`）。 |

因此，现有证据支持小中型功能正确性和格式互通，不支持“LAWA 已完成生产规模验证”的结论（`docs/lawa-test-todo.md:80`）。
