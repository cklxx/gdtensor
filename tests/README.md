# 测试与真实文件验证

存储与缓存测试使用真实文件。计算端使用独立数组作为输入和预期值。抽象接口保留设计定义；具体 session 提供文件 accessor。

## 运行流程

1. 本地修改代码并暂存需要提交的文件。
2. `remote_gate.sh` 打包暂存树，发送到 `ssh digest`。
3. digest 重建 Git 树，并核对树哈希。
4. digest 使用 `uv`、`uv.lock` 和独立虚拟环境验证。
5. 提交钩子验证相同暂存树。正常提交后核对提交树，再推送 GitHub main。

macOS 上的 `precommit.sh` 和 `full_check.sh` 自动转发到 digest。Linux CI 直接运行锁定环境，不访问私有主机。未暂存修改不属于受测树，不随该提交发布。

```bash
# 在本地仓库调用；测试在 digest 执行。
scripts/precommit.sh
scripts/full_check.sh

# 在 digest 的受测快照目录运行独立检查。
uv run --locked --extra torch-cpu python tests/mutation_check.py --group notorch
uv run --locked --extra torch-cpu python tests/mutation_check.py --group torch
bash scripts/formal_gate.sh
```

## 接口与文件实现

| 层级 | 实现 | 实际行为 |
|---|---|---|
| 核心接口 | `gdtensor/access.py::DTensorDataAccessor` | 抽象 `read_to_buffer`、`write_from_buffer`；不绑定文件格式 |
| safetensors | `gdtensor_sim/access.py::StorageDataAccessor` | callback 与 buffer 交互；`FileIO` 执行 `os.pread/os.pwrite`；session 校验文件和 slot 范围 |
| PyTorch DCP | `gdtensor_sim/dcp_backend.py` | process 模式写真实 stage 文件；commit 调用 DCP save；load session 初始化时调用 DCP load 读盘，accessor 再选择加载结果中的范围 |
| 文件缓存测试 | `tests/support/cache_oracle.py::FileCache` | 每 slot 独立文件；flush/fsync、pread；记录 rank ACK，完成后删除 slot 文件 |
| 计算端独立 oracle | `tests/support/oracle.py::ArrayFabric` | 稠密全局下标生成预期值，校验每 rank 局部数组；存储与缓存仍真实落盘 |

DCP 的读盘发生在 load session 初始化。不能把每次 accessor callback 都描述为直接 pread。

## 验收分层

- `core/`：设计导出、签名、具体类、builder、约束、view map。
- `io/`：公开 save/load/chain、分片覆盖、非交并、错误清理、缓存节点和容量/ACK 行为。
- `checkpoint/`：真实文件、换拓扑、并行模式、合并、独立 CRC 与 arange oracle。
- `torch/`：原生 DTensor 桥、CPU/gloo 多进程 LAWA、TP/DP/EP/PP 与换拓扑。
- `sim/`：具体文件实现、真实 DCP、生命周期、故障、CLI 子进程。
- `support/`：共享 fixture；测试文件之间不互相导入。

### 新增真实文件验收

`sim/test_real_file_accessors.py` 有 6 个参数展开用例。

- 具体 session 实现抽象 read/write callback；写入、关闭、重开、独立解码一致。
- 写 callback 返回 Err 时，存储端不写 payload。
- 文件截短后，读取返回 Err，且不调用成功 callback。
- 公开 rank_save/rank_load 对 safetensors 和真实 DCP 完成落盘、重开、CRC、arange 对拍。
- 物理 payload 翻转后，独立 CRC 与 arange oracle 检出损坏。

`io/test_cache_mutation_detectors.py` 有 2 个真实文件缓存用例。

- 单元预算按 (step, slot) 单元格对全部张量求和：同一单元格的两个张量可同时驻留（peak 2），验证修复①的聚合语义。
- ACK：两个 rank 分别读取同一 slot。第一个完成后保留文件，全部完成后释放，且只 fill 一次。

`io/test_cache_propagation.py` 用独立 oracle 检查节点级传播计划：两层扩散链路逐 rank 投递、节点间流量按边计算、同节点源端读取去重、常驻字节不超过统计值、源端读单位不超过一个 segment、缺席消费者按超时报错；另有对抗式回归：多张量预算按 (s,c) 聚合、缺席消费者 join 缺员即明确报错（不挂起）、每 rank 一进程经共享文件协调（`GDTENSOR_CACHE_COORDINATOR`）、在途慢填充计为进展、计划缓存弱引用不 pin cache view、`check_constraints` 前置校验 mixed_nodes 与 rank 越界。

这些测试检出缓存 fill job 丢失 file_path 的缺陷。修复保留源文件路径及每个合法 range，逐 slot 读取，避免跨 slot 合并成非法文件范围。修复未新增公开 API。

## CRC 验证与混合部署模拟

CRC 是测试断言，不改变公开 IO 接口。arange 及其确定性切片继续使用原有逐元素或逐字节断言，不额外加 CRC。原有 checkpoint CRC 验证保持不变。

`support/crc.py` 提供 CRC-32 校验。测试必须在操作前保存基线，或从独立输出 oracle 生成基线；不能对同一输出计算两次 CRC 后比较。原有数值精度与逐字节断言继续执行。

- `core/test_payload_crc.py` 检查标准向量、原样传输，以及翻转、截短、交换和追加损坏。失败路径不得打印 PASS。
- `sim/test_dcp_codec.py` 检查非 arange BF16 常量和随机载荷的 codec/DCP 往返。
- `torch/test_lawa_dtype.py` 在累加前记录噪声输入 CRC，累加后检查输入未变；BF16 输出与独立量化均值做 CRC 和精确对拍。允许误差的 F32 累加保留容差断言，不强制 CRC 相等。
- `torch/test_lawa.py` 的非 arange bias 在加载后、累加前与独立输入做 CRC 对拍。

`io/test_colocated_cluster_sim.py` 使用每节点独立缓存内存、有效位和 NIC，以及每 rank 独立计算内存。它检查每 rank 的 arange 分片与未写入保护字节。NIC 计数只记录跨节点传输，全覆盖场景要求网络为零。

部分覆盖、单 DN 和两层场景的四项流量 xfail 已移除。段填充只由本节点的消费 rank 执行；没有本地消费者时指派本节点 rank 服务未消费段。内部缓存 range 携带 `cache_node`，让 accessor 定位节点，公开 range 字段不变。缓存节点不属于 rank 节点集合时，保留原 `cache_subset` 拒绝条件，不扩展公开部署范围。

服务 rank 在自己分片完成后继续填充服务段。完成段按 generation 留痕，避免段释放后被晚到 helper 重填。缺席本地 filler 有界报错，不退回远端填充。新增小预算、串行、缺席 filler、非法 DN 拒绝与多进程共享内存测试。内存/NIC 模拟不证明多物理机器部署。

`io/test_random_cache_stress.py` 按 seed 随机生成合法部署和缓存策略。随机轴包括：每节点 rank 数、节点数、一层或两层缓存、各层 DN 节点集合、steps、并发宽度、张量数与大小、分片或复制、file 或 thread 协调器、线程启动顺序与延迟、NIC 带宽、重复加载次数。payload 为随机字节，存储为单个拼接文件。独立 oracle 不调用规划器，按以下各项检查：

- 每 rank 输出逐字节相等，CRC 一致，保护字节不变；
- 缓存读取命中目标节点；末层 drain 读取的节点等于 oracle 推导的服务节点；
- 缓存写入的执行节点等于 DN 节点；
- 每个被需求的缓存字节在每代只填充一次；
- 存储只由第一层 DN 节点读取，每字节每代一次；
- 统计入站、出站流量等于模拟 NIC 字节加存储读取字节。

每代开始前清空缓存字节和全部计数器，每代单独检查，因此跨代 2+0 的填充分布会失败。

默认 128 个 seed。`GDTENSOR_STRESS_SEEDS` 设置 seed 数，`GDTENSOR_STRESS_BASE` 设置起始 seed。五个变异接入 `mutation_check.py`，分别破坏填充节点、helper 节点、存储偏移、代际重填和统计流量。

## 进程退出与重启

文件协调器为每次缓存加载持有一个生命周期文件锁。进程退出时内核释放锁；其他调用检测到未完成调用的锁已释放后，返回 `cache_peer_down`。健康的慢 IO 没有租约到期时间。

失败状态保留在同一 chain 的协调状态文件中。同目录重试明确报错，不自动复用可能不完整的缓存。所有旧调用停止后，使用新 `GDTENSOR_CACHE_DIR` 重新加载；同进程切换目录也会创建新协调表。不要在旧调用运行时删除或替换锁文件。

状态协议升级为 v3。旧协议或损坏状态返回 `cache_state`；升级时全部参与进程必须使用同一版本和新协调目录，不支持混合版本运行。跨主机还要求共享文件系统提供可靠的跨客户端 `flock` 和原子替换。主机断联、文件系统阻塞及活进程永久阻塞不在本次进程退出检测的保证范围内。

`io/test_colocated_cluster_sim.py` 检查源读取中途杀进程、健康慢读取、同目录失败和新目录恢复。`io/test_cache_owner_lifecycle.py` 检查 fork 后锁隔离、状态提交失败和同进程切换目录。`io/test_random_cache_stress.py` 增加第二代更换源数据与末层旧数据负对照；`io/test_cache_reuse_layout.py` 检查两层缓存实际删除和复用文件、慢消费者、二维 F32 非均匀列分片。后两项分别覆盖 file/thread 协调器，不等同于跨机器验证。

## 本轮实际结果

受测树：`42d03d8`（分支 `fix/cache-propagation-plan`，digest 干净归档树）。

- digest 全量 pytest：1488 passed、0 failed、0 skipped（含真实 2–32 rank torch/LAWA）。
- 格式、lint、字节编译通过；ty 与 Pyrefly 均为 0 诊断。
- 独立变异检测：11 个核心变异全部检出（含新的字节预算准入与 owed ACK 释放变异）。
- 独立形式化复验：Lean 5 个正向文件通过、6 个负向夹具被拒绝；TLA+ 30 个检查通过（CopyPlan 10、AtomicPublish 1、SlotLifecycle 11、CachePropagation 8）。

证据在 digest `log/fix-df1/`（pytest、mutation、formal）。日志不作为发布包内容。

## 证明边界

通过测试表示枚举输入满足断言，不表示代码完备。

- 独立 CRC 和 arange oracle 检测数据错误。公开 IO 不承诺自动 checksum 校验。
- Lean/TLA+ 验证指定代数、状态和有限模型；不证明全部 Python、OS 或文件系统行为。
- CPU/gloo 多进程不等于多物理机器或 GPU/NCCL 验证。
- 不承诺掉电原子性、崩溃恢复、对象存储重试或大模型规模吞吐。
- GitHub 托管 CI 是独立证据。digest 全绿不能替代最新 GitHub run 的结果。
