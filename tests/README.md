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
| PyTorch DCP | `gdtensor_sim/dcp_backend.py` | process 模式写真实 stage 文件；commit 调用 DCP save；load session 延迟读取，仅实际源读取触发 DCP load |
| 文件缓存测试 | `tests/support/cache_oracle.py::FileCache` | 每 slot 独立文件；flush/fsync、pread；记录 rank ACK，完成后删除 slot 文件 |
| 计算端独立 oracle | `tests/support/oracle.py::ArrayFabric` | 稠密全局下标生成预期值，校验每 rank 局部数组；存储与缓存仍真实落盘 |

DCP 仅在源 accessor 首次读取时加载。每个 filler session 仍全量读取；缓存填充字节不等于 DCP 的全部底层 IO。

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

常见形态采样 `_draw_common` 覆盖均匀采样到不了的规模：2 或 4 个节点、每节点 4 或 8 个 rank（world 最多 32）、张量 256–1024 B、steps 4 或 8、并发达 3、两层缓存、file 协调器为主。它用的检查与均匀采样完全相同。默认 8 个 seed（`GDTENSOR_STRESS_COMMON_SEEDS`），单个 seed 约 5 秒，所以默认不放大批量。实测 64 个 seed 全部通过，用时 312 秒；64 个里 32 个同时命中"两层 + DN 节点少于总节点"。

## 无缓存冷读基准（多轮）

`io/test_cold_read_baseline.py` 从无缓存冷读起步。随机 12 个 seed，覆盖节点 1–4、每节点 rank 2/4/8、张量 1–3 个、64–2048 B、20% 复制、2–4 轮。

每轮把输出重置为保护字节，所以每轮都是冷读。断言：输出逐字节等于源分片，保护字节不变；存储读取等于所有 rank 需求字节之和，无跨 rank 去重；各轮存储字节完全相同。

同形状对照用一层或两层缓存，各轮不清缓存。断言分两段：第 1 轮存储读取等于冷读字节；暖轮存储读取为 0，缓存读取大于 0，填充条数保持第 1 轮的水平。

跨轮复用读取缓存 accessor 的有效范围和源身份。`has_bytes` 默认返回 False。文件后端自动使用完整路径和 stat；调用方不用传版本号。同名文件变更会失效。DCP 身份包含 `.metadata`。内部 range 传递 `source_identity`，后端必须按身份隔离字节；未知身份仅允许当轮复用。视图结构保持不变。

实测：36 个用例全部通过（12 个冷读、24 个缓存对照，四个层组合各 12 个）。包括冷读与压测在内的 `tests/io` 两组共 181 个用例通过，用时 100 秒。

## 进程退出与重启

文件协调器为每次缓存加载持有一个生命周期文件锁。进程退出时内核释放锁；其他调用检测到未完成调用的锁已释放后，返回 `cache_peer_down`。健康的慢 IO 没有租约到期时间。

失败状态保留在同一 chain 的协调状态文件中。同目录重试明确报错，不自动复用可能不完整的缓存。所有旧调用停止后，使用新 `GDTENSOR_CACHE_DIR` 重新加载；同进程切换目录也会创建新协调表。不要在旧调用运行时删除或替换锁文件。

状态协议升级为 v3。旧协议或损坏状态返回 `cache_state`；升级时全部参与进程必须使用同一版本和新协调目录，不支持混合版本运行。跨主机还要求共享文件系统提供可靠的跨客户端 `flock` 和原子替换。主机断联、文件系统阻塞及活进程永久阻塞不在本次进程退出检测的保证范围内。

`io/test_colocated_cluster_sim.py` 检查源读取中途杀进程、健康慢读取、同目录失败和新目录恢复。`io/test_cache_owner_lifecycle.py` 检查 fork 后锁隔离、状态提交失败和同进程切换目录。`io/test_random_cache_stress.py` 增加第二代更换源数据与末层旧数据负对照；`io/test_cache_reuse_layout.py` 检查两层缓存实际删除和复用文件、慢消费者、二维 F32 非均匀列分片。后两项分别覆盖 file/thread 协调器，不等同于跨机器验证。

## 本轮实际结果

受测树：`cea07b4`（分支 `fix/cache-propagation-plan`，digest 干净归档树）。本段文档更新单独提交，不影响测试结果。

- digest 全量 pytest：1710 passed、0 failed、0 skipped（含真实 torch/LAWA）。
- 随机缓存策略压测：5000 个 seed 全部通过（`io/test_random_cache_stress.py`，1103 秒）。
- 格式、lint、字节编译通过；ty 与 Pyrefly 均为 0 诊断。
- 独立变异检测：notorch 组 19/19、torch 组 1/1 全部检出。
- 独立形式化复验：Lean 5 个正向文件通过，NegativeFixtures 按预期被拒绝；TLA+ 31 个检查通过，含新增 `cp_peer_down`。
- 形式化工具：digest 直连 GitHub 取固定 jar 返回 403。改用已校验缓存的同一 jar（sha256 `011267fc…`）补跑。脚本仍校验固定哈希，不匹配即中止。

跨轮复用改动后的复验（同一 digest 工作树）：

- 全量 pytest：1758 passed、0 failed、0 skipped（含真实 torch/LAWA）。
- 格式与 lint 通过；ty 与 Pyrefly 均 0 诊断。
- 变异检测：19/19 检出，基线先跑通。
- 形式化：Lean 5 个正向文件编译无空洞，NegativeFixtures 按预期拒绝；TLA+ 31 个检查通过。固定 jar 在 mac 侧下载并校验后传入 digest。

证据在 digest `log/fix-df1/`（pytest、mutation、formal）。日志不作为发布包内容。

## 固定黄金测试

在 digest 的仓库根目录执行，不传测试开关。

```bash
PYTHONPATH=. uv run --locked --extra torch-cpu python log/golden_chain.py
```

入口固定 100 个逻辑节点，每节点 1 rank。文件是 32 张量、1.6 GB 的 F16 safetensors。计算视图使用 `Shard(0)`。每次执行全部 0/1/2 层、随机两层、同预算策略选择及正负控制。独立 oracle 检查每个 rank 的精确字节、覆盖、边界及非法写。

源设备模型为 256 MiB/s、200 µs/op；缓存和 NIC 为 4096 MiB/s。缓存延迟为 50 µs/op。源和缓存都实际 pread/pwrite。未清 OS page cache；计时包含模型延迟、计划、session 和执行，不等于物理磁盘性能。

账本分离 payload、header、缓存消费与物理 pread、TX/RX、队列和文件分配。校验整段的读放大会如实记录。每轮分别输出增量与累计。计划与 session 秒数按 rank 合计，有重叠，不能加成墙钟。

热点定义为节点需求字节超过层均值的两倍。随机层只随机节点集合；parent 仍本机优先、其余轮转。同预算候选固定为三组随机 60/85 节点和连续 60/85 节点。评分先比较 L0 峰值，再比较跨节点 parent 边。不承诺全局最优或时延更优。

执行器准入峰值按节点检查 budget。持久文件有效字节单独记录。前者不包含 callback 暂存与 RSS；后者跨轮保留，可以大于 active budget。

双机入口 `log/p2p_verify.py` 固定执行 digest/v100 的 0/1/2 层、8 MiB 往返及损坏拒绝。0 层本机直读；1 层通过 SSH 读写 v100；2 层执行本机 cache→v100 cache→rank。它使用一个逻辑 rank 和本机协调器。测试清理远端 helper 和专用数据目录。SSH 仅作为受控测试后端，不提供生产 peer 发现或跨机协调。

## 证明边界

通过测试表示枚举输入满足断言，不表示代码完备。

- 独立 CRC 和 arange oracle 检测数据错误。公开 IO 不承诺自动 checksum 校验。
- Lean/TLA+ 验证指定代数、状态和有限模型；不证明全部 Python、OS 或文件系统行为。
- CPU/gloo 多进程不等于多物理机器或 GPU/NCCL 验证。
- 不承诺掉电原子性、崩溃恢复、对象存储重试或大模型规模吞吐。
- GitHub 托管 CI 是独立证据。digest 全绿不能替代最新 GitHub run 的结果。
