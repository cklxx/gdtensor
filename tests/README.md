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

- 字节预算：第一个 rank 持有真实读取时，第二个 tensor 不能新增 slot 文件。
- ACK：两个 rank 分别读取同一 slot。第一个完成后保留文件，全部完成后释放，且只 fill 一次。

`io/test_cache_propagation.py` 用独立 oracle 检查节点级传播计划：两层扩散链路逐 rank 投递、节点间流量按边计算、同节点源端读取去重、常驻字节不超过统计值、源端读单位不超过一个 segment、缺席消费者按超时报错。

这些测试检出缓存 fill job 丢失 file_path 的缺陷。修复保留源文件路径及每个合法 range，逐 slot 读取，避免跨 slot 合并成非法文件范围。修复未新增公开 API。

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
