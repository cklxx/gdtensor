# 缓存读链修复与验收

日期：2026-11-11。测试和性能任务均在 digest 的隔离目录执行。

## 结论

- 热轮已改为批量检查和批量读取。不能一次装入全部数据的计划也走批量路径。
- 冷轮从 L0 向 L1 填充时，父层按批读取。range 不拆分；超过批次上限的 range 单独成批。
- 默认批次保持 4 MiB。对照中它的冷轮最快，热轮接近 16 MiB，内存低于 16 MiB。
- 100-rank 热轮：一层 27.365 秒，两层 29.059 秒。相对本任务修复前记录分别约为 9.22 倍和 9.42 倍。
- 两层冷轮仍需 661.082 秒，慢于无缓存 direct 的 203.869 秒。当前收益主要在热轮；计划构建和两层冷填充仍是瓶颈。
- 真实 Qwen3-0.6B 文件的 safetensors 和 DCP 写入、独立字节校验、direct、cold、hot 全部通过。热轮源读取为 0。

## 修复内容

1. 热轮先批量检查缓存是否存在，再调用 `read_bytes_batch`。非 all-fit 计划在 `_step` 中直接把已完成的检查结果交给批量 drain。
2. 冷轮 `_fill_parent_batch` 批量读取父层，再逐段填充子层。segment 仍是填充、驻留、ACK 和释放单位；slot 仍只负责容量预算和运行时预留。
3. 文件缓存只有在批次内全部 range 通过长度和 checksum 检查后，才向 consumer 发布数据。覆盖部分 miss、多 range 损坏和 malformed batch。
4. 普通 I/O 错误先发布到协调器，再归还 in-flight。提交中断时保留持久化 owner，让后续 peer 得到 `cache_peer_down`。
5. 文件正常提交时，filled state 与 owner activity 在同一事务提交。热属主标记全部上游 owned duties 完成。
6. 同一 chain 的并发 `_slots` 调用共享正在计算的 fingerprint。最后一个调用结束后删除共享项；后续调用重新计算。fork 后重建锁并清空活动状态，避免继承死锁或永不完成的 Future。
7. 单 rank all-fit 且终端全热时，不再执行不需要的 parent preflight 或 parent fill。
8. DCP identity 包含 `.metadata` 的 `storage_data` 引用的全部 payload。layout index 未列出的实际 payload 变更也会使旧缓存失效。
9. safetensors identity 对照路径 stat 与 pooled fd 的 fstat。session 打开后发生原子替换时，加载返回错误，不把旧 fd 的字节发布到新 identity 下。fd 校验只覆盖本 rank `my_files()` 中已打开的 shard；其他 shard 只做 stat，避免 identity 新增 R×F 个描述符。

协调协议保持 `gdtensor-coord-v4`。公开调用方仍收到原 Error 对象；协调器内部保存副本。

## 真实模型与测试配置

源文件：`Qwen3-0.6B/model.safetensors`。官方文件名仍是 0.6B。

- 311 个 BF16 张量。
- 751,632,384 个元素，约 0.752B；按一位小数可写 0.8B。
- 张量 payload：1,503,264,768 B。
- 性能任务独占、顺序执行，未清 OS page cache。
- 100-rank profile 使用单主机 100 个 thread ranks、`GDTENSOR_SOURCE_NODES=0,1`，不执行训练。
- 8-rank 格式测试使用单主机 thread ranks、`no_dist=True`、thread coordinator、`GDTENSOR_CACHE_WAIT_SECONDS=600`。
- 格式测试固定 `OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。

这些是单机逻辑 rank 验证，不是多物理机、GPU 或 NCCL 验证。

## 100-rank：0/1/2 层

下表比较本任务修复前记录与最终 `review24q`。每项是单次墙钟结果，没有误差区间。

| 路径 | 阶段 | 修复前秒数 | 最终秒数 | 修复前/最终 |
|---|---|---:|---:|---:|
| direct | load | 205.469048 | 203.869384 | 1.01× |
| 一层 | cold | 389.159030 | 183.353995 | 2.12× |
| 一层 | hot | 252.375606 | 27.365256 | 9.22× |
| 两层 | cold | 848.497760 | 661.082012 | 1.28× |
| 两层 | hot | 273.713191 | 29.058793 | 9.42× |

- direct 从源读取 1,503,264,768 B payload，逐 rank 字节校验通过。
- 一层 cold：缓存填充 1,503,264,768 B，校验通过。
- 一层 hot：源填充为 0，缓存读取 1,503,264,768 B，校验通过。
- 两层 cold：L0 和 L1 各填充 1,503,264,768 B，校验通过。
- 两层 hot：填充为 0，L1 缓存读取 1,503,264,768 B，校验通过。
- 一层和两层热轮的 SHA256 调用分别降至 297,287 和 315,528 次；修复前记录约为 2540 万次。
- 一层和两层的缓存损坏负对照均被拒绝。

两层 cold 仍慢于 direct。该轮同时承担两层写入和计划成本，不能把热轮加速比套到冷轮。

## 8-rank：真实 safetensors 与 DCP

所有阶段均逐张量独立校验。save 后另从落盘 artifact 解码并检查 311 个张量。

| 格式 | save 秒 | 独立校验秒 | direct 秒 | cold 秒 | hot 秒 |
|---|---:|---:|---:|---:|---:|
| safetensors | 11.756665 | 1.429012 | 6.345971 | 7.543626 | 1.221653 |
| DCP | 20.185419 | 3.154625 | 10.478939 | 11.896724 | 1.406557 |

落盘大小：

- safetensors artifact：1,503,300,423 B。
- DCP artifact：1,503,826,867 B。

读取计数：

| 检查项 | safetensors | DCP |
|---|---|---|
| direct 源读取 | FileIO payload 1,503,264,768 B | `torch.load` 8 次，逻辑张量 12,026,118,144 B |
| cold 源读取 | FileIO payload 1,503,264,768 B | `torch.load` 8 次，逻辑张量 12,026,118,144 B |
| hot 源读取 | FileIO payload 0 B | `torch.load` 0 次 |
| cold cache fill/read | 各 1,503,264,768 B | 各 1,503,264,768 B |
| hot cache fill/read | 0 / 1,503,264,768 B | 0 / 1,503,264,768 B |
| hot 批量读取 | 240 次 | 240 次 |
| hot 单 range `read_to_buffer` | 0 次 | 0 次 |

DCP 的 8 次 `torch.load` 对应 8 个 rank 各自加载完整逻辑张量。DCP payload 不经过 gdtensor FileIO，所以 FileIO/pread 为 0 不表示 DCP 没有源读取。

DCP direct 后同一进程的 VmHWM 为 24,395,272,192 B。这是进程生命周期高水位，包含前面的阶段，不能全部归因于缓存。

## 1/4/16 MiB 批次对照

配置：15 ranks、thread coordinator、真实 Qwen payload。每组均通过 payload 校验。cold 的 source/fill/read 均为 1,503,264,768 B；hot 的 source/fill 为 0，read 为 1,503,264,768 B。

| 批次 | cold 秒 | hot1 秒 | hot2 秒 | hot 批量调用 | hot SHA256 调用 |
|---|---:|---:|---:|---:|---:|
| 1 MiB | 13.126769 | 3.117075 | 3.065701 | 1,067 | 106,933 |
| 4 MiB | 12.873299 | 2.642961 | 2.715949 | 268 | 82,963 |
| 16 MiB | 14.082676 | 2.637332 | 2.533842 | 90 | 77,623 |

memory 轮的内存高水位：

| 批次 | Python peak B | ru_maxrss B |
|---|---:|---:|
| 1 MiB | 1,507,408,047 | 3,298,758,656 |
| 4 MiB | 1,513,779,583 | 3,237,466,112 |
| 16 MiB | 1,553,655,802 | 3,304,169,472 |

4 MiB 的 hot1 比 16 MiB 慢 0.005629 秒，hot2 慢 0.182107 秒；cold 快 1.209377 秒，Python peak 少 39,876,219 B。保留 4 MiB 是本轮数据的折中，不表示它在所有机器上最优。

## 计划构建诊断

以下秒数是诊断脚本中的累计阶段时点，不是独立阶段耗时。

| ranks | full_plan 累计秒 | Python peak B | ru_maxrss B | segments |
|---:|---:|---:|---:|---:|
| 20 | 17.657475 | 25,879,896 | 111,489,024 | 6,164 |
| 100 | 112.739073 | 88,966,962 | 295,002,112 | 28,014 |

100-rank 的 `chain_fingerprint` 累计时点为 24.245264 秒，`source_identity` 为 25.086848 秒。计划构建仍是明显成本。后续优化应先处理计划和两层冷填充，不应继续扩大读取批次。

## 最终正确性门禁

证据：`gdtensor-cache-review24y-*.log`。

- API conformance 与受影响缓存测试：441 passed，52.32 秒。
- 完整 `tests/io`：582 passed，14 warnings，120.14 秒。
- 随机缓存压测：8 轮均为 169 passed，共 1,352 项执行。
- 变异检测：notorch 64/64，torch 1/1，全部检出。
- Ruff format：120 files already formatted。
- Ruff check、ty：通过。
- Pyrefly：0 errors，6 suppressed，另有 1 条未显示的 warning。
- `gdtensor/internal/api.py` SHA256：`5ef2460bbcda6731ecaa120a20ef4c75ef8240e1711360af03b8d7511e4f0b61`，与 mutation 冻结值一致。

`review24y` 未重跑全仓 pytest。后续 CI 修复后的 digest 单提交门禁为 1100 passed、138 warnings、153.27 秒。更新后的 formal gate 通过 5 个 Lean 正向文件、预期拒绝的负例和 31 个 TLC 检查；运行时哈希负对照按预期失败。证据为远端 `gdtensor-cache-review25-formal.log`、`gdtensor-cache-review25-formal-pin-negative.log` 和快照 `gdtensor-formal-review25-6DH5EHOc`。

source identity 和 fd 生命周期修复后的 digest 门禁为 1104 passed、140 warnings、159.50 秒。真实 DCP metadata-only payload 变更、safetensors 原子替换和非本地 shard fd 共 5 个定向回归通过；恢复旧 accessor 后前 4 项失败，fd 回归在上一提交也失败。最新全仓结果仍以最终 PR 的 GitHub CI 为准。

## 边界与已知限制

- 最终性能证据来自 `review24q`；批次对照来自 `review24p-focused15-{1,4,16}`。最终门禁来自独立的 `review24y`。旧 `review24u` 的重复运行日志不作为最终证据。
- 100-rank profile 未显式设置 coordinator，使用默认配置；8-rank 格式测试和 1/4/16 MiB 对照显式使用 thread coordinator。两类结果不能直接横向比较。
- 首轮 file-coordinator safetensors hot 为 50.995420 秒。该结果字节正确，但不代表最终 thread-coordinator 配置。
- SSHCache 仍使用 `subprocess.run(..., capture_output=True)`。响应在长度和 checksum 检查前可能无上限累积内存。本轮未修复该限制。
- `_content_digest` 对 cyclic 或 recursion-limited 结构保留整根 pickle 回退，可能保留 alias topology。identity-only object token 不支持。
- 历史一轮磁盘满后出现 native SIGSEGV；另一轮历史 100-rank full3 也曾在 CPython frame 路径 SIGSEGV。根因均未确定。最终 `review24q` 全部返回 0，不证明这些历史故障已修复。
- DCP 首轮曾在后续 direct 前出现 `libgomp: Thread creation failed`。save 和独立字节校验当时已通过。最终单线程 OpenMP 配置全流程通过。
- 源在加载期间变化会返回 Err；不承诺回滚已写 destination。缓存重启后无索引则冷读，不承诺掉电恢复或持久索引恢复。

远端证据目录：`/data00/home/chenkailun.c/gdtensor-cache-review24-YKtMLn`。性能数据目录已在检查目标后清理，源模型和代码目录未删除。
