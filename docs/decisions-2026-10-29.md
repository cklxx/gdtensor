# Decisions 2026-10-29

## 缓存读链：节点级传播计划

结论：读链的规划、统计和执行改为同一份节点级传播计划
（`gdtensor/internal/propagation.py`）。统计值即执行器强制的上限。

### 修复的缺陷（a8a2b8a 上实测）

| 缺陷 | 旧行为 | 新行为 |
|---|---|---|
| 两层缓存扩散（small 在 node0，big 在 node0、node1） | 约束检查通过，rank 2、3 报 `1 is not in list` | 执行通过；node0 的 small 层只读一次存储 |
| outbound 口径 | 同节点 cache→rank 记入 outbound（`[4,4]`） | 同节点边不计流量（`[0,0]`） |
| inbound 与源端读取 | 按 rank 累加（`[8,8]`），源端实读 16 字节 | 按节点去重（`[4,4]`），源端实读 8 字节 |
| 内存统计 | 只算本节点 rank 需求；扩散链路 `[8,4]` | 计入转发给子节点的字节；扩散链路 `[12,4]` |
| 源端读缓冲 | 整段连续文件 run 拼接，S=8 时 32 字节 | 按 segment 读取，不超过 cell 大小 |
| 协调状态 | 线程锁加 `except Exception` 静默回退 | 解析失败直接报错；`GDTENSOR_CACHE_COORDINATOR` 可选 `file`（默认，跨进程共享文件+flock）或 `thread` |
| 缺席消费者 | 永久等待 | 预算超需求且段被多 rank 消费时，join 缺员按 `GDTENSOR_CACHE_WAIT_SECONDS` 明确报 `cache_wait`（不挂起）；在途填充（`inflight`）计为进展，不触发超时 |

### 计划的定义

- 父节点：下层节点若在上层节点集中，则读本地；否则按排序后的上层节点轮转分配。最后一层按同一规则服务 rank 节点（rank r 在 node `r // num_ranks_per_node`）。
- 需求：最后一层持有所服务 rank 在 compute view 上的元素并集；上层持有其子节点需求的并集。
- cell：一个（temporal step，concurrency slot）box。segment：cell 与需求的交集中的一段连续字节。
- segment 切分点：第 0 层按存储 slot 边界切，一个 segment 只读一个存储 range；下层按父节点 segment 切，一个 segment 只有一个父 segment。
- 预算：每个（层，节点）取最大的 C 个 cell 字节之和（所有张量合计）。执行器只在常驻字节不超预算时接纳 segment；一个 segment 不超过其 cell，因此空层总能接纳。

### 统计口径（取代 decisions-2026-10-27 的待决项）

- `mem_overhead`：每节点各层预算之和。它是硬上限，实测峰值可以更低。
- `inbound`：存储→第 0 层的字节计入填充节点；跨节点的 cache→cache、cache→rank 边计入接收节点。
- `outbound`：跨节点边计入发送节点。存储没有节点 id，不计入。
- 同节点边不计流量；每个 segment 每条边只搬一次。

concurrency 语义按 3.4 定为“切片”：C 个槽把一个 step 切成 C 段；预算容纳 C 个 cell，
即可同时在途的槽数。`num_steps` 增大时 cell 变小，预算随之变小。

### 边界

- 默认协调器为共享文件 + `fcntl.flock`（`_FileSlotTable`，目录为 `GDTENSOR_CACHE_DIR` 或系统临时目录），每 rank 一进程可经 `GDTENSOR_CACHE_COORDINATOR=file` 协调；`thread` 保留进程内线程模式。尚未做多机部署测试。
- 计划在第一次 `chain_load` 时按链构建并缓存，键为链指纹；缓存对 cache view 持弱引用（`weakref.finalize`），view 被回收时缓存删除，不 pin 任何 cache view。
- `check_constraints` 前置校验：同一 cache view 的所有张量必须覆盖同一节点集且宽度一致；`rank` 越出链覆盖范围在 `chain_load` 入口报 `InvalidArgumentError`。
