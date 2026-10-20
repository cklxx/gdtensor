# gdtensor：OmniFS 丐版

结论：gdtensor 是通用分布式张量库。核心是 DTensorsView 抽象：统一描述计算 mesh、存储槽位和 reshard。它内置 OmniFS 后端，在无 GPU 环境下复刻多 rank 并发 pwrite 与 L1/L2/L3 读路径。存储链路按 OmniFS-Store 方案 2.1 实现。



1. 多 rank 并发 pwrite 同一个 HF 文件。写完即终态，免二次 merge。

2. 读取走 L1 本机缓存 → L2 P2P → L3 回源。回源只读一次，读放大为 1.0。

3. 写入按 64 MiB chunk 切分。每个 chunk 是一个 KeySet，每次 pwrite 是一个变长 KV。

4. 快照按 prepared → committed 发布。沉降把 SSD 数据重组为 HDFS 上的 HF 目录，不改 manifest。

65 项测试全部通过。

## 1 快速使用



```
pip install numpy safetensors pytest
python3 -m pytest -q tests                  # 49 passed

# 文件后端：TP4 DP4 PP2，32 rank 写共享 HF 文件
python3 -m gdtensor.sim --root /tmp/ckpt --layers 4 --hidden 1024 --ffn 2816 \
  --vocab 8192 --tp 4 --dp 4 --pp 2 --optimizer --layout hf --hf-shards 4 \
  --ranks-per-node 8 --report examples/tp4dp4pp2_hf

# OmniFS 后端 + EIC 半托管内存缓存（safetensors 3 副本，JSON 10 副本）+ 沉降
python3 -m gdtensor.sim --root /tmp/om --tp 4 --dp 4 --pp 2 --optimizer \
  --backend omnifs --n-dn 8 --sink --cache eic --ranks-per-node 8 \
  --report examples/omnifs_sink_eic

# TP 组内汇聚开关：dim1 张量先汇聚再整块写
python3 -m gdtensor.sim --root /tmp/om2 ... --backend omnifs --aggregate-tp
```

输出目录含 `report.json`、`write_trace.csv`、`read_trace.csv`。trace 列为 `t_us, rank, op, level, file, offset, size, chunk`。

## 2 分层与模块



| 层           | 职责                   | 模块                               | 依据                       |
| ----------- | -------------------- | -------------------------------- | ------------------------ |
| 逻辑层         | 张量、mesh、槽位           | `mesh.py` `dtensor.py`           | DTensorsView B.2.3、B.3.2 |
| 物理层         | 槽位到文件                | `layout.py`                      | B.2.2.2、B.2.2.3          |
| 执行层         | plan、check、主 API     | `plan.py` `api.py`               | B.1、B.3.1                |
| 视图          | 视图与 accessor         | `views.py`                       | B.2.1.1、B.2.2.1          |
| 文件 session  | safetensors / distcp | `formats.py`                     | B.2.2.5、B.2.2.6          |
| OmniFS 存储后端 | manifest、chunk、KV、沉降 | `omnifs.py` `omnifs_view.py`     | OmniFS-Store 方案 2.1      |
| 读缓存         | L1 / L2、单飞、副本上限      | `iobackend.py`                   | 需求三、热点扩散                 |
| 业务层         | 模型、布局构造器、仿真          | `models.py` `runner.py` `sim.py` | 核心之外                     |

对外 API 只有 `d_tensors_view_rank_save / load / reshard`。plan、check 等是内部函数。

## 3 核心数据结构（DTensorsView）



* `StorageMesh` 解析成槽位编号数组，编号 = `t * W + w`。`t` 是张量行，`w` 是列（rank）。

* `DisjointStorageMeshList` 用若干不重叠矩形覆盖一个文件的槽位。

* `Shard(axis)` 按 torch.chunk 切分。多个 mesh 维切同一轴时嵌套。`Replicate` 保留全量。`Partial` 不能持久化。

* `RangeFromStorageView` 只带 `contiguous_payload_offset`。session 负责映射成实际偏移。空槽占 0 字节。

* plan 是纯函数，不打开 session。主 API 是薄执行器，不重选区间。

## 4 存储表与 plan



| 模式       | 存储 placement           | 用途                   |
| -------- | ---------------------- | -------------------- |
| `hf`     | `Shard` 统一为 `Shard(0)` | OmniFS 主场景，多 rank 合写 |
| `mirror` | 与写者一致                  | DCP 类布局              |



| plan    | 方向                | 选择规则                    |
| ------- | ----------------- | ----------------------- |
| save    | compute → storage | 写本 rank 持有元素。复制维去重，写者轮转 |
| load    | storage → compute | 填满本 rank 持有元素           |
| reshard | 任意组合              | 填满目标份额。跨 rank 读计为通信     |

## 5 OmniFS 存储后端（方案 2.1：EIC-KV ChunkStore）

### 5.1 映射链（全部确定性推算）



```
HF file --manifest--> blob -- offset // 64MiB --> chunk -- == KeySet
        一次 pwrite 片段 --> 一个变长 KV -- crc32(key) % n_dn --> DN
```



* 一个文件一个 blob。blob 到 chunk 按规则推算，不存储。

* chunk 到 DN 非亲和放置。读取按 Layout 跨 DN 并行取 KV 再拼装（方案二）。

* KV key 含偏移和长度，天然唯一。同 key 重写且内容相同 = 重试幂等。内容不同报冲突。两个 KV 区间重叠报错。

* 跨 chunk 边界的写拆成多次。KV 不要求按 1 MiB 对齐。

* 固定两级目录：`project/step-N`。没有通用目录树。

### 5.2 快照协议（manifest 是唯一发布点）



1. rank0 建 manifest（state = prepared）。失败残留可重写。已 committed 不可重写。

2. barrier 后各 rank 直写 store，不经 host。

3. commit：各 rank fsync 自己的 KV 和 layout 日志，barrier 投票。

4. rank0 检查全覆盖，整写 JSON blob（index.json），manifest 改 committed 后原子替换。

5. 未提交快照不可见。任一 rank abort，全部 rank 停止发布。

6. 加载要求 committed。host 经 L1/L2/L3 读 header，共享 header 由 P2P 分摊。

### 5.3 沉降 SSD → HDFS（DMC）



* 以已提交快照为单位。manifest 只读，绝不修改。

* 按 KeySet Layout 重组变长 KV 为连续 chunk。以 chunk 为粒度 append 临时文件，再 concat 成完整 blob。

* 输出是用户可直接读的 HF 目录：`hdfs_root/project/step-N/model-*.safetensors` + `index.json`。

* 校验大小和 CRC 后发布 `sink.json`（存储侧状态），再回收 SSD KV。

* 读路径按 `sink.json` 路由 SSD 或 HDD。SSD 未命中不代表已沉降，不会自动透读 HDFS。

### 5.4 读缓存与副本



* L1 每节点 1 MiB 分块 LRU。L2 读其他节点副本。L3 回源。同一块并发未命中单飞。

* `--cache host`：副本随读者扩散（omnifs-host P2P）。

* `--cache eic`：副本有上限（方案 2.1：safetensors 3，热点 JSON 10）。超出节点走 L2 远端读，不留副本。

* 1 MiB 只是读缓存粒度。SSD 写入仍是变长 KV。

## 6 示例结果（`examples/`，本机 14 核，本地 SSD）

模型：4 层，hidden 1024，FFN 2816，词表 8192，含 ZeRO m/v（BF16 + F32）。并行 TP4 DP4 PP2，共 32 rank。

### 6.1 文件后端



| 指标                   | 值                    |
| -------------------- | -------------------- |
| pwrite 次数 / 字节       | 34117 / 681681616 B  |
| 写大小 P50 / P99        | 1408 B / 720896 B    |
| < 64 KiB 碎写占比        | 96.93%               |
| 单 64 MiB chunk 最大写者数 | 32                   |
| 读放大                  | 1.0                  |
| L1 / L2 / L3 服务占比    | 65.0% / 6.8% / 28.2% |

### 6.2 OmniFS 后端



| 指标                        | 不汇聚    | `--aggregate-tp` |
| ------------------------- | ------ | ---------------- |
| 逻辑 pwrite                 | 34116  | 1356（降 25 倍）     |
| KV Put                    | 34125  | 1365             |
| KV 大小 P50                 | 1408 B | 256 KiB          |
| < 64 KiB 碎写占比             | 96.93% | 22.12%           |
| 每个 64 MiB chunk 最大写者数     | 32     | 32               |
| 读放大（host P2P / EIC / 沉降后） | 1.0    | 1.0              |
| 读回逐字校验                    | 通过     | 通过               |

汇聚模式的跨 rank gather 字节与不汇聚一致。代价是机内拷贝和内存，收益是碎写数量。方案文档明确业务侧当前不做汇聚。

### 6.3 与文档中真实数值的对应（Qwen3-235B，TP=4，plan 级校验）



* q\_proj `[8192,4096]`（dim0）：每 rank 一段 16 MiB。

* o\_proj `[4096,8192]`（dim1）：每 rank 每行一段 4 KiB，共 4096 段。

* q\_norm `[128]` BF16 = 256 B，只由一个 rank 写。

* ZeRO m（F32）拍平后按 TP=4 切：每份 128 B。

* 文件偏移 = 8 + N + data\_offsets。变长 header N 使所有张量起点偏离 1 MiB 边界。

对应测试：`test_doc_patterns.py`。

## 7 测试覆盖（49 项）



| 类别                | 文件                     | 覆盖点                                                                                                      |
| ----------------- | ---------------------- | -------------------------------------------------------------------------------------------------------- |
| DTensorsView 用例矩阵 | `test_matrix.py`       | 每 rank 一文件、多 rank 一文件、MoE、PP stage、空槽、跨文件、复制去重、三种 reshard、混格式、disjoint/covering、布局确定性、session 生命周期、多进程并发 |
| 读缓存               | `test_read_cache.py`   | 单飞、P2P、读放大 < 2                                                                                           |
| OmniFS 后端         | `test_omnifs.py`       | chunk 切分、非亲和 DN、prepared/committed、不可见可重写、已提交不可写、重试幂等、冲突和重叠拒绝、空洞检测、沉降不改 manifest、按 sink.json 路由、EIC 副本上限 |
| 文档数值              | `test_doc_patterns.py` | q/o\_proj 段大小与段数、非对齐、单 chunk 多写者、ZeRO 128 B、汇聚开关                                                         |

校验路径与写路径独立。safetensors 用官方库解码。沉降结果按 HF 目录校验。数据逐字节比对确定性模式。

## 8 与规范的偏差（显式记录）



| 项                         | 规范                         | 本实现                | 原因                |
| ------------------------- | -------------------------- | ------------------ | ----------------- |
| accessor 回调               | `(buffer_ptr, buffer_len)` | 一个 `memoryview`    | memoryview 带指针和长度 |
| `DisjointStorageMeshList` | 只有 mesh 列表                 | 多 `tensor_ids` 行标签 | leader 函数只收 list  |
| DCP `.metadata`           | pickle                     | JSON               | 只需完成标记语义          |
| 设备显存                      | 框架管理                       | 进程内模拟全部 rank       | 无 GPU             |
| `begin_load`              | 只收 rank                    | 可选 `needed_files`  | 只读需要的 header      |
| KV 物理布局                   | EIC LavaKV / DN 网络         | 本地日志文件 + CRC 路由    | 复刻 pattern，不做真实存储 |

## 9 非目标与已知限制



* 不模拟网络带宽、时延、HPN 拓扑；耗时只反映本机 CPU 和磁盘。

* 不做对象存储 multipart、真实 P2P 传输、跨地域复制、沉降限流和后台调度。

* 不实现 EFS / EBS / FAFUSE 备选方案。只实现方案 2.1 的 EIC-KV 路径。

* 多进程模式只支持文件后端 save/load。OmniFS 后端用线程模拟多 rank。

* 下一步：接入 M15 trace 校准 IO size 和并发度；把 KV 后端替换为真实 EIC / HDFS SDK。