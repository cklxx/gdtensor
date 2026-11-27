# 跨计算和存储的 DTensorsView 设计

本文说明 DTensorsView 如何同时描述计算侧、存储侧和缓存侧的张量，以及如何在它们之间搬运数据。类型定义见附录 B，Python 映射见附录 C；本文解释模型、边界、执行顺序和不变量，不重复类型定义，也不记录变更历史。

读者对象是第一次接触这套设计的实现者和调用方。文中的类型名以附录 B 为准；术语见附录 A。

## 1 问题

本文面向写训练、评测和权重处理代码的人。他手上同时有两样东西：计算侧一组已经按某种并行方式切开、绑在设备上的 `torch.DTensor`；存储侧磁盘上的 checkpoint，里面是一个或多个文件，按文件格式装着同一批张量。

他反复要做三件事：把 checkpoint 读成计算侧的 `torch.DTensor`，把计算侧的结果写成 checkpoint，在两侧切分方式不同时把数据搬成目标切分。麻烦的地方是两侧的切分方式可以完全不同：

- 写 checkpoint 时可能一个 rank 一个文件，可能多个 rank 合写一个文件，也可能一个张量拆到多个文件里；读的时候并行度还可以和写的时候不一样，用户不该为此改代码。
- 一个文件里不一定装全部张量：可能只装本 pp stage 的张量，也可能 replicated 的张量在整组文件里只存一份。读的时候要能接受某个文件里没有某个张量，跳过它，不能报错，也不能把别的张量读错位置。
- 一个逻辑张量可能跨多个文件，读的时候要能拼回一个张量。
- 保存要能安全发布：读的人要么看到完整的新 checkpoint，要么看到旧的，不能读到写了一半的文件；不同格式的完成标记也不一样。
- 同一份张量在两侧的名字和全局元数据必须对得上，切分方式和落在哪些设备上可以不同；对不上时要报错，不能悄悄读错。
- 一次读或写可以在存储和计算之间经过一到两层节点本地缓存；用户希望能用同一套调用表达这条路径，并且在搬之前知道占多少内存、走多少流量。

具体一点的例子是 LAWA：权重由训练集群按当时的并行度写成 checkpoint；事后要对最近 k 个 step 做逐元素平均，而处理往往不在原集群上——临时集群的卡数、切分方式和训练时都不一样，训练时的拓扑信息也未必拿得到。用户希望读、算、写三步都不依赖原集群长什么样：这里读出的是 storage 侧的切分，要落到的是临时集群这份 mesh 上的切分，中间的换算不该由用户写。

用户不想为每种拓扑、每种文件格式、每种缓存层数各写一套搬运代码，也不想关心文件里第几个字节对应哪个张量。他需要一套统一的说法：有哪些张量、它们在两侧分别怎么切、怎么搬；搬运计划和偏移由库推出来，文件格式的差异留给格式实现。

本文出现的 DCP 指 PyTorch distributed checkpoint；safetensors 是单文件张量格式；PP stage 指流水线并行的一段。

设计目标是让用户只给出张量清单和两侧的 view（计算侧、存储侧），就能表达读和写；读的同时把 storage 侧的切分换成 compute 侧的切分，写时也支持通过缓存层将细碎计算分片聚合成连续段后再落盘。核心不决定布局策略，也不做数据的序列化。

## 2 心智模型

把系统分成三层：

| 层 | 回答的问题 | 主要类型 |
|---|---|---|
| 逻辑层 | 有哪些张量、槽位怎么排 | `DTensorTable`、`NamedDTensor`、`StorageMesh`、`CacheMesh` |
| 物理层 | 槽位落在哪个文件、什么格式 | `PhysicalLayoutTable`、`DisjointStorageMeshList`、`ContainerFormat` |
| 执行层 | 怎么把源 range 搬到目标 range | `DTensorRange`、`CopyPlan`、plan、session、accessor、读写链 |

view 是这一模型的句柄：

- `ComputeDTensorsView`：`tensor_ids` + `DTensorTable[ComputeMesh]` + `DTensorDataAccessor[RangeFromComputeView]`；
- `StorageDTensorsView`：`tensor_ids` + `DTensorTable[SingleTensorStorageMesh]` + `DTensorDataAccessor[RangeFromStorageView]` + `PhysicalLayoutTable` + `begin_save` / `begin_load`；
- `CacheDTensorsView`：`tensor_ids` + `DTensorTable[CacheMesh]` + `DTensorDataAccessor[RangeFromCacheView]`。缓存是计算节点上的伴生存储；`RangeFromCacheView` 就是 `RangeFromComputeView`（B.3.1），所以缓存不需要新的 range 类型，也不需要 session 和 `PhysicalLayoutTable`。

三个 view 都有 `tensor_ids`、逻辑表和 accessor。`DTensorsView` 是三支的 oneof：`ComputeDTensorsView | StorageDTensorsView | CacheDTensorsView`；内部的 plan 用它表达读写链每一段的两端，对外的 load / save 各自收具体的 view 类型。存储侧多一个物理布局表和 session，原因是文件是多个张量共享的线性容器，而且文件有 header / metadata 需要显式发布；cache 和 compute 都不需要这一层，两者只差 mesh 的参与者：compute 是 rank / `world_size`，cache 是 node / `num_nodes`。带缓存的 load / save 统一由读写链 `DTensorsRWChain` 描述，执行时拆成若干段，每段是一份 `CopyPlan`，见 5.8。

计算侧的数据处理不走 `GenericDTensor` 本身，而是走 compute/torch 桥：`compute_d_tensors_view_get_gd_tensor` 按 tensor id 取 `GenericDTensor[ComputeMesh]`，`gd_tensor_get_d_tensor` 取出 `torch.DTensor`，处理完用 `gd_tensor_reset_d_tensor` 写回。storage 侧不提供数据处理 API；`GenericDTensor[StorageMesh]` 先经 load 或读链变成 `GenericDTensor[ComputeMesh]`，再进入这座桥。cache 侧也没有数据处理 API：缓存不参与计算，只作为中转被复制。

## 3 逻辑模型

### 3.1 slot

`slot`（槽位）是逻辑概念，不是文件或容器。把存储计划看成一个 `num_tensors × world_size` 的表：每一行是一个张量，每一列是一个全局 rank，每个格子是一个 slot；总 slot 数 = `num_tensors * world_size`。

格子可以为空：该 rank 不持有这个张量、这个张量不在该 pp stage，或者它只存一份而其余列留空。同一行的非空格子按列顺序拼接，构成这个张量的存储区域。

### 3.2 DTensorTable

`DTensorTable[MeshT]` 是有序的 `NamedDTensor[MeshT]` 列表。列表顺序就是容器 header 中张量的顺序；不需要第二份 id 顺序。

`NamedDTensor` / `GenericDTensor` / `NaiveDTensor` 在计算侧和存储侧共用；两侧的差别是 `MeshT` 和 `RangeT`。

### 3.3 StorageMesh

`StorageMesh` 是存储侧唯一的 mesh 概念，和计算侧的 `ComputeMesh` 对称。它的 `SlotOuterProductMesh` 分支同时描述两个维度：

- `tensor_slots`：`ArrangedMesh["tensor_count"]`，哪些张量；
- `content_slots`：`ContentSlotsMesh`，即 `LiteralMesh["world_size"] | ArrangedMesh["world_size"]`，每个张量的哪些槽位；`LiteralMesh` 直接列出 rank，`ArrangedMesh` 用 `"world_size"` 表示列数。

`ReshapedMesh` / `SlicedMesh` 在这两个维度上继续变形。

计算侧的 `ComputeMesh` 也有 `LiteralMesh["world_size"]` 分支，用来接住任意 rank 的 device mesh。

mesh 相关的 Oneof 分支都是命名类型，所以不带 tag；`type()` 或 `isinstance(x, list)` 就能分出是哪一支。

按 `TensorSlotsMeshT` 分两种：

- `SingleTensorStorageMesh`：tensor slots 是 `ArrangedMesh["constant_one"]`，只覆盖一个张量，用于 `DTensorTable`；
- `MultiTensorsStorageMesh`：tensor slots 是 `ArrangedMesh["tensor_count"]`，覆盖多个张量，用于 `DisjointStorageMeshList`。

一个张量的单张量 mesh 必须等于多张量 mesh 中该张量所在的那一行：

```python
single_tensor_mesh = multi_tensors_mesh[
  tensor_order_value_of(tensor_id) : tensor_order_value_of(tensor_id) + 1,
  :
]
```

不能用一个覆盖多个张量槽的 mesh 代替单张量 mesh。`tensor_order_value_of` 是内部 helper，返回张量在 `DTensorTable.named_d_tensors` 里的顺序。

### 3.4 CacheMesh

`CacheMesh` 是缓存侧的 mesh 概念，和 `ComputeMesh` / `StorageMesh` 并列。它的参与者是计算侧的节点（node），不是 rank，所以这三个 mesh 类型不能互换。

它有三个维度，`StCacheOuterProductMesh` 把它们相乘，`St` 是 spatiotemporal。`SlotOuterProductMesh` 是二维（tensor slots × content slots），这里是三维（step × node × buffer）；名字都带 `OuterProduct`，表示参与者是各维度各选一部分之后的坐标：

- `temporal`：`TemporalCacheMesh`，`num_steps` 个 step；`num_steps` 是时间维复用的粒度：同一块缓存内存在 `num_steps` 个 step 之间复用，取值越大，摊到每个 step 上的占用越小；
- `spatial`：`SpatioCacheMesh`，`num_nodes` 个节点；
- `concurrency`：`ConcurrencyMesh`，即 `ArrangedMesh["concurrency_width"]`，`concurrency_width` 个并发缓冲槽。

三个维度的放置符由 `check_constraints` 约束：`temporal` 和 `concurrency` 固定为 `Shard`，`spatial` 允许 `Replicate` 或 `Shard`（`DTensorsRWChain` 上，读方向要求 `spatial` 为 `Replicate`，写方向要求 `spatial` 为 `Shard`）。原因在 `spatial`：读侧缓存要消减的是读热点，存储侧数据只有一份，计算侧却可能有 24k 个 rank 都要它，直连时这 24k 个读者全压在存储的那一份上，`spatial` 在读方向逐级用 `Replicate` 放大副本，把读者摊开（一传十、十传百）；而在写方向，`spatial` 使用 `Shard` 让各缓存节点持有互不重叠的数据分片，把细碎计算切片路由并合并为更大的连续段后收敛写盘。`temporal` 和 `concurrency` 切的是不同的数据片（不同的 step、不同的缓冲槽），不是同一份数据的副本，所以它们始终是 `Shard`。

`Replicate` 逐级放大意味着每个节点都留一份完整副本，集群缓存内存会跟着涨；压内存靠 `temporal`：同一块缓存内存在 `num_steps` 个 step 之间复用，`num_steps` 越大摊到每个 step 上的占用越小，`num_steps = 1` 时这块内存只服务一个 step、占用最大。`concurrency` 再把同一个 step 的副本拆成 `concurrency_width` 个缓冲槽：执行层可以在填一个槽的同时读另一个槽，让填充的等待被流水掩盖，槽越多同时在途的段越多。双缓冲只是其中最简单的一种，即 `concurrency_width = 2`；例如 `[Shard(0), Replicate(), Shard(0)]` 配 `TemporalCacheMesh(10) * SpatioCacheMesh(10) * ConcurrencyMesh(2)`，最后一个维度就在当 double buffer 用。

`CacheMesh` 顶层只有 `StCacheOuterProductMesh` 和 `PermutedMesh[CacheMesh]` 两个分支；`temporal` / `spatial` 各自可以 `ReshapedMesh` 成对方，用来换展开顺序。`ReshapedMesh` 保持元素总数不变，所以 temporal 与 spatial 之间的 reshape 只在两侧总数相等时成立；reshape 只改展开顺序，不改放置符。`CacheMesh` 没有 `SlicedMesh` 分支。

node 和 rank 之间的换算由读链的 `num_ranks_per_node` 给出（5.8）：node `n` 覆盖 rank 区间 `[n * num_ranks_per_node, (n + 1) * num_ranks_per_node)`，且 `num_ranks_per_node * num_nodes == world_size`。`temporal` 和 `concurrency` 不增加 rank：它们是同一批 node 上的缓冲槽。

## 4 物理模型

### 4.1 PhysicalLayoutTable

`PhysicalLayoutTable` 是纯配置数据，不是策略对象；生成后可以直接序列化。

- `container_format`：整个表只有一种 `ContainerFormat`。所有文件同格式；一个 `StorageDTensorsView` 持有一份布局表，所以一次操作只有一种格式，混格式拆成多次操作、多份布局。
- `tensor_entries`：`dict[TensorComponentPos, DisjointStorageMeshList]`。键是 `(tensor_id, rank)`；值是它所属的文件级逻辑分组；没有键表示这个张量在这个 rank 上没有物理文件。
- `file_paths`：`dict[TensorComponentPos, $file_path str]`。键是 list 的 leader component pos，值是文件路径。

遍历从逻辑层开始：先用 `(tensor_id, rank)` 在 `tensor_entries` 里找到 `DisjointStorageMeshList`，再用 `leader_tensor_component_pos_of(list)` 在 `file_paths` 里找到路径。`file_path_of(layout, pos)` 是内部 helper，封装这两步。

`DisjointStorageMeshList` 对应一个物理文件，里面的 mesh 互不重叠；不同文件的 mesh 也互不重叠。每个 slot 最多被一个 mesh 覆盖。

### 4.2 文件、格式与会话

`ContainerFormat` 是计划级属性，取 `safetensors` 或 `distcp`。格式决定文件层的发布方式：

- safetensors：header 在文件开头，包含相对 data section 的 `data_offsets`；header 先写，data 后写，`commit` 做 fsync + rename。
- distcp：payload 文件先写，`.metadata` 最后写；`.metadata` 的发布是完成标记。

这些差异由 `StorageSaveSession` / `StorageLoadSession` 的具体实现负责。`begin_save` / `begin_load` 返回 session；具体实现是 `SafeTensorsSession` / `DcpTensorsSession` 这类业务 session。

- save session：`begin_save` 建临时文件、按格式写 header；`commit` 发布；`abort` 丢弃临时文件；
- load session：`begin_load` 读 header / metadata；`finish` 关闭；
- save session 的终态是 `commit` 或 `abort`，两者都会关闭 session，所以没有单独的 `finish`；load session 只读，终态是 `finish`，没有 `commit` / `abort`；
- commit 结果 unknown 时保留 payload，`abort` 不能删。

`StorageSaveSession` / `StorageLoadSession` 的 `init_data_accessor` 返回绑定到该 session 的 `DTensorDataAccessor[RangeFromStorageView]`；只拿到 session 的回调（例如 `d_tensors_view_map` 的 session 转换）用它读写 payload。`StorageDTensorsView.data_accessor` 是 view 级的 accessor；两者的 range 类型相同，区别是 offset 映射绑定到哪个 session。

### 4.3 构造边界

对象按 protobuf 风格构造：`new_*` 创建、`set_*` / `add_*` / `mutable_*` 填字段、`check_constraints` 检查。核心不决定布局策略；`PhysicalLayoutTable` 的 `container_format`、`tensor_entries`、`file_paths` 仍由业务决定，再按 protobuf 风格写入。

一个 rank 一个文件、多个 rank 一个文件、多个 rank 多个文件、每个 pp stage 一个文件，以及 safetensors / distcp，都是构造器生成数据时的选择。表一旦生成，拓扑和格式就固定在数据里。

### 4.4 逻辑层和物理层的边界

- `DTensorTable` 只回答有哪些张量、顺序是什么；不出现文件路径、文件大小、container。
- `StorageMesh` 只描述 tensor slots 和 content slots；不出现文件路径、文件格式。
- `DisjointStorageMeshList` 是一个物理文件的逻辑分组；它本身不带文件格式和路径。路径存在 `file_paths` 里，用 `leader_tensor_component_pos_of(list)` 作键查得；调用方通常直接用 `file_path_of(layout, pos)`，它封装了这条链。
- `PhysicalLayoutTable` 是逻辑到物理的查表配置数据；没有函数字段，可以直接序列化。
- placement 在 `NamedDTensor` 里；`StorageMesh` 不重复 placement。
- `StorageDTensorsView` 是消费方句柄：张量子集 + 逻辑表 + accessor + 物理布局表 + session。它不带 rank；rank 是 `d_tensors_view_rank_*` 的参数。
- `CacheDTensorsView` 没有 `PhysicalLayoutTable`，也没有 session：缓存在节点本地内存里，不是文件，读写由 accessor 负责；它同样不带 rank，参与者是 node。

## 5 执行模型

### 5.1 `CopyPlan` 和 `DTensorRange`

`DTensorRange` 表示一次复制的一侧：

```text
DTensorRange =
    Oneof
  | RangeFromComputeView
  | RangeFromStorageView
```

`CopyPlan` 是 `list[($src DTensorRange, $dst DTensorRange)]`，每一对表示一次源 range 到目标 range 的复制。`CopyPlan` 是唯一的 pair 类型；compute/storage 的差异由 `DTensorRange` 的分支类型表达：`RangeFromComputeView` / `RangeFromStorageView`。cache 一侧用 `RangeFromCacheView`，它就是 `RangeFromComputeView` 的别名（B.3.1），所以 `DTensorRange` 只有两个分支，缓存不引入新的 range 类型。类型上允许四种 branch 组合；具体 plan API 用前置条件限制合法组合：`d_tensors_view_rank_plan_load` 产生 storage→compute，`d_tensors_view_rank_plan_save` 产生 compute→storage，`d_tensors_view_rank_plan_chain_load` 与 `d_tensors_view_rank_plan_chain_save` 均接收统一的 `DTensorsRWChain` 并按对应方向为每一段各产生一份 `CopyPlan`，返回同构的 `CopyPlanChain`。

### 5.2 plan API

`d_tensors_view_rank_plan_load`、`d_tensors_view_rank_plan_save` 返回 `Result[CopyPlan]`；`d_tensors_view_rank_plan_chain_load`、`d_tensors_view_rank_plan_chain_save` 均返回 `Result[CopyPlanChain]`。四者都是纯函数，不打开 session，负责 range selection。

`CopyPlanChain` 与 `DTensorsRWChain` 同构：分支一一对应，但每个分支里的字段不是 view，而是一段数据流的 `CopyPlan`，按数据流顺序排。`Direct` 一个（`src_to_dst`），`OneLayerCached` 两个（`src_to_cache`、`cache_to_dst`），`TwoLayerCached` 三个（`src_to_first_cache`、`first_cache_to_second_cache`、`second_cache_to_dst`）；`Mixed` 与多段链对应，每段一条 concrete plan 链。

`d_tensors_view_rank_load`、`d_tensors_view_rank_save`、`d_tensors_view_rank_chain_load`、`d_tensors_view_rank_chain_save` 是不带 plan 的 API，只做薄执行器：调用各自的 plan API，按 `CopyPlan` 执行复制，不重新选择区间。

读写链每一段的 `tensor_ids`、端视图方向与 `global_tensor_meta` 前置条件分别放在 `d_tensors_view_rank_plan_chain_load` 和 `d_tensors_view_rank_plan_chain_save` 的 `Assert` 里；`global_tensor_meta_of` 从 `NamedDTensor` 取出全局 meta，naive 和 reshaped 两支都归到同一个全局 meta。取 `NamedDTensor` 时用 `tensor_order_value_of(d_tensor_table, tensor_id)` 得到列表下标，不能把 `tensor_id` 直接当 `named_d_tensors` 的下标。不带 plan 的 `d_tensors_view_rank_chain_load` / `d_tensors_view_rank_chain_save` 只执行。

### 5.3 偏移：payload 偏移 vs 文件偏移

`RangeFromStorageView` 的偏移是 `contiguous_payload_offset`，不是 `file_offset`：

- 在同一个文件里，所有 payload 假定连续存放；空槽占 0 字节；
- 它由 slot、`GlobalTensorMeta` 和 rank 推出，不包含 header 和对齐；
- `data_accessor` 只认这个偏移；session / format 再把它映射成实际文件偏移（`get_file_offset` 返回的值）。

例：一个文件里依次是 A0=100、A1=空、A2=100、B0=200。

```text
contiguous_payload_offset: A0=0, A2=100, B0=200；A1 没有 range
data_start = 208（8 字节长度 + 200 字节 header/padding）
get_file_offset: A0=208, A2=308, B0=408
```

空槽 A1 占 0 字节，所以 A2 的偏移是 100、B0 是 200，不会因为 A1 被推到 300。

read/write buffer 完全由 `view.data_accessor` 负责；session 只负责 `get_file_offset` 和生命周期，不做 payload 序列化 / 反序列化。

### 5.4 save / load 的执行顺序

save：

1. `plan = d_tensors_view_rank_plan_save(rank, compute_view, storage_view)`；
2. `session = storage_view.begin_save(rank)`；format 由 `physical_layout.container_format` 决定，session 在 begin 里写 header / 建临时文件；
3. 对 `plan` 里的每个 `(src, dst)`：`src` 是 compute range，`dst` 是 storage range；`compute_view.data_accessor.read_to_buffer(src)` 读，`storage_view.data_accessor.write_from_buffer(dst)` 写。`dst` 的 `contiguous_payload_offset` 由 plan 算好；session / format 在内部把它映射成实际文件偏移；
4. `session.commit()` 或 `session.abort()`。

load：

1. `plan = d_tensors_view_rank_plan_load(rank, compute_view, storage_view)`；
2. `session = storage_view.begin_load(rank)`；begin 里读 header / metadata；
3. 对每个 `(src, dst)`：`src` 是 storage range，`dst` 是 compute range；`storage_view.data_accessor.read_to_buffer(src)` 读，`compute_view.data_accessor.write_from_buffer(dst)` 写；
4. `session.finish()`。

带缓存的读走读链，带缓存的写走写链，执行顺序见 5.8。每次 load 本身就把 storage 侧的切分换成 compute 侧的切分，所以核心不再提供单独的换切分 API；写分为直接写 `d_tensors_view_rank_save` 和经由写链聚合的 `d_tensors_view_rank_chain_save`。

check 函数也收 `CopyPlan`，按 range 类型过滤出 compute 或 storage range，再做 disjoint / covering 检查。check 是内部一致性校验，不是 IO API。

### 5.5 d_tensors_view_map

`d_tensors_view_map` 是 view 转换 API，Python 入口是 `gdtensor.builder.d_tensors_view_map`：它把一个 `DTensorsView` 转成同种类的另一个 `DTensorsView`，只做 view / session 的结构转换。它不选 range，不做 load / save / 读链的跨 view 搬运，也不承担归约、平均这类数据处理；这些计算走 5.6。

`DTensorsViewMapCore` 是调用方提供的五个可选回调：

- `convert_each_compute_view`：compute view 的每个 `GenericDTensor[ComputeMesh]` 转成一个新的 `GenericDTensor[ComputeMesh]`；
- `convert_each_storage_view`：storage view 的每个 `GenericDTensor[StorageMesh]` 转成一个新的 `GenericDTensor[StorageMesh]`；
- `convert_each_cache_view`：cache view 的每个 `GenericDTensor[CacheMesh]` 转成一个新的 `GenericDTensor[CacheMesh]`；
- `convert_each_storage_load_session`：storage view 的每个 `StorageLoadSession` 转成一个新的 `StorageLoadSession`；
- `convert_each_storage_save_session`：storage view 的每个 `StorageSaveSession` 转成一个新的 `StorageSaveSession`。

这五个字段是 product，但都可以为 nil：tensor 字段为 nil 表示 identity，session 字段为 nil 表示跳过对应的 session。`d_tensors_view_map` 按 `origin` 的类型只处理对应字段。

执行顺序：

1. `origin` 是 `ComputeDTensorsView`：对 `DTensorTable[ComputeMesh]` 的每个 `NamedDTensor[ComputeMesh]` 调 `convert_each_compute_view`；该字段为 nil 时 tensor 原样保留；没有 session。
2. `origin` 是 `StorageDTensorsView`：对 `DTensorTable[SingleTensorStorageMesh]` 的每个 `NamedDTensor[SingleTensorStorageMesh]` 调 `convert_each_storage_view`；该字段为 nil 时 tensor 原样保留。再处理 load / save session：字段非 nil 时打开对应 session 并调用，字段为 nil 时跳过该 session。session 的打开、提交、中止由 `d_tensors_view_map` 管；转换函数只替换 session。
3. `origin` 是 `CacheDTensorsView`：对 `DTensorTable[CacheMesh]` 的每个 `NamedDTensor[CacheMesh]` 调 `convert_each_cache_view`；该字段为 nil 时 tensor 原样保留；没有 session。
4. 返回的 view 与 `origin` 同种类；`tensor_ids` 不变，`global_tensor_meta` 由转换函数决定。

任一回调返回 `Err` 时，`d_tensors_view_map` 原样返回那个 `Error`，不再调用后面的回调。

`d_tensors_view_map` 不承担归约、平均这类计算。需要计算时走 5.6：先把 `GenericDTensor[StorageMesh]` 变成 `GenericDTensor[ComputeMesh]`，再取 `torch.DTensor`。

### 5.6 Compute 与 torch.DTensor 的桥

计算侧的数据处理不在 `GenericDTensor[ComputeMesh]` 上做，而是转到 `torch.DTensor` 上做。`gd_tensor` 是 `GenericDTensor[ComputeMesh]` 的简称，`d_tensor` 指 `torch.DTensor`。五个 API 构成这座桥：

- `compute_d_tensors_view_get_tensor_ids(view)`：列出 compute view 的 tensor id；
- `compute_d_tensors_view_get_gd_tensor(view, tensor_id)`：按 id 取 `GenericDTensor[ComputeMesh]`；
- `gd_tensor_make(d_tensor)`：从 `torch.DTensor` 取数据、`GlobalTensorMeta`、placements 和 device mesh，并把 device mesh 转成 `ComputeMesh`，造一个 `GenericDTensor[ComputeMesh]`；任意 rank 的 device mesh 用 `LiteralMesh["world_size"]`；
- `gd_tensor_get_d_tensor(gd_tensor)`：取底层 `torch.DTensor`；返回 `nil` 表示这个 generic tensor 只有元数据、没有 materialize；
- `gd_tensor_reset_d_tensor(gd_tensor, d_tensor)`：把处理后的 `torch.DTensor` 写回 generic tensor。

Python 里这五个函数放顶层，在 `gdtensor` 包内按“动词 + 对象”简写，去掉 `compute_d_tensors_view_` / `gd_tensor_` 这类主体前缀：`gdtensor.get_tensor_ids`、`gdtensor.get_gd_tensor`、`gdtensor.make_gd_tensor`、`gdtensor.get_d_tensor`、`gdtensor.reset_d_tensor`。viba 全名不改，完整对照见 C.3。

典型流程：

1. `compute_d_tensors_view_get_tensor_ids(view)` 拿到要处理的 id；
2. `compute_d_tensors_view_get_gd_tensor(view, tensor_id)` 拿到 `GenericDTensor[ComputeMesh]`；
3. `gd_tensor_get_d_tensor` 取出 `torch.DTensor`；返回 `nil` 时先 materialize（例如 load），或用 `gd_tensor_make`；
4. 在 `torch.DTensor` 上做计算；
5. `gd_tensor_reset_d_tensor` 写回；
6. 用 save 把结果写回 storage。

storage 侧没有对应的数据处理 API。`GenericDTensor[StorageMesh]` 总是先经 load 或读链变成 `GenericDTensor[ComputeMesh]`，再进入这座桥；处理完经 save 回到 storage。`d_tensors_view_map` 只做 view / session 的结构转换，不参与这里的计算。

`gd_tensor_reset_d_tensor` 要求新的 `torch.DTensor` 的 `global_tensor_meta`、`placements`、`mesh` 与原 `gd_tensor` 一致；不一致时调用方自己在计算侧重新分配，或重新 `gd_tensor_make`。

### 5.7 类的 API

所有数据 `Object` / `Oneof`（session / access 接口除外）都按 protobuf 生成的 API 约定，不另设独立的 builder 对象；build 接口放在 `<ClassName>BuilderMixin`（附录 C）：

- `new_<type>()`：创建空对象；
- `get_<field>()` / `set_<field>(value)`：读写字段；
- `add_<field>(value)`：向 repeated 字段追加；
- `mutable_<field>()`：取嵌套对象 / map 的可写句柄；
- `clear_<field>()`：清空字段；
- `check_constraints()`：返回 `Result[void]`。递归规则：
  - `Object`：逐个字段调用该字段的 `check_constraints`；
  - `Oneof`：调用当前分支的 `check_constraints`；
  - `list[T]` / `dict[K, V]`：对每个元素 / 每个 value 调用；
  - `nil`：跳过；
  - 任意子对象返回 `Err`，父对象立即返回同一个错误；父对象不包装、不改写它。

`check_constraints` 失败时，`Err` 里的 `value` 是 `Error` 的 `ConstraintViolationError` 分支（B.3.4）：`type_name` 是出错对象的类型名，`constraint_name` 是它没通过的那条约束，`msg` 说明原因。

构造 view 按这个约定：

```python
view = new_compute_d_tensors_view()
view.set_data_accessor(accessor)
view.mutable_d_tensor_table().add_named_d_tensor(named)
must_ok(view.check_constraints())
```

`StorageDTensorsView` 还 `set_physical_layout` / `set_begin_save` / `set_begin_load`；`PhysicalLayoutTable` 的 `tensor_entries` / `file_paths` 用 `mutable_*` 写入。所有 `check_constraints` 返回 `Result[void]`；失败时 `Err` 里是 `ConstraintViolationError`，带约束名和 `msg`。

`check_constraints` 不替代 `plan` / `check_*`：前者检查对象自身的数据是否完整、合规；后者检查一次 `CopyPlan` 的访问是否 disjoint / covering。

Python 实现里，build 接口放在 `<ClassName>BuilderMixin`；数据类写成 `class Foo(FooBuilderMixin)`；见附录 C。

### 5.8 读写链与缓存

一次 load 可以在 storage 和 compute 之间经过一到两层缓存；对称地，一次 save 也可以在 compute 和 storage 之间经过一到两层缓存。缓存由 `CacheDTensorsView` 描述：它落在计算侧的节点上，accessor 用 compute range，所以既没有 session 也没有 `PhysicalLayoutTable`；mesh 的参与者是 node，见 3.4。

读侧要解决的是热点：同一份存储数据可能有几万个 rank 要读，这部分压力靠存储自身消化不了，只能在计算侧用缓存逐级分担。`spatial` 在读方向固定为 `Replicate`，逐级放大副本分散读者，一传十、十传百；代价是内存，靠 `temporal` 在多个 step 之间复用同一块缓存内存、靠 `concurrency` 提供若干缓冲槽让填充和读取重叠（3.4）。

写侧除去重（同一份 replicated payload 只写一次）外，当计算端切分在盘上行主序下表现为多段细碎小分片（例如 2D 列切分 `Shard(1)` 造成的逐行跳空，或高 DP/小张量下的大量极短分片）时，缓存沿着 `spatial` 维采用 `Shard` 让各缓存节点持有互不重叠的数据分片，将邻接分片聚合成连续行段或跨卡块并逐级汇聚到 `second_cache` 连续 `pwrite` 落盘，既保留 `gdtensor` 三层 View 与纯函数 `CopyPlan` 抽象，又将小分片随机写转化为顺序连续大块写。

`DTensorsRWChain` 统一描述一次带缓存 load 或 save 的数据流，分四支：

- `DirectDTensorsRWChain`：`src` → `dst`，没有缓存；
- `OneLayerCachedDTensorsRWChain`：`src` → `cache` → `dst`；
- `TwoLayerCachedDTensorsRWChain`：`src` → `first_cache` → `second_cache` → `dst`；
- `MixedDTensorsRWChain`：若干 `DisjointTensorIdsAndRWChain`，每段一条 concrete 链，段的 `tensor_ids` 两两不重叠且数据方向一致。

`src` 与 `dst` 为 `(StorageDTensorsView | ComputeDTensorsView)`，读方向 `src` 为 `StorageDTensorsView`、`dst` 为 `ComputeDTensorsView`；写方向 `src` 为 `ComputeDTensorsView`、`dst` 为 `StorageDTensorsView`。concrete 链指前三支；`Mixed` 链的每一段只能放 concrete 链，所以混合不会嵌套。`get_tensor_ids_of_rw_chain`（Python 顶层 `gdtensor.get_chain_tensor_ids`）返回整条 RW 链覆盖的 tensor id：concrete 支取计算侧端点的 `tensor_ids`（与 `src` / `dst` 一致）；`Mixed` 支按列表顺序拼接各段的 `tensor_ids`。

`d_tensors_view_rank_chain_load` 的执行顺序：

1. `plan_chain = d_tensors_view_rank_plan_chain_load(rank, rw_chain)`：校验 `rw_chain` 为读方向及各段元数据一致，按数据流把链拆成段，每段一份 `CopyPlan`，放在与 `DTensorsRWChain` 同构的 `CopyPlanChain` 里。`Direct` 一段，`OneLayerCached` 两段（`src → cache`、`cache → dst`），`TwoLayerCached` 三段（`src → first_cache → second_cache → dst`）；`Mixed` 先拆成互不重叠的子链，每个子链再拆段。
2. 逐段执行：执行器按 `DTensorsRWChain` 和 `CopyPlanChain` 的同构分支取段；每段是一份 `CopyPlan`，里面若干对 range，每对把源 view 的一段 range 搬到目标 view 的一段 range，两端的切分可以不同。session 由 `src` 的 `StorageDTensorsView.begin_load` 提供，cache 和 compute 一侧的段直接用两端 view 的 `data_accessor`。
3. 段与段按数据流顺序推进，前一段准备好后才消费后一段；`src` 的 session 在该子链所有段执行完后 `finish`。
4. 任一步返回 `Err` 时原样返回那个 `Error`，并 `finish` 已经打开的 session。

`d_tensors_view_rank_chain_save` 执行写方向 `DTensorsRWChain`：

1. `plan_chain = d_tensors_view_rank_plan_chain_save(rank, rw_chain)`：校验 `rw_chain` 为写方向、`CacheMesh.spatial` 为 `Shard`、各段张量与元数据一致，并推导同构 `CopyPlanChain`（`Direct` 一段，`OneLayerCached` 两段，`TwoLayerCached` 三段 `src → first_cache → second_cache → dst`）。
2. 为每个 concrete 子链打开 `dst.begin_save(rank)`，各计算 rank 按下游 `spatial: Shard` 归属将区间 stage 进底层 cache；同一段的所有指定 producer 到齐且字节全部就位后，才由负责 flush 的本地（或指定服务）rank 读出连续块并写入下一层 `Shard` 节点或最终存储区间。
3. 每次仅在节点字节预算范围内准入新的写段并在 flush 完成后立即释放预算。所有参与 rank 都参与 `session.commit()`；任一环节失败时向同组 rank 传播精确错误并 `abort()` 已打开的 save session。

三个 `get_stat_*_of_rw_chain` 是纯函数：只看链结构和各 view 的 mesh / `GlobalTensorMeta`，不打开 session，也不读 payload。它们返回 `list[int]`，下标是 node id，单位是字节，长度是链里出现的节点数：

- `get_stat_mem_overhead_of_rw_chain`：每个 node 上这份缓存占的内存。`temporal` 是复用维，同一块缓存内存在 `num_steps` 个 step 之间复用，所以这个值随 `num_steps` 增大而变小；多一个 `concurrency_width` 缓冲槽就多占一份；
- `get_stat_outbound_traffic_of_rw_chain`：每个 node 发出的字节数；
- `get_stat_inbound_traffic_of_rw_chain`：每个 node 收到的字节数。

三个函数按段累加：每段的字节数由该段要搬的张量大小和该段在两侧 mesh 上的分片决定；同时按 node 归属记账，同一 node 内部的段不计流量，跨 node 的段分别计入发送 node 的 outbound 和接收 node 的 inbound。调用方可以先用这三个函数判断缓存装不装得下、流量是否超预算，再决定用哪一支链。

这三个函数在这里固定下来的是口径：一个 node 一个值、单位字节、同一 node 内的段不计流量、`mem_overhead` 随 `num_steps` 增大而变小。具体的换算式（`num_steps`、`concurrency_width`、spatial 各级参与数怎么折算成字节）和两个流量函数的分摊方式由实现给出，本文不写式子。

## 6 不变量

- `PhysicalLayoutTable` 对同一个 `DTensorTable[SingleTensorStorageMesh]` 必须是确定的：`container_format`、`tensor_entries` 的键、`file_paths` 的键、header 字节和区间偏移都一致。
- 一个 `PhysicalLayoutTable` 只有一个 `container_format`；所有文件同格式，混格式拆成多份布局。
- header 的顺序只来自 `DTensorTable[SingleTensorStorageMesh].named_d_tensors` 的列表顺序。
- 共享文件的表不能把 rank 相关信息写进 header。
- `file_paths` 的键必须是某个 `DisjointStorageMeshList` 的 leader `TensorComponentPos`；同一个 list 只有一个路径。
- `tensor_entries` 里的每个 list，其 leader `TensorComponentPos` 必须出现在 `file_paths` 的键里。
- 所有 `DisjointStorageMeshList` 里的 mesh 两两不重叠；不同文件的 mesh 也不重叠；每个 slot 最多被一个 mesh 覆盖。
- 每个空 content slot 占用 0 字节，不贡献物理区间，也不参与 offset 累加；`tensor_entries` 里没有对应的键。
- 读写链要求 `src` 与 `dst` 的 `tensor_ids` 相同，且同一个 `tensor_id` 的 `global_tensor_meta` 相同；placement 和 mesh 可以不同。
- `CacheDTensorsView.tensor_ids` 与它的 `d_tensor_table` 顺序一致；`data_accessor` 非空。
- 一条链里 `src.tensor_ids == dst.tensor_ids`；有 cache 时 `cache.tensor_ids` 也相同，两层缓存时 `first_cache` / `second_cache` 的 `tensor_ids` 也相同。
- 有 cache 时 `num_ranks_per_node * num_nodes == world_size`；读方向 `first_cache ⊆ second_cache ⊆ dst`，写方向 `second_cache ⊆ first_cache ⊆ src`。
- cache 张量的放置符在 `CacheMesh` 中按维度约束：`temporal` / `concurrency` 是 `Shard`，`spatial` 允许 `Replicate` 或 `Shard`；在 `DTensorsRWChain` 上，读方向要求 `spatial` 为 `Replicate`，写方向要求 `spatial` 为 `Shard`；只有 `spatial` 维的 node 展开成 rank。
- `MixedDTensorsRWChain` 各段的 `tensor_ids` 两两不重叠且数据方向一致；`DisjointTensorIdsAndRWChain.tensor_ids` 等于对应子链的 `get_chain_tensor_ids` 结果。
- 段只搬运字节，不改写 payload；缓存里的副本和 compute / storage 里的对应区间逐字节相同。

## 7 用例矩阵

这张表是设计覆盖矩阵；实现时每个格点对应一个集成测试。

| 用例 | 物理布局 | session / 格式 | 关键检查 |
|---|---|---|---|
| 每 rank 一个 `.distcp` | 每 rank 一个 `DisjointStorageMeshList` | `DcpTensorsSession` | 每 rank 路径独立；slot 顺序拼接 |
| 每 rank 一个 safetensors | 每 rank 一个 list | `SafeTensorsSession` | header 先写；每 rank 数据偏移正确 |
| 多 rank 合写一个 safetensors | 多个 entry 指向同一个 list | `SafeTensorsSession` | header 唯一；并发写不重叠 |
| 一个 rank 多个文件 | 一个 rank 的多个 pos 指向不同 list | 两种 session 均可 | `file_paths` 一 list 一路径 |
| 每个 pp stage 一个文件 | `tensor_slots` 按 stage 切 | 两种 session 均可 | 空槽 0 字节；stage 边界不串 |
| 空槽 | `tensor_entries` 没有对应键 | 两种 session 均可 | offset 不被空槽推动 |
| 张量跨多个文件 | 一个张量对应多个 list | 两种 session 均可 | 每个 list 的 row slice 是单张量 mesh |
| replicated 张量只存一份 | 一个非空 list，其余槽为空 | 两种 session 均可 | 不重复写同一份 payload |
| 读链 storage→compute（直连） | 源一份布局表 + compute mesh | 源端 session | `CopyPlan` 配对；meta 一致 |
| 写链 compute→storage（直连 / 缓存汇聚） | 目标一份布局表 + compute mesh | 目标 save session | `CopyPlanChain` 配对；相邻碎分片经 `spatial: Shard` 缓存汇聚成连续写 |
| 单层缓存 load | storage + 一份 cache | 源端 session | 两段按序执行；cache 的 mesh 展开后是 dst 的子集 |
| 两层缓存 load | storage + first_cache + second_cache | 源端 session | 三段按序执行；`first_cache ⊆ second_cache ⊆ dst`（展开后） |
| 单层 / 两层缓存 save | compute + cache + storage | 目标 save session | 两段/三段按 `spatial: Shard` 汇聚并在凑齐后 flush；peak <= mem_overhead |
| 缓存只留一个 step（`num_steps = 1`） | cache mesh 的 temporal 取 `ArrangedMesh["num_steps"]`，size 为 1 | 源端 / 目标 session | 同一块缓存内存只服务一个 step；单 step 占用最大 |
| 缓存留住 N 个 step（`num_steps = N`） | 同一个 temporal 分支，size 为 N | 源端 / 目标 session | N 个 step 复用同一块缓存内存；单 step 占用是 1 时的 1/N |
| 读热点消减（spatial 逐级放大） | 各级 cache 的 spatial 都放 `Replicate`，参与数逐级变大 | 源端 session | 存储只被一级缓存读；每个末级缓存承担的读者是 1 / 末级参与数 |
| 多缓冲（`concurrency_width = n`，2 就是双缓冲） | `StCacheOuterProductMesh` 的 concurrency 维为 n，放置轴 0 的 `Shard` | 源端 / 目标 session | 一个槽在填、别的槽在读，填充被流水掩盖 |
| 放置符不合规 | temporal / spatial / concurrency 不满足 `Shard` / `(Replicate \| Shard)` / `Shard`，或读写链方向与 `spatial` 不匹配 | 无 | `check_constraints` 返回 `ConstraintViolationError` |
| mixed 链 | 按 tensor id 切成互不重叠的同方向子链 | 每段自己的源端 / 目标 session | 段间 tensor id 不重叠；段内顺序同单支 |
| 链 load / save 统计 | 同一份链结构 | 无 | 三个 stats 只看 mesh 和 `GlobalTensorMeta` |
| 对象存储 multipart | 同一份布局表 | safetensors session 的 multipart 实现 | payload 偏移仍连续；part 边界由 session 处理 |
| 混格式 | 拆成多次操作，每次一份布局表 | 各自 session | 单个 view / plan 只有一种 format |

`num_steps = 1` 和 `num_steps = N` 是缓存内存复用的两端：取 1 时同一块缓存内存只服务一个 step，单 step 占用最大；取 N 时 N 个 step 复用这块内存，单 step 占用降到 1/N，代价是每个 step 要等轮到自己。另一个开关是 `concurrency_width`：它把同一个 step 的副本拆成几个缓冲槽，2 就是双缓冲，也可以取更多；多一个槽多占一份，换来填充和读取的重叠。`num_steps` / `concurrency_width` 都是 cache mesh 的一部分，所以不同的取值是几份不同的 cache view，不是同一个 view 上可以随时改的字段。选哪一种可以在搬之前用 `get_stat_mem_overhead_of_rw_chain` 和两个流量函数算出来（5.8），按内存和流量预算定。

## 8 IO、实现者与内部边界

| 层 | 内容 | 说明 |
|---|---|---|
| IO API | `d_tensors_view_rank_load`、`d_tensors_view_rank_save`、`d_tensors_view_rank_chain_load`、`d_tensors_view_rank_chain_save` | 这四个是跨 view 的搬运；plan 是内部 key function；session 由存储视图提供，格式实现者实现；`d_tensors_view_rank_chain_*` 把 `DTensorsRWChain` 拆成若干段，每段是一份 `CopyPlan`；单段和 `d_tensors_view_rank_*` 等价 |
| 统计与查询 API | `get_tensor_ids_of_rw_chain`、`get_stat_mem_overhead_of_rw_chain`、`get_stat_outbound_traffic_of_rw_chain`、`get_stat_inbound_traffic_of_rw_chain` | 纯函数；接收统一 `DTensorsRWChain`；只看链结构和各 view 的 mesh / `GlobalTensorMeta`，不打开 session，不读 payload |
| 数据处理 API | `compute_d_tensors_view_get_tensor_ids`、`compute_d_tensors_view_get_gd_tensor`、`gd_tensor_make`、`gd_tensor_get_d_tensor`、`gd_tensor_reset_d_tensor` | compute 侧 view 与 `torch.DTensor` 的桥；所有数据处理都走这里；storage 侧和 cache 侧不提供数据处理 API |
| 构造 API | `d_tensors_view_map` | 只构造 / 转换 view，不做文件 I/O |
| 类 API | 每个数据 `Object` / `Oneof`（session / access 接口除外）的 `new_*` / `get_*` / `set_*` / `add_*` / `mutable_*` / `clear_*`；build 接口在 `<ClassName>BuilderMixin`，数据类写成 `class Foo(FooBuilderMixin)` | protobuf 风格；不生成 `has_*` / `which_*`，判空用 `get_*() is None`、判分支用 `type(get_*())`；`check_constraints` 在数据类上，递归检查自身和所有嵌套数据 |
| Helper API | `global_tensor_meta_of`、`tensor_order_value_of`、`file_path_of`、`leader_tensor_component_pos_of`、`is_cache_mesh_subset_of_mesh`、`legs_of` | 放已有的 `builder` namespace；不是主要 API，不放顶层；`is_cache_mesh_subset_of_mesh` 与 `legs_of` 供读写链校验与规划使用 |
| 实现者接口 | `StorageSaveSession`、`StorageLoadSession`、`ContainerFormat`、布局构造器、`DTensorDataAccessor` | 格式实现者和布局构造者实现；session 管生命周期、offset，并通过 `init_data_accessor` 提供绑定到该 session 的 data accessor |
| 内部构造 | `d_tensors_view_rank_plan_load`、`d_tensors_view_rank_plan_save`、`d_tensors_view_rank_plan_chain_load`、`d_tensors_view_rank_plan_chain_save`、`CopyPlan`、`CopyPlanChain`、`DTensorRange`、check 函数 | 内部使用；不是 IO API |

附录 B 的 API 章节按“IO API / 统计 API / 数据处理 API / 构造 API / 类 API / Helper API / plan API / check API”分类，是为了完整描述核心；调用方需要 IO API、数据处理 API、构造 API 和类 API，Helper API 按需使用。plan API 是内部 key function，check 是内部一致性校验。

## 9 结论与非目标

计算侧、存储侧和缓存侧共用 `NamedDTensor` / `GenericDTensor` / `NaiveDTensor`，各自只有一个 mesh 概念：`ComputeMesh`、`StorageMesh` 和 `CacheMesh`。缓存是第三种 view：读方向 `spatial` 逐级 `Replicate` 放大副本消减读热点，写方向 `spatial` 使用 `Shard` 先在缓存节点汇聚细碎分片再顺序落盘，`temporal` / `concurrency` 把峰值内存压回可接受范围；带缓存的读写统一由 `DTensorsRWChain` 描述，执行时拆成若干次 range 到 range 的复制，所以 `DTensorRange`、`CopyPlan` 都不为缓存加分支。load 本身就把 storage 侧的切分换成 compute 侧的切分，所以核心不单独提供换切分 API。存储侧多一个 `PhysicalLayoutTable`，因为文件是多个张量共享的线性容器；多一个 session 层，因为文件格式和 header / metadata 需要显式发布。

非目标：

- 不在核心里做 payload 序列化 / 反序列化；read/write 是 raw bytes。
- 不在核心里提供 reduce API；归约、平均、Welford 这类数据处理都在 `torch.DTensor` 上做，核心只提供 5.6 的桥。
- 不在核心里实现对象存储 multipart、分布式 barrier、重试；这些在 session / 实现层。
- 不在核心里决定布局构造策略；对象只保存业务给出的 entry 和属性，拓扑仍由业务层决定。
- 不在核心里决定缓存的换入换出、预取、淘汰策略，也不决定 `num_steps` 对应哪些 step、缓存放在哪种介质上；这些由业务构造 cache view 和链时决定。
- 不支持 compute→compute 的换切分、storage→storage 的复制。这两类是按当前需求裁掉的，不是待补的缺口。

## 10 LAWA 示例

LAWA 对最近 k 个 checkpoint 的权重做逐元素平均。核心不决定“最近 k 个”是哪些；调用方按 step / 目录顺序选出 k 个 `StorageDTensorsView`。两个版本的 load / save、tensor id 遍历、`gdtensor.get_d_tensor` / `gdtensor.reset_d_tensor` 都相同，只有 `torch.DTensor` 上的平均公式不同。

两个代码块都省略 `Result` 错误处理；`must_ok` 是示例里自己写的辅助函数，用来取 `Ok` 的值，不属于这套 API。

### 10.1 朴素版本

先累加，最后除以 k：

```python
# must_ok unwraps Ok; error handling omitted.
import gdtensor
import gdtensor.io as io

rank = ...
ckpts = latest_k_checkpoints(run_dir, k)  # caller selects the latest k
k = len(ckpts)
acc_view = make_compute_view_like(ckpts[0])  # framework allocates
out_storage = make_storage_view_for_output(...)  # business constructs

must_ok(io.rank_load(rank, acc_view, ckpts[0]))
tensor_ids = gdtensor.get_tensor_ids(acc_view)

acc = {}
for tensor_id in tensor_ids:
    gd = must_ok(gdtensor.get_gd_tensor(acc_view, tensor_id))
    d_tensor = gdtensor.get_d_tensor(gd)
    assert d_tensor is not None
    acc[tensor_id] = d_tensor

for ckpt in ckpts[1:]:
    tmp_view = make_compute_view_like(acc_view)  # framework allocates
    must_ok(io.rank_load(rank, tmp_view, ckpt))
    for tensor_id in tensor_ids:
        gd = must_ok(gdtensor.get_gd_tensor(tmp_view, tensor_id))
        d_tensor = gdtensor.get_d_tensor(gd)
        assert d_tensor is not None
        acc[tensor_id] = acc[tensor_id] + d_tensor

for tensor_id in tensor_ids:
    gd = must_ok(gdtensor.get_gd_tensor(acc_view, tensor_id))
    gdtensor.reset_d_tensor(gd, acc[tensor_id] / k)

must_ok(io.rank_save(rank, acc_view, out_storage))
```

- `acc` 从第一个 checkpoint 的 tensor 开始；每个后续 checkpoint 加到 `acc`；最后 `acc / k`。
- 实现简单；k 大时累加和可能损失精度或溢出。

### 10.2 Welford 版本

每个 checkpoint 到达时更新 mean；不做最终除法：

```python
# must_ok unwraps Ok; error handling omitted.
import gdtensor
import gdtensor.io as io

rank = ...
ckpts = latest_k_checkpoints(run_dir, k)  # caller selects the latest k
acc_view = make_compute_view_like(ckpts[0])  # framework allocates
out_storage = make_storage_view_for_output(...)  # business constructs

must_ok(io.rank_load(rank, acc_view, ckpts[0]))
tensor_ids = gdtensor.get_tensor_ids(acc_view)

mean = {}
for tensor_id in tensor_ids:
    gd = must_ok(gdtensor.get_gd_tensor(acc_view, tensor_id))
    d_tensor = gdtensor.get_d_tensor(gd)
    assert d_tensor is not None
    mean[tensor_id] = d_tensor

n = 1
for ckpt in ckpts[1:]:
    tmp_view = make_compute_view_like(acc_view)  # framework allocates
    must_ok(io.rank_load(rank, tmp_view, ckpt))
    n += 1
    for tensor_id in tensor_ids:
        gd = must_ok(gdtensor.get_gd_tensor(tmp_view, tensor_id))
        d_tensor = gdtensor.get_d_tensor(gd)
        assert d_tensor is not None
        delta = d_tensor - mean[tensor_id]
        mean[tensor_id] = mean[tensor_id] + delta / n

for tensor_id in tensor_ids:
    gd = must_ok(gdtensor.get_gd_tensor(acc_view, tensor_id))
    gdtensor.reset_d_tensor(gd, mean[tensor_id])

must_ok(io.rank_save(rank, acc_view, out_storage))
```

- `n` 是已经并入的 checkpoint 数；`mean` 是当前的平均。
- 新 checkpoint 的更新：`n += 1`；`delta = d_tensor - mean`；`mean = mean + delta / n`。
- 没有累加和，也没有最后的 `/ k`；k 大时比朴素版本稳。
- 这里只需要 mean，不需要 Welford 的 M2；M2 只在算方差时用。
- 所有 checkpoint 都包含同一个 tensor 时，`n` 是标量；缺 tensor 时，`n` 要是和 tensor 同形的 `torch.DTensor`，按元素计数。

### 10.3 共用边界

- `latest_k_checkpoints`、`make_compute_view_like`、`make_storage_view_for_output` 都在调用方或业务层；core 只看到 `StorageDTensorsView` / `ComputeDTensorsView`。
- `make_compute_view_like` / `make_storage_view_for_output` 用 protobuf 风格 API 构造：前者 `new_compute_d_tensors_view` + `mutable_d_tensor_table().add_named_d_tensor`；后者 `new_physical_layout_table` + `new_storage_d_tensors_view`；构造后调用 `check_constraints`。
- `d_tensors_view_rank_load` 把每个 checkpoint 读进 compute view；`d_tensors_view_rank_save` 把平均结果写回 storage。
- 所有 checkpoint 必须覆盖同一组 `tensor_ids`，且同名 tensor 的 `GlobalTensorMeta` 相同；否则对齐后再读，或拒绝。
- 缺 tensor 或按元素加权时，用额外的 mask / weight `torch.DTensor`；core 不提供专门的归约 API。
- 需要缓存时，读侧把 `io.rank_load(rank, acc_view, ckpt)` 换成 `io.chain_load(rank, rw_chain)`，写侧把 `io.rank_save(rank, acc_view, out_storage)` 换成 `io.chain_save(rank, rw_chain)`：链由业务按 checkpoint 目录和可用缓存构造，平均逻辑不变。搬之前可以用 `io.chain_stat_mem_overhead` 等三个统计函数判断缓存和流量是否在预算内。
- storage 侧没有 LAWA 专用 API；它只提供 load / save，数据处理统一走 5.6 的桥。

## 附录 A 术语

| 术语 | 含义 |
|---|---|
| slot（槽位） | `num_tensors × world_size` 表中的一个格子；可以为空；空槽占 0 字节 |
| rank | 全局进程编号；在 slot 表里是列号 |
| `world_size` | 每个张量的 slot 数，也是 rank 数 |
| `TensorComponentPos` | `(tensor_id, rank)`，逻辑张量在某个 rank 上的组件位置 |
| placement | `Shard` / `Replicate` / `Partial`；张量维度在参与者之间的分配方式 |
| replicated | 每个参与者都持有完整张量 |
| DCP | PyTorch distributed checkpoint；常按 rank 写 `.distcp`，`.metadata` 是完成标记 |
| safetensors | 单文件张量格式；header 在文件开头 |
| PP stage | 流水线并行的一段；通常对应一部分 rank 和一部分张量 |
| payload | 文件里张量数据的字节；不含 header / metadata |
| 配置数据 | 描述布局、路径、格式、映射的数据；不含张量 payload |
| `contiguous_payload_offset` | 同一个文件内，所有 payload 连续拼接后的偏移；不含 header / 对齐 |
| `file_offset` | 实际文件偏移；等于 data_start + `contiguous_payload_offset` |
| header | 文件内的元数据；safetensors 的 header 在文件开头 |
| metadata | 独立于 payload 文件的元数据；distcp 的 `.metadata` 是完成标记 |
| `ContainerFormat` | 计划级容器格式；一份布局表只有一种 |
| `DisjointStorageMeshList` | 一个物理文件的逻辑分组；组内和跨组的 mesh 都不重叠 |
| leader component pos | `DisjointStorageMeshList` 的规范键；由 `leader_tensor_component_pos_of` 得到 |
| `DTensorRange` | 一次复制的一侧；`RangeFromComputeView` 或 `RangeFromStorageView` |
| `CopyPlan` | 源 range 到目标 range 的列表；唯一的 pair 类型 |
| plan | 纯函数，生成 `CopyPlan`；读写链的 plan 均生成 `CopyPlanChain`；都不打开 session |
| `CopyPlanChain` | 与 `DTensorsRWChain` 同构的计划结构：分支一一对应，每个分支按数据流顺序放该段的 `CopyPlan` |
| session | 文件生命周期 + offset；save 用 commit / abort，load 用 finish |
| view（视图） | 张量句柄；`tensor_ids` + `DTensorTable` + accessor，存储侧多物理布局表和 session |
| mesh（网格） | 张量在参与者维度上的排布；`ComputeMesh` / `StorageMesh` / `CacheMesh` |
| node（节点） | 计算侧的一台机器；cache view 的参与者；`num_nodes` 是节点数 |
| `num_ranks_per_node` | 每个节点上的 rank 数；node `n` 覆盖 rank 区间 `[n * num_ranks_per_node, (n + 1) * num_ranks_per_node)`；`num_ranks_per_node * num_nodes == world_size` |
| cache（缓存） | 计算节点上的伴生存储；由 `CacheDTensorsView` 描述，不改变张量内容 |
| `CacheMesh` | 缓存侧 mesh；`temporal`（`num_steps`）× `spatial`（`num_nodes`）× `concurrency`（`concurrency_width`） |
| `StCacheOuterProductMesh` | 把缓存的 temporal、spatial、concurrency 三个维度相乘；`St` 是 spatiotemporal |
| concurrency | 缓存的并发缓冲维度；`concurrency_width` 是缓冲槽数，取 2 就是双缓冲，也可以取更多 |
| 读热点（hot spot） | 大量 rank 同时读同一份数据；缓存用 spatial 逐级 `Replicate` 放大副本把它消减掉 |
| temporal | 缓存的 step 维度；`num_steps` 是 step 数 |
| spatial | 缓存的空间维度；`num_nodes` 是节点数 |
| 读写链（RW chain） | 统一表达带缓存 load / save 的双向数据流结构；`DTensorsRWChain` |
| concrete 链 | 不含 `Mixed` 的链：`Direct` / `OneLayerCached` / `TwoLayerCached` |
| 段（segment） | 读写链拆出来的一段，对应一份 `CopyPlan`：里面每对是源 view 的一段 range 到目标 view 的一段 range |
| 统计 API | 三个 `get_stat_*_of_rw_chain`；纯函数，不打开 session |
| `LiteralMesh` | 显式的 rank 列表，`$max` 是 rank 的排他上界；表示无法用 `ArrangedMesh` 写出的 device mesh |
| range（区间） | 一段连续字节的定位；`RangeFromComputeView` / `RangeFromStorageView` |
| accessor | 裸字节读写接口；`DTensorDataAccessor[RangeT]` |
| `torch.DTensor` | PyTorch 的 DTensor；计算侧实际数据处理的句柄 |
| `gd_tensor` | `GenericDTensor[ComputeMesh]` 的简称；compute/torch 桥的参数名 |
| compute/torch 桥 | 5.6 的五个 API；在 compute view、`GenericDTensor[ComputeMesh]` 和 `torch.DTensor` 之间取和写 |
| `check_*` | 内部一致性校验；disjoint / covering，不是 IO API |
| `check_constraints` | 每个数据 `Object` / `Oneof` 的方法；session / access 接口除外；递归检查自身和所有嵌套数据 |
| `DTensorsView` | `ComputeDTensorsView` / `StorageDTensorsView` / `CacheDTensorsView`；内部 plan 的 view 参数类型，读链每一段的两端各是其中一支 |
| `GlobalTensorMeta` | 全局张量的各维长度、dtype、itemsize |
| multipart | 对象存储的分段上传；payload 偏移仍连续，段边界由 session 处理 |
| barrier | 分布式同步点；多 rank 在提交或读取前对齐 |
| 布局构造器 | 业务实现，生成 `PhysicalLayoutTable`；核心不提供策略类型 |
| `Error` | 顶层错误类型；`Result` 的 `Err` 分支；每个分支是带 `msg` 的 Object（B.3.4） |
| `Result` | 成功是 `Ok T`，失败是 `Err Error`（B.3.4） |

## 附录 B 类型定义

本附录的类型定义用 viba DSL 写。viba（vibe adt）是给 vibe coding 用的意图描述 DSL，定义见 [lixinqi/viba](https://github.com/lixinqi/viba)。本附录是类型定义清单；设计说明见正文。这里用平铺的 viba 全名，没有 namespace；Python 侧的短名和 namespace 见 C.3。全文的 viba 代码块统一使用 `viba` 围栏并遵循 `viba.lint` 版式与单定义规范。

### B.1 API

#### B.1.1 IO APIs

##### `d_tensors_view_rank_load`

```viba
# Thin executor of d_tensors_view_rank_plan_load; range selection lives in
# the plan API only.
d_tensors_view_rank_load =
     Result[void]
  <- $rank int
  <- $compute_view ComputeDTensorsView
  <- $storage_view StorageDTensorsView
```

##### `d_tensors_view_rank_save`

```viba
# Thin executor of d_tensors_view_rank_plan_save.
d_tensors_view_rank_save =
     Result[void]
  <- $rank int
  <- $compute_view ComputeDTensorsView
  <- $storage_view StorageDTensorsView
```

##### `d_tensors_view_rank_chain_load`

```viba
# Thin executor of a read-direction RW chain. Legs and their ranges come from
# d_tensors_view_rank_plan_chain_load; this only executes them in order.
d_tensors_view_rank_chain_load =
     Result[void]
  <- $rank int
  <- $rw_chain DTensorsRWChain
```

##### `d_tensors_view_rank_chain_save`

```viba
# Thin executor of a write-direction RW chain. Legs and their ranges come from
# d_tensors_view_rank_plan_chain_save; this only executes them in order.
d_tensors_view_rank_chain_save =
     Result[void]
  <- $rank int
  <- $rw_chain DTensorsRWChain
```

#### B.1.2 Plan APIs

##### `d_tensors_view_rank_plan_load`

```viba
# Internal key function behind d_tensors_view_rank_load. Keep load a thin
# executor of this plan; range selection must live here only.
d_tensors_view_rank_plan_load =
     Result[CopyPlan]
  <- $rank int
  <- $compute_view ComputeDTensorsView
  <- $storage_view StorageDTensorsView
```

##### `d_tensors_view_rank_plan_save`

```viba
# Internal key function behind d_tensors_view_rank_save.
d_tensors_view_rank_plan_save =
     Result[CopyPlan]
  <- $rank int
  <- $compute_view ComputeDTensorsView
  <- $storage_view StorageDTensorsView
```

##### `d_tensors_view_rank_plan_chain_load`

```viba
# Internal key function behind d_tensors_view_rank_chain_load. It returns one
# CopyPlanChain, isomorphic to the read-direction RW chain: one CopyPlan per
# leg, in data-flow order. The tensor/meta precondition is checked per leg, and
# the plan-less executor stays thin. The Assert resolves each tensor id to its
# row with tensor_order_value_of; tensor_id is never used as a subscript of
# named_d_tensors (see 5.2).
d_tensors_view_rank_plan_chain_load =
     Result[CopyPlanChain]
  <- $rank int
  <- $rw_chain DTensorsRWChain
  <- Assert[{
    src is StorageDTensorsView and dst is ComputeDTensorsView and
    forall leg in legs_of(rw_chain):
      leg.src.tensor_ids == leg.dst.tensor_ids
      and forall tensor_id in leg.src.tensor_ids:
        global_tensor_meta_of(leg.src.d_tensor_table.named_d_tensors[tensor_order_value_of(leg.src.d_tensor_table, tensor_id)])
        == global_tensor_meta_of(leg.dst.d_tensor_table.named_d_tensors[tensor_order_value_of(leg.dst.d_tensor_table, tensor_id)])
  }]
```

##### `d_tensors_view_rank_plan_chain_save`

```viba
# Internal key function behind d_tensors_view_rank_chain_save. Returns one
# CopyPlanChain isomorphic to the write-direction RW chain.
d_tensors_view_rank_plan_chain_save =
     Result[CopyPlanChain]
  <- $rank int
  <- $rw_chain DTensorsRWChain
  <- Assert[{
    src is ComputeDTensorsView and dst is StorageDTensorsView and
    forall leg in legs_of(rw_chain):
      leg.src.tensor_ids == leg.dst.tensor_ids
      and forall tensor_id in leg.src.tensor_ids:
        global_tensor_meta_of(leg.src.d_tensor_table.named_d_tensors[tensor_order_value_of(leg.src.d_tensor_table, tensor_id)])
        == global_tensor_meta_of(leg.dst.d_tensor_table.named_d_tensors[tensor_order_value_of(leg.dst.d_tensor_table, tensor_id)])
  }]
```

#### B.1.3 Check APIs

##### `check_compute_d_tensors_access_disjoint`

```viba
check_compute_d_tensors_access_disjoint =
     Result[void]
  <- $d_tensor_table DTensorTable[ComputeMesh]
  <- $copy_plan CopyPlan
```

##### `check_compute_d_tensors_access_full_covering`

```viba
check_compute_d_tensors_access_full_covering =
     Result[void]
  <- $d_tensor_table DTensorTable[ComputeMesh]
  <- $copy_plan CopyPlan
```

##### `check_cache_d_tensors_access_disjoint`

```viba
check_cache_d_tensors_access_disjoint =
     Result[void]
  <- $d_tensor_table DTensorTable[CacheMesh]
  <- $copy_plan CopyPlan
```

##### `check_cache_d_tensors_access_full_covering`

```viba
check_cache_d_tensors_access_full_covering =
     Result[void]
  <- $d_tensor_table DTensorTable[CacheMesh]
  <- $copy_plan CopyPlan
```

##### `check_storage_d_tensors_access_disjoint`

```viba
check_storage_d_tensors_access_disjoint =
     Result[void]
  <- $d_tensor_table DTensorTable[SingleTensorStorageMesh]
  <- $physical_layout PhysicalLayoutTable
  <- $file_paths list[str]
  <- $copy_plan CopyPlan
```

##### `check_storage_d_tensors_access_full_covering`

```viba
check_storage_d_tensors_access_full_covering =
     Result[void]
  <- $d_tensor_table DTensorTable[SingleTensorStorageMesh]
  <- $physical_layout PhysicalLayoutTable
  <- $file_paths list[str]
  <- $copy_plan CopyPlan
```

##### `check_storage_mesh_disjoint`

```viba
check_storage_mesh_disjoint[TensorSlotsMeshT] =
     Result[void]
  <- $lhs StorageMesh[TensorSlotsMeshT]
  <- $rhs StorageMesh[TensorSlotsMeshT]
```

#### B.1.4 Compute/Torch APIs

##### `compute_d_tensors_view_get_tensor_ids`

```viba
# List tensor ids in a compute view.
compute_d_tensors_view_get_tensor_ids =
     list[str]
  <- $view ComputeDTensorsView
```

##### `compute_d_tensors_view_get_gd_tensor`

```viba
# Get a generic compute tensor by tensor id. Use tensor_order_value_of for
# the lookup; a missing id returns Err with NotFoundError.
compute_d_tensors_view_get_gd_tensor =
     Result[GenericDTensor[ComputeMesh]]
  <- $view ComputeDTensorsView
  <- $tensor_id str
```

##### `gd_tensor_make`

```viba
# Wrap a torch.DTensor as a generic compute tensor. The torch tensor provides
# the data, global meta, placements and device mesh; gd_tensor_make converts
# the device mesh to the viba ComputeMesh.
gd_tensor_make =
     GenericDTensor[ComputeMesh]
  <- $d_tensor torch.DTensor
```

##### `gd_tensor_get_d_tensor`

```viba
# Get the underlying torch.DTensor; nil means metadata-only.
gd_tensor_get_d_tensor =
     (torch.DTensor | nil)
  <- $gd_tensor GenericDTensor[ComputeMesh]
```

##### `gd_tensor_reset_d_tensor`

```viba
# Replace the underlying torch.DTensor.
gd_tensor_reset_d_tensor =
     void
  <- $gd_tensor GenericDTensor[ComputeMesh]
  <- $d_tensor torch.DTensor
```

#### B.1.5 View constructors

##### `d_tensors_view_map`

```viba
# Convert each tensor of origin. Callbacks are optional: a nil tensor
# callback means identity; a nil session callback means skip that session.
# No range selection.
d_tensors_view_map =
     Result[DTensorsView]
  <- $origin DTensorsView
  <- $map_core DTensorsViewMapCore
```

#### B.1.6 Stats APIs

各函数在读写链结构上纯静态计算，不打开 session 也不读取 payload；返回值按节点 id 索引，单位为字节。

##### `get_stat_mem_overhead_of_rw_chain`

```viba
# Memory the cache occupies on each node.
get_stat_mem_overhead_of_rw_chain =
     list[int]
  <- $rw_chain DTensorsRWChain
```

##### `get_stat_outbound_traffic_of_rw_chain`

```viba
# Bytes each node sends out.
get_stat_outbound_traffic_of_rw_chain =
     list[int]
  <- $rw_chain DTensorsRWChain
```

##### `get_stat_inbound_traffic_of_rw_chain`

```viba
# Bytes each node takes in.
get_stat_inbound_traffic_of_rw_chain =
     list[int]
  <- $rw_chain DTensorsRWChain
```

### B.2 Entities

实体分四侧：Compute、Storage、Cache、Common；`check_constraints` 的额外检查单列在 B.2.5。Python 实现的 `<ClassName>BuilderMixin` 见附录 C。`Slot` 是逻辑概念，`Layout` 是物理概念；类型名用 `Logical` / `Physical` 区分。

#### B.2.1 Compute side

##### B.2.1.1 `ComputeDTensorsView`

```viba
ComputeDTensorsView =
    Object
  * $tensor_ids (
        list[str]
      * { Tensors described by this view; the rank is an API argument, not a field. }
    )
  * $d_tensor_table (
        DTensorTable[ComputeMesh]
      * { Compute DTensor table ordered consistently with tensor_ids. }
    )
  * $data_accessor (
        DTensorDataAccessor[RangeFromComputeView]
      * { Host-side access to the local tensor ranges. }
    )
```

##### B.2.1.2 `ComputeMesh`

```viba
# Branches are named types; no tags, the payload type is the branch.
ComputeMesh =
    Oneof
  | LiteralMesh["world_size"]
  | ArrangedMesh["world_size"]
  | ReshapedMesh[ComputeMesh]
  | PermutedMesh[ComputeMesh]
  | SlicedMesh[ComputeMesh]
```

#### B.2.2 Storage side

`slot`（槽位）是逻辑概念，不是文件或容器。把存储计划看成一个 `num_tensors × world_size` 的表：每一行是一个张量，每一列是一个全局 rank，每个格子是一个 slot；总 slot 数 = `num_tensors * world_size`。格子可以为空：该 rank 不持有这个张量、这个张量不在该 pp stage，或者它只存一份而其余列留空。同一行的非空格子按列顺序拼接，构成这个张量的存储区域。`StorageMesh` 的 `SlotOuterProductMesh` 分支里，`tensor_slots` 选择行（张量），`content_slots` 选择列（slot）；`content_slots` 是 `ContentSlotsMesh`，即 `LiteralMesh["world_size"] | ArrangedMesh["world_size"]`：`LiteralMesh` 直接给出 rank 列表，`ArrangedMesh` 的 `"world_size"` 是列数。

##### B.2.2.1 `StorageDTensorsView`

```viba
StorageDTensorsView =
    Object
  * $tensor_ids (
        list[str]
      * { Tensors this view covers; a subset of the storage DTensors. }
    )
  * $d_tensor_table (
        DTensorTable[SingleTensorStorageMesh]
      * { Storage DTensor table; named_d_tensors order is the header order. }
    )
  * $data_accessor (
        DTensorDataAccessor[RangeFromStorageView]
      * { Read/write raw storage ranges; the range carries a file path and offset. }
    )
  * $physical_layout (
        PhysicalLayoutTable
      * { Physical file table; lookup instead of computation. }
    )
  * $begin_save (
        (
             Result[StorageSaveSession]
          <- $self StorageDTensorsView
          <- $rank int
        )
      * { Create a save session; safetensors writes header first and returns session. }
    )
  * $begin_load (
        (
             Result[StorageLoadSession]
          <- $self StorageDTensorsView
          <- $rank int
        )
      * { Create a load session and read the container header before get_file_offset calls. }
    )
```

##### B.2.2.2 `PhysicalLayoutTable`

```viba
# Pure configuration data; produced by a business layout constructor from
# DTensorTable and the chosen topology / format.
PhysicalLayoutTable =
    Object
  * $container_format (
        ContainerFormat
      * { One format for every file in this plan. }
    )
  * $tensor_entries (
        dict[TensorComponentPos, DisjointStorageMeshList]
      * { TensorComponentPos -> the list that contains it; missing means no physical file on this rank. }
    )
  * $file_paths (
        dict[TensorComponentPos, $file_path str]
      * { Leader TensorComponentPos -> physical file path. }
    )
  * Hint[{
    1. 系统可能会同时支持多种保存拓扑，比如每个 rank 存一个或多个文件，或者多个 rank 存一个文件，或者每个 pp stage 存一个文件，等等。
    2. 同一个 PhysicalLayoutTable 只对应一种 ContainerFormat；要混格式就拆成多份布局。
    3. 一个张量如果存在于多个 slots，那么这些 slots 逻辑上顺序拼接起来凑成整个张量的存储区域。
    4. 查表方式：先在 tensor_entries 里按 TensorComponentPos 找 disjoint_storage_meshes；没有这个键表示没有物理文件；再用 leader_tensor_component_pos_of(disjoint_storage_meshes) 在 file_paths 里找路径。
    5. file_paths 的键必须是某个 DisjointStorageMeshList 的 leader TensorComponentPos；同一个 list 只对应一个路径。
  }]
```

###### `ContainerFormat`

```viba
ContainerFormat = "safetensors" | "distcp"
```

###### `TensorComponentPos`

```viba
TensorComponentPos =
    Object
  * $tensor_id (
        str
      * { Global tensor identifier. }
    )
  * $rank (
        int
      * { Global rank index covering the component slot. }
    )
```

###### `leader_tensor_component_pos_of`

```viba
# Canonical TensorComponentPos of a DisjointStorageMeshList; used as its
# dict key.
leader_tensor_component_pos_of =
     TensorComponentPos
  <- $disjoint_storage_meshes DisjointStorageMeshList
```

###### `file_path_of`

```viba
# Path for a component pos: tensor_entries[pos] -> list -> leader ->
# file_paths. A missing key at either step returns Err with NotFoundError.
file_path_of =
     Result[str]
  <- $layout PhysicalLayoutTable
  <- $pos TensorComponentPos
```

##### B.2.2.3 `DisjointStorageMeshList`

```viba
DisjointStorageMeshList =
    Object
  * $disjoint_storage_mesh_list (
        list[MultiTensorsStorageMesh]
      * { One logical list per physical file; meshes inside and across files are disjoint. }
    )
  * Hint[{
    不同物理文件的 mesh 也互不重叠：每个 slot 最多被一个 mesh 覆盖。
    每个空 content slot 占用 0 字节；空槽不参与 offset 累加，所以 list 里有很多空槽也不会让后续张量的 offset 错位。
    single_tensor_mesh 应该等于
    multi_tensors_mesh[
      tensor_order_value_of(tensor_id) : tensor_order_value_of(tensor_id) + 1,
      :
    ]。
    也就是 SingleTensorStorageMesh 必须是 MultiTensorsStorageMesh 中该张量所在的那一行，不能覆盖多个张量槽。
  }]
```

##### B.2.2.4 `StorageMesh`

###### `SingleTensorStorageMesh`

```viba
SingleTensorStorageMesh = StorageMesh[ArrangedMesh["constant_one"]]
```

###### `MultiTensorsStorageMesh`

```viba
MultiTensorsStorageMesh = StorageMesh[ArrangedMesh["tensor_count"]]
```

###### `StorageMesh` body

```viba
# Branches are named types; no tags.
StorageMesh[TensorSlotsMeshT] =
    Oneof
  | SlotOuterProductMesh[TensorSlotsMeshT]
  | ReshapedMesh[StorageMesh[TensorSlotsMeshT]]
  | SlicedMesh[StorageMesh[TensorSlotsMeshT]]
```

###### `SlotOuterProductMesh`

```viba
# The two dimensions: which tensors, which content slots.
SlotOuterProductMesh[TensorSlotsMeshT] =
    Object
  * $tensor_slots (
        TensorSlotsMeshT
      * { Selected tensor rows covered by this storage mesh. }
    )
  * $content_slots (
        ContentSlotsMesh
      * { Selected content slot columns covered by this storage mesh. }
    )
```

###### `ContentSlotsMesh`

```viba
# Content slot selector: an explicit rank list, or an arranged world_size.
ContentSlotsMesh =
    Oneof
  | LiteralMesh["world_size"]
  | ArrangedMesh["world_size"]
```

##### B.2.2.5 `StorageSaveSession`

```viba
# Interface type; a concrete session (e.g. SafeTensorsSession or
# DcpTensorsSession) implements get_file_offset / init_data_accessor /
# commit / abort.
# Save session has two terminal operations: commit (publish, then close) and
# abort (discard, then close). There is no separate finish.
# Offset, lifecycle, and a session-bound data accessor. Payload I/O goes
# through DTensorDataAccessor, not through the offset methods directly.
StorageSaveSession =
    Object
  * $get_file_offset (
        (
             Result[int]
          <- $self StorageSaveSession
          <- $pos TensorComponentPos
        )
      * { Actual file offset (header/alignment included); valid between begin_save and commit/abort. }
    )
  * $init_data_accessor (
        (
             Result[DTensorDataAccessor[RangeFromStorageView]]
          <- $self StorageSaveSession
        )
      * { Bind buffer I/O to this save session and return its data accessor. }
    )
  * $commit (
        (
             Result[void]
          <- $self StorageSaveSession
        )
      * { Publish the save (e.g. fsync + rename), then close; keep payloads if outcome is unknown. }
    )
  * $abort (
        (
             Result[void]
          <- $self StorageSaveSession
        )
      * { Discard temporary files and close; never delete payloads after an uncertain commit. }
    )
```

##### B.2.2.6 `StorageLoadSession`

```viba
# Interface type; a concrete session (e.g. SafeTensorsSession or
# DcpTensorsSession) implements get_file_offset / init_data_accessor / finish.
# Load session has one terminal operation: finish (close). It never
# publishes, so there is no commit or abort.
# Offset, lifecycle, and a session-bound data accessor. Payload I/O goes
# through DTensorDataAccessor, not through the offset methods directly.
StorageLoadSession =
    Object
  * $get_file_offset (
        (
             Result[int]
          <- $self StorageLoadSession
          <- $pos TensorComponentPos
        )
      * { Actual file offset (header/alignment included); valid between begin_load and finish. }
    )
  * $init_data_accessor (
        (
             Result[DTensorDataAccessor[RangeFromStorageView]]
          <- $self StorageLoadSession
        )
      * { Bind buffer I/O to this load session and return its data accessor. }
    )
  * $finish (
        (
             Result[void]
          <- $self StorageLoadSession
        )
      * { Close the read-only load session handles. }
    )
```

#### B.2.3 Cache side

##### B.2.3.1 `CacheDTensorsView`

```viba
# Node-local companion storage on the compute side. Its range type is
# RangeFromCacheView, which is RangeFromComputeView (B.3.1), so a cache view
# needs no session and no PhysicalLayoutTable.
CacheDTensorsView =
    Object
  * $tensor_ids (
        list[str]
      * { Tensors cached by this view. }
    )
  * $d_tensor_table (
        DTensorTable[CacheMesh]
      * { Cache mesh tensor table; its participants are nodes, not ranks. }
    )
  * $data_accessor (
        DTensorDataAccessor[RangeFromCacheView]
      * { Read/write raw cache ranges; same range type as the compute side. }
    )
```

缓存是计算节点上的伴生存储：数据在节点本地，读写不经文件，所以这个 view 的字段和 `ComputeDTensorsView` 相同，只把 mesh 换成 `CacheMesh`。

##### B.2.3.2 `CacheMesh`

```viba
# Branches are named types; no tags.
CacheMesh =
    Oneof
  | StCacheOuterProductMesh
  | PermutedMesh[CacheMesh]
```

###### `StCacheOuterProductMesh`

```viba
# St -- spatiotemporal. Three dimensions: temporal (steps), spatial (nodes)
# and concurrency (buffer slots).
StCacheOuterProductMesh =
    Object
  * $temporal (
        TemporalCacheMesh
      * { Temporal reuse dimension across num_steps steps (placement fixed to Shard). }
    )
  * $spatial (
        SpatioCacheMesh
      * { Spatial node dimension across num_nodes nodes (Replicate on read, Shard on write). }
    )
  * $concurrency (
        ConcurrencyMesh
      * { Concurrent buffer slot dimension across concurrency_width slots (placement fixed to Shard). }
    )
  * Hint[{
    1. 放置符按维度约束：temporal / concurrency 固定为 Shard，spatial 允许 Replicate 或 Shard（读方向为 Replicate，写方向为 Shard）；mesh 做了 PermutedMesh 时 placements 按同样的顺序排。check_constraints 拒绝别的组合。
    2. spatial 维在读方向逐级 Replicate 放大副本消减读热点（一传十、十传百）；在写方向使用 Shard 让各节点持互不重叠分片并逐级合并为更大连续段。
    3. spatial 各级参与数相乘会让缓存内存暴涨；temporal 让同一块缓存内存在 num_steps 个 step 之间复用，
       concurrency 用 concurrency_width 提供并发缓冲槽，2 对应双缓冲，也可以更多，执行层可以让填充和读取重叠。
    4. temporal / concurrency 切的是不同的数据片（不同 step、不同缓冲槽），不是同一份数据的副本，所以它们是 Shard。
    5. 参与者坐标是 (step, node, buffer)；只有 spatial 维的 node 展开成 rank，temporal / concurrency 是同一批 node 上的缓冲槽。
    6. ReshapedMesh 保持元素总数不变，所以 temporal 与 spatial 之间的 reshape 只在两侧总数相等时成立；reshape 只改展开顺序，不改放置符。
    7. CacheMesh 没有 SlicedMesh 分支。
  }]
```

###### `ConcurrencyMesh`

```viba
ConcurrencyMesh = ArrangedMesh["concurrency_width"]
```

###### `TemporalCacheMesh`

```viba
# Temporal dimension; num_steps is the number of steps.
TemporalCacheMesh =
    Oneof
  | ArrangedMesh["num_steps"]
  | ReshapedMesh[SpatioCacheMesh]
```

###### `SpatioCacheMesh`

```viba
# Spatial dimension; num_nodes is the number of nodes.
SpatioCacheMesh =
    Oneof
  | LiteralMesh["num_nodes"]
  | ReshapedMesh[TemporalCacheMesh]
```

##### B.2.3.3 `is_cache_mesh_subset_of_mesh`

```viba
# True when every rank covered by lhs is also covered by rhs. A cache mesh
# is expanded to ranks through its spatial dimension only: node n maps to
# [n * num_ranks_per_node, (n + 1) * num_ranks_per_node), and the temporal /
# concurrency dimensions are buffer slots on those same nodes, so they add no
# ranks. rhs is expanded the same way when it is a cache mesh, and is already
# in ranks when it is a compute mesh. Used by the read chain Assert in B.3.2.
is_cache_mesh_subset_of_mesh =
     bool
  <- $lhs CacheMesh
  <- $rhs (CacheMesh | ComputeMesh)
  <- $num_ranks_per_node int
```

#### B.2.4 Common

##### B.2.4.1 `DTensorsView`

```viba
DTensorsView =
    Oneof
  | ComputeDTensorsView
  | StorageDTensorsView
  | CacheDTensorsView
```

##### B.2.4.2 `DTensorDataAccessor`

```viba
DTensorDataAccessor[RangeT] =
    Object
  * $read_to_buffer (
        (
             Result[void]
          <- $global_range RangeT
          <- $read_to_buffer (
                  Result[void]
               <- $buffer_ptr int
               <- $buffer_len int
             )
        )
      * { Read one range owned by this view and hand each byte chunk to the callback. }
    )
  * $write_from_buffer (
        (
             Result[void]
          <- $global_range RangeT
          <- $write_from_buffer (
                  Result[void]
               <- $buffer_ptr int
               <- $buffer_len int
             )
        )
      * { Pull each byte chunk from the callback and write it into one range owned by this view. }
    )
```

##### B.2.4.3 `NamedDTensor`

```viba
NamedDTensor[MeshT] =
    Object
  * $global_tensor_id (
        str
      * { Stable tensor name inside its weight namespace. }
    )
  * $d_tensor (
        GenericDTensor[MeshT]
      * { Logical tensor view; its metadata is the global tensor metadata. }
    )
  * $alias (
        list[str]
      * { Other module paths that name the same payload. }
    )
```

##### B.2.4.4 `GenericDTensor`

```viba
GenericDTensor[MeshT] =
    Oneof
  | NaiveDTensor[MeshT]
  | ReshapedDTensor[MeshT]
```

###### `ReshapedDTensor`

```viba
# Element-order-preserving logical reshape; shape must keep the local
# ownership interval unchanged, otherwise redistribute first.
ReshapedDTensor[MeshT] =
    Object
  * $underlying_d_tensor (
        NaiveDTensor[MeshT]
      * { Underlying unreshaped tensor being logically reshaped. }
    )
  * $shape (
        list[int]
      * { Shape of the logical reshape; one entry per tensor dimension. }
    )
```

##### B.2.4.5 `NaiveDTensor`

```viba
NaiveDTensor[MeshT] =
    Object
  * $global_tensor_meta (
        GlobalTensorMeta
      * { Full logical tensor metadata. }
    )
  * $placements (
        list[Placement]
      * { One placement per mesh dimension, in mesh order. }
    )
  * $mesh (
        MeshT
      * { Mesh this tensor is placed on. }
    )
  * $torch_d_tensor (
        (torch.DTensor | nil)
      * { Underlying compute-side torch tensor; nil means metadata-only. }
    )
```

##### B.2.4.6 `GlobalTensorMeta`

```viba
GlobalTensorMeta =
    Object
  * $global_shape (
        list[int]
      * { Shape of the full logical tensor; one entry per dimension. }
    )
  * $dtype (
        str
      * { Framework dtype name. }
    )
  * $itemsize (
        int
      * { Bytes per element. }
    )
```

##### B.2.4.7 `Placement`

```viba
# Branches are named types; no tags.
Placement =
    Oneof
  | Shard
  | Replicate
  | Partial
```

###### `Shard`

```viba
# One contiguous logical slice along this tensor dimension.
Shard =
    Object
  * $axis (
        int
      * { Non-negative tensor axis index along which the tensor is sharded. }
    )
```

###### `Replicate`

```viba
# Every participant holds the full tensor. Must be a field-less type: the
# type itself carries the meaning.
Replicate = Object
```

###### `Partial`

```viba
# Every participant holds a reduction contribution; not persistable yet.
# Must be a field-less type, same as Replicate.
Partial = Object
```

##### B.2.4.8 `DTensorTable`

```viba
DTensorTable[MeshT] =
    Object
  * $named_d_tensors (
        list[NamedDTensor[MeshT]]
      * { Ordered DTensors; list order is the header order. }
    )
```

##### B.2.4.9 `global_tensor_meta_of`

```viba
global_tensor_meta_of[MeshT] =
     GlobalTensorMeta
  <- $named_d_tensor NamedDTensor[MeshT]
```

##### B.2.4.10 `tensor_order_value_of`

```viba
# List index of tensor_id in named_d_tensors; a missing id returns Err
# with NotFoundError.
tensor_order_value_of[MeshT] =
     Result[int]
  <- $d_tensor_table DTensorTable[MeshT]
  <- $tensor_id str
```

##### B.2.4.11 `DTensorsViewMapCore`

```viba
# Five optional callbacks for d_tensors_view_map. Compute, storage and cache
# tensor callbacks are separate because their tensor types differ; storage
# also converts sessions. A nil tensor callback means identity; a nil session
# callback means skip that session.
DTensorsViewMapCore =
    Object
  * $convert_each_compute_view (
        (
            Oneof
          | (
                 Result[GenericDTensor[ComputeMesh]]
              <- $tensor GenericDTensor[ComputeMesh]
            )
          | nil
        )
      * { Optional transform for each compute GenericDTensor; nil keeps identity. }
    )
  * $convert_each_storage_view (
        (
            Oneof
          | (
                 Result[GenericDTensor[StorageMesh]]
              <- $tensor GenericDTensor[StorageMesh]
            )
          | nil
        )
      * { Optional transform for each storage GenericDTensor; nil keeps identity. }
    )
  * $convert_each_cache_view (
        (
            Oneof
          | (
                 Result[GenericDTensor[CacheMesh]]
              <- $tensor GenericDTensor[CacheMesh]
            )
          | nil
        )
      * { Optional transform for each cache GenericDTensor; nil keeps identity. }
    )
  * $convert_each_storage_load_session (
        (
            Oneof
          | (
                 Result[StorageLoadSession]
              <- $session StorageLoadSession
            )
          | nil
        )
      * { Optional transform for each StorageLoadSession; nil skips session conversion. }
    )
  * $convert_each_storage_save_session (
        (
            Oneof
          | (
                 Result[StorageSaveSession]
              <- $session StorageSaveSession
            )
          | nil
        )
      * { Optional transform for each StorageSaveSession; nil skips session conversion. }
    )
```

### B.2.5 `check_constraints`

每个数据 `Object` / `Oneof`（session / access 接口除外）都有：

```viba
check_constraints = Result[void] <- $self T
```

`check_constraints` 先递归检查所有嵌套字段，再检查本类型自己的约束。递归规则见 5.7：`Object` 逐字段、`Oneof` 当前分支、`list` 逐元素、`dict` 逐 value、`nil` 跳过。表里的“额外检查”是在递归之外加的；未列出的类型只做递归，不做额外检查。类型定义里的 `Assert` 由 `check_constraints` 执行；`Hint` 只是说明，不执行。

| 类型 | 额外检查（递归之外） |
|---|---|
| `ComputeDTensorsView` | `tensor_ids` 与 `d_tensor_table` 的顺序一致；`data_accessor` 非空 |
| `StorageDTensorsView` | `tensor_ids` 与 `d_tensor_table` 的顺序一致；`physical_layout` 的 entry 的 `tensor_id` 在 table 里；`data_accessor` / `begin_save` / `begin_load` 非空 |
| `CacheDTensorsView` | `tensor_ids` 与 `d_tensor_table` 的顺序一致；`data_accessor` 非空；每个 cache 张量的 `placements` 与 mesh 三个维度对应：`temporal` / `concurrency` 是 `Shard`，`spatial` 是 `Replicate` 或 `Shard` |
| `DTensorTable` | `global_tensor_id` 不重复 |
| `NamedDTensor` | `global_tensor_id` 非空 |
| `NaiveDTensor` | `global_tensor_meta` / `placements` / `mesh` 一致；`torch_d_tensor` 可空 |
| `GlobalTensorMeta` | `global_shape` 非空；`itemsize > 0`；`dtype` 非空 |
| `PhysicalLayoutTable` | `container_format` 唯一；`file_paths` 的键是 leader pos；entry 的 leader 在 `file_paths` 里；mesh 两两不重叠 |
| `DisjointStorageMeshList` | list 非空；mesh 两两不重叠 |
| `StorageMesh` | `content_slots` 的 `ArrangedMesh` 的 size > 0 |
| `ComputeMesh` | `ArrangedMesh` 的 size > 0 |
| `CacheMesh` | `temporal` / `spatial` / `concurrency` 里的 `ArrangedMesh` 的 size > 0 |
| `LiteralMesh` | `ranks` 非空；每个 rank 满足 `0 <= rank < max`；`ranks` 无重复 |
| `Shard` | `axis >= 0` |
| `Err` | `value` 非空 |
| `Ok` / `Replicate` / `Partial` | 只做递归检查；`Ok` 的 `value` 可以是 `None`（`Result[void]`），`Replicate` / `Partial` 没有字段 |
| `DirectDTensorsRWChain` | `src` 与 `dst` 分属 `StorageDTensorsView` 与 `ComputeDTensorsView`；`src.tensor_ids == dst.tensor_ids` |
| `OneLayerCachedDTensorsRWChain` | `src` 与 `dst` 分属存储与计算侧；`src` / `cache` / `dst` 的 `tensor_ids` 相同；读方向 `cache.spatial` 为 `Replicate`，写方向为 `Shard`；`num_ranks_per_node * num_nodes == world_size`；`is_cache_mesh_subset_of_mesh(cache.mesh, compute.mesh, num_ranks_per_node)` |
| `TwoLayerCachedDTensorsRWChain` | `src` 与 `dst` 分属存储与计算侧；各视图 `tensor_ids` 相同；读方向 `spatial` 为 `Replicate` 且 `first_cache ⊆ second_cache ⊆ dst`，写方向 `spatial` 为 `Shard` 且 `second_cache ⊆ first_cache ⊆ src` |
| `MixedDTensorsRWChain` | 各段的 `tensor_ids` 两两不重叠；各段数据方向一致 |
| `DisjointTensorIdsAndRWChain` | `rw_chain` 为 `ConcreteDTensorsRWChain`；`tensor_ids` 等于 `get_tensor_ids_of_rw_chain(rw_chain)` |
| `ConstraintViolationError` / `CheckFailedError` / `NotFoundError` / `InvalidArgumentError` / `GDTensorIOError` | `msg` 非空 |

`OneLayerCachedDTensorsRWChain` / `TwoLayerCachedDTensorsRWChain` 的 mesh 约束里，`num_nodes` 取 cache mesh 的节点维度，`world_size` 取计算端视图（读方向 `dst` / 写方向 `src`）的 mesh 参与者总数；展开规则见 3.4。

`check_constraints` 不替代 `check_*`：前者递归检查对象自身和所有嵌套数据；后者检查一次 `CopyPlan` 的 disjoint / covering。

例如：`ComputeDTensorsView.check_constraints` 递归到 `d_tensor_table`，再逐层到 `NamedDTensor` / `GenericDTensor` / `NaiveDTensor` / `GlobalTensorMeta`；`PhysicalLayoutTable.check_constraints` 递归到每个 `DisjointStorageMeshList` 和其中的 `StorageMesh`。父对象的实现不能只查自己的字段。

Python 实现里，`check_constraints` 在数据类 `Foo` 上；build 接口在 `FooBuilderMixin`；见附录 C。

### B.3 Common types

#### B.3.1 Ranges

##### `CopyPlan`

```viba
# One source-to-destination copy; each side is a DTensorRange.
CopyPlan = list[($src DTensorRange, $dst DTensorRange)]
```

##### `DTensorRange`

```viba
# A range on either side of a copy: a compute, storage, or cache view. A
# cache view uses RangeFromComputeView, so there are only two branches.
DTensorRange =
    Oneof
  | RangeFromComputeView
  | RangeFromStorageView
```

##### `RangeFromCacheView`

```viba
# A cache range is a compute range: the cache lives on the compute side and
# its accessor takes the same range type, so DTensorRange has no cache
# branch.
RangeFromCacheView = RangeFromComputeView
```

##### `RangeFromComputeView`

```viba
RangeFromComputeView =
    Object
  * $global_tensor_id (
        str
      * { Global tensor identifier on the compute or cache side. }
    )
  * $global_tensor_offset (
        int
      * { Byte offset inside the global logical tensor. }
    )
  * $size (
        int
      * { Byte length of this slice. }
    )
```

##### `RangeFromStorageView`

```viba
RangeFromStorageView =
    Object
  * $file_path (
        str
      * { Path of the backing container file. }
    )
  * $contiguous_payload_offset (
        int
      * { Byte offset in the contiguous payload region (empty slots are 0 bytes; excludes header/alignment). }
    )
  * $size (
        int
      * { Byte length of this storage slice. }
    )
```

#### B.3.2 读写链

##### `DTensorsRWChain`

```viba
DTensorsRWChain =
    Oneof
  | DirectDTensorsRWChain
  | OneLayerCachedDTensorsRWChain
  | TwoLayerCachedDTensorsRWChain
  | MixedDTensorsRWChain
```

##### `get_tensor_ids_of_rw_chain`

```viba
# Tensor ids covered by the chain: compute endpoint's tensor_ids for the
# concrete branches; the concatenation of the segments in list order for Mixed.
get_tensor_ids_of_rw_chain =
     list[str]
  <- $rw_chain DTensorsRWChain
```

##### `legs_of`

```viba
# Views at the two ends of each leg, in data-flow order: Direct gives one
# pair (src, dst); OneLayerCached gives (src, cache) and (cache, dst);
# TwoLayerCached gives three pairs; Mixed concatenates its segments' pairs.
legs_of =
     list[($src DTensorsView, $dst DTensorsView)]
  <- $rw_chain DTensorsRWChain
```

##### `MixedDTensorsRWChain`

```viba
MixedDTensorsRWChain =
    Object
  * $disjoint_tensor_ids_and_rw_chain_list (
        list[DisjointTensorIdsAndRWChain]
      * { Disjoint tensor-id segments paired with their concrete RW chains. }
    )
  * Assert[{
      let tensor_ids = [
        x
        for segment in disjoint_tensor_ids_and_rw_chain_list
        for x in get_tensor_ids_of_rw_chain(segment.rw_chain)
      ]
      check len(tensor_ids) == len(set(tensor_ids))
      and all segment directions match (all read or all write)
    }]
```

##### `DisjointTensorIdsAndRWChain`

```viba
DisjointTensorIdsAndRWChain =
    Object
  * $tensor_ids (
        list[str]
      * { Disjoint subset of tensor ids covered by rw_chain. }
    )
  * $rw_chain (
        ConcreteDTensorsRWChain
      * { Concrete RW chain handling this tensor subset. }
    )
  * Assert[{
      tensor_ids == get_tensor_ids_of_rw_chain(rw_chain)
    }]
```

##### `ConcreteDTensorsRWChain`

```viba
# A concrete chain has no Mixed branch, so a Mixed chain does not nest.
ConcreteDTensorsRWChain =
    Oneof
  | DirectDTensorsRWChain
  | OneLayerCachedDTensorsRWChain
  | TwoLayerCachedDTensorsRWChain
```

##### `DirectDTensorsRWChain`

```viba
DirectDTensorsRWChain =
    Object
  * $src (
        (
            Oneof
          | StorageDTensorsView
          | ComputeDTensorsView
        )
      * { Source endpoint view (Storage on read, Compute on write). }
    )
  * $dst (
        (
            Oneof
          | StorageDTensorsView
          | ComputeDTensorsView
        )
      * { Destination endpoint view (Compute on read, Storage on write). }
    )
  * Assert[{
      src and dst pair StorageDTensorsView and ComputeDTensorsView
      and src.tensor_ids == dst.tensor_ids
    }]
```

##### `OneLayerCachedDTensorsRWChain`

```viba
# data-flow: src -> cache -> dst
OneLayerCachedDTensorsRWChain =
    Object
  * $src (
        (
            Oneof
          | StorageDTensorsView
          | ComputeDTensorsView
        )
      * { Source endpoint view (Storage on read, Compute on write). }
    )
  * $cache (
        CacheDTensorsView
      * { Single-layer cache view on compute nodes (spatial Replicate on read, Shard on write). }
    )
  * $dst (
        (
            Oneof
          | StorageDTensorsView
          | ComputeDTensorsView
        )
      * { Destination endpoint view (Compute on read, Storage on write). }
    )
  * $num_ranks_per_node (
        int
      * { Number of compute ranks per physical node. }
    )
  * Hint[{
      cache 落在计算侧的节点上：
      ComputeDTensorsView 的视角是 rank / world_size；
      CacheDTensorsView 的视角是 node / num_nodes；
      node n 覆盖 rank 区间 [n * num_ranks_per_node, (n + 1) * num_ranks_per_node)。
    }]
  * Assert[{
      src and dst pair StorageDTensorsView and ComputeDTensorsView
      and num_ranks_per_node * num_nodes == world_size
      and src.tensor_ids == cache.tensor_ids == dst.tensor_ids
      and is_cache_mesh_subset_of_mesh(
        cache.mesh, (dst if read else src).mesh, num_ranks_per_node
      )
    }]
```

##### `TwoLayerCachedDTensorsRWChain`

```viba
# data-flow: src -> first_cache -> second_cache -> dst
TwoLayerCachedDTensorsRWChain =
    Object
  * $src (
        (
            Oneof
          | StorageDTensorsView
          | ComputeDTensorsView
        )
      * { Source endpoint view (Storage on read, Compute on write). }
    )
  * $first_cache (
        CacheDTensorsView
      * { First cache layer along data flow (small on read, big on write). }
    )
  * $second_cache (
        CacheDTensorsView
      * { Second cache layer along data flow (big on read, small on write). }
    )
  * $dst (
        (
            Oneof
          | StorageDTensorsView
          | ComputeDTensorsView
        )
      * { Destination endpoint view (Compute on read, Storage on write). }
    )
  * $num_ranks_per_node (
        int
      * { Number of compute ranks per physical node. }
    )
  * Hint[{
      两层 cache 均落在计算侧节点上，按数据流顺序排列：
      读方向 src(Storage) -> first_cache(small) -> second_cache(big) -> dst(Compute)；
      写方向 src(Compute) -> first_cache(big) -> second_cache(small) -> dst(Storage)。
    }]
  * Assert[{
      src and dst pair StorageDTensorsView and ComputeDTensorsView
      and num_ranks_per_node * num_nodes == world_size
      and src.tensor_ids == dst.tensor_ids
        == first_cache.tensor_ids == second_cache.tensor_ids
      and (read:  first_cache ⊆ second_cache ⊆ dst;
           write: second_cache ⊆ first_cache ⊆ src)
    }]
```

##### `CopyPlanChain`

```viba
# Plan of an RW chain. Isomorphic to DTensorsRWChain: the branches match
# one to one, and each branch holds one CopyPlan per leg, in data-flow order.
# Direct has one leg, OneLayerCached two, TwoLayerCached three.
CopyPlanChain =
    Oneof
  | DirectCopyPlanChain
  | OneLayerCachedCopyPlanChain
  | TwoLayerCachedCopyPlanChain
  | MixedCopyPlanChain
```

##### `MixedCopyPlanChain`

```viba
MixedCopyPlanChain =
    Object
  * $disjoint_tensor_ids_and_copy_plan_chain_list (
        list[DisjointTensorIdsAndCopyPlanChain]
      * { Disjoint tensor-id segments paired with their concrete CopyPlan chains. }
    )
```

##### `DisjointTensorIdsAndCopyPlanChain`

```viba
DisjointTensorIdsAndCopyPlanChain =
    Object
  * $tensor_ids (
        list[str]
      * { Disjoint subset of tensor ids covered by copy_plan_chain. }
    )
  * $copy_plan_chain (
        ConcreteCopyPlanChain
      * { Concrete CopyPlan chain handling this tensor subset. }
    )
```

##### `ConcreteCopyPlanChain`

```viba
# Same nesting rule as the RW chain: a Mixed plan chain holds concrete ones.
ConcreteCopyPlanChain =
    Oneof
  | DirectCopyPlanChain
  | OneLayerCachedCopyPlanChain
  | TwoLayerCachedCopyPlanChain
```

##### `DirectCopyPlanChain`

```viba
DirectCopyPlanChain =
    Object
  * $src_to_dst (
        CopyPlan
      * { Copy plan from src directly to dst. }
    )
```

##### `OneLayerCachedCopyPlanChain`

```viba
# data-flow: src -> cache -> dst
OneLayerCachedCopyPlanChain =
    Object
  * $src_to_cache (
        CopyPlan
      * { Copy plan from src into cache. }
    )
  * $cache_to_dst (
        CopyPlan
      * { Copy plan from cache to dst. }
    )
```

##### `TwoLayerCachedCopyPlanChain`

```viba
# data-flow: src -> first_cache -> second_cache -> dst
TwoLayerCachedCopyPlanChain =
    Object
  * $src_to_first_cache (
        CopyPlan
      * { Copy plan from src into first_cache. }
    )
  * $first_cache_to_second_cache (
        CopyPlan
      * { Copy plan from first_cache into second_cache. }
    )
  * $second_cache_to_dst (
        CopyPlan
      * { Copy plan from second_cache to dst. }
    )
```

`num_nodes` 是 cache mesh 的 `spatial` 维（节点数），`world_size` 是计算侧（读方向 `dst` / 写方向 `src`）mesh 的参与者总数；`temporal` / `concurrency` 不加 rank，它们只是同一批 node 上的缓冲槽；`is_cache_mesh_subset_of_mesh` 见 B.2.3.3。

#### B.3.3 Meshes

##### `ArrangedMesh`

```viba
ArrangedMesh[tag_name] =
    Object
  * $size (
        tagged[tag_name, int]
      * { Number of participants on this mesh. }
    )
```

##### `LiteralMesh`

```viba
# Explicit rank list; used when a device mesh has no regular arranged form.
# max is tagged by the dimension it bounds: world_size, tensor_count, ...
LiteralMesh[tag_name] =
    Object
  * $ranks (
        list[int]
      * { Ranks covered by this mesh, in order. }
    )
  * $max (
        tagged[tag_name, int]
      * { Exclusive upper bound for every rank in ranks. }
    )
  * Assert[{
      forall x in ranks:
        check x >= 0 and x < self.max
      check len(ranks) == len(set(ranks))
  }]
```

##### `ReshapedMesh`

```viba
ReshapedMesh[T] =
    Object
  * $underlying_mesh (
        T
      * { Mesh before reshaping. }
    )
  * $shape (
        list[int]
      * { Shape of the mesh after reshaping; one entry per dimension. }
    )
```

##### `PermutedMesh`

```viba
PermutedMesh[T] =
    Object
  * $underlying_mesh (
        T
      * { Mesh before permuting dimensions. }
    )
  * $permutation (
        list[int]
      * { New order of mesh dimensions. }
    )
```

##### `SlicedMesh`

```viba
SlicedMesh[T] =
    Object
  * $underlying_mesh (
        T
      * { Mesh before fixing dimensions. }
    )
  * $slices (
        list[(Slice | nil)]
      * { One entry per underlying mesh dimension; nil keeps that dimension. }
    )
```

##### `Slice`

```viba
Slice =
    Object
  * $start (
        (int | nil)
      * { First index selected on this mesh dimension; nil starts at zero. }
    )
  * $stop (
        (int | nil)
      * { Index after the last selected one; nil runs to the end. }
    )
  * $step (
        (int | nil)
      * { Step between selected indices; nil means one. }
    )
```

#### B.3.4 `Error` 与 `Result`

`Error` 是顶层类型，`Result` 的 `Err` 分支用它，不用裸 `str`。每个分支都是一个 Object，至少带 `msg`；调用方按分支类型处理，不解析 `msg`。

##### `Error`

```viba
# Failure reported by any Result-returning API. Every branch is a named
# Object, so the branch type is the tag; callers switch on the value type.
Error =
    Oneof
  | ConstraintViolationError
  | CheckFailedError
  | NotFoundError
  | InvalidArgumentError
  | GDTensorIOError
```

##### `ConstraintViolationError`

```viba
# check_constraints rejected the data.
ConstraintViolationError =
    Object
  * $type_name (
        str
      * { Type name of the object whose check_constraints failed, e.g. "PhysicalLayoutTable". }
    )
  * $constraint_name (
        str
      * { Name of the constraint that failed on that object. }
    )
  * $msg (
        str
      * { Human-readable reason. }
    )
```

##### `CheckFailedError`

```viba
# A check_* validation failed: not disjoint, not covering, or overlapping
# storage meshes.
CheckFailedError =
    Object
  * $check_name (
        str
      * { Name of the check that failed. }
    )
  * $msg (
        str
      * { Human-readable reason. }
    )
```

##### `NotFoundError`

```viba
# A lookup by tensor id, dict key, or file path found nothing.
NotFoundError =
    Object
  * $key_name (
        str
      * { Field or table being looked up, e.g. "tensor_id", "tensor_entries", "file_paths". }
    )
  * $key (
        str
      * { The missing value, in its canonical form. }
    )
  * $msg (
        str
      * { Human-readable reason. }
    )
```

##### `InvalidArgumentError`

```viba
# A caller-supplied argument does not satisfy the documented contract.
InvalidArgumentError =
    Object
  * $argument_name (
        str
      * { Argument or precondition that failed, e.g. "tensor_ids". }
    )
  * $msg (
        str
      * { Human-readable reason. }
    )
```

##### `GDTensorIOError`

```viba
# File or buffer I/O failed.
GDTensorIOError =
    Object
  * $file_path (
        (str | nil)
      * { File path for file I/O; nil for in-memory buffer I/O. }
    )
  * $operation (
        str
      * { Operation that failed, e.g. "open", "read", "write", "fsync", "rename". }
    )
  * $msg (
        str
      * { Human-readable reason. }
    )
```

各分支的产生方：

| 分支 | 产生方 |
|---|---|
| `ConstraintViolationError` | 所有 `check_constraints`（B.2.5） |
| `CheckFailedError` | B.1.3 的 `check_*`：不 disjoint、不 covering、storage mesh 重叠 |
| `NotFoundError` | `compute_d_tensors_view_get_gd_tensor` 的 id、`tensor_order_value_of` 的 id、`file_path_of` 的 `tensor_entries` / `file_paths` 键 |
| `InvalidArgumentError` | `d_tensors_view_rank_plan_chain_load` 的 `Assert` 失败；session 在生命周期之外被调用（`begin_*` 之前，或 `commit` / `abort` / `finish` 之后） |
| `GDTensorIOError` | `begin_save` / `begin_load` / `commit` / `abort` / `finish` / `init_data_accessor` 的文件操作，以及 accessor 的 buffer 读写 |

##### `Result`

```viba
# Success or failure. Branches are named types; no tags.
Result[T] =
    Oneof
  | Ok[T]
  | Err
```

##### `Ok`

```viba
# Success; value is the payload, nil for void.
Ok[T] =
    Object
  * $value (
        T
      * { Payload value, or nil for Result[void]. }
    )
```

##### `Err`

```viba
# Failure; value is one of the Error branches.
Err =
    Object
  * $value (
        Error
      * { Concrete Error branch describing the failure. }
    )
```

## 附录 C Python 实现与代码生成指南

### C.1 Builder Mixin

数据类和 build 接口分开。每个数据类 `Foo` 的 build 接口放在 `FooBuilderMixin`，数据类写成：

```python
class Foo(FooBuilderMixin):
    def check_constraints(self) -> Result[None]: ...
```

`FooBuilderMixin` 提供：

- `get_<field>` / `set_<field>`；
- `add_<field>`（list）；
- `mutable_<field>`（嵌套对象 / map）；
- `clear_<field>`（可空字段 / oneof）。

不生成 `has_*` / `which_*`。判空用 `get_xxx() is None`；判分支用 `type(get_xxx())`，因为每个分支是一个独立的类（见 C.2）。

`FooBuilderMixin` 不持有数据，只读写 `self.<field>`；`new_foo()` 返回 `Foo()`。`check_constraints` 不放进 Mixin，留在 `Foo` 上，因为它检查数据，不是 build 接口。生成器为每个数据类生成一个 `<ClassName>BuilderMixin`，不要让数据类直接定义这些 build 方法。`session` / `access` 是抽象基类，不走这套 Mixin / check_constraints 约定（见 C.2）。

例如：

```python
class ComputeDTensorsViewBuilderMixin:
    def add_tensor_id(self, value: str) -> None: ...
    def set_data_accessor(self, value: DTensorDataAccessor) -> None: ...
    def mutable_d_tensor_table(self) -> DTensorTable: ...


class ComputeDTensorsView(ComputeDTensorsViewBuilderMixin):
    def check_constraints(self) -> Result[None]: ...
```

`StorageDTensorsView` 的 mixin 还包含 `set_physical_layout` / `set_begin_save` / `set_begin_load`；`PhysicalLayoutTable` 的 mixin 包含 `set_container_format` / `mutable_tensor_entries` / `mutable_file_paths`。

### C.2 类型映射

这里的类型名是 viba 名；Python 名字见用户主要 API 清单。

所有 product type（viba `Object`）映射成 Python 类型：

```python
class NamedDTensor: ...


class GlobalTensorMeta: ...
```

product 类按字段实现 `__eq__` / `__hash__`，因为它要当 dict 键：`PhysicalLayoutTable.tensor_entries` 的键是 `TensorComponentPos`。

类型别名和字符串字面量直接映射成 Python 原生类型，不生成新类，也不生成 `enum` / `Literal`：

```python
ContainerFormat = str  # 当前是 "safetensors" / "distcp"；新增格式不改这个类型
```

tuple 也直接写 Python 的 `tuple[...]`：

```python
CopyPlan = list[tuple[DTensorRange, DTensorRange]]
```

sum type（viba `Oneof`）统一映射成 `types.UnionType`；分支一律是命名类型，Python 侧就是分支自己的类，没有 tag。

```python
DTensorsView = ComputeDTensorsView | StorageDTensorsView | CacheDTensorsView
DTensorRange = RangeFromComputeView | RangeFromStorageView
GenericDTensor = NaiveDTensor | ReshapedDTensor
DTensorsRWChain = (
    DirectDTensorsRWChain
    | OneLayerCachedDTensorsRWChain
    | TwoLayerCachedDTensorsRWChain
    | MixedDTensorsRWChain
)
ConcreteDTensorsRWChain = (
    DirectDTensorsRWChain
    | OneLayerCachedDTensorsRWChain
    | TwoLayerCachedDTensorsRWChain
)
CopyPlanChain = (
    DirectCopyPlanChain
    | OneLayerCachedCopyPlanChain
    | TwoLayerCachedCopyPlanChain
    | MixedCopyPlanChain
)
CacheMesh = StCacheOuterProductMesh | PermutedMesh[CacheMesh]
Placement = Shard | Replicate | Partial
Error = (
    ConstraintViolationError
    | CheckFailedError
    | NotFoundError
    | InvalidArgumentError
    | GDTensorIOError
)
```

payload 是泛型 `T`、`int` 或 void 的分支也是命名类型，和别的 product 一样放 `gdtensor.builder`，有 `FooBuilderMixin` 和 `check_constraints`：

```python
class Ok[T](OkBuilderMixin):
    def check_constraints(self) -> Result[None]: ...


class Err(ErrBuilderMixin):
    def check_constraints(self) -> Result[None]: ...


class Shard(ShardBuilderMixin):
    def check_constraints(self) -> Result[None]: ...


class Replicate(ReplicateBuilderMixin):
    def check_constraints(self) -> Result[None]: ...


class Partial(PartialBuilderMixin):
    def check_constraints(self) -> Result[None]: ...
```

字段访问在 Mixin 里：`Ok` / `Err` 是 `get_value` / `set_value`，`Shard` 是 `get_axis` / `set_axis`。

`Replicate` / `Partial` 必须实现成无字段类型：类体里除 `check_constraints` 外不放任何成员，Mixin 里不生成 `get_*` / `set_*` / `add_*` / `mutable_*` / `clear_*`，也不为了“占位”加字段、不借用别的类型代替。它们的语义由类型本身表达，判别和比较都只看类型：`type(x) is Replicate`，两个 `new_replicate()` 造出来的对象相等。

`Result` 是类型别名：

```python
type Result[T] = Ok[T] | Err
```

这是 PEP 695 的 `type` 别名，需要 Python 3.12+；`Result[int]` 能直接当注解用。它是类型别名不是类，所以不能写 `isinstance(x, Result)`。`Result[void]` 写成 `Result[None]`，此时 `Ok.value` 是 `None`。

构造和别的数据类一样走 protobuf 风格：

```python
ok = new_ok()
ok.set_value(3)
err = new_err()
err.set_value(NotFoundError(...))
replicate = new_replicate()  # Replicate 没有字段
```

判别用 `type(x)`：`type(x) is Ok`、`type(placement) is Shard`、`type(placement) is Replicate`。`Ok` / `Err` 的 payload 在 `get_value()` 里；`Replicate` / `Partial` 没有字段，`type(x)` 就足够区分。`isinstance(x, Ok)` 可以（不带参数），`isinstance(x, Ok[int])` 不行（参数化泛型不能给 `isinstance`）。

```python
result = gdtensor.io.rank_load(rank, compute_view, storage_view)
if type(result) is Ok:
    value = result.get_value()
elif type(result) is Err:
    error = result.get_value()
```

viba 的 `Oneof` 只是特殊标记；protobuf 的 `oneof` 是关键字。这里不生成 protobuf 的 oneof API。判空用 `get_xxx() is None`。

inline product 先提成命名类型：`GenericDTensor` 的第二个分支 `(NaiveDTensor * $shape list[int])` 在 Python 侧是 `ReshapedDTensor`。同一个 Union 里不能有两个用 `type()` 分不开的分支。

`tagged[tag_name, T]` 是带名 tag 的标量，不是 Oneof 分支；只有它还进 `gdtensor.tag`。`tagged["world_size", int]` 对应 `tag.WorldSize[int]`，值是 `tag.WorldSize(4)`；同理 `tagged["num_steps", int]` 是 `tag.NumSteps[int]`，`tagged["num_nodes", int]` 是 `tag.NumNodes[int]`，`tagged["concurrency_width", int]` 是 `tag.ConcurrencyWidth[int]`，`tagged["constant_one", int]` 是 `tag.ConstantOne[int]`，`tagged["tensor_count", int]` 是 `tag.TensorCount[int]`。`ArrangedMesh` 的 `$size` 和 `LiteralMesh` 的 `$max` 都是这种用法：`$size tagged["world_size", int]` 写成 `size: tag.WorldSize[int]`，`LiteralMesh["world_size"]` 写成 `LiteralMesh[tag.WorldSize[int]]`。

这些标量 tag 不逐个手写：一个 `Box` 基类加一个 `make_box_subclass`，每个 tag 一行。

```python
import types


class Box:
    # 一个 tagged[...] 标量：tag 是类本身，值挂在 value 上。
    __slots__ = ("value",)
    __match_args__ = ("value",)

    def __init__(self, value=None):
        self.value = value

    def __class_getitem__(cls, item):
        args = item if isinstance(item, tuple) else (item,)
        return types.GenericAlias(cls, args)


def make_box_subclass(name):
    return type(name, (Box,), {"__module__": __name__, "__slots__": ()})


# gdtensor/tag.py
WorldSize = make_box_subclass("WorldSize")
ConstantOne = make_box_subclass("ConstantOne")
TensorCount = make_box_subclass("TensorCount")
NumSteps = make_box_subclass("NumSteps")
NumNodes = make_box_subclass("NumNodes")
ConcurrencyWidth = make_box_subclass("ConcurrencyWidth")
```

`Box` 提供构造、`value`、`__match_args__` 和 `__class_getitem__`；`make_box_subclass` 用 `type` 造子类，每个 tag 一行。`gdtensor.tag` 里只有一个基类、一个工厂和六个标量 tag 类。

mesh 的 Oneof 因此就是分支类型的 union，例如 `type ComputeMesh = LiteralMesh[tag.WorldSize[int]] | ArrangedMesh[tag.WorldSize[int]] | ReshapedMesh[ComputeMesh] | PermutedMesh[ComputeMesh] | SlicedMesh[ComputeMesh]`；缓存同理，`type CacheMesh = StCacheOuterProductMesh | PermutedMesh[CacheMesh]`。

viba 的 `Oneof` 只是特殊标记；protobuf 的 `oneof` 是关键字。这里不生成 protobuf 的 oneof API。判空用 `get_xxx() is None`。

inline product 先提成命名类型：`GenericDTensor` 的第二个分支 `(NaiveDTensor * $shape list[int])` 在 Python 侧是 `ReshapedDTensor`。同一个 Union 里不能有两个用 `type()` 分不开的分支。

`session` / `access` 是抽象基类，不按 product / sum 映射，不生成 BuilderMixin / check_constraints。附录 B 里它们写成 `Object`、字段是 callable；生成成 `abc.ABC` 子类加 `@abstractmethod`：字段名就是方法名，`$self` 就是 `self`，其余参数按顺序变成方法参数，返回类型就是方法的返回类型。

```python
import abc
from typing import Callable


class StorageSaveSession(abc.ABC):
    @abc.abstractmethod
    def get_file_offset(self, pos: TensorComponentPos) -> Result[int]: ...

    @abc.abstractmethod
    def init_data_accessor(
        self,
    ) -> Result[DTensorDataAccessor[RangeFromStorageView]]: ...

    @abc.abstractmethod
    def commit(self) -> Result[None]: ...

    @abc.abstractmethod
    def abort(self) -> Result[None]: ...


class StorageLoadSession(abc.ABC):
    @abc.abstractmethod
    def get_file_offset(self, pos: TensorComponentPos) -> Result[int]: ...

    @abc.abstractmethod
    def init_data_accessor(
        self,
    ) -> Result[DTensorDataAccessor[RangeFromStorageView]]: ...

    @abc.abstractmethod
    def finish(self) -> Result[None]: ...


# (buffer_ptr, buffer_len) -> Result[void]
BufferCallback = Callable[[int, int], Result[None]]


class DTensorDataAccessor[RangeT](abc.ABC):
    @abc.abstractmethod
    def read_to_buffer(
        self, global_range: RangeT, read_to_buffer: BufferCallback
    ) -> Result[None]: ...

    @abc.abstractmethod
    def write_from_buffer(
        self, global_range: RangeT, write_from_buffer: BufferCallback
    ) -> Result[None]: ...
```

每个方法都返回 `Result[...]`：成功是 `Ok`，失败是 `Err`；payload 用 `set_value` 写入。`get_file_offset` 的 payload 是文件偏移 `int`，`init_data_accessor` 的 payload 是绑定到该 session 的 `DTensorDataAccessor[RangeFromStorageView]`，`commit` / `abort` / `finish` 的 payload 是 `None`。

具体实现继承这些基类：`SafeTensorsSession` / `DcpTensorsSession` 继承对应的 session 基类，具体 accessor 继承 `DTensorDataAccessor`。

### C.3 用户主要 API 清单

假设 Python 包名是 `gdtensor`。namespace 只分 `io` / `builder` / `session` / `access` / `tag`；tensor 函数直接放顶层，表格第一列留空，顶层函数放表最前面。`d_tensors_view_map` 只构造 view，放 `builder`，不属于 `io`。下表只列用户会直接 import 的 API / 类 / 接口；内部构造（plan / check / CopyPlan / range）不在表里，见第 8 节。

这张表有两个读者。

- 用户看 `Python path`。用户怕麻烦，名字尽量短，靠 namespace 和参数定位；不需要背 `compute_d_tensors_view_` 这类主体前缀，参数已经说明主体。
- 实现者看 `full name`。这是附录 B 的 viba 定义名，平铺、精确；同一个类型在多个 API 里出现时名字完全一致，便于对照和搜索。内部开发要求精确，名字长一点没关系。

`full name` 不能取消，也不能用带 namespace 的 Python 名代替。viba 的定义里没有 namespace 这一层；如果设计一开始就要决定每个名字归哪个 namespace，类型定义会先被 Python 的模块划分绑住，后面改分层就要连带改名。所以规则是：viba 定义用平铺全名，namespace 只在 Python 侧出现；Python 名可以简写，全名不改。

| namespace | Python path | full name |
|---|---|---|
|  | `gdtensor.get_tensor_ids` | `compute_d_tensors_view_get_tensor_ids` |
|  | `gdtensor.get_chain_tensor_ids` | `get_tensor_ids_of_rw_chain` |
|  | `gdtensor.get_gd_tensor` | `compute_d_tensors_view_get_gd_tensor` |
|  | `gdtensor.make_gd_tensor` | `gd_tensor_make` |
|  | `gdtensor.get_d_tensor` | `gd_tensor_get_d_tensor` |
|  | `gdtensor.reset_d_tensor` | `gd_tensor_reset_d_tensor` |
|  | `gdtensor.Result` | `Result` |
|  | `gdtensor.Error` | `Error` |
| `io` | `gdtensor.io.rank_load` | `d_tensors_view_rank_load` |
| `io` | `gdtensor.io.rank_save` | `d_tensors_view_rank_save` |
| `io` | `gdtensor.io.chain_load` | `d_tensors_view_rank_chain_load` |
| `io` | `gdtensor.io.chain_save` | `d_tensors_view_rank_chain_save` |
| `io` | `gdtensor.io.chain_stat_mem_overhead` | `get_stat_mem_overhead_of_rw_chain` |
| `io` | `gdtensor.io.chain_stat_outbound_traffic` | `get_stat_outbound_traffic_of_rw_chain` |
| `io` | `gdtensor.io.chain_stat_inbound_traffic` | `get_stat_inbound_traffic_of_rw_chain` |
| `builder` | `gdtensor.builder.d_tensors_view_map` | `d_tensors_view_map` |
| `builder` | `gdtensor.builder.DTensorsView` | `DTensorsView` |
| `builder` | `gdtensor.builder.ComputeDTensorsView` | `ComputeDTensorsView` |
| `builder` | `gdtensor.builder.StorageDTensorsView` | `StorageDTensorsView` |
| `builder` | `gdtensor.builder.CacheDTensorsView` | `CacheDTensorsView` |
| `builder` | `gdtensor.builder.DTensorsViewMapCore` | `DTensorsViewMapCore` |
| `builder` | `gdtensor.builder.DTensorTable` | `DTensorTable` |
| `builder` | `gdtensor.builder.NamedDTensor` | `NamedDTensor` |
| `builder` | `gdtensor.builder.GenericDTensor` | `GenericDTensor` |
| `builder` | `gdtensor.builder.NaiveDTensor` | `NaiveDTensor` |
| `builder` | `gdtensor.builder.ReshapedDTensor` | `ReshapedDTensor` |
| `builder` | `gdtensor.builder.GlobalTensorMeta` | `GlobalTensorMeta` |
| `builder` | `gdtensor.builder.Placement` | `Placement` |
| `builder` | `gdtensor.builder.Ok` | `Ok` |
| `builder` | `gdtensor.builder.Err` | `Err` |
| `builder` | `gdtensor.builder.Shard` | `Shard` |
| `builder` | `gdtensor.builder.Replicate` | `Replicate` |
| `builder` | `gdtensor.builder.Partial` | `Partial` |
| `builder` | `gdtensor.builder.ComputeMesh` | `ComputeMesh` |
| `builder` | `gdtensor.builder.StorageMesh` | `StorageMesh` |
| `builder` | `gdtensor.builder.SlotOuterProductMesh` | `SlotOuterProductMesh` |
| `builder` | `gdtensor.builder.ContentSlotsMesh` | `ContentSlotsMesh` |
| `builder` | `gdtensor.builder.LiteralMesh` | `LiteralMesh` |
| `builder` | `gdtensor.builder.ArrangedMesh` | `ArrangedMesh` |
| `builder` | `gdtensor.builder.ReshapedMesh` | `ReshapedMesh` |
| `builder` | `gdtensor.builder.PermutedMesh` | `PermutedMesh` |
| `builder` | `gdtensor.builder.SlicedMesh` | `SlicedMesh` |
| `builder` | `gdtensor.builder.Slice` | `Slice` |
| `builder` | `gdtensor.builder.CacheMesh` | `CacheMesh` |
| `builder` | `gdtensor.builder.StCacheOuterProductMesh` | `StCacheOuterProductMesh` |
| `builder` | `gdtensor.builder.TemporalCacheMesh` | `TemporalCacheMesh` |
| `builder` | `gdtensor.builder.SpatioCacheMesh` | `SpatioCacheMesh` |
| `builder` | `gdtensor.builder.DTensorsRWChain` | `DTensorsRWChain` |
| `builder` | `gdtensor.builder.ConcreteDTensorsRWChain` | `ConcreteDTensorsRWChain` |
| `builder` | `gdtensor.builder.DirectDTensorsRWChain` | `DirectDTensorsRWChain` |
| `builder` | `gdtensor.builder.OneLayerCachedDTensorsRWChain` | `OneLayerCachedDTensorsRWChain` |
| `builder` | `gdtensor.builder.TwoLayerCachedDTensorsRWChain` | `TwoLayerCachedDTensorsRWChain` |
| `builder` | `gdtensor.builder.MixedDTensorsRWChain` | `MixedDTensorsRWChain` |
| `builder` | `gdtensor.builder.DisjointTensorIdsAndRWChain` | `DisjointTensorIdsAndRWChain` |
| `builder` | `gdtensor.builder.PhysicalLayoutTable` | `PhysicalLayoutTable` |
| `builder` | `gdtensor.builder.DisjointStorageMeshList` | `DisjointStorageMeshList` |
| `builder` | `gdtensor.builder.ContainerFormat` | `ContainerFormat` |
| `builder` | `gdtensor.builder.TensorComponentPos` | `TensorComponentPos` |
| `builder` | `gdtensor.builder.ConstraintViolationError` | `ConstraintViolationError` |
| `builder` | `gdtensor.builder.CheckFailedError` | `CheckFailedError` |
| `builder` | `gdtensor.builder.NotFoundError` | `NotFoundError` |
| `builder` | `gdtensor.builder.InvalidArgumentError` | `InvalidArgumentError` |
| `builder` | `gdtensor.builder.GDTensorIOError` | `GDTensorIOError` |
| `session` | `gdtensor.session.StorageSaveSession` | `StorageSaveSession` |
| `session` | `gdtensor.session.StorageLoadSession` | `StorageLoadSession` |
| `access` | `gdtensor.access.DTensorDataAccessor` | `DTensorDataAccessor` |
| `tag` | `gdtensor.tag.ConstantOne` | `tagged["constant_one", int]` |
| `tag` | `gdtensor.tag.TensorCount` | `tagged["tensor_count", int]` |
| `tag` | `gdtensor.tag.WorldSize` | `tagged["world_size", int]` |
| `tag` | `gdtensor.tag.NumSteps` | `tagged["num_steps", int]` |
| `tag` | `gdtensor.tag.NumNodes` | `tagged["num_nodes", int]` |
| `tag` | `gdtensor.tag.ConcurrencyWidth` | `tagged["concurrency_width", int]` |

每个数据类 `Foo` 都在 `gdtensor.builder` 提供 `FooBuilderMixin`。例如：

- `gdtensor.builder.DTensorsViewMapCoreBuilderMixin`
- `gdtensor.builder.ComputeDTensorsViewBuilderMixin`
- `gdtensor.builder.StorageDTensorsViewBuilderMixin`
- `gdtensor.builder.CacheDTensorsViewBuilderMixin`
- `gdtensor.builder.OkBuilderMixin`
- `gdtensor.builder.DTensorTableBuilderMixin`
- `gdtensor.builder.PhysicalLayoutTableBuilderMixin`

`Ok` / `Err` / `Shard` / `Replicate` / `Partial` 是 product，和别的数据类一样放 `gdtensor.builder`，有 BuilderMixin 和 `check_constraints`；`Replicate` / `Partial` 没有字段，构造只有 `new_replicate()` / `new_partial()`。`gdtensor.Result` 是顶层别名 `type Result[T] = Ok[T] | Err`；`gdtensor.Error` 是五个错误类的 union，放顶层方便调用方按 `Error` 做分类。

`gdtensor.tag` 只剩 `tagged[...]` 标量：`tag.WorldSize`、`tag.ConstantOne`、`tag.TensorCount`、`tag.NumSteps`、`tag.NumNodes`、`tag.ConcurrencyWidth`；表里 `full name` 列写的是对应的 `tagged["...", int]` 名字。Oneof 分支没有 tag，所以 `tag` 里不再有分支类。`check_constraints` 在 `Foo` 上，不在 Mixin；`new_foo()` 也在 `gdtensor.builder`。`DTensorsView` / `DTensorsRWChain` / `ConcreteDTensorsRWChain` / `CopyPlanChain` / `ConcreteCopyPlanChain` / `GenericDTensor` / `ComputeMesh` / `StorageMesh` / `CacheMesh` / `TemporalCacheMesh` / `SpatioCacheMesh` / `ContentSlotsMesh` / `Error` / `Placement` 是 `types.UnionType`，`Result` 是 `type` 别名，都没有 BuilderMixin；`ContainerFormat` 是类型别名，也没有 BuilderMixin；`TensorComponentPos`、五个 `*Error` 和 `Ok` / `Err` / `Shard` / `Replicate` / `Partial` 是 product，按类处理。`session` / `access` 是抽象基类，不放进 builder，不套 BuilderMixin / check_constraints；它们的 Python 形态见 C.2。

### C.4 Helper functions

helpers 放已有的 `builder` namespace，不新增 namespace，也不放顶层。它们不是主要 API，所以单列，不并进 C.3 的表。

| namespace | Python path | full name |
|---|---|---|
| `builder` | `gdtensor.builder.global_tensor_meta_of` | `global_tensor_meta_of` |
| `builder` | `gdtensor.builder.tensor_order_value_of` | `tensor_order_value_of` |
| `builder` | `gdtensor.builder.file_path_of` | `file_path_of` |
| `builder` | `gdtensor.builder.leader_tensor_component_pos_of` | `leader_tensor_component_pos_of` |
| `builder` | `gdtensor.builder.is_cache_mesh_subset_of_mesh` | `is_cache_mesh_subset_of_mesh` |
| `builder` | `gdtensor.builder.legs_of` | `legs_of` |
