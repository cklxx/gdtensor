# 跨计算和存储的 DTensorsView 设计

本文说明 DTensorsView 如何同时描述计算侧和存储侧的张量，以及如何在两者之间搬运数据。类型定义见附录 B，Python 映射见附录 C；本文解释模型、边界、执行顺序和不变量，不重复类型定义，也不记录变更历史。

读者对象是第一次接触这套设计的实现者和调用方。文中的类型名以附录 B 为准；术语见附录 A。

## 1 问题

一份逻辑张量，在计算侧和设备 mesh 绑定，在存储侧和文件、槽位绑定。两侧需要共同回答：

- 逻辑上有哪些张量、每个张量有多少槽位；
- 物理上这些槽位落在哪些文件、文件的哪个位置、文件路径和格式是什么；
- 一个文件可能由一个 rank 写，也可能由多个 rank 合写；
- 一个张量可能跨多个文件；
- 保存和加载的并行度可以不同，reshard 不能依赖某一种固定拓扑；
- 槽位可以为空。

本文出现的 DCP 指 PyTorch distributed checkpoint；safetensors 是单文件张量格式；PP stage 指流水线并行的一段。

设计目标是用一份逻辑表、一种复制计划、两种 view 覆盖这些情况；文件格式差异留在 session 的实现里，不进入核心类型。

## 2 心智模型

把系统分成三层：

| 层 | 回答的问题 | 主要类型 |
|---|---|---|
| 逻辑层 | 有哪些张量、槽位怎么排 | `DTensorTable`、`NamedDTensor`、`StorageMesh` |
| 物理层 | 槽位落在哪个文件、什么格式 | `PhysicalLayoutTable`、`DisjointStorageMeshList`、`ContainerFormat` |
| 执行层 | 怎么把源 range 搬到目标 range | `DTensorRange`、`CopyPlan`、plan、session、accessor |

view 是这一模型的句柄：

- `ComputeDTensorsView`：`tensor_ids` + `DTensorTable[ComputeMesh]` + `DTensorDataAccessor[RangeFromComputeView]`；
- `StorageDTensorsView`：`tensor_ids` + `DTensorTable[SingleTensorStorageMesh]` + `DTensorDataAccessor[RangeFromStorageView]` + `PhysicalLayoutTable` + `begin_save` / `begin_load`。

两边都有 `tensor_ids`、逻辑表和 accessor。`DTensorsView` 是两支的 oneof：`ComputeDTensorsView | StorageDTensorsView`，reshard 用它表达源和目标。存储侧多一个物理布局表和 session，原因是文件是多个张量共享的线性容器，而且文件有 header / metadata 需要显式发布；设备内存由框架管理，计算侧不需要这一层。

计算侧的数据处理不走 `GenericDTensor` 本身，而是走 compute/torch 桥：`compute_d_tensors_view_get_gd_tensor` 按 tensor id 取 `GenericDTensor[ComputeMesh]`，`gd_tensor_get_d_tensor` 取出 `torch.DTensor`，处理完用 `gd_tensor_reset_d_tensor` 写回。storage 侧不提供数据处理 API；`GenericDTensor[StorageMesh]` 先经 load / reshard 变成 `GenericDTensor[ComputeMesh]`，再进入这座桥。

## 3 逻辑模型

### 3.1 slot

`slot`（槽位）是逻辑概念，不是文件或容器。把存储计划看成一个 `num_tensors × world_size` 的表：每一行是一个张量，每一列是一个全局 rank，每个格子是一个 slot；总 slot 数 = `num_tensors * world_size`。

格子可以为空：该 rank 不持有这个张量、这个张量不在该 pp stage，或者它只存一份而其余列留空。同一行的非空格子按列顺序拼接，构成这个张量的存储区域。

### 3.2 DTensorTable

`DTensorTable[MeshT]` 是有序的 `NamedDTensor[MeshT]` 列表。列表顺序就是容器 header 中张量的顺序；不需要第二份 id 顺序。

`NamedDTensor` / `GenericDTensor` / `NaiveDTensor` 在计算侧和存储侧共用；两侧的差别是 `MeshT` 和 `RangeT`。

### 3.3 StorageMesh

`StorageMesh` 是存储侧唯一的 mesh 概念，和计算侧的 `ComputeMesh` 对称。它的 `OuterProductMesh` 分支同时描述两个维度：

- `tensor_slots`：`ArrangedMesh["tensor_count"]`，哪些张量；
- `content_slots`：`ContentSlotsMesh`，即 `LiteralMesh["world_size"] | ArrangedMesh["world_size"]`，每个张量的哪些槽位；`LiteralMesh` 直接列出 rank，`ArrangedMesh` 用 `"world_size"` 表示列数。

`ReshapedMesh` / `SlicedMesh` 在这两个维度上继续变形。

计算侧的 `ComputeMesh` 也有 `LiteralMesh["world_size"]` 分支，用来接住任意 rank 的 device mesh。

mesh 相关的 Oneof 分支都是命名类型，所以不带 tag；`type()` 或 `isinstance(x, list)` 就能分出是哪一支。

按 `TensorSlotsMeshT` 分两种：

- `SingleTensorStorageMesh`：tensor slots 是 `ArrangedMesh["constant_one"]`，只覆盖一个张量，用于 `DTensorTable`；
- `MultiTensorsStorageMesh`：tensor slots 是 `ArrangedMesh["tensor_count"]`，覆盖多个张量，用于 `DisjointStorageMeshList`。

一个张量的单张量 mesh 必须等于多张量 mesh 中该张量所在的那一行：

```bash
single_tensor_mesh = multi_tensors_mesh[
  tensor_order_value_of(tensor_id) : tensor_order_value_of(tensor_id) + 1,
  :
]
```

不能用一个覆盖多个张量槽的 mesh 代替单张量 mesh。`tensor_order_value_of` 是内部 helper，返回张量在 `DTensorTable.named_d_tensors` 里的顺序。

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

## 5 执行模型

### 5.1 CopyPlan 和 DTensorRange

`DTensorRange` 表示一次复制的一侧：

```bash
DTensorRange =
    Oneof
  | RangeFromComputeView
  | RangeFromStorageView
```

`CopyPlan` 是 `list[($src DTensorRange, $dst DTensorRange)]`，每一对表示一次源 range 到目标 range 的复制。`CopyPlan` 是唯一的 pair 类型；compute/storage 的差异由 `DTensorRange` 的分支类型表达：`RangeFromComputeView` / `RangeFromStorageView`。类型上允许四种 branch 组合；具体 plan API 用前置条件限制合法组合：`plan_save` 产生 compute→storage，`plan_load` 产生 storage→compute，`plan_reshard` 允许混合。

### 5.2 plan API

`d_tensors_view_rank_plan_load`、`d_tensors_view_rank_plan_save`、`d_tensors_view_rank_plan_reshard` 返回 `Result[CopyPlan]`。它们是纯函数，不打开 session，负责 range selection。

`d_tensors_view_rank_load`、`d_tensors_view_rank_save`、`d_tensors_view_rank_reshard` 是不带 plan 的 API，只做薄执行器：调用对应的 plan API，按 `CopyPlan` 执行复制，不重新选择区间。

`d_tensors_view_rank_reshard` 的 `tensor_ids` 和 `global_tensor_meta` 前置条件放在 `d_tensors_view_rank_plan_reshard` 的 `Assert` 里；`global_tensor_meta_of` 从 `NamedDTensor` 取出全局 meta，naive 和 reshaped 两支都归到同一个全局 meta。取 `NamedDTensor` 时用 `tensor_order_value_of(d_tensor_table, tensor_id)` 得到列表下标，不能把 `tensor_id` 直接当 `named_d_tensors` 的下标。不带 plan 的 `d_tensors_view_rank_reshard` 只执行。

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

### 5.4 save / load / reshard 的执行顺序

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

reshard：

1. `plan = d_tensors_view_rank_plan_reshard(rank, src_view, dst_view)`；`tensor_ids` / `global_tensor_meta` 的前置条件在 plan 的 `Assert` 里；
2. 源或目标是 storage view 时，分别 `begin_load` / `begin_save`；compute view 没有 session；
3. 对每个 `(src, dst)`：按 range 的类型选 accessor（`RangeFromComputeView` / `RangeFromStorageView`）；storage 侧用 session 做实际文件偏移映射；
4. 目标 `commit()`；源 `finish()`；失败时目标 `abort()`。

check 函数也收 `CopyPlan`，按 range 类型过滤出 compute 或 storage range，再做 disjoint / covering 检查。check 是内部一致性校验，不是 IO API。

### 5.5 d_tensors_view_map

`d_tensors_view_map` 是 view 转换 API，Python 入口是 `gdtensor.builder.d_tensors_view_map`：它把一个 `DTensorsView` 转成同种类的另一个 `DTensorsView`，只做 view / session 的结构转换。它不选 range，不做 load / save / reshard 的跨 view 搬运，也不承担归约、平均这类数据处理；这些计算走 5.6。

`DTensorsViewMapCore` 是调用方提供的四个可选回调：

- `convert_each_compute_view`：compute view 的每个 `GenericDTensor[ComputeMesh]` 转成一个新的 `GenericDTensor[ComputeMesh]`；
- `convert_each_storage_view`：storage view 的每个 `GenericDTensor[StorageMesh]` 转成一个新的 `GenericDTensor[StorageMesh]`；
- `convert_each_storage_load_session`：storage view 的每个 `StorageLoadSession` 转成一个新的 `StorageLoadSession`；
- `convert_each_storage_save_session`：storage view 的每个 `StorageSaveSession` 转成一个新的 `StorageSaveSession`。

这四个字段是 product，但都可以为 nil：tensor 字段为 nil 表示 identity，session 字段为 nil 表示跳过对应的 session。`d_tensors_view_map` 按 `origin` 的类型只处理对应字段。

执行顺序：

1. `origin` 是 `ComputeDTensorsView`：对 `DTensorTable[ComputeMesh]` 的每个 `NamedDTensor[ComputeMesh]` 调 `convert_each_compute_view`；该字段为 nil 时 tensor 原样保留；没有 session。
2. `origin` 是 `StorageDTensorsView`：对 `DTensorTable[SingleTensorStorageMesh]` 的每个 `NamedDTensor[SingleTensorStorageMesh]` 调 `convert_each_storage_view`；该字段为 nil 时 tensor 原样保留。再处理 load / save session：字段非 nil 时打开对应 session 并调用，字段为 nil 时跳过该 session。session 的打开、提交、中止由 `d_tensors_view_map` 管；转换函数只替换 session。
3. 返回的 view 与 `origin` 同种类；`tensor_ids` 不变，`global_tensor_meta` 由转换函数决定。

任一回调返回 `$err` 时，`d_tensors_view_map` 原样返回那个 `Error`，不再调用后面的回调。

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
6. 用 save / reshard 把结果写回 storage。

storage 侧没有对应的数据处理 API。`GenericDTensor[StorageMesh]` 总是先经 load / reshard 变成 `GenericDTensor[ComputeMesh]`，再进入这座桥；处理完再经 save / reshard 回到 storage。`d_tensors_view_map` 只做 view / session 的结构转换，不参与这里的计算。

`gd_tensor_reset_d_tensor` 要求新的 `torch.DTensor` 的 `global_tensor_meta`、`placements`、`mesh` 与原 `gd_tensor` 一致；不一致时调用方先 reshard，或重新 `gd_tensor_make`。

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
  - 任意子对象返回 `$err`，父对象立即返回同一个错误；父对象不包装、不改写它。

`check_constraints` 的 `$err` 是顶层的 `Error` 的 `ConstraintViolationError` 分支（B.3.3）：`type_name` 是出错对象的类型名，`constraint_name` 是它没通过的那条约束，`msg` 说明原因。

构造 view 按这个约定：

```python
view = new_compute_d_tensors_view()
view.set_data_accessor(accessor)
view.mutable_d_tensor_table().add_named_d_tensor(named)
must_ok(view.check_constraints())
```

`StorageDTensorsView` 还 `set_physical_layout` / `set_begin_save` / `set_begin_load`；`PhysicalLayoutTable` 的 `tensor_entries` / `file_paths` 用 `mutable_*` 写入。所有 `check_constraints` 返回 `Result[void]`；失败时 `$err` 是 `ConstraintViolationError`，带约束名和 `msg`。

`check_constraints` 不替代 `plan` / `check_*`：前者检查对象自身的数据是否完整、合规；后者检查一次 `CopyPlan` 的访问是否 disjoint / covering。

Python 实现里，build 接口放在 `<ClassName>BuilderMixin`；数据类写成 `class Foo(FooBuilderMixin)`；见附录 C。

## 6 不变量

- `PhysicalLayoutTable` 对同一个 `DTensorTable[SingleTensorStorageMesh]` 必须是确定的：`container_format`、`tensor_entries` 的键、`file_paths` 的键、header 字节和区间偏移都一致。
- 一个 `PhysicalLayoutTable` 只有一个 `container_format`；所有文件同格式，混格式拆成多份布局。
- header 的顺序只来自 `DTensorTable[SingleTensorStorageMesh].named_d_tensors` 的列表顺序。
- 共享文件的表不能把 rank 相关信息写进 header。
- `file_paths` 的键必须是某个 `DisjointStorageMeshList` 的 leader `TensorComponentPos`；同一个 list 只有一个路径。
- `tensor_entries` 里的每个 list，其 leader `TensorComponentPos` 必须出现在 `file_paths` 的键里。
- 所有 `DisjointStorageMeshList` 里的 mesh 两两不重叠；不同文件的 mesh 也不重叠；每个 slot 最多被一个 mesh 覆盖。
- 每个空 content slot 占用 0 字节，不贡献物理区间，也不参与 offset 累加；`tensor_entries` 里没有对应的键。
- reshard 要求两边 `tensor_ids` 相同，且同一个 `tensor_id` 的 `global_tensor_meta` 相同；placement 和 mesh 可以不同。

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
| reshard storage→storage | 源/目标各一份布局表 | 两端 session | `CopyPlan` 配对；meta 一致 |
| reshard storage→compute | 源布局表 + compute mesh | 源 session | 目标 accessor 写设备 |
| reshard compute→compute | 两侧 compute mesh | 无 storage session | 走通信，不走文件 |
| 对象存储 multipart | 同一份布局表 | safetensors session 的 multipart 实现 | payload 偏移仍连续；part 边界由 session 处理 |
| 混格式 | 拆成多次操作，每次一份布局表 | 各自 session | 单个 view / plan 只有一种 format |

## 8 IO、实现者与内部边界

| 层 | 内容 | 说明 |
|---|---|---|
| IO API | `d_tensors_view_rank_load`、`d_tensors_view_rank_save`、`d_tensors_view_rank_reshard` | 这三个是跨 view 的搬运；plan 是内部 key function；session 由存储视图提供，格式实现者实现 |
| 数据处理 API | `compute_d_tensors_view_get_tensor_ids`、`compute_d_tensors_view_get_gd_tensor`、`gd_tensor_make`、`gd_tensor_get_d_tensor`、`gd_tensor_reset_d_tensor` | compute 侧 view 与 `torch.DTensor` 的桥；所有数据处理都走这里；storage 侧不提供数据处理 API |
| 构造 API | `d_tensors_view_map` | 只构造 / 转换 view，不做文件 I/O |
| 类 API | 每个数据 `Object` / `Oneof`（session / access 接口除外）的 `new_*` / `get_*` / `set_*` / `add_*` / `mutable_*` / `clear_*`；build 接口在 `<ClassName>BuilderMixin`，数据类写成 `class Foo(FooBuilderMixin)` | protobuf 风格；不生成 `has_*` / `which_*`，判空用 `get_*() is None`、判分支用 `type(get_*())`；`check_constraints` 在数据类上，递归检查自身和所有嵌套数据 |
| Helper API | `global_tensor_meta_of`、`tensor_order_value_of`、`file_path_of`、`leader_tensor_component_pos_of` | 放已有的 `builder` namespace；不是主要 API，不放顶层 |
| 实现者接口 | `StorageSaveSession`、`StorageLoadSession`、`ContainerFormat`、布局构造器、`DTensorDataAccessor` | 格式实现者和布局构造者实现；session 管生命周期、offset，并通过 `init_data_accessor` 提供绑定到该 session 的 data accessor |
| 内部构造 | `d_tensors_view_rank_plan_load`、`d_tensors_view_rank_plan_save`、`d_tensors_view_rank_plan_reshard`、`CopyPlan`、`DTensorRange`、check 函数 | 内部使用；不是 IO API |

附录 B 的 API 章节按“IO API / 数据处理 API / 构造 API / 类 API / Helper API / plan API / check API”分类，是为了完整描述核心；调用方需要 IO API、数据处理 API、构造 API 和类 API，Helper API 按需使用。plan API 是内部 key function，check 是内部一致性校验。

## 9 结论与非目标

计算侧和存储侧共用 `NamedDTensor` / `GenericDTensor` / `NaiveDTensor`，各自只有一个 mesh 概念：`ComputeMesh` 和 `StorageMesh`。存储侧多一个 `PhysicalLayoutTable`，因为文件是多个张量共享的线性容器；多一个 session 层，因为文件格式和 header / metadata 需要显式发布。

非目标：

- 不在核心里做 payload 序列化 / 反序列化；read/write 是 raw bytes。
- 不在核心里提供 reduce API；归约、平均、Welford 这类数据处理都在 `torch.DTensor` 上做，核心只提供 5.6 的桥。
- 不在核心里实现对象存储 multipart、分布式 barrier、重试；这些在 session / 实现层。
- 不在核心里决定布局构造策略；对象只保存业务给出的 entry 和属性，拓扑仍由业务层决定。

## 10 LAWA 示例

LAWA 对最近 k 个 checkpoint 的权重做逐元素平均。核心不决定“最近 k 个”是哪些；调用方按 step / 目录顺序选出 k 个 `StorageDTensorsView`。两个版本的 load / save、tensor id 遍历、`gdtensor.get_d_tensor` / `gdtensor.reset_d_tensor` 都相同，只有 `torch.DTensor` 上的平均公式不同。

两个代码块都省略 `Result` 错误处理；`must_ok` 是示例里自己写的辅助函数，用来取 `$ok` 的值，不属于这套 API。

### 10.1 朴素版本

先累加，最后除以 k：

```python
# must_ok unwraps $ok; error handling omitted.
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
# must_ok unwraps $ok; error handling omitted.
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
- 所有 checkpoint 必须覆盖同一组 `tensor_ids`，且同名 tensor 的 `GlobalTensorMeta` 相同；否则先 reshard 或拒绝。
- 缺 tensor 或按元素加权时，用额外的 mask / weight `torch.DTensor`；core 不提供专门的归约 API。
- storage 侧没有 LAWA 专用 API；它只提供 load / save，数据处理统一走 5.6 的桥。

## 附录 A 术语

| 术语 | 含义 |
|---|---|
| slot（槽位） | `num_tensors × world_size` 表中的一个格子；可以为空；空槽占 0 字节 |
| rank | 全局进程编号；在 slot 表里是列号 |
| `world_size` | 每个张量的 slot 数，也是 rank 数 |
| `TensorComponentPos` | `(tensor_id, rank)`，逻辑张量在某个 rank 上的组件位置 |
| placement | `$shard` / `$replicate` / `$partial`；张量维度在参与者之间的分配方式 |
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
| plan | 纯函数，生成 `CopyPlan`；不打开 session |
| session | 文件生命周期 + offset；save 用 commit / abort，load 用 finish |
| view（视图） | 张量句柄；`tensor_ids` + `DTensorTable` + accessor，存储侧多物理布局表和 session |
| mesh（网格） | 张量在参与者维度上的排布；`ComputeMesh` / `StorageMesh` |
| `LiteralMesh` | 显式的 rank 列表，`$max` 是 rank 的排他上界；表示无法用 `ArrangedMesh` 写出的 device mesh |
| range（区间） | 一段连续字节的定位；`RangeFromComputeView` / `RangeFromStorageView` |
| accessor | 裸字节读写接口；`DTensorDataAccessor[RangeT]` |
| `torch.DTensor` | PyTorch 的 DTensor；计算侧实际数据处理的句柄 |
| `gd_tensor` | `GenericDTensor[ComputeMesh]` 的简称；compute/torch 桥的参数名 |
| compute/torch 桥 | 5.6 的五个 API；在 compute view、`GenericDTensor[ComputeMesh]` 和 `torch.DTensor` 之间取和写 |
| `check_*` | 内部一致性校验；disjoint / covering，不是 IO API |
| `check_constraints` | 每个数据 `Object` / `Oneof` 的方法；session / access 接口除外；递归检查自身和所有嵌套数据 |
| `DTensorsView` | `ComputeDTensorsView` / `StorageDTensorsView`；reshard 的源/目标 |
| `GlobalTensorMeta` | 全局张量的 shape、dtype、itemsize |
| multipart | 对象存储的分段上传；payload 偏移仍连续，段边界由 session 处理 |
| barrier | 分布式同步点；多 rank 在提交或读取前对齐 |
| 布局构造器 | 业务实现，生成 `PhysicalLayoutTable`；核心不提供策略类型 |
| `Error` | 顶层错误类型；`Result` 的 `$err` 分支；每个分支是带 `msg` 的 Object（B.3.3） |
| `Result` | 成功是 `$ok T`，失败是 `$err Error`（B.3.3） |

## 附录 B 类型定义

本附录的类型定义用 viba DSL 写。viba（vibe adt）是给 vibe coding 用的意图描述 DSL，定义见 [lixinqi/viba](https://github.com/lixinqi/viba)。本附录是类型定义清单；设计说明见正文。这里用平铺的 viba 全名，没有 namespace；Python 侧的短名和 namespace 见 C.3。全文的 viba 代码块统一把围栏语言写成 `bash`；那只是高亮，不是 shell 命令。

### B.1 API

#### B.1.1 IO APIs

```bash
# Thin executor of d_tensors_view_rank_plan_load; range selection lives in
# the plan API only.
d_tensors_view_rank_load =
    Result[void]
  <- $rank int
  <- $compute_view ComputeDTensorsView
  <- $storage_view StorageDTensorsView

# Thin executor of d_tensors_view_rank_plan_save.
d_tensors_view_rank_save =
    Result[void]
  <- $rank int
  <- $compute_view ComputeDTensorsView
  <- $storage_view StorageDTensorsView

# Thin executor of d_tensors_view_rank_plan_reshard.
d_tensors_view_rank_reshard =
    Result[void]
  <- $rank int
  <- $src_view DTensorsView
  <- $dst_view DTensorsView
```

#### B.1.2 Plan APIs

```bash
# Internal key function behind d_tensors_view_rank_load. Keep load a thin
# executor of this plan; range selection must live here only.
d_tensors_view_rank_plan_load =
    Result[CopyPlan]
  <- $rank int
  <- $compute_view ComputeDTensorsView
  <- $storage_view StorageDTensorsView

# Internal key function behind d_tensors_view_rank_save.
d_tensors_view_rank_plan_save =
    Result[CopyPlan]
  <- $rank int
  <- $compute_view ComputeDTensorsView
  <- $storage_view StorageDTensorsView

# Internal key function behind d_tensors_view_rank_reshard. The tensor/meta
# precondition lives here; the plan-less API stays a thin executor.
d_tensors_view_rank_plan_reshard =
    Result[CopyPlan]
  <- $rank int
  <- $src_view DTensorsView
  <- $dst_view DTensorsView
  <- Assert[{
    src_view.tensor_ids == dst_view.tensor_ids
    and forall tensor_id in src_view.tensor_ids:
      global_tensor_meta_of(src_view.d_tensor_table.named_d_tensors[tensor_id])
      == global_tensor_meta_of(dst_view.d_tensor_table.named_d_tensors[tensor_id])
  }]
```

#### B.1.3 Check APIs

```bash
check_compute_d_tensors_access_disjoint =
    Result[void]
  <- $d_tensor_table DTensorTable[ComputeMesh]
  <- $copy_plan CopyPlan

check_compute_d_tensors_access_full_covering =
    Result[void]
  <- $d_tensor_table DTensorTable[ComputeMesh]
  <- $copy_plan CopyPlan

check_storage_d_tensors_access_disjoint =
    Result[void]
  <- $d_tensor_table DTensorTable[SingleTensorStorageMesh]
  <- $physical_layout PhysicalLayoutTable
  <- $file_paths list[str]
  <- $copy_plan CopyPlan

check_storage_d_tensors_access_full_covering =
    Result[void]
  <- $d_tensor_table DTensorTable[SingleTensorStorageMesh]
  <- $physical_layout PhysicalLayoutTable
  <- $file_paths list[str]
  <- $copy_plan CopyPlan

check_storage_mesh_disjoint[TensorSlotsMeshT] =
    Result[void]
  <- $lhs StorageMesh[TensorSlotsMeshT]
  <- $rhs StorageMesh[TensorSlotsMeshT]
```

#### B.1.4 Compute/Torch APIs

```bash
# List tensor ids in a compute view.
compute_d_tensors_view_get_tensor_ids =
    list[str]
  <- $view ComputeDTensorsView

# Get a generic compute tensor by tensor id. Use tensor_order_value_of for
# the lookup; a missing id returns $err with NotFoundError.
compute_d_tensors_view_get_gd_tensor =
    Result[GenericDTensor[ComputeMesh]]
  <- $view ComputeDTensorsView
  <- $tensor_id str

# Wrap a torch.DTensor as a generic compute tensor. The torch tensor provides
# the data, global meta, placements and device mesh; gd_tensor_make converts
# the device mesh to the viba ComputeMesh.
gd_tensor_make =
    GenericDTensor[ComputeMesh]
  <- $d_tensor torch.DTensor

# Get the underlying torch.DTensor; nil means metadata-only.
gd_tensor_get_d_tensor =
    (torch.DTensor | nil)
  <- $gd_tensor GenericDTensor[ComputeMesh]

# Replace the underlying torch.DTensor.
gd_tensor_reset_d_tensor =
    void
  <- $gd_tensor GenericDTensor[ComputeMesh]
  <- $d_tensor torch.DTensor
```

#### B.1.5 View constructors

```bash
# Convert each tensor of origin. Callbacks are optional: a nil tensor
# callback means identity; a nil session callback means skip that session.
# No range selection.
d_tensors_view_map =
       Result[DTensorsView]
  <- $origin DTensorsView
  <- $map_core DTensorsViewMapCore
```

### B.2 Entities

实体分三侧：Compute、Storage、Common；`check_constraints` 的额外检查单列在 B.2.4。Python 实现的 `<ClassName>BuilderMixin` 见附录 C。`Slot` 是逻辑概念，`Layout` 是物理概念；类型名用 `Logical` / `Physical` 区分。

#### B.2.1 Compute side

##### B.2.1.1 ComputeDTensorsView

```bash
ComputeDTensorsView =
    Object
  * $tensor_ids list[str]
    # Tensors described by this view; the rank is an API argument, not a field.
  * $d_tensor_table DTensorTable[ComputeMesh]
    # Host-side access to the local tensor ranges.
  * $data_accessor DTensorDataAccessor[RangeFromComputeView]
```

##### B.2.1.2 ComputeMesh

```bash
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

`slot`（槽位）是逻辑概念，不是文件或容器。把存储计划看成一个 `num_tensors × world_size` 的表：每一行是一个张量，每一列是一个全局 rank，每个格子是一个 slot；总 slot 数 = `num_tensors * world_size`。格子可以为空：该 rank 不持有这个张量、这个张量不在该 pp stage，或者它只存一份而其余列留空。同一行的非空格子按列顺序拼接，构成这个张量的存储区域。`StorageMesh` 的 `OuterProductMesh` 分支里，`tensor_slots` 选择行（张量），`content_slots` 选择列（slot）；`content_slots` 是 `ContentSlotsMesh`，即 `LiteralMesh["world_size"] | ArrangedMesh["world_size"]`：`LiteralMesh` 直接给出 rank 列表，`ArrangedMesh` 的 `"world_size"` 是列数。

##### B.2.2.1 StorageDTensorsView

```bash
StorageDTensorsView =
    Object
    # Tensors this view covers; a subset of the storage DTensors.
  * $tensor_ids list[str]
    # Storage DTensor table; named_d_tensors order is the header order.
  * $d_tensor_table DTensorTable[SingleTensorStorageMesh]
    # Read/write raw storage ranges; the range carries a file path and offset.
  * $data_accessor DTensorDataAccessor[RangeFromStorageView]
    # Physical file table; lookup instead of computation.
  * $physical_layout PhysicalLayoutTable
    # Create a save session. Example (safetensors): compute relative
    # data_offsets from tensor sizes/order, serialize and pad the header,
    # then data_start = 8 + header_len; write the header first and return
    # the session. Data is written later through data_accessor.
  * $begin_save (
        Result[StorageSaveSession]
      <- $self StorageDTensorsView
      <- $rank int
    )
    # Create a load session. Example (safetensors): open the file and read
    # its header before any get_file_offset call.
  * $begin_load (
        Result[StorageLoadSession]
      <- $self StorageDTensorsView
      <- $rank int
    )
```

##### B.2.2.2 PhysicalLayoutTable

```bash
ContainerFormat = "safetensors" | "distcp"

TensorComponentPos = $tensor_id str * $rank int

# Pure configuration data; produced by a business layout constructor from
# DTensorTable and the chosen topology / format.
PhysicalLayoutTable =
    Object
    # One format for every file in this plan.
  * $container_format ContainerFormat
    # TensorComponentPos -> the list that contains it; a missing key means
    # this tensor has no physical file on this rank.
  * $tensor_entries dict[TensorComponentPos, DisjointStorageMeshList]
    # leader TensorComponentPos -> physical file path.
  * $file_paths dict[TensorComponentPos, $file_path str]
  * Hint[{
    1. 系统可能会同时支持多种保存拓扑，比如每个 rank 存一个或多个文件，或者多个 rank 存一个文件，或者每个 pp stage 存一个文件，等等。
    2. 同一个 PhysicalLayoutTable 只对应一种 ContainerFormat；要混格式就拆成多份布局。
    3. 一个张量如果存在于多个 slots，那么这些 slots 逻辑上顺序拼接起来凑成整个张量的存储区域。
    4. 查表方式：先在 tensor_entries 里按 TensorComponentPos 找 disjoint_storage_meshes；没有这个键表示没有物理文件；再用 leader_tensor_component_pos_of(disjoint_storage_meshes) 在 file_paths 里找路径。
    5. file_paths 的键必须是某个 DisjointStorageMeshList 的 leader TensorComponentPos；同一个 list 只对应一个路径。
  }]

# Canonical TensorComponentPos of a DisjointStorageMeshList; used as its
# dict key.
leader_tensor_component_pos_of =
    TensorComponentPos
  <- $disjoint_storage_meshes DisjointStorageMeshList

# Path for a component pos: tensor_entries[pos] -> list -> leader ->
# file_paths. A missing key at either step returns $err with NotFoundError.
file_path_of =
    Result[str]
  <- $layout PhysicalLayoutTable
  <- $pos TensorComponentPos
```

##### B.2.2.3 DisjointStorageMeshList

```bash
DisjointStorageMeshList =
    Object
    # One logical list per physical file. The meshes inside this list are
    # disjoint, and meshes from different files are disjoint too.
  * $disjoint_storage_mesh_list list[MultiTensorsStorageMesh]
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

##### B.2.2.4 StorageMesh

```bash
SingleTensorStorageMesh = StorageMesh[ArrangedMesh["constant_one"]]
MultiTensorsStorageMesh = StorageMesh[ArrangedMesh["tensor_count"]]

# Branches are named types; no tags.
StorageMesh[TensorSlotsMeshT] =
    Oneof
  | OuterProductMesh[TensorSlotsMeshT]
  | ReshapedMesh[StorageMesh[TensorSlotsMeshT]]
  | SlicedMesh[StorageMesh[TensorSlotsMeshT]]

# The two dimensions: which tensors, which content slots.
OuterProductMesh[TensorSlotsMeshT] =
    Object
  * $tensor_slots TensorSlotsMeshT
  * $content_slots ContentSlotsMesh

# Content slot selector: an explicit rank list, or an arranged world_size.
ContentSlotsMesh =
    Oneof
  | LiteralMesh["world_size"]
  | ArrangedMesh["world_size"]
```

##### B.2.2.5 StorageSaveSession

```bash
# Interface type; a concrete session (e.g. SafeTensorsSession or
# DcpTensorsSession) implements get_file_offset / init_data_accessor /
# commit / abort.
# Save session has two terminal operations: commit (publish, then close) and
# abort (discard, then close). There is no separate finish.
# Offset, lifecycle, and a session-bound data accessor. Payload I/O goes
# through DTensorDataAccessor, not through the offset methods directly.
StorageSaveSession =
    Object
    # Actual file offset (header/alignment included); valid after begin_save,
    # before commit/abort. safetensors: 8 + header_len + relative data_offset.
    # The data_accessor uses contiguous_payload_offset, not this value.
    # Example: data_start 208 + contiguous offset 100 = actual 308.
    # Empty slots occupy 0 bytes; offset accumulation skips them.
  * $get_file_offset (
        Result[int]
      <- $self StorageSaveSession
      <- $pos TensorComponentPos
    )
    # Bind buffer I/O to this session and return the accessor. The accessor
    # maps RangeFromStorageView.contiguous_payload_offset through this
    # session's get_file_offset; callers that only hold the session use this
    # accessor instead of StorageDTensorsView.data_accessor.
  * $init_data_accessor (
        Result[DTensorDataAccessor[RangeFromStorageView]]
      <- $self StorageSaveSession
    )
    # Publish the save, then close the session.
    # Example (safetensors): fsync model.safetensors.tmp, then rename it to
    # model.safetensors. Readers see the old file or no file before commit,
    # and the complete new file after.
    # Example (distcp): write .metadata.tmp, fsync, then rename it to
    # .metadata; the presence of .metadata is the checkpoint-complete marker.
    # If the outcome is unknown, keep payloads; do not delete them.
  * $commit (
        Result[void]
      <- $self StorageSaveSession
    )
    # Discard temporary files and close the session. Do not delete payloads
    # after an uncertain commit; the final path may already exist.
  * $abort (
        Result[void]
      <- $self StorageSaveSession
    )
```

##### B.2.2.6 StorageLoadSession

```bash
# Interface type; a concrete session (e.g. SafeTensorsSession or
# DcpTensorsSession) implements get_file_offset / init_data_accessor / finish.
# Load session has one terminal operation: finish (close). It never
# publishes, so there is no commit or abort.
# Offset, lifecycle, and a session-bound data accessor. Payload I/O goes
# through DTensorDataAccessor, not through the offset methods directly.
StorageLoadSession =
    Object
    # Actual file offset (header/alignment included); valid after begin_load,
    # before finish. safetensors: 8 + header_len + relative data_offset.
    # The data_accessor uses contiguous_payload_offset, not this value.
    # Example: data_start 208 + contiguous offset 100 = actual 308.
    # Empty slots occupy 0 bytes; offset accumulation skips them.
  * $get_file_offset (
        Result[int]
      <- $self StorageLoadSession
      <- $pos TensorComponentPos
    )
    # Bind buffer I/O to this session and return the accessor. The accessor
    # maps RangeFromStorageView.contiguous_payload_offset through this
    # session's get_file_offset; callers that only hold the session use this
    # accessor instead of StorageDTensorsView.data_accessor.
  * $init_data_accessor (
        Result[DTensorDataAccessor[RangeFromStorageView]]
      <- $self StorageLoadSession
    )
    # Close the load session. Example: close file handles; the read-only
    # side has nothing to publish, so there is no commit here.
  * $finish (
        Result[void]
      <- $self StorageLoadSession
    )
```

#### B.2.3 Common

##### B.2.3.1 DTensorsView

```bash
DTensorsView =
    Oneof
  | ComputeDTensorsView
  | StorageDTensorsView
```

##### B.2.3.2 DTensorDataAccessor

```bash
DTensorDataAccessor[RangeT] =
    Object
    # Read one range owned by this view and hand each chunk to the callback.
    # RangeT is the compute-side or storage-side range; units are bytes.
  * $read_to_buffer (
         Result[void]
      <- $global_range RangeT
      <- $read_to_buffer (Result[void] <- $buffer_ptr int <- $buffer_len int)
    )
    # Ask the callback for each chunk and write it into one range owned by
    # this view. RangeT is the compute-side or storage-side range; units are bytes.
  * $write_from_buffer (
         Result[void]
      <- $global_range RangeT
      <- $write_from_buffer (Result[void] <- $buffer_ptr int <- $buffer_len int)
    )
```

##### B.2.3.3 NamedDTensor

```bash
NamedDTensor[MeshT] =
    Object
    # Stable tensor name inside its weight namespace.
  * $global_tensor_id str
    # Logical tensor view; its metadata is the global tensor metadata.
  * $d_tensor GenericDTensor[MeshT]
    # Other module paths that name the same payload.
  * $alias list[str]
```

##### B.2.3.4 GenericDTensor

```bash
GenericDTensor[MeshT] =
    Oneof
  | NaiveDTensor[MeshT]
  | ReshapedDTensor[MeshT]

# Element-order-preserving logical reshape; shape must keep the local
# ownership interval unchanged, otherwise redistribute first.
ReshapedDTensor[MeshT] =
    Object
    # Tensor being reshaped.
  * $underlying_d_tensor NaiveDTensor[MeshT]
    # Shape of the logical reshape; one entry per tensor dimension.
  * $shape list[int]
```

##### B.2.3.5 NaiveDTensor

```bash
NaiveDTensor[MeshT] =
    Object
    # Full logical tensor metadata.
  * $global_tensor_meta GlobalTensorMeta
    # One placement per mesh dimension, in mesh order.
  * $placements list[Placement]
    # Mesh this tensor is placed on.
  * $mesh MeshT
    # Underlying torch tensor for compute-side processing; nil means
    # metadata-only. Only compute-side tensors set this; storage-side tensors
    # keep it nil. Use gd_tensor_get_d_tensor / gd_tensor_reset_d_tensor.
  * $torch_d_tensor (torch.DTensor | nil)
```

##### B.2.3.6 GlobalTensorMeta

```bash
GlobalTensorMeta =
    Object
    # Shape of the full logical tensor; one entry per dimension.
  * $global_shape list[int]
    # Framework dtype name.
  * $dtype str
    # Bytes per element.
  * $itemsize int
```

##### B.2.3.7 Placement

```bash
Placement =
    Oneof
    # One contiguous logical slice along this tensor dimension.
  | $shard ($axis int)
    # Every participant holds the full tensor.
  | $replicate ()
    # Every participant holds a reduction contribution; not persistable yet.
  | $partial ()
```

##### B.2.3.8 DTensorTable

```bash
DTensorTable[MeshT] =
    Object
    # Ordered DTensors; list order is the header order.
  * $named_d_tensors list[NamedDTensor[MeshT]]
```

##### B.2.3.9 GlobalTensorMetaOf

```bash
global_tensor_meta_of[MeshT] =
    GlobalTensorMeta
  <- $named_d_tensor NamedDTensor[MeshT]
```

##### B.2.3.10 TensorOrderValueOf

```bash
# List index of tensor_id in named_d_tensors; a missing id returns $err
# with NotFoundError.
tensor_order_value_of[MeshT] =
    Result[int]
  <- $d_tensor_table DTensorTable[MeshT]
  <- $tensor_id str
```

##### B.2.3.11 DTensorsViewMapCore

```bash
# Four optional callbacks for d_tensors_view_map. Compute and storage tensor
# callbacks are separate because their tensor types differ; storage also
# converts sessions. A nil tensor callback means identity; a nil session
# callback means skip that session.
DTensorsViewMapCore =
    Object
  * $convert_each_compute_view ((
          Result[GenericDTensor[ComputeMesh]]
        <- $tensor GenericDTensor[ComputeMesh]
      ) | nil)
  * $convert_each_storage_view ((
          Result[GenericDTensor[StorageMesh]]
        <- $tensor GenericDTensor[StorageMesh]
      ) | nil)
  * $convert_each_storage_load_session ((
          Result[StorageLoadSession]
        <- $session StorageLoadSession
      ) | nil)
  * $convert_each_storage_save_session ((
          Result[StorageSaveSession]
        <- $session StorageSaveSession
      ) | nil)
```

### B.2.4 check_constraints

每个数据 `Object` / `Oneof`（session / access 接口除外）都有：

```bash
check_constraints = Result[void] <- $self T
```

`check_constraints` 先递归检查所有嵌套字段，再检查本类型自己的约束。递归规则见 5.7：`Object` 逐字段、`Oneof` 当前分支、`list` 逐元素、`dict` 逐 value、`nil` 跳过。表里的“额外检查”是在递归之外加的；未列出的类型只做递归，不做额外检查。类型定义里的 `Assert` 由 `check_constraints` 执行；`Hint` 只是说明，不执行。

| 类型 | 额外检查（递归之外） |
|---|---|
| `ComputeDTensorsView` | `tensor_ids` 与 `d_tensor_table` 的顺序一致；`data_accessor` 非空 |
| `StorageDTensorsView` | `tensor_ids` 与 `d_tensor_table` 的顺序一致；`physical_layout` 的 entry 的 `tensor_id` 在 table 里；`data_accessor` / `begin_save` / `begin_load` 非空 |
| `DTensorTable` | `global_tensor_id` 不重复 |
| `NamedDTensor` | `global_tensor_id` 非空 |
| `NaiveDTensor` | `global_tensor_meta` / `placements` / `mesh` 一致；`torch_d_tensor` 可空 |
| `GlobalTensorMeta` | `global_shape` 非空；`itemsize > 0`；`dtype` 非空 |
| `PhysicalLayoutTable` | `container_format` 唯一；`file_paths` 的键是 leader pos；entry 的 leader 在 `file_paths` 里；mesh 两两不重叠 |
| `DisjointStorageMeshList` | list 非空；mesh 两两不重叠 |
| `StorageMesh` | `content_slots` 的 `ArrangedMesh` 的 size > 0 |
| `ComputeMesh` | `ArrangedMesh` 的 size > 0 |
| `LiteralMesh` | `ranks` 非空；每个 rank 满足 `0 <= rank < max`；`ranks` 无重复 |
| `ConstraintViolationError` / `CheckFailedError` / `NotFoundError` / `InvalidArgumentError` / `GDTensorIOError` | `msg` 非空 |

`check_constraints` 不替代 `check_*`：前者递归检查对象自身和所有嵌套数据；后者检查一次 `CopyPlan` 的 disjoint / covering。

例如：`ComputeDTensorsView.check_constraints` 递归到 `d_tensor_table`，再逐层到 `NamedDTensor` / `GenericDTensor` / `NaiveDTensor` / `GlobalTensorMeta`；`PhysicalLayoutTable.check_constraints` 递归到每个 `DisjointStorageMeshList` 和其中的 `StorageMesh`。父对象的实现不能只查自己的字段。

Python 实现里，`check_constraints` 在数据类 `Foo` 上；build 接口在 `FooBuilderMixin`；见附录 C。

### B.3 Common types

#### B.3.1 Ranges

```bash
# A range on either side of a copy: a compute view or a storage view.
DTensorRange =
    Oneof
  | RangeFromComputeView
  | RangeFromStorageView

# One source-to-destination copy; each side is a DTensorRange.
CopyPlan = list[(
  $src DTensorRange,
  $dst DTensorRange
)]

RangeFromComputeView =
    Object
  * $global_tensor_id str
  * $global_tensor_offset int
  * $size int

RangeFromStorageView =
    Object
  * $file_path str
    # Offset in the contiguous payload region of this file: all payloads
    # are assumed contiguous, and empty slots occupy 0 bytes. It is derived
    # from slots, global meta and rank; it is not the file offset (header
    # and alignment are not included).
    # Example: A0=100, A1=empty, A2=100, B0=200 in one file gives
    #   contiguous offsets A0=0, A2=100, B0=200; A1 has no range.
    #   With data_start=208, actual file offsets are 208, 308, 408.
  * $contiguous_payload_offset int
  * $size int
```

#### B.3.2 Meshes

```bash
ArrangedMesh[tag_name] =
    Object
    # Number of participants on this mesh.
  * $size tagged[tag_name, int]

# Explicit rank list; used when a device mesh has no regular arranged form.
# max is tagged by the dimension it bounds: world_size, tensor_count, ...
LiteralMesh[tag_name] =
    Object
    # Ranks covered by this mesh, in order.
  * $ranks list[int]
    # Exclusive upper bound for every rank in ranks.
  * $max tagged[tag_name, int]
  * Assert[{
      forall x in ranks:
        check x >= 0 and x < self.max
      check len(ranks) == len(set(ranks))
  }]

ReshapedMesh[T] =
    Object
    # Mesh before reshaping.
  * $underlying_mesh T
    # Shape of the mesh after reshaping; one entry per dimension.
  * $shape list[int]

PermutedMesh[T] =
    Object
    # Mesh before permuting dimensions.
  * $underlying_mesh T
    # New order of mesh dimensions.
  * $permutation list[int]

SlicedMesh[T] =
    Object
    # Mesh before fixing dimensions.
  * $underlying_mesh T
    # One entry per underlying mesh dimension; nil keeps that dimension.
  * $slices list[(Slice | nil)]

Slice =
    Object
    # First index selected on this mesh dimension; nil starts at zero.
  * $start (int | nil)
    # Index after the last selected one; nil runs to the end.
  * $stop (int | nil)
    # Step between selected indices; nil means one.
  * $step (int | nil)
```

#### B.3.3 Error 与 Result

`Error` 是顶层类型，所有 `Result` 的 `$err` 分支都用它，不用裸 `str`。每个分支都是一个 Object，至少带 `msg`；调用方按分支类型处理，不解析 `msg`。

```bash
# Failure reported by any Result-returning API. Every branch is a named
# Object, so the branch type is the tag; callers switch on the value type.
Error =
    Oneof
  | ConstraintViolationError
  | CheckFailedError
  | NotFoundError
  | InvalidArgumentError
  | GDTensorIOError

# check_constraints rejected the data.
ConstraintViolationError =
    Object
    # Type name of the object whose check_constraints failed, e.g.
    # "PhysicalLayoutTable".
  * $type_name str
    # Name of the constraint that failed on that object.
  * $constraint_name str
    # Human-readable reason.
  * $msg str

# A check_* validation failed: not disjoint, not covering, or overlapping
# storage meshes.
CheckFailedError =
    Object
    # Name of the check that failed.
  * $check_name str
    # Human-readable reason.
  * $msg str

# A lookup by tensor id, dict key, or file path found nothing.
NotFoundError =
    Object
    # Field or table being looked up, e.g. "tensor_id", "tensor_entries",
    # "file_paths".
  * $key_name str
    # The missing value, in its canonical form.
  * $key str
    # Human-readable reason.
  * $msg str

# A caller-supplied argument does not satisfy the documented contract.
InvalidArgumentError =
    Object
    # Argument or precondition that failed, e.g. "tensor_ids".
  * $argument_name str
    # Human-readable reason.
  * $msg str

# File or buffer I/O failed.
GDTensorIOError =
    Object
    # File path for file I/O; nil for in-memory buffer I/O.
  * $file_path (str | nil)
    # Operation that failed, e.g. "open", "read", "write", "fsync", "rename".
  * $operation str
    # Human-readable reason.
  * $msg str
```

各分支的产生方：

| 分支 | 产生方 |
|---|---|
| `ConstraintViolationError` | 所有 `check_constraints`（B.2.4） |
| `CheckFailedError` | B.1.3 的 `check_*`：不 disjoint、不 covering、storage mesh 重叠 |
| `NotFoundError` | `compute_d_tensors_view_get_gd_tensor` 的 id、`tensor_order_value_of` 的 id、`file_path_of` 的 `tensor_entries` / `file_paths` 键 |
| `InvalidArgumentError` | `d_tensors_view_rank_plan_reshard` 的 `Assert` 失败；session 在生命周期之外被调用（`begin_*` 之前，或 `commit` / `abort` / `finish` 之后） |
| `GDTensorIOError` | `begin_save` / `begin_load` / `commit` / `abort` / `finish` / `init_data_accessor` 的文件操作，以及 accessor 的 buffer 读写 |

```bash
Result[T] = $ok T | $err Error
```

## 附录 C Python 实现与代码生成指南

### C.1 Builder Mixin

数据类和 build 接口分开。每个数据类 `Foo` 的 build 接口放在 `FooBuilderMixin`，数据类写成：

```python
class Foo(FooBuilderMixin):
    def check_constraints(self) -> Result[None]:
        ...
```

`FooBuilderMixin` 提供：

- `get_<field>` / `set_<field>`；
- `add_<field>`（list）；
- `mutable_<field>`（嵌套对象 / map）；
- `clear_<field>`（可空字段 / oneof）。

不生成 `has_*` / `which_*`。判空用 `get_xxx() is None`；判分支用 `type(get_xxx())`，因为每个 tag 是一个独立的类（见 C.2）。

`FooBuilderMixin` 不持有数据，只读写 `self.<field>`；`new_foo()` 返回 `Foo()`。`check_constraints` 不放进 Mixin，留在 `Foo` 上，因为它检查数据，不是 build 接口。生成器为每个数据类生成一个 `<ClassName>BuilderMixin`，不要让数据类直接定义这些 build 方法。`session` / `access` 是抽象基类，不走这套 Mixin / check_constraints 约定（见 C.2）。

例如：

```python
class ComputeDTensorsViewBuilderMixin:
    def add_tensor_id(self, value: str) -> None: ...
    def set_data_accessor(self, value: DTensorDataAccessor) -> None: ...
    def mutable_d_tensor_table(self) -> DTensorTable: ...

class ComputeDTensorsView(ComputeDTensorsViewBuilderMixin):
    def check_constraints(self) -> Result[None]:
        ...
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
ContainerFormat = str          # 当前是 "safetensors" / "distcp"；新增格式不改这个类型
```

tuple 也直接写 Python 的 `tuple[...]`：

```python
CopyPlan = list[tuple[DTensorRange, DTensorRange]]
```

sum type（viba `Oneof`）统一映射成 `types.UnionType`。带 tag 的分支在 `gdtensor.tag` 里各有一个类：类名是 tag 名的 PascalCase，payload 是泛型参数。

```python
DTensorsView = ComputeDTensorsView | StorageDTensorsView
DTensorRange = RangeFromComputeView | RangeFromStorageView
GenericDTensor = NaiveDTensor | ReshapedDTensor
Placement = tag.Shard[int] | tag.Replicate | tag.Partial
Error = (
    ConstraintViolationError | CheckFailedError | NotFoundError
    | InvalidArgumentError | GDTensorIOError
)
```

情况分两类：分支 payload 是独立命名类型的（`DTensorsView`、`DTensorRange`、`GenericDTensor`、`Error`、mesh 各 Oneof）直接用分支自己的类，不带 tag；payload 是 `int` / void / 泛型 `T` 的（`Placement`、`Result`）必须带 tag，tag 类和 `tagged[...]` 字段都进 `gdtensor.tag`。

mesh 的 Oneof 因此就是分支类型的 union，例如 `type ComputeMesh = LiteralMesh[tag.WorldSize[int]] | ArrangedMesh[tag.WorldSize[int]] | ReshapedMesh[ComputeMesh] | PermutedMesh[ComputeMesh] | SlicedMesh[ComputeMesh]`。

viba 的 `tagged[tag_name, T]`（不是 Oneof，是带名 tag 的标量）也进 `gdtensor.tag`：`tagged["world_size", int]` 对应 `tag.WorldSize[int]`，值是 `tag.WorldSize(4)`。`ArrangedMesh` 的 `$size` 和 `LiteralMesh` 的 `$max` 都是这种用法：`$size tagged["world_size", int]` 写成 `size: tag.WorldSize[int]`，`LiteralMesh["world_size"]` 写成 `LiteralMesh[tag.WorldSize[int]]`。

tag 类不逐个手写：一个 `Box` 基类加一个 `make_box_subclass`，每个 tag 一行。

```python
import types

class Box:
    # 一个 tag：tag 是类本身，payload 挂在 value 上。
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
Ok = make_box_subclass("Ok")
Err = make_box_subclass("Err")
Replicate = make_box_subclass("Replicate")
Partial = make_box_subclass("Partial")
```

`Box` 提供构造、`value`、`__match_args__` 和 `__class_getitem__`；`make_box_subclass` 用 `type` 造子类，每个 tag 一行，子类不再写任何东西。`tag` 里只多一个基类和一个工厂，不随着 tag 数量增长。Oneof 的分支 tag 和 `tagged[tag_name, T]` 字段都走这一套。

tag 类同时支持构造和参数化：

```python
tag.Ok(3)                    # 值：成功，payload 是 3
tag.Ok[int]                  # 注解：成功，payload 是 int
tag.Err(NotFoundError(...))  # 值：失败
tag.Err[Error]               # 注解：失败，payload 是 Error
tag.Replicate()              # 值：void payload，value 是 None
```

payload 挂在 `value` 上：`tag.Ok(3).value == 3`。所有 tag 都有 `value`；void payload 的 tag（`tag.Replicate`、`tag.Partial`）的 `value` 是 `None`，和 `tag.Ok(None)` 靠类区分。

判别一律用 `type(x)`：`type(view) is ComputeDTensorsView`、`type(x) is tag.Shard`、`type(placement) is tag.Replicate`。`get_xxx()` 返回哪一类就说明是哪一支；`tag.Ok` 的 payload 里是计算结果，`tag.Err` 的 payload 里是 `Error` 的某个分支类。`isinstance(x, tag.Ok[int])` 不行（参数化泛型不能给 `isinstance`），所以判分支一律用 `type(x) is`。

`Result[T]` 在顶层定义：

```python
type Result[T] = tag.Ok[T] | tag.Err[Error]
```

这是 PEP 695 的 `type` 别名，需要 Python 3.12+；`Result[int]` 能直接当注解用。它是类型别名，不是类，所以不能写 `isinstance(x, Result)`，判别仍用 `type(x)`。`Result[void]` 写成 `Result[None]`。用法：

```python
result = gdtensor.io.rank_load(rank, compute_view, storage_view)
match result:
    case tag.Ok(value):
        ...
    case tag.Err(NotFoundError() as error):
        ...
```

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

每个方法都返回 `Result[...]`：成功是 `tag.Ok(payload)`，失败是 `tag.Err(某个错误类)`。`get_file_offset` 的 payload 是文件偏移 `int`，`init_data_accessor` 的 payload 是绑定到该 session 的 `DTensorDataAccessor[RangeFromStorageView]`，`commit` / `abort` / `finish` 的 payload 是 `None`。

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
|  | `gdtensor.get_gd_tensor` | `compute_d_tensors_view_get_gd_tensor` |
|  | `gdtensor.make_gd_tensor` | `gd_tensor_make` |
|  | `gdtensor.get_d_tensor` | `gd_tensor_get_d_tensor` |
|  | `gdtensor.reset_d_tensor` | `gd_tensor_reset_d_tensor` |
|  | `gdtensor.Result` | `Result` |
|  | `gdtensor.Error` | `Error` |
| `io` | `gdtensor.io.rank_load` | `d_tensors_view_rank_load` |
| `io` | `gdtensor.io.rank_save` | `d_tensors_view_rank_save` |
| `io` | `gdtensor.io.rank_reshard` | `d_tensors_view_rank_reshard` |
| `builder` | `gdtensor.builder.d_tensors_view_map` | `d_tensors_view_map` |
| `builder` | `gdtensor.builder.DTensorsView` | `DTensorsView` |
| `builder` | `gdtensor.builder.ComputeDTensorsView` | `ComputeDTensorsView` |
| `builder` | `gdtensor.builder.StorageDTensorsView` | `StorageDTensorsView` |
| `builder` | `gdtensor.builder.DTensorsViewMapCore` | `DTensorsViewMapCore` |
| `builder` | `gdtensor.builder.DTensorTable` | `DTensorTable` |
| `builder` | `gdtensor.builder.NamedDTensor` | `NamedDTensor` |
| `builder` | `gdtensor.builder.GenericDTensor` | `GenericDTensor` |
| `builder` | `gdtensor.builder.NaiveDTensor` | `NaiveDTensor` |
| `builder` | `gdtensor.builder.ReshapedDTensor` | `ReshapedDTensor` |
| `builder` | `gdtensor.builder.GlobalTensorMeta` | `GlobalTensorMeta` |
| `builder` | `gdtensor.builder.Placement` | `Placement` |
| `builder` | `gdtensor.builder.ComputeMesh` | `ComputeMesh` |
| `builder` | `gdtensor.builder.StorageMesh` | `StorageMesh` |
| `builder` | `gdtensor.builder.OuterProductMesh` | `OuterProductMesh` |
| `builder` | `gdtensor.builder.ContentSlotsMesh` | `ContentSlotsMesh` |
| `builder` | `gdtensor.builder.LiteralMesh` | `LiteralMesh` |
| `builder` | `gdtensor.builder.ArrangedMesh` | `ArrangedMesh` |
| `builder` | `gdtensor.builder.ReshapedMesh` | `ReshapedMesh` |
| `builder` | `gdtensor.builder.PermutedMesh` | `PermutedMesh` |
| `builder` | `gdtensor.builder.SlicedMesh` | `SlicedMesh` |
| `builder` | `gdtensor.builder.Slice` | `Slice` |
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
| `tag` | `gdtensor.tag.Ok` | `$ok` |
| `tag` | `gdtensor.tag.Err` | `$err` |
| `tag` | `gdtensor.tag.Shard` | `$shard` |
| `tag` | `gdtensor.tag.Replicate` | `$replicate` |
| `tag` | `gdtensor.tag.Partial` | `$partial` |
| `tag` | `gdtensor.tag.ConstantOne` | `tagged["constant_one", int]` |
| `tag` | `gdtensor.tag.TensorCount` | `tagged["tensor_count", int]` |
| `tag` | `gdtensor.tag.WorldSize` | `tagged["world_size", int]` |

每个数据类 `Foo` 都在 `gdtensor.builder` 提供 `FooBuilderMixin`。例如：

- `gdtensor.builder.DTensorsViewMapCoreBuilderMixin`
- `gdtensor.builder.ComputeDTensorsViewBuilderMixin`
- `gdtensor.builder.StorageDTensorsViewBuilderMixin`
- `gdtensor.builder.DTensorTableBuilderMixin`
- `gdtensor.builder.PhysicalLayoutTableBuilderMixin`

`gdtensor.tag` 里的类是一对一翻译过来的 tag：类名是 tag 名的 PascalCase，payload 是泛型参数；`gdtensor.Result` 是顶层别名 `type Result[T] = tag.Ok[T] | tag.Err[Error]`；`gdtensor.Error` 是五个错误类的 union，放顶层方便调用方按 `Error` 做分类。tag 在 viba 里不是类型名，表里 `full name` 列写的是对应的 viba 写法：Oneof 分支的 tag，或者 `tagged[...]` 的名字。`check_constraints` 在 `Foo` 上，不在 Mixin；`new_foo()` 也在 `gdtensor.builder`。`DTensorsView` / `GenericDTensor` / `ComputeMesh` / `StorageMesh` / `ContentSlotsMesh` / `Error` / `Placement` 是 `types.UnionType`，`Result` 是 `type` 别名，都没有 BuilderMixin；`ContainerFormat` 是类型别名，也没有 BuilderMixin；`TensorComponentPos` 和五个 `*Error` 是 product，按类处理。`session` / `access` 是抽象基类，不放进 builder，不套 BuilderMixin / check_constraints；它们的 Python 形态见 C.2。

### C.4 Helper functions

helpers 放已有的 `builder` namespace，不新增 namespace，也不放顶层。它们不是主要 API，所以单列，不并进 C.3 的表。

| namespace | Python path | full name |
|---|---|---|
| `builder` | `gdtensor.builder.global_tensor_meta_of` | `global_tensor_meta_of` |
| `builder` | `gdtensor.builder.tensor_order_value_of` | `tensor_order_value_of` |
| `builder` | `gdtensor.builder.file_path_of` | `file_path_of` |
| `builder` | `gdtensor.builder.leader_tensor_component_pos_of` | `leader_tensor_component_pos_of` |
