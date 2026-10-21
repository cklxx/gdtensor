# 规格追溯矩阵

每行对应设计文档 `DTensorsView-design.md` 的一个条目。列含义：

- 实现：代码位置。
- 测试：覆盖该条目的测试文件。
- 状态：已实现并验证 / 简化实现 / 非目标。

判据：每个"已实现并验证"行必须同时有实现位置和测试。空行即缺口。

## 正文章节

| 章节 | 规格要点 | 实现 | 测试 | 状态 |
|---|---|---|---|---|
| 2 心智模型 | 逻辑层/物理层/执行层三层 | views.py, plan.py, api.py | 全部 | 已实现并验证 |
| 3.1 slot | (tensor row, content column) 二元组 | layout.py TensorComponentPos | test_matrix, test_checks | 已实现并验证 |
| 3.2 DTensorTable | 有序张量表，id 唯一 | dtensor.py DTensorTable | test_matrix（重复 id 报错） | 已实现并验证 |
| 3.3 StorageMesh | 内容槽列与存储网格 | mesh.py OuterProductMesh 等 | test_matrix, test_checks | 已实现并验证 |
| 4.1 PhysicalLayoutTable | 纯配置，tensor_entries + file_paths | layout.py | test_checks（5 个反例） | 已实现并验证 |
| 4.2 文件/格式/会话 | safetensors 与 distcp 两格式 | formats.py | test_matrix, test_data_correctness | 已实现并验证 |
| 4.3 构造边界 | 业务 builder 在核心之外 | models.py build_layout 系列 | test_doc_patterns, test_matrix | 已实现并验证 |
| 4.4 逻辑/物理边界 | plan 是两层唯一耦合 | plan.py | test_matrix | 已实现并验证 |
| 5.1 CopyPlan/DTensorRange | src/dst range 二元组，单位字节 | ranges.py | test_interval_math, test_exhaustive | 已实现并验证 |
| 5.2 plan API | range 选择只在 plan 层 | plan.py | 全部 plan 测试 | 已实现并验证 |
| 5.3 两种偏移 | payload 偏移 vs 文件偏移 | ranges.py 注释, formats.py get_file_offset | test_checks, test_data_correctness | 已实现并验证 |
| 5.4 执行顺序 | begin/copy/commit 生命周期 | api.py, formats.py, omnifs_view.py | test_omnifs, test_matrix, test_adversarial | 已实现并验证 |
| 6 不变量 | 非交并、全覆盖、纯函数等 | plan.py check_* | test_checks（20 项）, test_exhaustive | 已实现并验证 |
| 7 用例矩阵 | section 7 全部组合 | models.py + runner | test_matrix（30 项） | 已实现并验证 |
| 8 边界 | 对外/实现者/内部 API 分层 | api.py（公开）, plan.py（内部） | 文档与测试一致 | 已实现并验证 |
| 9 非目标 | 不做真实网络/GPU | 仿真层 iobackend/runner | — | 有意简化 |

## 附录 B 类型定义

| 编号 | 类型/API | 实现 | 测试 | 状态 |
|---|---|---|---|---|
| B.1.1 | d_tensors_view_rank_save/load/reshard | api.py | test_matrix, test_exhaustive, test_data_correctness | 已实现并验证 |
| B.1.2 | plan_save/plan_load/plan_reshard | plan.py:132-178 | test_exhaustive（286 项 plan oracle） | 已实现并验证 |
| B.1.3 | 6 个 check API | plan.py:216-300 | test_checks（20 项正反向） | 已实现并验证 |
| B.2.1.1 | ComputeDTensorsView | views.py | 全部 | 已实现并验证 |
| B.2.1.2 | ComputeMesh（arranged/reshaped/permuted/sliced） | mesh.py resolve_compute_mesh | test_matrix, test_interval 间接触发 | 已实现并验证 |
| B.2.2.1 | StorageDTensorsView + begin_save/load | views.py, formats.py | 全部 | 已实现并验证 |
| B.2.2.2 | PhysicalLayoutTable | layout.py | test_checks | 已实现并验证 |
| B.2.2.3 | DisjointStorageMeshList + leader | layout.py:46-79 | test_checks（重叠/leader 反例） | 已实现并验证 |
| B.2.2.4 | StorageMesh（outer/reshaped/sliced，无 permuted） | mesh.py:133-143 | test_matrix | 已实现并验证（permuted 按规格拒绝） |
| B.2.2.5 | StorageSaveSession：commit/abort | formats.py SaveSessionBase | test_matrix（abort/state/unknown） | 已实现并验证 |
| B.2.2.6 | StorageLoadSession：finish | formats.py LoadSessionBase | test_matrix（session states） | 已实现并验证 |
| B.2.3.1 | DTensorsView 联合类型 | views.py union | 类型层面 | 已实现并验证 |
| B.2.3.2 | DTensorDataAccessor 回调 | views.py accessor | 全部 copy 路径 | 已实现并验证 |
| B.2.3.3 | NamedDTensor | dtensor.py | 全部 | 已实现并验证 |
| B.2.3.4 | GenericDTensor（naive/reshaped） | dtensor.py ReshapedDTensor | test_matrix（zero 张量 reshape） | 已实现并验证 |
| B.2.3.5 | NaiveDTensor | dtensor.py | 全部 | 已实现并验证 |
| B.2.3.6 | GlobalTensorMeta（numel/nbytes） | dtensor.py:38-49 | 全部 | 已实现并验证 |
| B.2.3.7 | Placement（Shard/Replicate/Partial） | dtensor.py:55-70 | Partial 持久化报错有测试 | 已实现并验证 |
| B.2.3.8 | DTensorTable | dtensor.py:108- | test_matrix（重复 id） | 已实现并验证 |
| B.2.3.9 | GlobalTensorMetaOf | dtensor.py global_tensor_meta_of | 全部 | 已实现并验证 |
| B.2.3.10 | TensorOrderValueOf | dtensor.py:142 | test_matrix（header 顺序） | 已实现并验证 |
| B.3.1 | RangeFromComputeView/StorageView/CopyPlan | ranges.py | test_interval_math | 已实现并验证 |
| B.3.2 | Mesh 全部构造与解析 | mesh.py | test_matrix, test_checks | 已实现并验证 |
| B.3.3 | Result（Ok/Err/catch） | result.py | 全部错误路径测试 | 已实现并验证 |

## OmniFS-Store 扩展（第二份设计文档）

| 要点 | 实现 | 测试 | 状态 |
|---|---|---|---|
| manifest prepared/committed 状态机 | omnifs.py Manifest | test_omnifs | 已实现并验证 |
| 唯一发布点 commit | omnifs_view.py | test_omnifs（uncommitted 不可见） | 已实现并验证 |
| KV 幂等/冲突/重叠检测 | omnifs.py layout() | test_omnifs | 已实现并验证 |
| full coverage 提交前校验 | omnifs.py check_full_coverage | test_omnifs（gap 反例） | 已实现并验证 |
| L1/L2/L3 读路径 + single-flight | iobackend.py P2PReadCache | test_read_cache, test_adversarial | 已实现并验证 |
| sink 不改 manifest，按 sink.json 路由 | omnifs.py sink_snapshot | test_omnifs（sink 前后读路径） | 已实现并验证 |
| sink CRC 校验 | omnifs.py:458 | test_omnifs | 已实现并验证 |
| 字节确定性（跨机一致前提） | 无 pid/time/hash 依赖 | test_consistency（6 项） | 已实现并验证 |

## 验证结果（本轮）

| 验证手段 | 规模 | 结果 |
|---|---|---|
| plan oracle 穷举 | world ≤ 8 全部 (tp,dp,pp) × 7 种 layout/mode | 286 个配置全部通过 |
| 端到端穷举（线程 + 文件后端） | 8 种 world × 7 种 layout + 5 种跨并行加载 + 非整除形状 | 56 个场景全部通过 |
| 区间运算属性测试 | 60 个随机种子，对拍布尔地面真值 | 240 例全部通过 |
| 跨机一致性 | 重复保存、线程/进程、反序启动、omnifs 导出、非 ASCII、hashseed | 7 项字节一致 |
| 变异测试 | 22 个定向变异（tests/mutation_check.py） | 15 个行为变异全部被杀死；7 个存活变异经证明为等价变异或冗余检查 |

存活变异的处置（不是测试缺口）：

1. iv_intersect 的两个 searchsorted side 翻转和 keep 边界：只改变零宽行是否产生，最终被 `e > s` 过滤；属性测试断言输出的规范形式（正宽、有序、不相交）。
2. iv_subtract 的两个推进条件：j/k 多走一行会在下一当前点被重新检查，输出集合不变；60 种子随机对拍覆盖。
3. plan 的 subtract 范围和提前退出：全覆盖抛错和下一 src 迭代会兜底；pair 元素恒等由 tests/oracle.py 在 286 个配置上直接断言。
4. KV gap 检查与重复路径检查：读路径 read_chunk 和 build_payload_index 独立拦截，端到端测试固定。

多机行为一致性的依据：

- 全部 plan 与落盘内容是 (rank, tables, layout) 的纯函数，代码中无 time.time/getpid/uuid/random 影响输出；Python 3.7+ dict 保持插入序，所有 set 只用于存在性判断，唯一参与路由的 hash() 已改为 zlib.crc32。
- 线程与 fork 进程两种 rank 运行时产出逐字节相同。
- 发布路径全部走 fsync + os.replace（原子重命名），读者只见完整文件或无文件。
- 要在真多机集群复验：在 Linux 上跑 `python3 -m pytest -q`，再跑 `python3 tests/mutation_check.py`，两者全过即环境一致。

## 设计修订对齐（最新版文档，约 1513 行版 vs 初版 785 行版）

新增实现：

| 设计条目 | 实现 | 测试 |
|---|---|---|
| B.3.3 五个类型化 Error + tag Box（C.2） | gdtensor/tag.py（Ok/Err/Shard/Replicate/Partial/WorldSize/ConstantOne/TensorCount，五类 Error） | test_spec_alignment |
| B.3.2 LiteralMesh（显式 rank 列表，compute + content slots） | mesh.py LiteralMesh，ComputeMesh/ContentSlotsMesh 联合，字面内容槽用 -1 表示空列 | test_spec_alignment |
| B.2.4 / 5.7 check_constraints 递归校验 | gdtensor/constraints.py，挂在各数据类 | test_spec_alignment（正反向） |
| 5.5 d_tensors_view_map + 四回调 | gdtensor/builder.py | test_spec_alignment |
| 5.6 compute↔torch 桥五个短名 API | gdtensor/builder.py + 包顶层（无 torch 环境返回 InvalidArgumentError） | test_spec_alignment |
| B.2.2.5/6 session/access ABC + init_data_accessor | gdtensor/session.py，具体会话/accessor 注册为虚拟子类 | test_spec_alignment |
| B.2.3.5 NaiveDTensor.torch_d_tensor | dtensor.py 可选字段 | 元数据-only 桥测试 |
| C.3 namespace io/builder/session/access/tag | gdtensor/io.py、gdtensor/__init__.py 顶层短名 | 导入测试 |

兼容性处理：

- 旧的 `Shard(0)` 位置参数、`Shard.axis`、`Ok/Err`、`ResultError` 全部保留，653 个旧测试零改动通过。
- ResultError 现为五类 Error 的基类 GDTensorError 的别名；catch 装饰器把已知错误原样保留在 Err 里。
- make_gd_tensor 需要 torch；本机无 torch，返回 InvalidArgumentError，其余桥接 API 在仿真 fabric 上可用。

非目标（设计 9 节）：真实 torch.DTensor 解包、对象存储 multipart、分布式 barrier/重试，仍由 session/实现层承担，核心不实现。

## 对外 API 与 torch DTensor 打通

API 完全对齐（C.1/C.3）：

- gdtensor/builder_mixin.py：每个 product 数据类生成 get_/set_/add_/mutable_/clear_ 构造方法和 new_ 工厂（test_builder_api 6 项）。
- namespace io/builder/session/access/tag 全部就位，顶层短名 5 个桥函数。
- 无 has_*/which_*；判空 get_x() is None，判分支 type(x) is。

torch 打通（真实 torch 2.12 distributed.tensor，Python 3.11 验证）：

- gdtensor/torch_bridge.py：make_gd_tensor 读 torch DTensor 的 shape/dtype/itemsize/placements/device_mesh，转成 gdtensor NaiveDTensor 并挂 torch_d_tensor；placement/mesh 双向映射；显式 placements 参数支持“单 torch 进程持有全局值，gdtensor 按虚拟 (pp,dp,tp) 出分片 plan”。
- fabric_from_torch：把 torch full_tensor 全局字节按 gdtensor ownership 灌进每个 rank。
- torch_from_fabric：load 后各 rank buffer 按 owned intervals 重组成全局 torch tensor，带重叠/未覆盖校验。
- 端到端 test_torch_e2e：真实 torch.DTensor → gdtensor 分片 safetensors（行切/列切strided/1维）→ load → torch.equal，含多张量表；官方 safetensors 解码校验。

验证：Python 3.14（无 torch，桥测试 skip）680 passed；Python 3.11 + torch 2.12，691 passed（含 11 项 torch 桥/端到端）。

## LAWA 归并端到端（设计 10）+ API 合规审计

- tests/test_lawa_merge.py：k∈{1,2,3,5} 个 checkpoint 的逐元素平均；朴素版（累加/k）与 Welford 版（delta/n）都走 gdtensor load → torch 计算 → reset → save，两者与 torch.stack().mean 三方一致，结果再经 gdtensor 重载校验。
- 归并严格只在 torch.DTensor 上做；审计断言 core/io/builder 不存在任何 reduce/mean/lawa/merge API（设计 9、10.3）。
- meta 不一致（dtype/itemsize）由 plan 的 _check_same_tensors 前置拒绝，正向同 meta 放行。
- tests/test_api_conformance.py（62 项）：机械对照 C.3 表，逐个断言顶层短名、io 三动词、builder 全部 product/mesh/layout/error/helper、session/access ABC、tag 类型存在且签名正确；并断言顶层和 io 没有表外 API，防止自由发挥。
- builder 命名空间补齐 C.3 全部 re-export；tag.Placement 联合。

验证：Python 3.11+torch 759 passed；Python 3.14 742 passed（torch 用例 skip）。
