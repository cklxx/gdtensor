# gdtensor

结论：gdtensor 是通用分布式张量库，严格实现 DTensorsView 设计的抽象。三层为逻辑层（`DTensorTable`、`NamedDTensor`、mesh）、物理层（`PhysicalLayoutTable`、`DisjointStorageMeshList`、`ContainerFormat`）、执行层（`DTensorRange`、`CopyPlan`、plan、session、accessor）。

公开 API 以设计附录 C.3 / C.4 为准，未列出的符号均为内部实现（模块名带 `_` 前缀）。

## 公开 API

| namespace | 内容 |
|---|---|
| `gdtensor` | `get_tensor_ids` `get_gd_tensor` `make_gd_tensor` `get_d_tensor` `reset_d_tensor` `Result` `Error` `Ok` `Err` |
| `gdtensor.io` | `rank_load` `rank_save` `rank_reshard` |
| `gdtensor.builder` | view / table / tensor / mesh / layout 产品类型、五个 typed error、`d_tensors_view_map` 与四个 C.4 helper |
| `gdtensor.session` | `StorageSaveSession` `StorageLoadSession` |
| `gdtensor.access` | `DTensorDataAccessor` |
| `gdtensor.tag` | `Ok` `Err` `Shard` `Replicate` `Partial` `ConstantOne` `TensorCount` `WorldSize` `Box` |

不在公开 API：业务模型构造（`gdtensor._models`）、仿真 CLI（`gdtensor._sim`）、多 rank runner、文件 IO、plan/check、区间与所有权运算、torch 虚拟拓扑辅助 `gd_for_layout`。

## 文件格式

设计只定义两种格式实现，核心类型不出现格式分支：

- safetensors：header 先写，data 后写；`commit` 做 fsync + 原子 rename。多 rank 合写一个文件时，并发写区间互不重叠，barrier 后由文件 leader 发布。
- distcp：payload 先写，`.metadata` 最后发布；`.metadata` 存在即完成标记。

save 终态为 commit 或 abort；load 只有 finish。commit 结果未知时保留 payload，abort 不删除最终路径。

## 仿真 CLI（内部工具）

```bash
python3 -m gdtensor._sim --root /tmp/ck --tp 4 --dp 4 --pp 2 --optimizer \
  --layout hf --format safetensors
```

输出目录含 `report.json`、`write_trace.csv`、`read_trace.csv`。列：`t_us, rank, op, file, offset, size`。

## 测试

| 测试文件 | 覆盖 |
|---|---|
| `test_checks.py` | B.1.3 disjoint / covering 正反例、`check_physical_layout` |
| `test_matrix.py` | 设计第 7 节用例矩阵：每 rank 文件、多 rank 合写、一空槽、跨文件、reshard、生命周期 |
| `test_exhaustive.py` | world≤8 共 286 种配置穷举 |
| `test_adversarial.py` | 空张量、标量、全 replicate、单 rank、reshard 自拷贝、abort 后重写 |
| `test_data_correctness.py` | safetensors CRC、arrange 张量在不同并行度下往返 |
| `test_consistency.py` | plan 纯函数、线程/进程字节一致、反序启动、PYTHONHASHSEED 跨进程一致 |
| `test_spec_alignment.py` | C.2/C.3/B.2.4 产品类、tag、约束递归、view map、桥、session/access ABC |
| `test_api_conformance.py` | C.3 符号与签名机械审计 |
| `test_builder_api.py` | C.1 BuilderMixin get/set/add/mutable/clear |
| `test_torch_bridge.py` `test_torch_e2e.py` `test_lawa_merge.py` | 真实 torch.DTensor（2.12+）桥与 LAWA，门控 `GDTENSOR_TORCH_TESTS=1` |
| `tests/mutation_check.py` | 核心区间/plan/layout 变异，13 个有效变异全部被杀 |

形式化验证：`formal/tla/CopyPlan.tcl`（TLC）建模 save/load/reshard 的非交并覆盖与原子发布；`formal/lean/ShardAlgebra.lean` 机器证明分片代数。

## 边界

- 不在核心做 payload 序列化：read/write 是 raw bytes。
- 不提供 reduce API：归约、LAWA 平均在 `torch.DTensor` 上做（设计 9 / 10）。
- 不实现对象存储 multipart、分布式 barrier、重试：这些留给 session 实现层（本仓库的本地文件 session 自带线程/进程 barrier）。
- 不决定布局策略：`PhysicalLayoutTable` 由业务构造器填充。
- 形式化证明只覆盖抽象模型和分片代数；Python 字节码不在证明范围内。
