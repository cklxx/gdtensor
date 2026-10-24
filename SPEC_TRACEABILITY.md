# 规格追溯

测试的领域、层级和缺口见 [测试矩阵](tests/README.md)。

唯一设计依据：[docs/DTensorsView-design.md](docs/DTensorsView-design.md)，随 main 的 `12ceb37` 合入。

| 设计条目 | 实现 | 通过设计 API 验证 |
|---|---|---|
| C.3 / C.4 namespace、导出与签名 | 顶层及 io / builder / session / access / tag | [`test_api_conformance.py`](tests/core/test_api_conformance.py) |
| C.1 产品类、BuilderMixin、new/get/set/add/mutable/clear | `internal/builder_mixin.py` 及产品定义 | [`test_api.py`](tests/core/test_api.py)、[`test_api_conformance.py`](tests/core/test_api_conformance.py) |
| C.2 Python 3.12 Result 别名、Error union、tag、抽象接口 | `__init__.py`、`tag.py`、`session.py`、`access.py` | [`test_api.py`](tests/core/test_api.py) |
| B.1.1 IO 与 session 生命周期 | `internal/api.py` | [`test_io_contract.py`](tests/io/test_io_contract.py)；初始化、读写、finish、commit 失败与清理 |
| B.1.2 区间配对、所有权与偏移 | `internal/plan.py`、`internal/dtensor.py`、`internal/layout.py` | [`test_data_correctness.py`](tests/checkpoint/test_data_correctness.py)、[`test_matrix.py`](tests/checkpoint/test_matrix.py)、[`test_random.py`](tests/checkpoint/test_random.py)；独立切片、CRC、逐字节与覆盖计数 |
| B.1.2 / B.1.3 内部接口声明 | `internal/plan.py` 的 3 个 plan 和 5 个 check | [`test_api_conformance.py`](tests/core/test_api_conformance.py) 只解析源码，与设计逐项核对名称、参数和 Result 注解，不导入内部 API |
| B.1.3 per-rank 运行时不变量 | `internal/plan.py` 的 `validate_rank_save/load/reshard`，由 `internal/api.py` 在 plan 生成后、开 session 前调用 | [`test_runtime_invariants.py`](tests/io/test_runtime_invariants.py)；逐 rank 精确覆盖、replicate 每副本独立读全量、缺列/缺源拒绝且不开 session。5 个聚合 check 的跨 rank 并集行为未接入，缺口见[测试矩阵 B.1.3](tests/README.md) |
| plan 的公开可观测行为 | `rank_save/load/reshard` → accessor | [`test_plan_behavior.py`](tests/io/test_plan_behavior.py)；手算 2×5 I16 区间、mesh 转置和步长切片、乱序 rank、空分片、packed 偏移、replica 去重、表顺序和缺源时不开 session |
| TP / DP / EP / PP 及混合配置的 checkpoint 搬运 | 测试调用方构造四维 mesh，核心仍只处理设计中的 view | [`test_parallel_modes.py`](tests/checkpoint/test_parallel_modes.py)；60 个跨拓扑 / 格式 / 文件布局用例（四轴全部 15 种非空组合）及 2 个交换 expert 负例；每 rank 实际值、CRC、覆盖一次、串行整文件一致；线程模拟，不覆盖训练通信 |
| 多 checkpoint 组合读写与调用方合并 | `rank_save/load/reshard`；数值合并由测试调用方执行 | [`test_checkpoint_merge_crc.py`](tests/checkpoint/test_checkpoint_merge_crc.py)；多 rank / tensor / 文件、跨格式重排、普通与加权平均、拆分再合并、源文件不变、等长 payload 跨文件交换 |
| 串行/并行读写一致性与分片重建 | `rank_save/load/reshard`，无数值平均 | [`test_serial_parallel_crc.py`](tests/checkpoint/test_serial_parallel_crc.py)；独立重建张量与单 rank 基线对照、统一布局后整文件逐字节对照、交叉读取、顺序/并发读取对照、交换 rank 分片负例 |
| B.1.4 / 5.6 torch 桥 | `internal/torch_bridge.py`，仅顶层导出 | [`test_torch_bridge.py`](tests/torch/test_torch_bridge.py)、[`test_lawa.py`](tests/torch/test_lawa.py)；真实原生 mesh、dtype、placements、reset 拒绝不一致输入 |
| B.2.3.2 分块 buffer 回调 | `access.py`、`internal/api.py` | [`test_io_contract.py`](tests/io/test_io_contract.py)；读写分块边界不同、空分块、长度不足、越界、负长度 |
| B.2.4 递归约束与同一错误对象返回 | `internal/constraints.py` | [`test_constraints.py`](tests/core/test_constraints.py) |
| 第 7 节文件布局场景 | 业务或测试构造 PhysicalLayoutTable | [`test_data_correctness.py`](tests/checkpoint/test_data_correctness.py)、[`test_matrix.py`](tests/checkpoint/test_matrix.py)；每 rank 文件、共享文件、多文件、按 stage、跨 world |
| 第 10 节 LAWA | 测试调用方在 torch.DTensor 上计算 | [`test_lawa.py`](tests/torch/test_lawa.py)；朴素 / Welford、k=1/2/3/5、2–32 个真实进程 |
| 格式生命周期及 metadata 发布 | `gdtensor_sim/formats.py` | [`test_sim_formats.py`](tests/sim/test_sim_formats.py)；元数据错误、metadata 延迟、leader 发布失败 |
| 删除领域越界 | 模型、布局策略、文件格式、模拟内存与协调放入 `gdtensor_sim/` | [`test_api_conformance.py`](tests/core/test_api_conformance.py) 的定义及依赖检查 |

[`test_api_conformance.py`](tests/core/test_api_conformance.py) 还检查 core / sim / tests 的代码块缩进不超过三层。`scripts/review.sh` 使用小数据并打印中间执行结果。环境、依赖和格式化统一由 uv 管理。

核心测试的 session、文件 fixture 和内存 fixture 只依赖设计 API。格式实现测试调用 `gdtensor_sim` 中的具体实现；独立数据 oracle 不调用这些实现的区间或布局算法。

设计方已明确两条规则，并已同步到唯一设计文档：

- `DisjointStorageMeshList.tensor_ids` 是表顺序的行标签。布局构造器填充，list 校验长度，storage view 校验与表的顺序一致。leader 通过 `tensor_ids[row]` 取得 ID。
- view map 只包装 `begin_save/load`。rank IO 打开 session 后执行转换，使用返回的替换 session。转换失败立即清理原 session；正常提交或结束由 rank IO 执行。

[`test_constraints.py`](tests/core/test_constraints.py) 验证行数、表顺序和跨张量文件 leader。[`test_view_map.py`](tests/core/test_view_map.py) 验证延迟转换、替换 session、转换失败和清理失败。sim 与测试均由 AST 检查禁止导入核心内部模块。

内部 check 的源码审查修复：未知 tensor / file、负 offset / size 和越界访问返回 `CheckFailedError`；storage range 按文件的连续 payload 校验，不额外限制它不能跨相邻 slot；空 payload 的 `(0, 0)` range 合法。covering 检查覆盖范围，重复访问由独立的 disjoint 检查判断。删除未列入设计且无调用方的 `check_physical_layout`，保留对象的 `check_constraints`。

这 5 个内部 check 当前没有对外 API 调用路径。按所有测试只用对外 API 的要求，本轮对它们做源码与边界推导审查，并检查声明，不宣称做了直接行为测试。plan 的行为通过公开 IO 测试验证。无效物理 mesh 在公开 `check_constraints` 路径返回 `ConstraintViolationError`，不再向调用方抛出 Python 异常。

形式化制品仅验证抽象模型。本次重构的正确性证据是设计 API 测试、独立数据 oracle、真实 torch 进程和行为变异；不宣称完成了真实 1000 rank 或完整 DeepSeek 模型验证。

## 本轮门禁已验证范围（2026-10-24，证据在 log/gate/ 与 formal/）

- 形式化：`formal/lean/ShardAlgebra.lean`（工具链 `leanprover/lean4:v4.34.0`，`formal/lean/check.sh` 拒绝 sorry/admit/axiom，编译 exit 0）；`formal/tla/CopyPlan.tla`（S∈{2,3}×N∈{1,2,3,5,6} 含 N=1 零 size 边界）与 `AtomicPublish.tla`（`formal/tla/run_tlc.sh`，全部不变量通过，log 落 `log/tla/`）。形式化仅证抽象模型。
- 行为变异：`tests/mutation_check.py` 9/9 killed（log/gate/mutation_final.txt）。pyrefly 已 0（core 将 negative issubclass 反射与构造分独立测试，未 ignore/未改产品 API）。
- 单元：tests/core + tests/io 独立复跑 486 passed / 0 skip；torch env 全量 1314 passed / 0 skip（log/gate/full_check_final2.txt）；precommit.sh 本体 exit 0（557 passed）；ty/pyrefly 全绿。
- CI：checks job 仅允许 torch importorskip；torch job 跑 tests/torch+sim+checkpoint+demo 且强制 zero skip；collection 计数已预验（no-torch 1255、torch 816 含 distcp）。

## 已决设计项（2026-10-25 更新）

- `DisjointStorageMeshList.tensor_ids` 行标签已恢复：storage view / table 严格按 full table 行序，helper 单参返回真实 ID；反序 labels、未知 tensor boundary 均有正反例（core 479 过）。
- DCP `get_file_offset`：nonzero real raw supported case 已实测（gloo 2-rank row+col offset/path/pread，3 tests 过）。跨 chunk boundary 选择待 Main 批准；不擅新增 multi-segment 核心 API。

## 待决设计项（不记 pass）

- concurrency 槽 C 对 `mem_overhead` 的贡献：3.4（temporal/concurrency 分片、非副本）与 5.8（多一槽多占一份）存在表述张力，待设计最小确认；最终验收以真实峰值与 accessor 实际 bytes 为准，不接受跨时间 box union 或 stats 自证。
- cache 覆盖缺口（返修中，独立 oracle 实测值）：multi-tensor 链、`PermutedMesh`/`ReshapedMesh[CacheMesh]`、Mixed chain 段间失败生命周期、world=4/k=2 真峰值反例。独立观察测试 `tests/io/test_cache_observation.py` 实测：
  - multi-tensor 峰值（reference ResidentCache，r3）：stats `mem_overhead=[4,4]` vs per-node allocator 实测 peak_by_node=[5,4]——node 0 stats 低估 1B。
  - delayed read：read callback parked 期间生产已推进后续 leg/finish 事件（4 个事件在 barrier 释放前出现）。
  - world=4/k=2：outbound stats [4,4] == 观测 [4,4]；inbound stats [4,4] vs 观测 [8,8]（inbound 少计 2×）。

