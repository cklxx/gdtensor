# 规格追溯

设计文档：`跨计算和存储的 DTensorsView 设计`（1513 行，2026-10-21 版）。本表列出设计要点、实现位置与验证方式。

## 核心类型与 API

| 设计条目 | 实现 | 验证 |
|---|---|---|
| B.1.1 三个 IO API（薄执行器） | `internal/api.py`，`io.py` 仅导出 rank_load/rank_save/rank_reshard | test_matrix、test_adversarial、test_consistency |
| B.1.2 plan API（纯函数，不打开 session） | `internal/plan.py` | test_checks、test_exhaustive（286 配置） |
| B.1.3 五个 check API | `internal/plan.py` | test_checks |
| B.1.4 compute/torch 桥五个 API | `builder.py` 顶层短名，`internal/torch_bridge.py` 实现 | test_torch_bridge、test_torch_e2e、test_lawa_merge、test_public_api（reset 校验 meta/placements/mesh 一致） |
| B.1.5 d_tensors_view_map(origin, map_core) | `builder.py` | test_spec_alignment |
| 五个 namespace 的全部公开符号（`__all__` 共 66 个） | 公开命名空间 | test_public_api（逐符号，含三个 io API 端到端、reset 回写与 meta 一致性校验） |
| B.2 / C.3 产品类型与五个 namespace | `builder.py`、`session.py`、`access.py`、`tag.py` | test_api_conformance（机械审计）、test_spec_alignment |
| C.1 `<ClassName>BuilderMixin` + new/get/set/add/mutable/clear | `internal/builder_mixin.py` 按字段生成 | test_builder_api |
| C.2 标签 Box 与 Result；`type(x) is` 判别 | `tag.py` 纯标签；Error 产品类在 `internal/result.py` | test_spec_alignment |
| B.2.4 check_constraints 递归与额外检查 | `internal/constraints.py` | test_spec_alignment |
| B.2.2.5/B.2.2.6 session 生命周期与 init_data_accessor | `internal/formats.py` 继承 ABC | test_matrix、test_spec_alignment |
| B.2.3.2 buffer 回调 `(buffer_ptr, buffer_len)` | `internal/views.py` 用 buffer 地址回调 | 全量端到端 |
| 5.3 contiguous_payload_offset 与空槽 0 字节 | `internal/layout.py` PayloadIndex | test_matrix::test_empty_slot_offsets |
| 5.4 commit 顺序 / abort / 未知结果保留 payload | `internal/formats.py` | test_matrix、test_adversarial::abort 后重写 |

## 格式（仅两种）

| 设计条目 | 实现 | 验证 |
|---|---|---|
| 4.2 safetensors header 先写 + fsync/rename | `internal/formats.py` SafeTensorsSaveSession | test_matrix、test_data_correctness |
| 4.2 distcp `.metadata` 完成标记 | `internal/formats.py` DcpSaveSession | test_matrix |
| 第 7 节用例矩阵 | `internal/models.py` 业务布局构造器 | test_matrix 全格点 |

## 非目标（设计第 9 节）

- 核心无 reduce / LAWA API：归约只在 `torch.DTensor` 上做（test_lawa_merge 朴素/Welford 一致）。
- 无对象存储 multipart / 重试 / 真实分布式 barrier 之外的存储实现。
- 不内置布局策略：布局由业务层构造。

## 形式化验证（抽象层）

| 制品 | 覆盖 |
|---|---|
| `formal/tla/CopyPlan.tcl` + `AtomicPublish.tla` | TLC 全网格：CopyPlan 非交并覆盖（save/load/reshard）、发布状态机 writing→done |
| `formal/lean/ShardAlgebra.lean` | 分片相邻、单调、满射、唯一、非交并覆盖（clean build，无 sorry/axiom） |

诚实边界：形式化证明只针对抽象模型与分片代数，不针对 Python 字节码。Python 侧以穷举（world≤8）、独立 plan oracle、属性对拍、变异测试（13/13 有效变异被杀）补足。

## 多机一致性

plan 与发布字节是 (rank, tables, layout) 的纯函数；选路无 `hash()`/进程随机盐。`test_plan_and_bytes_are_hash_seed_independent` 用三个 `PYTHONHASHSEED` 子进程固定 plan repr 与文件字节一致；线程与 fork 进程产出字节相同。

## 已删除的偏离

本次重构按设计删除：非设计的 manifest/chunk/KV/沉降存储后端与节点读缓存、`rank_reshard` 的额外合并参数与批量 IO、`begin_load` 的额外文件参数、Error 异常基类与 `unwrap()` 等 Result 方法（仅测试 shim 保留）、构造期重复 id 抛错、旧 `max_`/字符串 tag 形式、内部模块公开导入。
