# 规格追溯

唯一设计依据：[docs/DTensorsView-design.md](docs/DTensorsView-design.md)，随 main 的 `12ceb37` 合入。

| 设计条目 | 实现 | 通过设计 API 验证 |
|---|---|---|
| C.3 / C.4 namespace、导出与签名 | 顶层及 io / builder / session / access / tag | `test_api_conformance.py` |
| C.1 产品类、BuilderMixin、new/get/set/add/mutable/clear | `internal/builder_mixin.py` 及产品定义 | `test_api.py`、`test_api_conformance.py` |
| C.2 Python 3.12 Result 别名、Error union、tag、抽象接口 | `__init__.py`、`tag.py`、`session.py`、`access.py` | `test_api.py` |
| B.1.1 IO 与 session 生命周期 | `internal/api.py` | `test_io_contract.py`；初始化、读写、finish、commit 失败与清理 |
| B.1.2 区间配对、所有权与偏移 | `internal/plan.py`、`internal/dtensor.py`、`internal/layout.py` | `test_data_correctness.py`、`test_matrix.py`、`test_random.py`；独立切片、CRC、逐字节与覆盖计数 |
| 多 checkpoint 组合读写与调用方合并 | `rank_save/load/reshard`；数值合并由测试调用方执行 | `test_checkpoint_merge_crc.py`；多 rank / tensor / 文件、跨格式重排、普通与加权平均、拆分再合并、源文件不变、等长 payload 跨文件交换 |
| B.1.4 / 5.6 torch 桥 | `internal/torch_bridge.py`，仅顶层导出 | `test_torch_bridge.py`、`test_lawa.py`；真实原生 mesh、dtype、placements、reset 拒绝不一致输入 |
| B.2.3.2 分块 buffer 回调 | `access.py`、`internal/api.py` | `test_io_contract.py`；读写分块边界不同、空分块、长度不足、越界、负长度 |
| B.2.4 递归约束与同一错误对象返回 | `internal/constraints.py` | `test_constraints.py` |
| 第 7 节文件布局场景 | 业务或测试构造 PhysicalLayoutTable | `test_data_correctness.py`、`test_matrix.py`；每 rank 文件、共享文件、多文件、按 stage、跨 world |
| 第 10 节 LAWA | 测试调用方在 torch.DTensor 上计算 | `test_lawa.py`；朴素 / Welford、k=1/2/3/5、2–32 个真实进程 |
| 格式生命周期及 metadata 发布 | `gdtensor_sim/formats.py` | `test_sim_formats.py`；元数据错误、metadata 延迟、leader 发布失败 |
| 删除领域越界 | 模型、布局策略、文件格式、模拟内存与协调放入 `gdtensor_sim/` | `test_api_conformance.py` 的定义及依赖检查 |

`test_api_conformance.py` 还检查 core / sim / tests 的代码块缩进不超过三层。`scripts/review.sh` 使用小数据并打印中间执行结果。环境、依赖和格式化统一由 uv 管理。

核心测试的 session、文件 fixture 和内存 fixture 只依赖设计 API。格式实现测试调用 `gdtensor_sim` 中的具体实现；独立数据 oracle 不调用这些实现的区间或布局算法。

设计方已明确两条规则，并已同步到唯一设计文档：

- `DisjointStorageMeshList.tensor_ids` 是表顺序的行标签。布局构造器填充，list 校验长度，storage view 校验与表的顺序一致。leader 通过 `tensor_ids[row]` 取得 ID。
- view map 只包装 `begin_save/load`。rank IO 打开 session 后执行转换，使用返回的替换 session。转换失败立即清理原 session；正常提交或结束由 rank IO 执行。

`test_constraints.py` 验证行数、表顺序和跨张量文件 leader。`test_io_contract.py` 验证延迟转换、替换 session、转换失败和清理失败。sim 与测试均由 AST 检查禁止导入核心内部模块。

形式化制品仅验证抽象模型。本次重构的正确性证据是设计 API 测试、独立数据 oracle、真实 torch 进程和行为变异；不宣称完成了真实 1000 rank 或完整 DeepSeek 模型验证。
