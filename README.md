# gdtensor

结论：gdtensor 是通用分布式张量库，严格实现 DTensorsView 设计的抽象。三层为逻辑层（`DTensorTable`、`NamedDTensor`、mesh）、物理层（`PhysicalLayoutTable`、`DisjointStorageMeshList`、`ContainerFormat`）、执行层（`DTensorRange`、`CopyPlan`、plan、session、accessor）。

公开 API 以设计附录 C.3 / C.4 为准。发布包 `gdtensor/` 只含设计核心，实现细节在 `gdtensor/internal/`，不对外承诺稳定。

## 公开 API

| namespace | 内容 |
|---|---|
| `gdtensor` | `get_tensor_ids` `get_gd_tensor` `make_gd_tensor` `get_d_tensor` `reset_d_tensor` `Result` `Error` `Ok` `Err` |
| `gdtensor.io` | `rank_load` `rank_save` `rank_reshard` |
| `gdtensor.builder` | view / table / tensor / mesh / layout 产品类型、五个 typed error、`d_tensors_view_map` 与四个 C.4 helper |
| `gdtensor.session` | `StorageSaveSession` `StorageLoadSession` |
| `gdtensor.access` | `DTensorDataAccessor` |
| `gdtensor.tag` | `Ok` `Err` `Shard` `Replicate` `Partial` `ConstantOne` `TensorCount` `WorldSize` `Box` |

不在发布包：业务模型构造、仿真 CLI、多 rank runner、线程/进程协调在测试侧的 `gdtensor_sim/` 包（设计 4.3：builder 在核心之外）。文件 IO、plan/check、区间与所有权运算、torch 虚拟拓扑辅助 `gd_for_layout` 是 `gdtensor` 的内部实现。

## 文件格式

设计只定义两种格式实现，核心类型不出现格式分支：

- safetensors：header 先写，data 后写；`commit` 做 fsync + 原子 rename。多 rank 合写一个文件时，并发写区间互不重叠，barrier 后由文件 leader 发布。
- distcp：payload 先写，`.metadata` 最后发布；`.metadata` 存在即完成标记。

save 终态为 commit 或 abort；load 只有 finish。commit 结果未知时保留 payload，abort 不删除最终路径。

## 仿真 CLI（测试侧工具，不在发布包内）

```bash
python3 -m gdtensor_sim.sim --root /tmp/ck --tp 4 --dp 4 --pp 2 --optimizer \
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
| `test_data_correctness.py` | 两种格式、五种布局的独立 CRC / arange 校验，8→6→8 rank 往返、四种 reshard 方向、元数据错误与发布失败 |
| `test_public_io_contract.py` | 仅实现公共 ABC 的 session/accessor、错误清理、buffer 长度、view map session 回调 |
| `test_consistency.py` | plan 纯函数、线程/进程字节一致、反序启动、PYTHONHASHSEED 跨进程一致 |
| `test_spec_alignment.py` | C.2/C.3/B.2.4 产品类、tag、约束递归、view map、桥、session/access ABC |
| `test_api_conformance.py` | C.3 符号与签名机械审计 |
| `test_builder_api.py` | C.1 BuilderMixin get/set/add/mutable/clear |
| `test_torch_bridge.py` `test_torch_e2e.py` `test_lawa_merge.py` | 真实 torch.DTensor（2.12+）桥与 LAWA，门控 `GDTENSOR_TORCH_TESTS=1` |
| `test_lawa_public.py` | 两个真实 torch rank，通过公共桥和 IO 执行朴素 / Welford LAWA；k=1/2/5，F32/F64、非整除列分片、DP/TP 与逆序 mesh |
| `tests/mutation_check.py` | 核心区间/plan/layout 变异，13 个有效变异全部被杀 |

## 自动化检测

提交前高速门禁（约 3 秒，无需 torch）：

```bash
scripts/precommit.sh        # ruff F401 + 字节编译 + import 冒烟 + 核心/API pytest（约 3 秒）
scripts/full_check.sh       # 全量 760（含 GDTENSOR_TORCH_TESTS=1 门控 torch 用例）
```

本地 git 钩子已用 `git config core.hooksPath .githooks` 启用，提交时自动跑 precommit.sh；其他克隆需执行同一条 config。

远端：`.codebase/pipelines/ci.yaml` 在目标为 main 的 MR 上自动跑 ruff F401、字节编译、import 冒烟与全量非 torch pytest（torch 用例在无 torch 镜像中自动 skip）。

形式化验证：`formal/tla/CopyPlan.tla`（TLC）建模 save/load/reshard 的非交并覆盖与原子发布；`formal/lean/ShardAlgebra.lean` 机器证明分片代数。

运行含真实 torch 分布式用例的全套测试：

```bash
GDTENSOR_TORCH_TESTS=1 python3 -m pytest -q
```

公共桥直接保留 torch mesh 的 rank 排列、维度和 placements。IO 执行器使用 session 的 `init_data_accessor()`，自定义 accessor 无需实现内部 `index` / `bind`。保存成功前，各 rank 同步 payload 发布与元数据发布的结果。CRC 是测试侧校验；raw IO 不提供 payload 校验和。

## 边界

- 不在核心做 payload 序列化：read/write 是 raw bytes。
- 不提供 reduce API：归约、LAWA 平均在 `torch.DTensor` 上做（设计 9 / 10）。
- 不实现对象存储 multipart、分布式 barrier、重试：这些留给 session 实现层（本仓库的本地文件 session 自带线程/进程 barrier）。
- 不决定布局策略：`PhysicalLayoutTable` 由业务构造器填充。
- 形式化证明只覆盖抽象模型和分片代数；Python 字节码不在证明范围内。
