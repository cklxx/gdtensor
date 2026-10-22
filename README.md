# gdtensor

通用分布式张量的 view、布局和 IO 核心。唯一设计依据是 [DTensorsView-design.md](docs/DTensorsView-design.md)。实现与测试的对应关系见 [SPEC_TRACEABILITY.md](SPEC_TRACEABILITY.md)。

| namespace | 导出范围 |
|---|---|
| `gdtensor` | `get_tensor_ids`、`get_gd_tensor`、`make_gd_tensor`、`get_d_tensor`、`reset_d_tensor`、`Result`、`Error` |
| `gdtensor.io` | `rank_load`、`rank_save`、`rank_reshard` |
| `gdtensor.builder` | C.3 产品类型、BuilderMixin、`new_*`、view map，以及 C.4 的四个 helper |
| `gdtensor.session` | `StorageSaveSession`、`StorageLoadSession` |
| `gdtensor.access` | `DTensorDataAccessor` |
| `gdtensor.tag` | 八个 tag 类，以及 C.2 的 `Box`、`make_box_subclass` |

`Ok` / `Err` 只从 `gdtensor.tag` 导入。五个 torch 桥函数只从顶层导入。session / accessor 接口只从各自的 namespace 导入。`Result` 使用设计 C.2 指定的 Python 3.12 类型别名。

`gdtensor/internal/` 保存 plan、range、约束和区间运算。`gdtensor_sim/` 保存业务模型、布局策略、文件格式实现、模拟内存和多 rank runner。核心不导出 `ComputeFabric`，不决定 PP/DP/TP 拓扑或文件布局，不执行 LAWA。

IO 使用 session 的 `init_data_accessor()`。自定义 accessor 不需要内部 `index` 或 `bind`。buffer 回调支持分块，读写两侧可以使用不同的分块大小。LAWA 在 `torch.DTensor` 上计算，通过顶层桥函数取出和写回张量。

## 测试

测试只导入核心的设计 API。`test_api_conformance.py` 检查导出、函数与 BuilderMixin 签名、产品字段、领域边界，并用 AST 检查测试和 sim 没有导入 `gdtensor.internal`。

| 文件 | 验证内容 |
|---|---|
| `test_api.py`、`test_constraints.py` | 类型别名、tag、helper、构造方法、哈希、递归约束及错误原样传递 |
| `test_io_contract.py` | 自定义 session/accessor、六种 IO 方向、分块回调、失败清理、提交结果未知、view map |
| `test_data_correctness.py` | 独立稠密切片与文件解码，CRC、arange、每元素恰好存一次；五种布局、两种格式、跨 world 加载与四种 reshard 方向；最多 32 rank / 64 tensor |
| `test_matrix.py` | world≤8 的 38 种 PP/DP/TP 配置 × 七种布局模式 × 两种格式；自拷贝、启动顺序、空张量 |
| `test_random.py` | 32 个固定随机种子 × 两种格式，随机形状与 rank 排列 |
| `test_torch_bridge.py`、`test_lawa.py` | 真实 torch.DTensor；2/4/8/16/32 个进程，最多 64 个 tensor；k=1/2/3/5，朴素与 Welford 平均，原生及逆序 mesh |
| `test_walkthrough.py` | 12 个 arange 元素，打印实际字节、CRC 和跨 world 分片 |
| `test_sim_formats.py`、`test_sim_runtime.py` | 仿真格式元数据、发布顺序和发布失败；独立 oracle 校验 sim 跨 rank 读写、真实子进程和失败回收 |
| `mutation_check.py` | 在临时副本中执行九个行为变异，检测器只调用设计 API |

CRC 和期望字节由独立的 NumPy 切片与文件解析得到，不调用核心的 ownership、interval 或 payload index。distcp 仿真实现使用 JSON metadata，不是 PyTorch DCP 的 pickle 格式。

## 本地检查

需要 Python 3.12+。torch 只用于桥和真实多进程测试。

```bash
uv sync --locked --extra torch
uv run --locked ruff format gdtensor gdtensor_sim tests
uv run --locked --extra torch ty check --error-on-warning
uv run --locked --extra torch pyrefly check --min-severity=warn
scripts/review.sh
scripts/precommit.sh
scripts/full_check.sh
uv run --locked python tests/mutation_check.py
```

`pyproject.toml` 声明 Python 3.12 和依赖，`uv.lock` 固定版本，`.venv` 由 uv 创建。numpy / safetensors 是运行依赖，pytest / ruff / ty / pyrefly 是开发依赖。两个类型检查器都使用项目 `.venv`，并从 `tests` 解析测试辅助模块；完整类型检查需要安装 torch extra。

`review.sh` 用小数据逐步打印：12 个 arange 元素、完整 header / CRC、2→3 rank 加载及 reshard；LAWA 使用两个真实 rank、三个小张量，打印 checkpoint、实际值与期望值。

`precommit.sh` 检查 fmt、lint、编译、导入和 API 合约。`full_check.sh` 还检查 ty（警告视为失败）和 Pyrefly，启用真实 torch 测试并运行全部 pytest。两者显示测试名称和执行阶段。缩进检查覆盖 core、sim 和 tests，代码块最多三层；括号内的换行对齐不计层级。

CI 用 `uv sync --locked` 安装依赖并检查格式，不安装 torch。`.githooks/pre-commit` 调用快速检查，可用 `git config core.hooksPath .githooks` 启用。

`formal/` 中的 TLA+ / Lean 制品描述抽象模型，不证明 Python 实现。

## 仿真 CLI

```bash
uv run --locked python -m gdtensor_sim.sim --root /tmp/gdtensor-ck \
  --tp 4 --dp 4 --pp 2 --optimizer --layout hf --format safetensors
```

CLI 会重建指定的 `--root` 目录。报告和读写 trace 位于默认的 `<root>_report/`。
