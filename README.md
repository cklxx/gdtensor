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
| `test_checkpoint_merge_crc.py` | 3 个 checkpoint，3→5→2 / 8→6→4 / 16→12→8 rank，16/24/32 个小张量；混合格式重排、普通/加权平均、多文件拆分与单文件合并；CRC、逐字节、源文件不变及等长 payload 跨文件错位检测 |
| `test_matrix.py` | world≤8 的 38 种 PP/DP/TP 配置 × 七种布局模式 × 两种格式；自拷贝、启动顺序、空张量 |
| `test_random.py` | 32 个固定随机种子 × 两种格式，随机形状与 rank 排列 |
| `test_torch_bridge.py`、`test_lawa.py` | 真实 torch.DTensor；2/4/8/16/32 个进程，最多 64 个 tensor；k=1/2/3/5，朴素与 Welford 平均，原生及逆序 mesh |
| `test_walkthrough.py` | 12 个 arange 元素，打印实际字节、CRC 和跨 world 分片 |
| `test_sim_formats.py`、`test_sim_runtime.py` | 仿真格式元数据、发布顺序和发布失败；独立 oracle 校验 sim 跨 rank 读写、真实子进程和失败回收 |
| `mutation_check.py` | 在临时副本中执行九个行为变异，检测器只调用设计 API |

CRC 和期望字节由独立的 NumPy 切片与文件解析得到，不调用核心的 ownership、interval 或 payload index。distcp 仿真实现使用 JSON metadata，不是 PyTorch DCP 的 pickle 格式。

### CRC/arange checkpoint 合并测试的行为

`tests/test_checkpoint_merge_crc.py` 有 14 个数据用例：12 个完整读写合并用例，以及 2 个跨文件损坏用例。完整用例是下表的三种配置 × 两种起始格式 × 两种平均方法。

| pytest 配置名 | 初始保存 rank 数 | 重排、读回和数值合并 rank 数 | 最终合文件、读回 rank 数 | tensor 数 |
|---|---:|---:|---:|---:|
| `source_p0` | 3 | 5 | 2 | 16 |
| `source_p1` | 8 | 6 | 4 | 24 |
| `source_p2` | 16 | 12 | 8 | 32 |

rank 由线程模拟。三个阶段使用不同的 PP/DP/TP 组合或 world size。一个 rank 可以没有某个张量的元素，也可以持有空分片。真实 torch 多进程验证由 `test_lawa.py` 执行。

每个基础张量 `A` 使用 `arange + seed`，数据类型包含 I64、F64、F32、F16。形状包含 5×7、3×4×5、逻辑 reshape 后的 3×7、单元素和空张量；每个张量最多 60 个元素。使用三份 checkpoint：`A`、`A+3`、`A+9`。数值可直接展开查看，也能发现元素次序、字节偏移和张量 ID 错位。

一次完整用例按以下顺序执行：

| 阶段 | 调用与数据去向 | 本阶段检查 |
|---|---|---|
| 1. 保存三份 checkpoint | `rank_save`：初始 compute 分片 → 每 rank 多文件；三份输入交替使用 safetensors / distcp | 每份 checkpoint 的字节、CRC 和覆盖次数正确；确实存在同 rank 多文件、同 tensor 跨文件、同文件多 tensor；源 compute 值不变 |
| 2. 每份输入重排并读回 | `rank_reshard`：原多文件 → 另一格式的共享文件；`rank_load`：共享文件 → 第二组 rank | 新文件仍表示同一份完整张量；新的分片轴和 rank 数下，每个本地元素都正确；load 前后文件字节不变 |
| 3. 调用方合并张量 | 测试在第二组 rank 的本地 buffer 上用 float64 累加，再转回原 dtype | 等权平均期望为 `A+4`；权重 1/2/3 的平均期望为 `A+5.5`；整数结果按 dtype 转换规则截断。期望值由全局公式直接生成，与实际读回路径分开 |
| 4. 保存数值合并结果 | `rank_save`：合并后的 compute 分片 → 每 rank 多文件 | 全部输出字节、CRC、覆盖次数正确；再次检查多文件分布 |
| 5. 合并物理文件并读回 | `rank_reshard`：上一步多文件 → 最后一组 rank 协作写一个文件；`rank_load`：单文件 → 最后一组 compute 分片 | 恰好输出一个 payload 文件；数值仍等于阶段 3 的期望；原 checkpoint 和作为合文件输入的目录字节不变 |

例如，第一个张量是从 0 开始的 I64 arange。普通平均的前三个期望值是 `[4, 5, 6]`；加权平均先得到 `[5.5, 6.5, 7.5]`，转回 I64 后为 `[5, 6, 7]`。日志中的 `base_arange_first12` 展示原始输入，`CHECKPOINT_INPUT` 展示各份 checkpoint；`merged_expected_first12` 和 `loaded_rank0_first12` 分别展示合并后的期望值与实际读回值。

这里有两种 merge。**张量 merge** 是阶段 3 的调用方数值平均。**文件 merge** 是阶段 5 的存储布局转换，保持张量值不变。核心只调用设计中的 `rank_save/load/reshard`；没有新增 merge API。

CRC 只计算 payload 字节。独立 oracle 同时逐字节比较，并为每个全局元素计数，要求恰好存一次。这能检查丢片、重复写、错 tensor、错 offset 和分片次序错误。仅比较整文件 CRC 会把 header 差异也算进去，因此本测试还会分别检查文件格式、shape、dtype 和数据区域。

损坏用例先正常保存，再交换两个文件中长度相同、内容不同的 payload 区域。文件名、header 和区域长度保持不变。测试要求独立 CRC 检查失败，再执行 `rank_load`，要求读回值检查也失败。此用例显示 `PASSED` 表示两个检测器都抓到了损坏。当前 IO 按 range 搬运字节，测试中的独立 oracle 执行数据校验。

相关日志统一以 `[CRC+ARANGE]` 开头。`[CASE][START]` 给出模拟 rank 数、tensor 数和权重；`[SAVE_CHECKPOINT]`、`[REPACK_CHECKPOINT]`、`[SAVE_MERGED]`、`[CONSOLIDATE_FILES]` 给出文件数、格式和 payload CRC。`[VERIFY][PASS]` 表示 CRC、精确字节和元素覆盖检查通过；`[LOAD_REPACKED][PASS]` / `[LOAD_CONSOLIDATED][PASS]` 表示各 rank 读回值和源文件不变检查通过。`[TENSOR_MEAN][PASS]` 给出数值合并公式。`[CASE][INPUT]` 打印基础 arange，`[CHECKPOINT_INPUT]` 打印每份输入；`[CASE][PASS]` 打印前 12 个合并期望值和 rank 0 实际读回值。`[CORRUPTION][PASS] crc=detected arange=detected` 表示两个检测器都按预期发现了损坏。每条 `PASS` 都在对应断言通过后输出。

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
最后还运行 3→5→2 个模拟 rank、16 个小张量的加权 checkpoint 合并，打印各阶段文件 CRC 和前 12 个实际值。新增 CRC 合并矩阵使用线程模拟 rank；真实 torch 进程测试仍在 `test_lawa.py`。

`precommit.sh` 检查 fmt、lint、编译、导入和 API 合约。`full_check.sh` 还检查 ty（警告视为失败）和 Pyrefly，启用真实 torch 测试并运行全部 pytest。两者显示测试名称和执行阶段。缩进检查覆盖 core、sim 和 tests，代码块最多三层；括号内的换行对齐不计层级。

CI 用 `uv sync --locked` 安装依赖并检查格式，不安装 torch。`.githooks/pre-commit` 调用快速检查，可用 `git config core.hooksPath .githooks` 启用。

`formal/` 中的 TLA+ / Lean 制品描述抽象模型，不证明 Python 实现。

## 仿真 CLI

```bash
uv run --locked python -m gdtensor_sim.sim --root /tmp/gdtensor-ck \
  --tp 4 --dp 4 --pp 2 --optimizer --layout hf --format safetensors
```

CLI 会重建指定的 `--root` 目录。报告和读写 trace 位于默认的 `<root>_report/`。
