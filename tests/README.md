# 测试矩阵

设计依据是 [DTensorsView-design.md](../docs/DTensorsView-design.md)。设计条目到实现的映射见 [规格追溯](../SPEC_TRACEABILITY.md)。本文件记录测试组织、判断标准和未覆盖范围。

## 目录与层级

| 领域 | 目录 | 层级 | 测试函数 | 参数展开用例 |
|---|---|---|---:|---:|
| 核心公开合约、构造、约束、view map | `core/` | 声明检查、对象行为 | 49 | 289 |
| IO 执行与 plan 可观测行为 | `io/` | 内存集成、失败清理、手算区间 | 22 | 181 |
| checkpoint 数据正确性 | `checkpoint/` | 文件端到端、线程 rank、独立 oracle | 16 | 714 |
| torch 桥与 LAWA | `torch/` | 原生 DTensor、CPU/gloo 多进程 | 7 | 24 |
| sim 格式与执行器 | `sim/` | 适配器集成、子进程 | 8 | 28 |
| 小数据演示 | `demo/` | 打印 12 个 arange 元素与 CRC | 1 | 2 |
| 合计 | | | 103 | 1238 |

2026-10-23 本机验证：`scripts/full_check.sh` 通过，1238 passed；9/9 行为变异被检测。使用 uv，启用真实 torch 测试。原有参数用例保留；新增 64 个参数用例及 24 个文件扫描检查。

数量来自本轮 pytest collection。参数展开用例包含文件级 AST 检查，不能等同于独立行为数量或代码覆盖率。未测量行覆盖率、分支覆盖率。

`support/` 只放共享 fixture 与 oracle，不收集测试。`memory.py` 实现公开 accessor/session；`storage.py` 构造文件布局和 session；`oracle.py` 用稠密数组计算独立期望；`schemas.py` 构造通用张量及解析文件；`checkpoint.py` 构造小张量及读取文件快照。测试文件不能互相导入。

## 领域 × 行为矩阵

“已验证”仅表示表内范围有行为断言。内部函数声明检查单列，不能替代行为验证。所有核心行为测试只走设计中的公开 API。sim 自身的实现测试可以调用 `gdtensor_sim`。

| 领域与层级 | 正常行为 | 边界或错误行为 | 证据 | 判断 |
|---|---|---|---|---|
| 公开 namespace / 声明 | 顶层及 5 个模块的精确导出、签名、字段、Mixin、factory | 禁止跨域导出、额外业务 API、内部导入；递归扫描各级目录 | [core/test_api_conformance.py](core/test_api_conformance.py) | 声明已验证 |
| 构造、tag、helper / 对象行为 | new/get/set/add/mutable/clear、union/tag、相等与 hash、表顺序与文件 leader | 找不到 tensor / entry / leader；返回结构化错误 | [core/test_api.py](core/test_api.py) | 已验证 |
| `check_constraints` / 递归边界 | 有效 compute/storage、行标签和表顺序、跨张量文件 | 12 类额外约束见下一表；错误对象原样返回 | [core/test_constraints.py](core/test_constraints.py) | 已验证列出的约束 |
| view map / 延迟转换 | compute/storage 分支隔离、保持 origin、延迟 begin、替换 session | begin 错误、转换 Err/异常/非 Result、清理失败不覆盖原错 | [core/test_view_map.py](core/test_view_map.py) | 已验证 |
| rank IO / 内存集成 | save/load、reshard 的 CC/CS/SC/SS；使用 session 返回的 accessor | 错 view、schema 不同、begin/init/read/write/finish/commit 失败及清理；提交结果未知 | [io/test_io_contract.py](io/test_io_contract.py) | 已验证 |
| buffer / 回调边界 | 读写使用不同分块，允许零长分块 | 未调用 callback、总长度短/长、负长度、空序列 | [io/test_io_contract.py](io/test_io_contract.py) | 已验证 |
| plan / 公开可观测区间 | 手算 2×5 I16；乱序 rank、空 rank、packed 偏移、replica 去重、表顺序 | 缺源在 begin 前失败、未访问 buffer；schema 保持不变 | [io/test_plan_behavior.py](io/test_plan_behavior.py) | 经公开 IO 验证 |
| mesh / 实际搬运 | Arranged/Literal/Reshaped/Sliced；新增 PermutedMesh 与 step=2、显式倒序切片 | 切片未选中的 rank 不搬运，未写区域不变 | [io/test_plan_behavior.py](io/test_plan_behavior.py)、[checkpoint/test_random.py](checkpoint/test_random.py) | 已验证这些组合 |
| arange / 文件端到端 | 跨 world load，CC/CS/SC/SS、跨格式与布局；最多 32 rank / 64 tensor | 字节翻转必须被 CRC 和读回值检测 | [checkpoint/test_data_correctness.py](checkpoint/test_data_correctness.py) | 已验证 |
| 布局 / 参数矩阵 | 38 种 PP/DP/TP 配置 × 7 种布局 × 2 种格式＝532；另有 9 个边界用例 | 自拷贝、启动顺序、空张量 | [checkpoint/test_matrix.py](checkpoint/test_matrix.py) | 枚举范围内已验证 |
| 随机形状 / 固定种子 | 32 seeds × 2 formats，5 个张量，随机 shape 与 rank 排列 | 不整除切片由独立稠密 oracle 对照 | [checkpoint/test_random.py](checkpoint/test_random.py) | 固定样本已验证 |
| 多 checkpoint / 调用方合并 | 3→5→2、8→6→4、16→12→8 rank；16/24/32 tensor；等权与加权平均、拆分再合文件 | 跨文件等长 payload 交换；读源文件不变 | [checkpoint/test_checkpoint_merge_crc.py](checkpoint/test_checkpoint_merge_crc.py) | 12 正例、2 损坏负例 |
| 多 checkpoint × EP / 调用方合并 | 5 种跨 EP 的三段拓扑（EP、EP×TP、PP×EP、DP×EP×TP、PP×DP×EP×TP）；11 tensor 含 5 个 expert；等权/加权 × 2 格式 | 跨文件交换两个 expert payload，CRC 与读回值必须同时失败 | [checkpoint/test_merge_crc_axes.py](checkpoint/test_merge_crc_axes.py) | 20 正例、2 损坏负例 |
| 串行 vs 并行 / 文件一致性 | 单 rank 基线 vs 3/8/16 线程 rank；独立重建、统一布局后整文件相同、交叉读、顺序读 vs 并发读 | 交换同一 tensor 的两个 rank 分片必须检出拼错 | [checkpoint/test_serial_parallel_crc.py](checkpoint/test_serial_parallel_crc.py) | 12 正例、2 损坏负例 |
| TP/DP/EP/PP / 四轴组合 | 全部 15 种非空组合 × 2 formats × 2 layouts＝60；换拓扑 load/reshard，optimizer 与 expert 归属 | 2 个 expert payload 交换负例；1 个组合完整性守卫 | [checkpoint/test_parallel_modes.py](checkpoint/test_parallel_modes.py) | 线程 checkpoint 验证 |
| torch 桥 / 原生对象 | 9 种 dtype，原生 mesh/placements、identity、Naive/Reshaped reset | plain tensor；shape/dtype/placements/mesh 不同则拒绝且保留原 handle | [torch/test_torch_bridge.py](torch/test_torch_bridge.py) | 需要 torch 开关 |
| LAWA / CPU 多进程 | 7 种配置，2/4/8/16/32 进程，最多 64 tensor；DP/TP/DP+TP，正序/逆序 mesh；朴素/Welford，k=1/2/3/5 | 各 rank 本地值、保存结果与独立平均值对照 | [torch/test_lawa.py](torch/test_lawa.py) | 需要 torch 开关 |
| LAWA × 四轴 / CPU 多进程 | 9 种 (PP,DP,EP,TP) 配置，2–16 进程；PP、EP、PP×EP、EP×TP、PP×DP×EP×TP；3 例换拓扑（TP→DP×TP、PP×DP×TP→PP×TP、PP×EP×TP→PP×DP×EP）；子 mesh 上的原生 DTensor（stage / expert owner），optimizer 走 ReshapedDTensor；朴素/Welford，k=1/2/3 | 每 rank 本地值、单文件解码值与独立 numpy 平均对照；每元素恰好存一次 | [torch/test_lawa_axes.py](torch/test_lawa_axes.py) | 需要 torch 开关 |
| sim 格式 / 生命周期 | safetensors、仿真 distcp，metadata 发布后才返回 | dtype/shape/size 不符、leader 发布失败传到所有 rank、终态后拒绝访问 | [sim/test_sim_formats.py](sim/test_sim_formats.py) | 适配器范围内已验证 |
| sim 执行器 / 子进程 | 线程弹性 IO、真实子进程写读与独立 oracle | worker 失败时回收其他子进程 | [sim/test_sim_runtime.py](sim/test_sim_runtime.py) | 已验证 |
| 检测器 / 行为变异 | 临时副本执行 9 项变异；每个 detector 先通过原始代码 | 偏移、尾分片、漏拷贝/清理、错误 accessor、越界导出等变异必须导致断言失败 | [mutation_check.py](mutation_check.py) | 与 pytest 分开执行 |

四轴的具体源/目标拓扑及归属规则见 [TP / DP / EP / PP 测试解释](../README.md#tp--dp--ep--pp-测试的含义)。每例使用 11 个 I64 张量，每张量最多 15 个元素，含 5 个 expert、单元素和空张量。它覆盖轴组合，不穷举所有 world size、mesh 或布局的笛卡尔积。

## `check_constraints` 条目矩阵

以下按设计 B.2.4 逐项核对。负例均要求公开 `check_constraints()` 返回 `Err(ConstraintViolationError)`，不直接调用内部实现。

| 设计类型 | 已覆盖正常及失败条件 |
|---|---|
| ComputeDTensorsView | ID 顺序与 table 一致；accessor 非空 |
| StorageDTensorsView | ID 顺序；entry 的 tensor 在 table；行标签按 table；accessor / begin_save / begin_load 非空 |
| DTensorTable | tensor ID 唯一；重复 ID 拒绝 |
| NamedDTensor | 非空 ID；空 ID 拒绝 |
| NaiveDTensor | shape / placements / mesh 维度与 shard axis；torch handle 可空 |
| GlobalTensorMeta | shape 非空、dtype 非空、itemsize 正数 |
| PhysicalLayoutTable | format 唯一；file_paths 的键是 leader；leader 存在；同 leader / 不同 leader 的 list 重叠均拒绝 |
| DisjointStorageMeshList | list 非空；同 list mesh 不重叠；行标签数量正确 |
| StorageMesh | 嵌套 content_slots 的 size=0 拒绝；非法 reshape、空 slice、step=0 返回约束错误 |
| ComputeMesh | Arranged size=0 / -1 拒绝；有效 transformed mesh 经 IO 对照 |
| LiteralMesh | 非空；rank 不重复；rank=-1 / rank=max 拒绝 |
| 5 个 Error product | msg 非空；空 msg 拒绝 |

递归错误用对象 identity 断言，确保父对象不重新包装错误。此矩阵不等于 B.1.3 的内部 `check_*` 验证。

## 未覆盖或只做静态检查

| 缺口 | 当前证据 | 处理边界 |
|---|---|---|
| 5 个内部 `check_*` 的 disjoint / covering 行为 | 源码声明、参数与 Result 注解检查；没有公开调用路径 | 保留“只通过公开 API 测试”的要求，因此不能声明这些函数已有行为测试；需要设计提供调用路径才能闭合 |
| `Partial` 的持久化与搬运语义 | 仅 tag/union 合约 | 设计未规定 IO 行为；需要明确语义后补测试，不能按当前实现自行定义 |
| GPU / NCCL；真实进程 PP/EP 的 IO 搬运 | CPU/gloo 已跑四轴 LAWA；PP/EP 的纯 checkpoint 矩阵仍使用线程 | 需要对应设备与执行场景；线程搬运不证明训练、token routing 或梯度通信 |
| 对象存储 multipart、网络中断与服务端重试 | 本地文件与注入的 session/accessor 失败 | 设计将其放在 session 实现层；当前没有对象存储适配器集成用例 |
| PyTorch DCP 外部互操作 | sim distcp 使用 JSON metadata | 不能据此声明支持 PyTorch DCP pickle checkpoint |
| 1000 个真实 rank、完整 DeepSeek 模型容量 | 本轮最大 32 个真实 torch 进程；大矩阵使用小张量 | 未验证容量、峰值内存或吞吐；正确性矩阵不能替代规模测试 |
| 崩溃重启、掉电后的文件原子性 | API 失败与清理已测，未做故障注入重启 | 不声明持久化故障恢复完整 |

结论：公开 API 的声明、主要成功路径、失败清理、分片重建和线程并发对照已有测试；上述缺口仍在，不能把整套测试称为完整验证。

## 运行与日志

```bash
uv sync --locked --extra torch
uv run --locked pytest -v -s tests/core tests/io
uv run --locked pytest -v -s tests/checkpoint
uv run --locked pytest -v -s tests/sim tests/demo
GDTENSOR_TORCH_TESTS=1 uv run --locked --extra torch pytest -v -s tests/torch
scripts/precommit.sh
scripts/full_check.sh
uv run --locked python tests/mutation_check.py
scripts/review.sh
```

`full_check.sh` 依次检查格式、lint、两个类型检查器、编译和全部 pytest，并启用 torch。CI 只安装默认依赖，不启用 torch，因此会跳过 `torch/` 的 24 个用例。变异测试由独立命令执行，不随普通 pytest 自动运行。

`review.sh` 分 7 阶段演示小数据：合约、arange/CRC、LAWA、checkpoint 平均、串行/并行文件一致性、plan/约束、四轴混合拓扑。日志中 `[CRC+ARANGE]` 标识文件与元素正确性，`[lawa]` 标识数值平均，`[PLAN]` 标识实际访问区间。损坏用例的 `PASS` 表示检测器抓到了预期错误。

复核时分别看：每 rank 局部值是否对应正确全局下标；完整 tensor 的 shape/dtype/全部字节是否一致；每元素是否恰好存一次；统一布局后的文件是否逐字节相同。CRC 由独立 oracle 计算，公开 IO 不承诺自动校验 checksum。
