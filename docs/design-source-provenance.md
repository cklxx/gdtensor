# Design Source Provenance（云设计原文溯源）

日期：2026-10-26。本文仅纠正/记录 vendored 云设计文档的来源与 hash 事实，不改动设计语义、不还原 vendored 字节、不覆盖云原文。

## 来源

| 项 | 值 |
|---|---|
| 云文档标题 | DTensorsView-design.md |
| wiki 入口 | https://bytedance.larkoffice.com/wiki/ONvawkdlniQZT8kuF09ctT0un0f |
| 解析后类型 | `file`（非 docx；`docs +fetch` 不支持，须 `drive +preview --type source_file`） |
| file_token | `OZK3bpqz1o8YwwxgWDccAApcnab` |
| canonical URL | https://bytedance.larkoffice.com/file/OZK3bpqz1o8YwwxgWDccAApcnab |
| publish_time / create_time | 2026-10-20T17:12:48+08:00（create_time epoch 1792487568） |
| 本地产物副本（只读，未回写） | `log/audit-df1/cloud-DTensorsView-design.md` |

## Hash 对照（实测，2026-10-26）

| 副本 | 字节数 | md5 | sha256 |
|---|---|---|---|
| 云原文副本 | 109401 | `9e0155f34f4f5872caf46c74f2507c19` | — |
| 仓库 vendored `docs/DTensorsView-design.md` | 109406 | `f3dc57cc5aca7b6170f52ed786a2a3f3` | `098adb05a505cadb24c0a1b59191df288a2188e09641076812fb0281204932d1` |

两者 **md5 不等、字节数差 5**。已归一化（去空白）后 md5 仍不等。

## 差异性质（语义等价，非 byte-exact）

逐行 diff（`log/fix-df1/provenance/cloud-vs-vendored.diff`，共 81 行）显示差异全部为**格式重排**，无任何领域语义改动：

1. 两处代码示例中分号被拆成换行（云为 `new_ok(); ok.set_value(3)` 一行，vendored 拆为两行；`err()` 示例同理）。
2. 若干签名行的 `...` 省略号：云另起一行，vendored 与签名同行。
3. 多余空行、注释对齐空格、`|` 联合类型换行折行。

结论：**截至 2026-10-26，内容语义与云设计一致；2026-10-27 的 B.1.2 行下标修正见“本地 delta”。** "md5 未变 / byte-exact / 字节原样" 的旧声明不准确，予以撤回。vendored 副本不做字节还原。

## 本地 delta（已批准，另见 decisions）

本轮 leader 批准的本地 delta（API 字段、slot 调度顺序）记录在 `docs/decisions-2026-10-25.md` / `docs/decisions-2026-10-26.md`，与云设计原文相互独立，不构成对云原文的修改。

2026-10-27 又加了一条本地修正：B.1.2 的 Assert 改用 `tensor_order_value_of` 取行下标，见 `docs/decisions-2026-10-27.md`。这一条与云原文不同，云原文待同步；其余语义仍未改动。

## 已知 warnings 说明（本轮不改代码）

- DCP save warning 437 / load warning 438：为 torch 内部在「单进程、无 dist 初始化、显式覆盖已有文件」路径下的提示，属预期行为，非缺陷。
- `torch.frombuffer` readonly 缓冲：已通过显式 `.copy()` 规避只读视图写风险，不视为生产 hazard。
- BF16 安全修复由 DCP owner 负责，与本记录无冲突。

以上仅为文档说明；代码/测试/CI/formal 本轮不动。
