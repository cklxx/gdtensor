# 本地决策记录（用户批准 delta，不改动云原文）

日期：2026-10-25。本文件为本地工程决策快照；云设计文档（vendored）保持原样，不覆盖。

## 已决（用户批准）

1. **`DisjointStorageMeshList.tensor_ids` 行标签**：helper 单参返回真实 tensor ID；storage view 与 table 严格按 full table 行序。反序 labels / 未知 tensor boundary 均有正反例。
2. **DCP `get_file_offset`**：nonzero real raw 已实测支持 case（gloo 2-rank row+col get_file_offset / file_path_of / pread，3 tests 过；zipfile `data/<N>` local header `data_start`）。跨 chunk boundary（pos 对 existing chunk InvalidArg vs layout 每 pos single-chunk）待 Main 批准选择，不擅新增 multi-segment 核心 API。
3. **world=4/k=2 inbound 统计**：每 rank stage 4 bytes、每 node 2 ranks，实际 8；producer 已修 `_node_cache_pull_bytes` 4→8，独立 observer 复跑 world4 项 PASSED。

## 待决（不记 pass）

1. **concurrency C 跨 step 实际峰值口径**：§3.4（temporal/concurrency 分片、非副本）与 §5.8（多槽多占一份）存在张力。最终验收以真实 allocator peak 与 accessor 实际 bytes 为准。
2. **多 chunk fqn 覆盖**：需真实文件反例，不以自实现 dict 覆盖当 API 能力。

## 独立 observer 实测反例（log/cache-observer-r2.txt）

- **multi-tensor 峰值**：stats `mem_overhead=[4,4]` vs reference allocator（per-tensor region 独立 live list）per-node allocator 实测 peak_by_node=[5,4]——node 0 stats 低估 1B。
- **delayed read**：read callback parked 期间生产已推进后续 leg/finish（barrier 释放前 4 个事件出现）——有界 C "槽驻留到 dst callback 完成" 未实现。

测试清单：3 passed / 2 failed（slot_steps、mixed、world4 过；multitensor、delayed 反例）。
