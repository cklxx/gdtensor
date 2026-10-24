# 本地决策记录（用户批准 delta，不改动云原文）

日期：2026-10-25。本文件为本地工程决策快照；云设计文档（vendored）保持字节原样（md5 未变），不覆盖。

## 已决（用户批准）

1. **`DisjointStorageMeshList.tensor_ids` 行标签**：helper 单参返回真实 tensor ID；storage view 与 table 严格按 full table 行序。反序 labels / 未知 tensor boundary 均有正反例。
2. **DCP `get_file_offset`**：supported case 已实测——gloo 2-rank row+col public session、file_path_of、os.pread 真实 archive；world=1 满 tensor 跨 2 文件返回 offset error、torch 读全量正确（3 个公共 session 测试过）。跨 chunk / 跨 slot boundary 选择待 Main 最小能力边界批准，不擅新增 multi-segment 核心 API。
3. **world=4/k=2 inbound 统计**：每 rank stage 4 bytes、每 node 2 ranks，实际 8；producer 已修 `_node_cache_pull_bytes` 4→8，独立 observer 复跑 world4 项 PASSED。

## 待决（不记 pass）

1. **concurrency C 跨 step 实际峰值口径**：§3.4（temporal/concurrency 分片、非副本）与 §5.8（多槽多占一份）存在张力。最终验收以真实 allocator peak 与 accessor 实际 bytes 为准。
2. **bounded C cache pipeline**：两次实现尝试失败后回退 serial（world4/k2 fill node box off 0:4、rank0 drain 0:2、rank1 drain 2:4，需 node 级 slot state across calls），当前未实现。
3. **Mixed shared source**：begin1/finish1 同源 identity group 尚无合法 test 验证。
4. **多 chunk fqn 覆盖**：需真实文件反例，不以自实现 dict 覆盖当 API 能力。

## 独立 observer 实测（log/cache-observer-r6.txt，5/5 通过）

reference ResidentCache 按 (node, global_tensor_id, offset) 三维隔离，write retain 到同 (node,tid) reader ack 才 free；peak_by_node 逐 node 记录。

- multi-tensor 峰值：stats `mem_overhead=[4,4]` 与 per-node allocator 实测 peak_by_node=[4,4] 一致（producer 修 inbound 计数后）。
- delayed read：barrier 释放前同 slot 下游 leg/finalize 不发生；释放后 cache.live 为空且 rank-0 shard 字节完整。（此前"释放前 4 事件提前推进"为误判，已撤回。）
- world=4/k=2：outbound [4,4] 一致；inbound 实测 [8,8]，stats 已对齐。
