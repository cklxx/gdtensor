# Decisions 2026-10-27

## B.1.2 读链 Assert 的行下标

`docs/DTensorsView-design.md` 的 B.1.2 里，`d_tensors_view_rank_plan_chain_load`
的 Assert 原来写 `named_d_tensors[tensor_id]`，把 tensor id 直接当列表下标。
5.2 明确禁止这种写法。现在改为
`named_d_tensors[tensor_order_value_of(table, tensor_id)]`，与实现里的
`named_of` 一致。

- 这是对 vendored 设计副本的本地修正；云原文仍保留旧写法，等设计所有者更新。
- 回归用例：`tests/core/test_api_conformance.py::test_read_chain_assert_indexes_tensor_rows_by_order_not_by_id`。

## `gdtensor.io` 的导出范围

- 实际导出：`rank_load`、`rank_save`、`chain_load`、`chain_stat_mem_overhead`、
  `chain_stat_outbound_traffic`、`chain_stat_inbound_traffic`。
- 没有 `rank_reshard`。设计第 9 节不支持 storage→storage 复制，也不支持
  compute→compute 换切分。存储到存储的重排由 `rank_load` 后接 `rank_save` 完成。
- README、SPEC_TRACEABILITY、`internal/plan.py` 的说明和 sim 注释已按此对齐；
  `internal/plan.py` 里三个只服务于 reshard 的校验函数已删除。
- 回归用例：`tests/core/test_api_conformance.py::test_readme_io_row_matches_io_exports`。

## 缓存容量口径（待决）

现状（实测）：

- `get_stat_mem_overhead_of_chain_load` 对每个 node、每个张量返回
  “第 s 个 temporal 切片与 node 所要元素的交集”在各 s 上的最大值，单位字节。
  当 node 需要整个张量时，这个值是各 temporal 切片字节数的最大值；
  `concurrency_width` 只把同一个 s 的切片再切成 C 段，不改变并集，所以统计值
  不随 C 变。同一 node 上若有多层独立缓存，每层各占一块区域，统计值按层相加。
- `tests/io/test_real_file_cache_strategy.py` 固定了 `stat = TOTAL / S`，
  并记录串行填充时的实测峰值 `TOTAL / (S * C)`。
- 用公开 API 复算：12 字节张量、compute 侧 `Replicate`（每个 node 需要整个
  张量）、S=1..6、C=1..3 时，统计值只随 S 变：12、6、4、3、2 字节；C 取 1、
  2、3 时相同。

与文档的冲突：

- 3.4 说 temporal / concurrency 是 Shard，切的是不同的数据片，不是副本。
- 5.8 和同节末段说“多一个 `concurrency_width` 缓冲槽就多占一份”。按这句话，
  统计值应是 `C * max_s(切片字节数)`，与上面的实测相反。
- `docs/decisions-2026-10-25.md` 待决第 1 条已经记录了这个矛盾，至今未决。

要定的两件事：

1. concurrency 的语义：C 个槽是把同一个 step 的数据切成 C 段（总占用不变），
   还是 C 份各自完整的缓冲（总占用乘 C）。当前代码和
   `test_real_file_cache_strategy.py` 是前者；5.8 和 decisions-10-26 的
   `peak <= C * ceil(...)` 写的是后者。
2. “step”指什么。公开 API `chain_load(rank, read_chain)` 没有 step 参数；
   实现里 `_cells_for_nd` 在一次调用中枚举全部 S 个 temporal 坐标。因此现在
   temporal 是“把张量沿 placement 指定的轴切成 S 段”，不是“第 s 个 step 复用
   第 s-1 个 step 的缓冲”。若设计要的是后者，需要给出 step 的来源，并把
   fill → drain（读者 ACK）→ release 的槽生命周期写进设计；现在这条规则只在
   decisions-2026-10-26 和 `formal/tla/SlotLifecycle.tla` 里。

补充一条实测边界：缓存执行器目前是串行的。单 rank 调用里，一个槽填完、读者
读完并 ACK 之后才开始下一个槽；`test_concurrent_ranks_share_allocator_drain_and_peak`
的文档也写明“The executor is currently serial”。因此 5.8 说的“填一个槽的同时读
另一个槽”在一个 rank 内没有发生；`concurrency_width` 现在限制的是同一 node 上
同时在途的槽数，只有多个 rank 同时进入时才有并行入口。
