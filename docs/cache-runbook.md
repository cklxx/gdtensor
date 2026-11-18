# 缓存部署与恢复 Runbook

日期：2026-11-18。

本文适用于 `GDTENSOR_CACHE_COORDINATOR=file` 的共享文件缓存。线程协调器只用于同一进程内的 rank。`SSHCache` 是受控测试后端，不提供生产 peer discovery 或跨机协调。

## 1. 部署前检查

1. 所有 rank 使用相同的 commit/tree、依赖锁、协调协议和 checkpoint。
2. 每个 job 使用独立的新 `GDTENSOR_CACHE_DIR`。不同 job 不共享协调目录。
3. 所有客户端对协调目录和缓存目录具有读写、创建和原子 rename 权限。
4. 设置 `GDTENSOR_CACHE_COORDINATOR=file`。多进程不得使用 `thread`。
5. 设置 launcher 总时限。主机断联和文件系统永久阻塞由 launcher 终止，缓存层不承诺检测这两类故障。
6. `GDTENSOR_CACHE_WAIT_SECONDS` 默认 60 秒。健康慢 IO 处于 in-flight 时不会因该时限失败，不得用租约替换此语义。

在每个共享文件系统客户端并发执行 `scripts/cache_fs_preflight.py`：

```bash
uv run --locked python scripts/cache_fs_preflight.py \
  --root /shared/job-coord \
  --job JOB_ID \
  --chain CHAIN_FINGERPRINT \
  --namespace CACHE_NAMESPACE \
  --client CLIENT_INDEX \
  --clients CLIENT_COUNT
```

所有客户端使用相同的 `--root`、`--job`、`--chain`、`--namespace` 和 `--clients`。`--client` 从 0 开始且不重复。`--job` 必须是新标识。

预检检查协议、chain、namespace、目录、客户端数、跨客户端 `flock` 和原子发布。任一客户端失败即把该部署记为 `blocked`，不得作为 skipped pass，也不得启动多客户端缓存加载。

## 2. 运行记录

每个 run 保存以下证据：

- commit/tree、模型 digest、协议、协调器、transport 和共享文件系统配置；
- hostname、PID、rank、node、layer 和目标设备；
- CUDA、NCCL 和 driver 版本（适用时）；
- source/cache/remote bytes、TX/RX、wall time、RSS/VmHWM 和错误；
- generation、resident/budget、in-flight 和协调目录代次（适用时）。

性能测量不得与其他测试、mutation、profile 或 benchmark 重叠。每个样本保留原始值，并报告中位数、最小值和最大值。

## 3. 错误处理

### `cache_peer_down`

一个 owner 在未完成调用前退出，或其生命周期锁提前释放。停止整个 job。不要在同一协调目录重试，不要删除仍在使用的锁文件。

### `cache_state`

协调状态损坏、过大或协议不兼容。检查所有进程的版本和目录是否一致。停止整个 job，使用新目录恢复。旧状态不自动修复或清除。

### `cache_wait`

在无 in-flight 工作时，等待超过配置时限。检查缺席 rank、计划 cohort 和 launcher 日志。错误消息包含 rank、layer、node、segment、resident/budget 和 owed consumers。健康慢 IO 不适用固定租约；不要在慢 IO 尚未失败时强制终止它。

### 源身份变化

源文件在加载前后或加载期间变化时，加载返回错误。关闭旧 session，确认源已静止，再创建新 session。目标的部分写入不自动回滚。

## 4. 新目录恢复

1. 停止所有旧调用和进程，确认旧 owner、in-flight 和 launcher worker 已退出。
2. 保留旧目录用于诊断。不要清理、替换或复用旧状态。
3. 创建新的空 `GDTENSOR_CACHE_DIR`。
4. 重新执行共享文件系统预检，并启动完整 cold load。
5. 校验每个 rank 的字节、CRC 和 guard。cold 轮必须读取新源。
6. 在不修改源的第二轮验证 hot load。源读取必须为 0。

## 5. 版本升级

1. 停止旧版本的全部 rank。
2. 所有客户端安装同一 commit/tree 和锁定依赖。
3. 使用新的协调目录，不混用协议版本。
4. 重新执行预检和 cold/hot 正确性检查。
5. 保留升级前后的报告和 SHA。报告缺失或 SHA 不符时禁止发布。

## 6. 清理

只有 job 完全停止且证据已保存后，才能显式清理该 job 的专用目录。不得按模糊路径删除，不得在 rank 运行时清理。

`SSHCache.close()` 只清本地记录。远端专用测试目录由显式 `cleanup()` 删除；一个 rank 的 `close()` 不得删除其他 rank 使用的 root。soak 或集成测试结束时必须确认残留 helper 和 SSH 进程为 0。

## 7. 验证边界

运行时故障语义见 `tests/README.md`。形式化模型只覆盖有界 ACK 和依赖投影，不证明文件锁、文件系统、IO、GPU 或操作系统行为。单主机线程、单机多进程和单 rank SSH 结果不得作为多物理机验收。
