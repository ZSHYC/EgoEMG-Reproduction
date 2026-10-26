# EMG2Pose Byte-Exact Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 按原始 tar 字节获取、独立 xz6 分块持久保存并可恢复，形成后续流式解包的可核验源；本阶段不启动全量接收。

**Architecture:** 新增独立模块，固定官方 full/mini URL 与字节数。每个完整 HTTP range 经协议与长度检查后才压缩、事务提交；恢复逐块验证并从原始前缀重建 SHA256。reader 默认拒绝不完整源，工程前缀读取必须显式声明。

**Tech Stack:** Python 标准库（lzma/hashlib/json/fcntl/io）、现有 requests、pytest、本地真实 HTTP server；不新增依赖。

**Spec:** `docs/superpowers/specs/2026-10-26-full-paper-reproduction-design.md`；`docs/superpowers/plans/2026-10-26-emg2pose.md` Task 6。

## Global Constraints

- 官方 full 源唯一为 `https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar`，精确 `462824048640` bytes；mini 为同目录 `emg2pose_dataset_mini.tar`，精确 `647198208` bytes。
- 原始字节、tar padding、NaN payload、所有用户/字段均保留；没有 HDF5 repack、删行、重采样或降精度。
- 默认 chunk=16 MiB，预算=340 GiB，实际 free reserve=15 GiB；不要求同时能放下 431 GiB 裸 tar。压缩比外推是预测，不能承诺全量容量。
- `max_chunks` 是每次调用最多新增块数，只能形成明确 incomplete 的有界前缀；全部源字节连续覆盖且复核后才 complete，完整 SHA256 不得提前给出。
- 最多一个 16 MiB full 真实范围探测，其余均离线；不启动全量下载/训练，不使用 GPU，不提交 git。根代理统一全套测试、审阅与发布。

## Review Focus

- HTTP 200/HTML、错误范围/总长/长度、编码、变更 ETag 与短读/超长内容一律不得成为已完成块；仅暂态错误有限重试。
- 中断恰好发生在块文件与 manifest 提交之间时，孤儿块/part 保留到 quarantine 并明确报错，不自动信用或覆盖。
- manifest 改写路径、范围缺口/重排、压缩文件或原始 hash 篡改、压缩炸弹/尾随流必须拒绝，不能读出“完整源”。
- 两个进程、reader 与 writer 并发必须有真实 flock 协调；symlink 不能绕过受控目录写入。
- 压缩率变差、实际 free 变化、预算耗尽时保留已验证前缀并停止；不缩小 source expected size，不伪称 complete。

### Task 1: 协议、事务归档与恢复

**Files:** Create `egoemg_repro/emg2pose_archive.py`; Test `tests/test_emg2pose_archive.py`。

**Interfaces:** `acquire_archive(output, *, asset='full', chunk_bytes=16*1024**2, max_chunks=None, budget_bytes=340*1024**3, reserve_bytes=15*1024**3) -> dict`。固定 asset registry；离线夹具只替换注册源到 localhost，不 mock range/hash/compression 算法。返回 manifest 状态、真实已收 bytes、压缩 bytes、观测比率与非保证容量预测；原始数据身份的 SHA256 与对象 ETag 分别命名。

- [x] 先写本地 HTTP fixture 与真实小 tar 夹具，包含随机浮点原始位、NaN payload、负零、非零 tar padding；测试逐字节 roundtrip。
- [x] 写失败测试覆盖 HEAD 大小/编码/ETag、206 精确范围、200/HTML/短读/长读、ETag 改变、有限 503 重试、If-Match/identity 请求头。
- [x] 写失败测试覆盖 interrupt/resume、完整源 SHA 重建、显式 bounded 不伪完成、预算/预测/free reserve 停止、孤儿/part quarantine、块篡改与 manifest 非连续范围、两个 writer 拒绝。
- [x] 执行 `PYTHONPATH=. <python> -m pytest tests/test_emg2pose_archive.py -q`，保存 RED 日志，确认缺失模块导致失败。
- [x] 实现强 ETag HEAD 身份；有界 body 验证后 xz6；fsync+原子 rename/manifest；非阻塞 writer flock；恢复复核连续所有块及原始/压缩 SHA。协议失败不追加；网络/503 最多三次尝试。
- [x] 实现硬预算、每次写入前及写入过程中实际 free reserve 检查、观测比率的完整容量预测；容量不足返回明确 incomplete/stop_reason 并保留前缀。
- [x] 执行针对测试，保存 GREEN 与修改范围证据。不得修改其他模块或提交。

### Task 2: 流式 reader 与有界真实探测

**Files:** 同上；私有证据写入 `tmp/emg2pose-archive-*`。

**Interfaces:** `read_archive(output, *, allow_incomplete=False) -> BinaryIO`，支持 `read/readinto` 供 `tarfile.open(mode='r|', fileobj=...)`；默认只接收 complete，工程前缀对象明确 `complete=False`。CLI 只提供显式 bounded 获取，长下载须根代理另行裁决。

- [x] 在实现前写 reader 失败测试：真实 tar streaming 解出原文件全部字节；不完整默认拒绝，显式前缀可读；missing/tampered/reordered 块与伪 complete 拒绝。
- [x] 实现单块有界解压与受控文件名，shared reader flock 阻止 writer；上下文关闭释放锁。完整与前缀读取不落全 tar 或第二份 HDF5。
- [x] 跑针对全部测试；独立真实 full 范围最多 `max_chunks=1`、16 MiB，记录 HTTP 身份、原始/压缩摘要与耗时、压缩率、实际空间，不外推为已保证全量可行。
- [x] 保存私有报告及完整针对日志；说明正式全量未开始，root 负责合并环境中的全套/审阅。计划勾选只反映实际证据。

## 执行证据（2026-10-26）

API 缺失 RED 为 1 failed；容量声明边界 RED 为 2 failed；quarantine fsync 回归 RED 为 1 failed。最终针对测试 **42 passed in 7.00s，无警告**，日志 `tmp/emg2pose-archive-green.log`。唯一真实 full 范围为首 16 MiB，压缩后 11,861,836 bytes、ratio 0.7070205，归档明确 incomplete；未启动完整获取。详细私有报告 `tmp/emg2pose-archive-report.md`。完整合并测试与独立审阅由根代理执行，本代理未提交 git。
