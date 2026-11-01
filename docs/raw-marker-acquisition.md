# Legacy 双手 marker 与腕部来源接收

> **最新状态：Drive目前返回配额页面，接续已停止。** 原1GiB经过完整SHA准入后，另目录保全前缀并建立独立可写副本；[独立接续实际失败](../results/legacy-marker-prefix-continuation-stop-20261102.json)于17:30:46 UTC退出，4拥有者关闭，新增提交字节0，旧失败状态与旧部分文件九属性未变。[相同16MiB范围的一次只读诊断](../results/legacy-marker-original-range-status-probe-20261102.json)于17:32:04 UTC实际收到HTTP200／2,040字节HTML，标题为Google Drive - Quota exceeded并包含配额提示；这证实诊断当时的响应，不能倒推前两次未记录的具体HTTP状态。先前64字节HTTP206不代表完整块可下载，未将HTML追加为数据；不再继续本轮下载请求，八个完整文件仍为0。

> **较早64字节只读诊断**：[停止位置Range检查](../results/legacy-marker-failed-offset-probe-20261102.json)在原退出后约44分钟仅发出一次64字节请求，实际HTTP206／精确Content-Range及16,672,754,700总字节吻合，未写入新原始数据且原1GiB九属性未变。说明本次检查时端点重新接受范围请求，不能反推之前非206的具体状态码或原因；独立接续另行登记。

> **首次接收的失败快照。** [非206停止及部分文件保全证据](../results/legacy-marker-non206-stop-20261102.json)记录2026-11-01 16:44:04 UTC原任务exit1、全部4拥有者关闭，1,073,741,824bytes（1GiB）部分文件完整保留。严格传输拒收了非206响应；具体HTTP状态码/正文未被原日志记录，不能认定为已证实配额。八个最终文件仍为0，独立全量读回未开始，该次记录时未重试。下文是启动时的范围与计划验收流程。

2026-11-02，新增的 [8 文件清单](../assets_markers_manifest.json)逐项匹配固定 legacy manifest 的 dtype/shape 和官方公开 Drive 文件元数据。共 **44,196,032,300 bytes**，每个字段为 66,161,725 对齐行，41 个 episode。这里的坐标已经按发布格式对齐，不能称为未处理的原硬件采样记录。

| 新接收字段 | 每手 shape / 字节 |
| --- | --- |
| 左、右 `mocap_*_keypoints` | 各 `[66161725,21,3]` float32／16,672,754,700 |
| 左腕 `mocap_left_wrist_rigid_markers` | `[66161725,5,3]` float32／3,969,703,500 |
| 右腕 `mocap_right_wrist_rigid_markers` | `[66161725,4,3]` float32／3,175,762,800 |
| 左、右 `mocap_*_wrist_position` | 各 `[66161725,3]` float32／793,940,700 |
| 左、右 `mocap_*_wrist_orientation` | 各 `[66161725,4]` float32／1,058,587,600 |

两个 `mocap_*_valid` 掩码已在先前几何接收中取得，本次不再下载。掩码不能替代上述三维坐标，已有 MANO pose 也不是原始 marker 观测。

## 已发生的启动证据

[真实启动及进度快照](../results/legacy-marker-intake-launch-20261102.json)记录服务器上独立的 recorder/controller、下载 worker/guardian 的精确 PID/start 身份。原控制器已脱离 SSH，PPID=1、无终端；流程不依赖本机继续开机。快照实际观察到左手坐标 `.part` 已写入 **301,989,888 bytes**，不是仅创建任务后声称下载成功。源码胶囊、清单和固定 manifest 均已核对哈希。

Google Drive 连接能够读取元数据，但完整原始坐标单文件 16.67 GB 超过其 256 MiB 下载限制，实际调用返回 413。随后服务器与本机对同一公开文件请求 `bytes=0-63`，均返回精确 `206`、总长 16,672,754,700，64 字节 SHA256 同为 `f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b`。服务器因此使用既有的公开 HTTPS 范围接收器，无需新的本机中转或登录凭据。

## 沿用的接收和验收边界

使用未修改的 `download_one`、`StrictRangeDownloader` 和 `BoundedRangeDownloader`，顺序接收八个原文件，保留至少 15 GiB 余量。每次最多 16 MiB，检查 HTTP206、Content-Range、总长度、Content-Length、二进制类型及完整响应后才追加。瞬时连接中断最多三次尝试、等待 1/2 秒；HTTP 配额、权限、格式、容量或证书错误会停止，保留已收到的部分，不自动另开新下载任务。

每个文件仅在完整尺寸和整文件 SHA256 核对后转为最终文件名并写入 acquisition ledger。全数组统计逐块读取实际 float32，记录有限值、NaN、正负无穷与零值数量，不修复或删掉异常值。八个文件全部收到后，另一个进程会重新读取全部 SHA256、数值统计与文件属性，再核对原下载进程已经退出。最终仍需实际终态及独立退出后收集，启动快照不是完成证明。

完整文件的 SHA256 证明本次接收字节身份，作者没有公开可认证的 bulk 参考 SHA256。取得这些字段后仍需核验坐标单位、左右/marker 顺序、有效性和腕角生成入口。markers2mano 模型包、训练配置与权重缺失的问题不因下载输入而消失；本流程没有执行标注模型、生成标签或修改既有训练数据。

需要在另一独立目录复用时，可按[下载指南的范围接收示例](download.md#bounded-transient-recovery-for-the-reviewed-geometry-selection)，将清单改为 `assets_markers_manifest.json`，仍使用相同固定 legacy manifest。运行中的目标目录只能由原任务写入，不要启动第二个写进程。
