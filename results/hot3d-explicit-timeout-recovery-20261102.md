# HOT3D完整训练来源：超时后明确接续

[实际失败／来源核验／新进程和首包证据](hot3d-explicit-timeout-recovery-20261102.json) · [未修改的公开接收器](../scripts/paper/hot3d_full_train_receiver.py) · [原完整清单启动记录](hot3d-full-train-acquisition-start-20261102.md)。

原接收器在2026-11-01 23:26:16UTC连接超时后停止。实际完成日志201归档／20,849,530,880bytes；原receiver和所有native拥有者均已关闭。原aria2日志是errorCode2 Timeout，外层数值退出码没有保存，不能补写成已观察的exit1。没有覆盖旧失败目录或删除旧文件。

原版另有一个已复制但尚未按顺序登记的Quest种子，共202个完整候选。新进程逐一核对官方LFS完整SHA、复制并再核对副本；共20,951,971,840bytes，使用复制而非硬链接，旧完成归档及全部旧metadata文件代际均未变。准备阶段实际退出0／两拥有者关闭后，才启动原接收器。

接续前对原超时文件做了两次不同边界的HEAD元数据核验，均未下载payload。最终CDN的ETag是本次观察到的Xet hash704536…，不是文件LFS SHA；首次部署检查误将两者相等作为前提，在任何远程目录／进程创建前已被挡下。按照[上游metadata读取实现](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/file_download.py)，重新核对原Hub的`X-Repo-Commit`、`X-Linked-Etag`和`X-Linked-Size`，均与固定清单一致。此修正只涉及部署元数据判断，原下载代码和完整payload SHA校验未变；它不是原网络超时原因的证明。

2026-11-01 23:53:19UTC新的独立后台流程启动。23:54:59UTC原拥有者精确PID／启动时钟仍存活、无终端，recorder父进程为1；10个冻结成员匹配。原超时`train_aria/clip-002050.tar`已真正新下载109,578,240bytes，native退出0后独立全SHA匹配`f0e6bc2e1115b7d33cccbcc8aead002dfada094b52b9ccde23b3ff23b10ad300`，随后进入下一包。

当时新日志202条＝201个已完成Aria种子＋1个新增归档；另一个Quest种子仍待按清单顺序登记。因此复制种子数和当前日志复用数不能直接混为总覆盖。目标仍为同一revision的2,804归档／294,201,743,360bytes，按1连接、32MiB/s、每次native最多1次尝试执行，保留200GiB磁盘；本次是失败关闭及来源复核后的明确接续，没有添加自动重试循环。

完整下载、全部归档退出后核验及完整参数数值资格仍待完成。202份完整字节副本不等于202份MANO参数／mesh资格；原来只有两个片段的参数资格仍按原范围记录。作者混合、模型及全文复现均未完成。

## 在其它私有目录复读接收器

将公开接收器逐字节复制到新的私有接收目录作为`worker.py`，将仓库`assets_hot3d_train_manifest.json`复制为同目录`manifest.json`，将[配置模板](../configs/annotation/hot3d_full_train_receiver.example.json)复制为`config.json`。填写本机真实aria2c路径、版本和文件SHA；模板里的binary SHA记录本次环境，不能用它冒充另一个二进制。准备空的`archives/`目录，模板的seed列表为空，对应下载全部固定训练清单；若预置seed，必须先逐个全文件SHA验收并精确列出所有既存归档路径。不要把失败partial或`.aria2`控制文件当完整seed。

使用Python3.11以上及独立后台控制器运行私有目录的`worker.py`。本次外层控制器另外记录拥有进程及真实退出状态，原接收器仍逐包全SHA并同步日志。脚本自身不提供跨失败重试或自动partial续传；出现失败后必须先审计真实退出、保留已有文件和范围身份，再决定明确接续。执行前需要剩余全清单容量加200GiB余量。

可先只运行四项拒绝检查，不产生下载：

```bash
EGOEMG_HOT3D_RECEIVER_FILE=scripts/paper/hot3d_full_train_receiver.py \
EGOEMG_HOT3D_MANIFEST_FILE=assets_hot3d_train_manifest.json \
  python -B scripts/paper/check_hot3d_full_train_receiver.py
```
