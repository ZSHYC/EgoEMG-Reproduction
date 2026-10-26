# 全论文扩展的发布来源

本次扩展已通过专用 SSH deploy key 正常推送到公开分支 [codex/full-paper-reproduction](https://github.com/ZSHYC/EgoEMG-Reproduction/tree/codex/full-paper-reproduction)，首次发布头为 `7295891eed32bad84db4ccb28080e92cf6a24ddc`。截至 **2026-10-26 03:30 UTC** 已正常快进至 `2a779dfccb0efec952c28c014d23c8aedc325826`，该提交的实际 CI 成绩见下文。推送保留原始提交身份，没有使用 API 镜像。`main` 仍为 `4a583bea483b9f0a431cea6431577625139a025b`；[全论文覆盖状态](full-paper-status.md)记录了仍在进行的工作。

## 已冻结的来源

Git commit 标识包含文件树、父提交、作者和时间等信息。即使两个提交的文件树完全相同，提交标识仍可能不同。训练环境记录的实际执行提交必须保留，不得改写为稍后发布的镜像提交。

| 用途 | 本地提交 | Git tree | 公开对应提交 |
| --- | --- | --- | --- |
| EMGFormer-L 冻结训练来源 | `7b805d428957e62cc93ff48f34e954c64159691e` | `240d9f5778c4105564f7760ba22fe8651f77244d` | [同一本地原提交](https://github.com/ZSHYC/EgoEMG-Reproduction/commit/7b805d428957e62cc93ff48f34e954c64159691e) |
| ResNet-50 / ResNet-152 冻结训练来源 | `a357423b722b8df3e15b78b88b897c6087a04884` | `b3741c28dc30d05ef3cc9245f31fec28282f3324` | [同一本地原提交](https://github.com/ZSHYC/EgoEMG-Reproduction/commit/a357423b722b8df3e15b78b88b897c6087a04884) |
| 文档、形状统计与机器路径脱敏后的发布输入快照；不代表新增训练执行 | `b857a49c912149c25fde74e4e1c48c3b2aaa4090` | `5e33dfcd99c9814e8ba64bb081123c5684202d23` | [同一本地原提交](https://github.com/ZSHYC/EgoEMG-Reproduction/commit/b857a49c912149c25fde74e4e1c48c3b2aaa4090) |
| 首次公开分支头，包含本页的初始版本 | `7295891eed32bad84db4ccb28080e92cf6a24ddc` | `eda046192842d6c07ede34eeca356c28d68c2def` | [同一本地原提交](https://github.com/ZSHYC/EgoEMG-Reproduction/commit/7295891eed32bad84db4ccb28080e92cf6a24ddc) |

本页初始版本在 `b857a49` 输入快照之后添加，由 `7295891` 首次公开；本次成功状态说明属于后续文档更新。最新文件树以远端分支的实际提交为准。官方模型源码继续固定为 `f7cbd048b7092139852270491d3859cd472e0311`。

原始开发历史随正常 Git 推送保留，包含文档汇总祖先 `66eb0840bcdc14c7cddf278a9564f996654e7527`。`b857a49` 已从当前计划文件移除不必要的机器路径；这不表示旧历史被改写。两个冻结训练来源的新增文件已检查，没有这类机器路径。

## 本次实际发布检查

- 历史尝试：HTTPS 推送缺少可用凭据；GitHub 工具创建 blob 和分支均返回 `403 Resource not accessible by integration`；先前的其他仓库 SSH 身份被本仓库拒绝。这些失败未创建扩展分支。
- 用户随后为本仓库添加带写权限的专用 deploy key。验证其公开指纹后，仅使用这把 key、禁用 SSH agent 并保留严格主机校验，正常推送成功。
- 推送前远端扩展分支不存在，成功创建后指向精确的本地 `7295891`；`main` 未改变。没有强制推送，也没有把其他未提交的模型开发内容加入发布。
- PR 创建与托管 CI 成绩须分别读取 GitHub 的实际状态，不能由 SSH 推送成功推断。
- 后续正常快进已公开至 `2a779dfccb0efec952c28c014d23c8aedc325826`。对应[Actions运行40264031015](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40264031015)实读为 success：官方CPU job 398 passed、无skip，轻量job 350 passed/48可选运行时skip，shell语法检查均通过。之前失败保留为历史；该成绩不延伸为其他提交或完整实验成绩。
- GitHub连接创建PR的请求仍返回403，SSH deploy key的Git推送权限不等同于PR API权限；当前代码已公开，尚无PR。
- 后续`128acf1`已公开官方损失合成与旧legacy差异审计，`79153e1e3895af2f216cebcc927528cff8e0b1a4`已公开Task3官方EMG2Pose入口/真实mini工程证据。后者[Actions运行40268110037](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40268110037)实读success：官方CPU446 passed、无skip，轻量379 passed/67依赖skip。新工程检查不证明mini优化或完整训练，旧训练来源身份未替换。
- `a203fa6dca67c0cb3eac879e8e406e0cb2ffd3a4`公开semantic定义/帧统计及classic模型接口，[CI40270328170](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40270328170)实际官方CPU519 passed/无skip、轻量433 passed/86依赖skip。
- `07019609583640d685f724041b022d10b4831e61`公开两步优化/全窗口重载入口，[CI40271080945](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40271080945)实际官方CPU551 passed/无skip、轻量440 passed/111依赖skip；随后真实mini S/M/L各2更新/526窗×2来自这一实际来源，[聚合证据](../results/emg2pose-mini-optimization.md)保持原source SHA，不用后来的文档头替换。新classic数据/中心审计实现的本机637 passed/1既有内存skip、轻量480 passed/158依赖skip只对相应文件快照成立；后续托管CI待对应提交实读。

## 后续更新验收

当前公开来源和本地来源是同一组原始提交，无须建立镜像映射。后续报告继续保留各训练环境记录的真实执行提交、文件与数据摘要；文档更新后的分支头不能取代训练来源。

继续使用扩展分支，不修改 `main`；更新前核对远端旧提交，只允许预期的正常快进，遇到不符先检查远端变化。草稿 PR 与该公开提交的实际 CI 状态另行核验。本地测试、统计复算和训练进度不等同于托管 CI 通过，也不等同于[全论文复现](full-paper-status.md)完成。
