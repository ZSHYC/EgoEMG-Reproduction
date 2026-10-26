# 全论文扩展的发布来源

本次扩展已通过专用 SSH deploy key 正常推送到公开分支 [codex/full-paper-reproduction](https://github.com/ZSHYC/EgoEMG-Reproduction/tree/codex/full-paper-reproduction)，首次发布头为 `7295891eed32bad84db4ccb28080e92cf6a24ddc`。本次新增组发布前，已正常快进至`ae3031cb10d0f6d154b539db96a75c863f5e6455`；RN152产物继续绑定原训练来源a357423，新增L仍绑定7b805，实际CI按提交分别列出。推送保留原始提交身份，没有使用 API 镜像。`main` 仍为 `4a583bea483b9f0a431cea6431577625139a025b`；[全论文覆盖状态](full-paper-status.md)记录了仍在进行的工作。

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
- `f6c293d6308dc0ead082b7afd2c70936901d31dd`的[CI40273189915](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40273189915)已实读成功：官方CPU638 passed/无skip、轻量480 passed/158依赖skip，新reader参考依赖实际安装且pip check通过。[真实中心点读](../results/dataset-semantics/center-coverage.md)亦执行该源的Git归档，源包SHA44437a902e284bf7993ecc37c70c922df14e56f6e2ac0dc57332bfa4f32284c5；实际审计与公开聚合有独立核验。中心文案stride误写2000已据原RED修正1200，计算/JSON未改；原审查历史保留。

## 后续更新验收

后续新增组的实际本机检查绑定66个源码/配置/测试文件，快照SHA256 `6ca23e7191d33cafee1e21a02f1ac83854e8a6cf9ea3af283695af6ab93e1380`：879 passed/3 skipped（284.01s），轻量609 passed/273依赖skip（20.12s）。包含classic全队列CPU准入、canonical CUDA限定检查、预取和派生TaskA；不包含派生worker及正式artifact WIP，也没有CPU冒充的CUDA正例。实际classic两配方全量数据准入/完整重算的独立[聚合证据](../results/classic-cohort-cpu-engineering.md)另绑定72267d源码，不能用后续发布头替换；新托管CI按发布提交另读。

后续实际公开提交`8ef6aeec24d7f9f38bf67bb93db83b89187e0355`包含RN50已审计结果和CI成功case临时目录保留策略。其[CI40278377967](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40278377967)实读success：官方CPU673 passed/无skip（268.78s），轻量490 passed/183可选运行时skip（19.38s），pip check通过；结束后/tmp可用7.8GiB、保留pytest目录92KiB。这个已完成运行不包括后续新增代码，下面的旧失败和诊断保留为历史。

`ec2d5b127828578dd3c125c44d7293dfab105c4c`已正常快进公开classic CPU工程训练与严格epoch恢复；对应本机57文件快照为672 passed/1既有内存skip，轻量490 passed/183依赖skip。其[托管CI40276917815](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40276917815)实际官方CPU为672 passed/1 failed：旧视觉迁移恢复测试写检查点失败，并非新增classic行为断言失败；轻量job成功。CI没有当时的磁盘快照，不能把具体原因写成已证实。独立测量本机整轮保留5.318GiB临时文件，其中新classic夹具占3.919GiB，与空间耗尽相容。后续仅将CPU CI设为pytest原生`tmp_path_retention_policy=failed`，每个成功case结束即释放其自有scratch、失败case保留，并新增always磁盘记录；生产代码/测试断言/容量guard不改，不跳过用例，后续CI另验。该提交只开放CPU工程入口，正式CUDA与科研训练仍待完成。新[RN50完整结果](../results/completed/resnet50/README.md)另保留原运行`a357423`来源、原严格导出字节和独立发布清单；发布结果的后续文档头不能代替训练来源。

当前公开来源和本地来源是同一组原始提交，无须建立镜像映射。后续报告继续保留各训练环境记录的真实执行提交、文件与数据摘要；文档更新后的分支头不能取代训练来源。

继续使用扩展分支，不修改 `main`；更新前核对远端旧提交，只允许预期的正常快进，遇到不符先检查远端变化。草稿 PR 与该公开提交的实际 CI 状态另行核验。本地测试、统计复算和训练进度不等同于托管 CI 通过，也不等同于[全论文复现](full-paper-status.md)完成。

`9b5f4931d53db0d536e0ebe64e386e8048872ab8`已正常公开。[CI40292505908](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40292505908)实际success：官方CPU880 passed/2 skipped（208.53s，仅两项真实CUDA正例），轻量609 passed/273依赖skip（21.25s），pip check及shell检查通过。它覆盖66文件组中的cohort/CUDA限定入口/预取/派生TaskA，后续worker和formal artifact WIP不在其中。

本次新增[RN152完整结果](../results/completed/resnet152/README.md)保留原a357423训练来源与原始metrics/audit字节。150轮、52,650更新、150次完整验证与独立复算已完成，全部/有图中心5.982738°/5.869706°；发布清单和曲线经独立数值、身份和展示检查。此结果与RN50一样属于明确声明的legacy派生配方，不改变原四模型/RN50发布清单，不由后续文档HEAD替换运行身份。

`ae3031cb10d0f6d154b539db96a75c863f5e6455`已正常公开RN152结果。[CI40298074155](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40298074155)实际success：官方CPU880 passed/2真实CUDA skip（222.86s），轻量609 passed/273依赖skip（22.15s），pip check与shell通过。它仍不覆盖之后的worker/artifact/capacity实现。

本次新增[EMGFormer-L完整结果](../results/completed/emgformer-l/README.md)保持原7b805训练来源、原导出字节与中心审计边界；200轮/25,400更新，全部中心16.612111°，原测试轨迹16.591133°未独立重推。七项legacy结果均不冒称论文原表等价，原四模型/RN50/RN152产物保持。新增[DINOv2-B/L CPU证据](../results/dinov2-base-large-pretrained.md)、[classic真实CUDA数值](../results/classic-cuda-numeric-engineering.md)及[容量证据](../results/classic-cuda-capacity-engineering.md)分别绑定其实际源码/资产/范围，绿CPU检查不替代这些实际执行。

容量工具原执行路径为私有tmp；现公开[tools/classic_capacity.py](../tools/classic_capacity.py)与实际b890源严格逐字节相同。它的公开位置不改写历史执行路径、base archive或模型运行身份。新formal-artifacts仅实现状态/日志契约，派生worker仅完成CPU与来源检查；正式runner仍未集成、未通过最终源码GPU验收，尚无新的classic科研完整训练结果。

新增公开capacity工具、worker与原生callback语义修正后的72文件冻结组，源码快照SHA9f879422b483206c0133b1bd095ce405503d14f962b5b183abfec1ea9f1d005d，本机完整CPU **1025 passed/3 skipped**（271.86s，两个真实CUDA正例及一个既有ViT-L内存准入skip），轻量 **690 passed/338依赖skip**（23.83s）。最终文件SHA逐一复核不变；该组明确排除仍在实现的formal runner及其测试，托管CI待对应发布提交实读。

该组公开为`6599e05da0edbec2adac570c9f75cc43f927c69b`后，[实际CI40306913437](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40306913437)失败，不能再记作等待或通过。官方CPU1013 passed/2 skipped/11 failed/2 errors，轻量678 passed/337 skipped/11 failed/2 errors；失败来自旧namespace夹具缺浅克隆历史对象及事务测试的临时磁盘前提。后续仅修复有界fetch和测试空间fixture，worker及生产15GiB门槛保持原字节，原失败记录保留。

后续正式classic入口5285与CI修复的75文件快照SHA为`d9f8283741000a5dfe151c13ae463b2e964de7669e7c5d170f42c4689208ade4`。实际本机完整**1076 passed/3 skipped**（267.26s）、轻量**727 passed/352依赖skip**（21.59s）；ViT-L本次实际通过，三项skip为两CUDA限定正例及新formal入口缺实际GPUbundle。源码与独立复审范围分别保留；这些CPU结果不能授权未运行的正式GPU训练，也不覆盖后续评价模块。新发布头的托管结果仍须按其实际运行核验。

该75文件组已公开为`2927aabfbe5a7a8e2f7003cc276ba729d0e24745`，[实际CI40310839399](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40310839399)成功：官方CPU1076 passed/3真实CUDA skip（300.14s），轻量727 passed/352依赖skip（17.34s），pip check及固定commit的有界fetch通过。旧6599失败不改写，此结果不覆盖后续artifact/evaluator修正。

后续原生AdamW参考修正及selected-best独立测试入口的77文件快照SHA为`d93e3d3824293c6355c01b92d9ad9e684e445172356e4cabc7329b1858eb559f`，本机完整1118 passed/3真实CUDA skip（271.95s）、轻量739 passed/382依赖skip（21.40s）。四个新增/修改源码与测试均冻结并独立复审，原七项结果不变；修前真实GPU保存失败保留，修后的新源码GPU验收与本次发布头托管CI须另读。CPU检查及独立测试入口实现不构成完整训练或论文数值完成。

该组已正常公开为`4925caada9b13acacc38cf347185f8eaf7be1f17`，[实际CI40319331373](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40319331373)两项success：官方CPU1118 passed/3实际CUDA skip（277.90s），轻量739 passed/382依赖skip（19.72s），pip check通过。

随后ModelCheckpoint原生不恢复current_score的比较修正，保留原checkpoint严格校验，新增原生Trainer恢复回归真实RED→GREEN；77文件快照SHA为`9cd7d7cc01a1aee1a8ff298b40b0e675011b90ebf56e8e5bd4149034247be498`。本机完整1119 passed/3实际CUDA skip（263.56s），轻量739 passed/383依赖skip（20.74s），源hash再次复核一致。新formal文件SHA为6adcadfe4f99c818560e0cc26272edea93205d9ab0d1b2130fe081eb1a98c1b6；此前5285+187b的fresh GPU恢复失败保留，新的实际GPU验收及本次发布头CI分别另读。
