# Classic 原生测试启动绑定：CPU证据

本阶段实现原生测试的父子进程／pipe／profile绑定及原始测试产物读取。**没有实际GPU校准、冻结profile、原生测试或新的科学成绩。** [结构化证据](classic-report-native-gate-cpu.json)绑定报告源码、测试源码和未改的formal／evaluator／artifacts／metrics摘要；[入口说明](../docs/classic-report.md)列出使用方式。

修复后的固定80成员完整CPU检查为 **1,351 passed／3 skipped**，106 warnings，283.35秒。3项skip要求实际CUDA或真实三进程GPU验收bundle。轻量无Torch检查为 **806 passed／501依赖skip**，1 warning，26.97秒。定向报告及原evaluator检查为211 passed，2 warnings，8.59秒。warning明细保存在原私有日志；没有据CPU检查宣布CUDA通过。

轻量首次命令误设`EGOEMG_UPSTREAM`，把两个既有视觉测试显式启用到没有Torch的环境，结果806 passed／499 skip／2 failed，失败均为缺Torch。随后使用与工作流一致的不启用该变量的命令通过；没有为此改测试、生产源码或依赖准入条件。原失败日志保留。

独立只读审查最初为“with fixes”，发现1项Important：回滚只核对目录inode，会误删同一目录内替换后的状态文件，并留下被移走的本次成功标记。4项真实文件系统回归先失败；一次修复后，result／status均在首次公开链接前保留创建FD与inode，失败时撤销本次inode，即使已被移走；只删除仍指向本次inode的公开名字，外来替换文件保留原字节。启动状态另存progress；最终文件不覆盖已有文件。修后未做第二次独立审查。

真实pipe子进程检查支持消息与放行顺序；真实父进程探针核对持久manifest后**故意以17退出**，没有产生被接纳的原生测试。原始读取正例显式替换终止／checkpoint权威，只验证元数据与文件机制，不能证明真实训练完成或CUDA评价。

本阶段源码的托管CI尚未观察。[b4608f8实际CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40347752496)的官方1,250 passed／3实际CUDA skip与轻量762 passed／444依赖skip属于前一验证阶段。该修复只将pytest临时目录从容量不足的tmpfs移到工作磁盘，原15GiB生产保留门槛未改。

实际GPU校准、正式原生测试、测试预测参考／独立复算、只读状态API和脱敏科学导出仍需完成；七项既有legacy结果及其运行来源不变。全文范围继续按[清单](../docs/full-paper-status.md)验收。
