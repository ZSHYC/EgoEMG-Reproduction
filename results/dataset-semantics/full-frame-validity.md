# Legacy 全帧角度有效性统计

完整扫描全部发布同步行，按既有 legacy 复现掩码统计。这个结果补齐 Table4/6 的逐手、逐角有效性覆盖；不表示标注精度、重新生成标签或与 PDF/current release 人口等价。

| 计数口径 | 数量 |
| --- | ---: |
| 共享同步行 | 66,161,725 |
| 左右手帧 | 132,323,450 |
| 可能的22角度元素 | 2,911,115,900 |
| 有效角度元素 | 2,804,767,094 |
| generated 标志有效的手帧 | 127,495,167 |
| 独立腕角标志有效的手帧 | 127,431,877 |
| 22角全部有效的手帧 | 127,431,877 |
| 至少一角有效的手帧 | 127,495,167 |

全部发布角度元素在原始单位及腕角转弧度后均为有限值。因此仅检查 finite 会把无效标签也算进去；这里仍使用原 generated/wrist 标志，再与逐角 finite 相与。指角磁盘为 rad、腕 pitch/yaw 为 deg。

| 手 | 手帧 | generated 有效 | 腕标志有效 / 22角全有效 | 有效角度元素 |
| --- | ---: | ---: | ---: | ---: |
| left | 66,161,725 | 63,675,430 | 63,622,882 | 1,400,754,364 |
| right | 66,161,725 | 63,819,737 | 63,808,995 | 1,404,012,730 |

20个指角各有127,495,167个有效手帧；两个腕角各有127,431,877个。逐角22项、完整60类及157条class/split/activity/hand统计均保存在[JSON](full-frame-validity.json)，raw50和未分配行没有删除。

| 发布 split | 共享同步行 | 手帧 | 22角全有效手帧 | 有效角度元素 |
| --- | ---: | ---: | ---: | ---: |
| both | 1,261,980 | 2,523,960 | 2,455,159 | 54,013,498 |
| gesture | 9,020,820 | 18,041,640 | 17,294,233 | 381,025,046 |
| train | 49,409,330 | 98,818,660 | 94,961,230 | 2,089,860,600 |
| user | 6,469,595 | 12,939,190 | 12,721,255 | 279,867,950 |

主扫描实际46.316秒exit0，按列bincount的独立全量重算51.171秒exit0；12个输入共12,306,080,850bytes，两次完整SHA均与既有transfer收据一致，九项输入属性未变。全部整数聚合精确一致，既有157条共享标签统计及33,508冻结中心的逐字段字节摘要也精确一致。成功与首个失败任务共六个记录进程均退出。首次隔离包漏带未改动data.py，发生在大型数组打开前，实际exit1和原包保留；补齐依赖的新隔离包成功。

一次新统计单元审查为0Critical/1Important/0Minor。独立校验脚本在优化模式下会丢失assert，已加显式启动拒绝；两个实际RED→GREEN检查覆盖-O与PYTHONOPTIMIZE=1。最终相关检查93通过、1个既有Torch可选检查跳过，未重复全量扫描或旧模型审查。实际执行的校验脚本与公开脚本仅差这一条提前拒绝，剩余AST一致，原执行收据不改写。

复用的官方定义固定于f7cbd048；标签仍是legacy发布的axis-angle projection结果，不改称PDF的landmark IK。腕角重生成、真实标注误差、遮挡和其他全论文缺项仍需独立完成。本次新增模型完成数为0。

复跑入口为`python -m egoemg_repro.semantic_frames --data-root DATA --upstream UPSTREAM --transfer-receipt-sha256 SHA256 --output NEW_PRIVATE_JSON`。输出含私有收据，公开时仅取public_projection。独立实现见[scripts/replay_frame_validity.py](../../scripts/replay_frame_validity.py)，按隔离目录private/request.json及先前完整收据运行；校验器要求非优化Python。

| 输入字段 | 完整字节 | SHA256 |
| --- | ---: | --- |
| label_gesture_class | 264,646,900 | `55a07d9d639483e54eef963a527d50554902d632eea4e42a29651f40f83b4212` |
| label_gesture_active | 66,161,725 | `27bfef6dc99d98110613acc8685dad1ee5321f119098d9bac772c27d4f77d7cb` |
| frame_split_id | 66,161,725 | `9ecef5c0057c639aa90e9574f9cf30f9de070b833f3a38bd2cd0f679c31b31ef` |
| generated_label_valid | 132,323,450 | `30486e402450e0a3eaddec6ab185ea6575bbc478fb41837a6ce981014d79eea2` |
| generated_joint_angles_left | 5,292,938,000 | `85d03d9e992d5d085000d0d26ab25ab5f1a015ee509460d0131fbbbb2252005f` |
| mocap_left_wrist_pitch | 264,646,900 | `325372afc973aa10680e73cd956cde4ff245ebf0cdee650fce63edd5aa450cee` |
| mocap_left_wrist_yaw | 264,646,900 | `d3d6936ba37ef517370f5f6c53ec5a6d6dd7667c7c619092e4faccb7c67d0730` |
| mocap_left_wrist_angles_valid | 66,161,725 | `d48e0430adac0b7ffd0e048e05cdd0fc084c90f28387ca79406dbf62bc9d9389` |
| generated_joint_angles_right | 5,292,938,000 | `383922f2a6f9a6ee5c5109e88d1b65c4d62c8a77435ee0fcbe7b2ef222f4d5e7` |
| mocap_right_wrist_pitch | 264,646,900 | `8c31710cd300927041524093ab151cd7bcc359924f4cb76fe582c492507b2d1b` |
| mocap_right_wrist_yaw | 264,646,900 | `bc157f87c16a51df507abdb7d83e23da91ce45a17954dabbeb383a777fdd5fdb` |
| mocap_right_wrist_angles_valid | 66,161,725 | `6175793773e197b2cdb825d0b131ad686afeaeb1180b9852e0953b7fd986349d` |
