# 原生 MANO→UmeTrack 全量输入与后续标签重生成

- [输入工程验证](input-engineering-tests-20261101.json)：40项新增真实文件检查通过、无跳过；工程源每手53行，没有真实IK优化或完整源producer替身。
- [完整双手输入双遍核验](full-input-double-admission-20261101.json)：每遍66,161,725行／手、25,406,102,400bytes姿态文件、6,351,525,600个值；两遍身份及JSON字节精确一致，共116.73秒。134份冻结部署文件、167份准入文件及完整姿态SHA在退出后独立重核；6个原拥有进程与复核进程均确认退出。
- [真实双手CPU工程执行](real-bilateral-engineering-20261101.json)：每手两帧、显式batch2，原优化器共122次外层调用／1313次closure；另一个禁止优化器的新进程直接重建全部几何数组与目标值，typed结果精确一致。6个原拥有进程及退出后收集进程关闭；再次完整重读25.406GB姿态来源及原始数据对应，源标签不变。
- [运行时来源15项＋工程前置7项检查](runtime-engineering-tests-20261101.json)：全部通过、无跳过，真实现有私有资产用于来源边界测试，不构造替身full-source producer。
- [方法、批次及边界](../../docs/native-mano-ik-full-data.md)；[执行计划](../../docs/superpowers/plans/2026-11-01-native-mano-ik-full-data.md)。

数据仍为可得旧版发布源。4,828,283个双手零姿态保留；不使用marker/label mask替代原脚本不存在的`valid.npy`。保留原默认六逻辑分区、batch10096、right→left及原IK几何设置。

**尚未完成batch10096优化资格或完整标签重生成。** 真实每手两帧是CPU工程链，不能代替完整原始批次、缺失的markers2mano、FE/RU腕角、原标注精度或全论文复现。
