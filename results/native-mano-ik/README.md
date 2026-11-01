# 原生 MANO→UmeTrack 全量输入与后续标签重生成

- [输入工程验证](input-engineering-tests-20261101.json)：40项新增真实文件检查通过、无跳过；工程源每手53行，没有真实IK优化或完整源producer替身。
- [完整双手输入双遍核验](full-input-double-admission-20261101.json)：每遍66,161,725行／手、25,406,102,400bytes姿态文件、6,351,525,600个值；两遍身份及JSON字节精确一致，共116.73秒。134份冻结部署文件、167份准入文件及完整姿态SHA在退出后独立重核；6个原拥有进程与复核进程均确认退出。
- [方法、批次及边界](../../docs/native-mano-ik-full-data.md)；[执行计划](../../docs/superpowers/plans/2026-11-01-native-mano-ik-full-data.md)。

数据仍为可得旧版发布源。4,828,283个双手零姿态保留；不使用marker/label mask替代原脚本不存在的`valid.npy`。保留原默认六逻辑分区、batch10096、right→left及原IK几何设置。

**尚未运行真实帧IK、batch10096优化资格或完整标签重生成。** 当前证据只证明输入，不能替代缺失的markers2mano、FE/RU腕角、原标注精度或全论文复现。
