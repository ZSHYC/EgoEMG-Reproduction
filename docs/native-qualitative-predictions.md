# 官方普通网格定性预测入口

前置[普通网格与配对输入](../results/native-qualitative/README.md)已经完整验收。本入口为 Figure7 默认抽样重新执行实际冻结预测；当前接入已完成的 ViT-S／EMGFormer-S／融合三模型，RN18兼容融合来源仍待训练结束。**这里的2,322个默认抽样用于定性筛选，不能替换全量表格评价。** 最终条件例图、完整新集成审查和 Figure7 验收尚未完成。

入口固定原始7,790窗口／普通步长，seed42每个split和手重置、每组最多500。六组各自B64，保留尾批，共38批，最后一批33个样本；不复用最近的密网格中心。所有80个缺少裁剪、64个无效标签、230个跨中心split窗口均保留，官方追踪过滤在全部推理结束后执行。

模型必须从真实完整来源载入：融合验证BEST108、视觉LAST124、独立8通道EMG验证BEST133；模型来源标识为`7543495ccd161d11bc2db25f277a7dfb1fddd4fd0051b9f1d0531e5faeea41fd`。视觉和融合使用实际训练时的16通道输入，独立EMG使用同一窗口的8通道输入与独立归一化，不把单点标签传给EMG前向来缩短轨迹。原生FP32／CPU前向、146步EMG输出的第73个位置、未遮罩22角MAE保持。

每次推理先重新核对完整54文件61.380GB输入。读取者逐点验证11个输入数组指纹；模型调用前再核对实际批次中的每个保留张量，不能仅信任读取子进程报告的哈希。原始包保存三预测、完整EMG轨迹、GT、标签／裁剪掩码和三误差。第二遍使用新模型实例，逐批与第一遍核对。模型状态前后、完整来源文件、实际拥有者及四个DataLoader子进程／组均需验证。

从本仓库根目录运行；替换为对应的实际完整来源路径。`infer`退出后再由独立进程调用`verify`；第二次`infer`换一个输出目录并通过`--replay-of`连接第一遍，最后使用`compare`核对两遍。

```bash
python scripts/paper/run_native_qualitative.py infer \
  --paired-root /path/to/completed-ordinary-paired-io \
  --upstream /path/to/pinned-official-source \
  --data /path/to/legacy --crops /path/to/crops \
  --fit /path/to/completed-native-fusion-fit \
  --output /path/to/new-pass-a

python scripts/paper/run_native_qualitative.py verify \
  --paired-root /path/to/completed-ordinary-paired-io \
  --upstream /path/to/pinned-official-source \
  --pass-root /path/to/new-pass-a --output /path/to/new-verification-a.json
```

实际来源证明固定为`ordinary-paired-inputs-complete-20261101.json`的完整SHA，不接受重新封装的小样本或借来的模型标识来授权推理。原始包工具的数值测试不授予训练来源许可。新增[27项检查](../results/native-qualitative/frozen-runtime-engineering-tests-20261101.json)已在既有服务器CPU环境通过，未安装或修改共享依赖，未重复旧模型数学套件。

原始追踪标志使用单独的真实布尔文件准入，新增[14项检查](../results/native-qualitative/flag-source-engineering-tests-20261101.json)已通过，包括全部2,322点与独立原始memmap读取逐项比较。新版脚本的`mocap_head_valid`显式映射到旧版`mocap_webcam_tracked`，`image_head_stale`映射到`image_webcam_stale`；左右手各自读取21个marker的`any()`，不使用生成标签或裁剪掩码替代。四个完整文件共2,911,115,900字节，须匹配已独立验收的旧版几何来源并在提取前后重新核对；当前字段映射不证明新版人口或物理相机等价。所有选中点的标志均保留，预测之后才过滤。

[实际预测快照](../results/native-qualitative/frozen-prediction-running-20261101.json)记录当前源版本的独立后台执行；完整两遍原始包、模型状态和进程关闭尚待验收，不能使用中间进度作为最终指标。
