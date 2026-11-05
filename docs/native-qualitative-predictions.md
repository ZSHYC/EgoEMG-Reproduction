# 官方普通网格定性预测入口

前置[普通网格与配对输入](../results/native-qualitative/README.md)已经完整验收。本页第一套入口为 ViT-S Figure7 默认抽样重新执行实际冻结预测；RN18完整模型及其2,208点精确缓存复用现已验收，使用下文单独的缓存导图入口。**这里的默认抽样用于定性筛选，不能替换全量表格评价。** ViT-S三类旧版例图已完成；RN18实际完整候选／条件例图、新整合验收及原论文人口等价仍欠。

## RN18已完成预测的缓存导图

[`render_rn18_qualitative.py`](../scripts/paper/render_rn18_qualitative.py)只消费[已独立验收的2,208点缓存](../results/native-qualitative/rn18-cache-source-read-20261105.json)。它核对完整历史代码包／原始数组／退出的生产者，重新准入原53文件输入，逐点验证11个实际输入张量后才将密网格键映射到原普通网格键。实际RGB、四组姿态、原始前20角网格和所有22角误差使用既有原生助手；源生产者身份与密网格原键单独保留，不执行神经网络预测。

[新增35项服务器检查及两张真实面板](../results/native-qualitative/rn18-cached-visual-engineering-20261105.json)已通过、0跳过；本机无Torch完整检查2701通过／1929可选或实际数据用例跳过。[完整120候选加两张显式失败诊断及独立重建](../results/native-qualitative/rn18-full-candidate-pool-complete-20261105.json)已完成退出后验收：122套PNG／NPZ／元数据字节一致，2,074数组及缓存姿态／源像素重查一致。三类人工例图及新整合审查仍欠，不能以工程面板或候选池宣布Figure7完成。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python scripts/paper/render_rn18_qualitative.py \
  --cache-root /path/to/accepted-rn18-ordinary-cache-source \
  --upstream /path/to/the-same-pinned-official-source \
  --data /path/to/the-original-legacy --crops /path/to/the-original-crops \
  --output /path/to/new-rn18-candidates --limit 120 --extra-ranks 1746 2119
```

数据、裁剪及上游路径须与缓存生产时实际相同。默认仍按原始未舍入收益排序选取前120个可显示候选；`--extra-ranks`显式追加诊断示例，保留原排名并在元数据中单独标记，不改变默认池或统计总体。当前2,119个追踪过滤后请求中的1,745改善／374变差排名全部保留；例图及这些跨模型比较不能单独证明EMG贡献。缺crop和无效标签在完整预测／缓存中保留，显示遵循原有追踪条件；不会生成虚构RGB。

## ViT-S原生普通网格预测及导图

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

[实际预测快照](../results/native-qualitative/frozen-prediction-running-20261101.json)已成为历史记录；[完整双遍结果](../results/native-qualitative/frozen-predictions-complete-20261101.json)现已验收，每遍2,322点/38批，全部原始数组精确重放、模型状态不变、60拥有者与退出后独立复核进程关闭。不能将定性抽样分母替换为全量表格分母。

[完整布尔来源闭环](../results/native-qualitative/boolean-flags-complete-20261101.json)现已完成：两个独立实际准入／导出、6个原始拥有者与退出后NumPy复核进程均已关闭，四文件完整字节与所有2,322点逐项一致。原始条件会保留2,242点；未追踪20、无有效手marker64、陈旧帧0，前两者存在重叠。所有2,322点继续进行预测，之后才按这些标志排序；不能把标签有效性直接当作该过滤条件。此闭环本身只证明过滤输入；两遍模型预测已由独立完整证明验收，三类例图与新版数据人口等价仍欠。

同点候选池入口如下。它只消费已独立验收并实际关闭的双遍预测来源；固定完整来源证明和所有原始文件须保持。每张候选保留实际RGB、四组22角、原始前20角网格、未舍入误差、公共视角及全部文件SHA。默认120只限制供人工检查的图片数，不改变2,242点完整收益排序；缺RGB样本仍留在排名中。不会自动赋予三类条件标签。

```bash
python scripts/paper/render_native_qualitative.py \
  --paired-root /path/to/completed-ordinary-paired-io \
  --upstream /path/to/pinned-official-source \
  --data /path/to/legacy --crops /path/to/crops \
  --geometry /path/to/legacy-geometry \
  --first /path/to/completed-neural-workflow/private/pass-a \
  --second /path/to/completed-neural-workflow/private/pass-b \
  --output /path/to/new-candidate-pool --limit 120
```

[21项来源／导图检查](../results/native-qualitative/render-source-engineering-tests-20261101.json)已通过、零跳过，包含真实完成来源的第一名可显示样例、GT与预测数组、RGB类型和尺寸、PNG及误差连接；也拒绝未注册对象和改写的完成证明。数值测试的合成RGB不会用作科研例图。原始网格采用每姿态0.09尺寸归一化，数值MAE仍包括两个腕角，显示不包括腕角；孤立网格共用显示视角，不声称是物理RGB重投影。

输入代码包准入现在会重新计算 `stage-receipt.json` 的完整主体SHA，并校验其每个源码／请求文件；全部文件继续受九项属性保护，不能只保留旧bundle字段或五份元数据就取得完整来源身份。读取历史完成预测时，`verify`／`compare`通过 `--producer-root /path/to/accepted-completed-neural-workflow` 指定已独立验收的生产代码包；同时核对固定完整证明、实际关闭、全部绑定源码及原始文件。生产身份保留历史真实SHA，当前读取器SHA单独放在 `reader_provenance` 中；只允许读取边界升级，其他数值助手必须匹配。历史证据只可读，不能开始新forward。新的 `infer` 使用当前生产身份并拒绝 `--producer-root`；候选导图从 `--first` 自动取得已完成的历史生产根目录。这样无需为一次来源校验修复重新执行已完成的模型推理。
