# FreiHAND固定来源几何复现

[全量结果](../results/freihand-official-geometry-20261102.md)覆盖32,560训练行和3,960评估行；[执行器](../scripts/paper/freihand_official_geometry.py)与实际全量worker逐字节相同，SHA256为`c601b7482c3a96179abb5dc439cafd07ea7951c84bcb9d45da3443d4ffff8ac2`。

## 输入和环境

使用[固定FreiHAND源码](https://github.com/lmb-freiburg/freihand/tree/5ea4ab9763fea0eec988a52bfa563333cb16523f)。私有public_root须包含`utils/model.py`复制为`model.py`、根`view_samples.py`、`utils/fh_utils.py`复制为`fh_utils.py`、`README.md`复制为`freihand-README.md`、`utils/mano_utils.py`复制为`mano_utils.py`和根`setup_mano.py`；每份完整Git blob由执行器核验，不能仅截取函数。

使用自己已获许可的五个MANO原文件：`models/MANO_RIGHT.pkl`和`webuser/{verts.py,posemapper.py,lbs.py,smpl_handpca_wrapper_HAND_only.py}`。原文件必须同时匹配官方setup的MD5及本次SHA；模型和许可源码不随仓库分发。在独立私有FreiHAND副本根目录用Python运行原`setup_mano.py /ABSOLUTE/PATH/TO/licensed-mano-source`，事先创建其输出父目录`data`和`utils/mano_core`。原setup会修改复制出的loader，不能对共享原MANO目录运行补丁。五份输出SHA见机器结果和请求模板。

实际环境是Linux／Python3.11.12、NumPy1.26.4、Chumpy0.70、Torch2.2.2+cu118、OpenCV4.10.0；本流程只用CPU。独立SMPL-X及Chumpy的全部27份实际Python provider源哈希见[请求模板](../configs/annotation/freihand_geometry_request.example.json)。模板保留实际来源SHA和字节数，只将私有路径替换为占位符；请复制到私有位置，逐项换为真实绝对路径。`package_paths`按Chumpy父目录、SMPL-X父目录排序，`chumpy_files`／`smplx_files`的路径须相应更换，哈希不得凭空填入。原始四字段K／mano／xyz／verts分别保留training、evaluation两角色。raw参数和GT均不改写。

固定资源门槛为256GiB可用主机内存加16GiB工作空间准入、`/data`空闲200GiB。脚本每个阶段完整读取原始JSON和执行整文件SHA；小内存本机不适合运行完整阶段。输出使用新的私有目录，不覆盖已有结果。

## 执行与退出后重读

准备好私有请求后，先在独立CPU子进程执行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -B scripts/paper/freihand_official_geometry.py \
  --request /ABSOLUTE/PATH/TO/private-request.json --phase qualification \
  --output /data/YOUR_PRIVATE_GEOMETRY/qualification
```

外层启动器须独立记录该真实子进程的PID和`/proc/PID/stat`启动时钟，并在`wait`确认其退出0后捕获`completed.json`完整SHA。随后以相同request、phase和output开启一个新进程，添加`--reopen-result /data/YOUR_PRIVATE_GEOMETRY/qualification-reopen.json --producer-completed-sha ACTUAL_CAPTURED_SHA --producer-pid ACTUAL_PID --producer-start ACTUAL_START_TICKS`。读取器要求原生产者已退出；不能从重新封装的结果自称已完成。

两行资格和新进程重读均成功后，再用`--phase full --output /data/YOUR_PRIVATE_GEOMETRY/full`执行全部36,520行；同样实际等待生产者退出、外层捕获身份及完成SHA，并以`--phase full`在新进程重读。输出包括两后端各一份`float64[36520,778,3]`顶点、`float64[36520,21,3]`关节、`float64[36520,6]`逐行误差和完整有序角色／行号。预分配文件长度不能证明覆盖；实际完成记录、全部成员和新进程重算必须同时通过。重读不导入Torch或Chumpy，不构造模型。

原始[source／closure／shape检查](../scripts/paper/check_freihand_official_geometry.py)可在已准备六份public源及五份许可原文件的目录上单独运行；这13项检查不会构造模型：

```bash
EGOEMG_FREIHAND_GEOMETRY_CHECK_ROOT=/ABSOLUTE/PATH/TO/private-check-root \
EGOEMG_FREIHAND_GEOMETRY_CHECK_WORKER=scripts/paper/freihand_official_geometry.py \
  python -B scripts/paper/check_freihand_official_geometry.py
```

其中private-check-root下需有`official/`和`licensed-mano-source/`两个已核验子目录。请求模板不是数据下载器或作者混合构建器；本单元不代替缺失的markers2mano模型、训练入口及权重。
