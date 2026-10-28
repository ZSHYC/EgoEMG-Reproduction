# 融合完整输入核验与原生加载器

本入口在[联合读取器](fusion-literal-data.md)之上，读取并hash所有实际会消费的输入文件，扫描所有可能抖动窗口的双手EMG和所有可能中心的22角目标，再绑定原生batch200加载器。它不构造模型，不授予训练或全文完成资格。

## 核验内容

`admit_fusion_cohort`绑定manifest、metadata、split、两手filtered EMG、20指角/腕角/生成mask/图像帧映射，以及所有消费episode的完整crop数据库。来源绑定覆盖本入口、读取/布局/变换/分区/文件守卫、本地依赖和固定官方源码/配置，并另包含实际使用的unified归一化文件。所有文件要求regular/nonsymlink；分块hash与扫描前后用文件身份/大小/mtime/ctime核对稳定性。

EMG扫描取每个角色全部可能抖动后的完整窗口并集；目标只扫描每个可能中心，而不是添加未消费位置。双手8通道源数值、每手20指角和2腕角均要求有限；即使生成mask=false也不清理或隐藏非有限数值。仅包含未消费标签端点的NaN不会被修复或变成新的样本筛选条件。chunk_rows严格为1..262144，默认262144；扫描按唯一行统计，不能当成一个实际训练epoch或评价有效分母。

身份记录完整文件SHA/大小、分区、重复消费与唯一手样本数、归一化明示适配、角色和读取器绑定。`expected`必须通过新一轮实际hash和扫描重算，重新给伪造记录计算摘要也不能代替真实字节检查。运行期间`check()`还核对角色、变换、通道、scalar、原类、memmap映射和图像根路径；重新解析所有crop来源，防止新增高优先级`.lmdb`静默替代已核验的`.data.mdb`。

当前登记旧版release的实际人口消费train/val/test为395,276/3,152/8,612，唯一手样本197,638/1,576/4,306。只有实际manifest、metadata、41条episode记录和完整计数同时匹配才标记`metadata_matches_registered_legacy_release=true`；该标记不是作者对payload的认证，也不代表完整合并语料或PDF条件等价。

## 使用

```python
from egoemg_repro.fusion_literal import compose_fusion
from egoemg_repro.fusion_literal_data import fusion_normalization
from egoemg_repro.fusion_literal_cohort import admit_fusion_cohort, native_fusion_loaders

config = compose_fusion(upstream)
normalization = fusion_normalization(upstream)
cohort = admit_fusion_cohort(data, crops, upstream=upstream,
    config=config, normalization=normalization)
cohort.check()
loaders = native_fusion_loaders(cohort)
```

三个角色均调用固定官方`make_data_module/_make_loader/_collate_fn`，实际batch200；声明的val_batch_size100没有参与。读取器已经重复2遍，加载器不额外重复，保留原全局RNG/训练shuffle与最后不足批次。这里明确采用单rank/workers0接入，不能据此声明多进程、多GPU或真实batch200 GPU容量。

核验不尝试预解码每张JPEG，原读取时坏JPEG错误仍会抛出；文件完整hash不能认证其作者来源或图像语义。已知旧版标签/划分/全episode滤波/固定pooled归一化及缺失ShowEE/Incre合并来源限制继续保留。

## 仍需完成

真实已训练分支来源及官方配置声明的选择角色（视觉last、EMGbest）、原生fusion训练/状态恢复及评价报告入口、实际batch200/bf16 GPU限定验收、150轮完整训练/验证选模/冻结测试/独立复算。这些未实际完成前，入口所有输出保持scientific_eligible/production_ready/full_paper_complete=false。

## 实际服务器核验

[完整输入证据](../results/native-fusion-cohort-full-20261028.json)与[完整安全身份](../results/native-fusion-cohort-full-identity-20261028.json)记录两次独立进程完整hash/扫描：54文件共61,379,659,366bytes，首次176.65秒、新进程105.07秒，身份逐字节一致。双手EMG并集65,171,929行/1,042,750,864元素，所有可能中心39,530,541行/1,739,343,804目标元素均有限。两个阶段及外层进程已独立确认退出，77个固定源码/执行文件前后hash不变；原生三角色batch200/workers0、RNG不变、无模型/CUDA。总链287.43秒actual exit0。一次整体审查仍待完成，未因此完成融合训练。
