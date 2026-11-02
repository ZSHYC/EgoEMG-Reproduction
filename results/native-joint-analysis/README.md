# 原生S/M/L逐关节分析与两种权重

三个原生200轮／25,000更新模型的完整2792手窗口均已从实际BF16原始预测包重新归约；同一模型的原生／独立重放记录一致。原中心验证BEST169／124／142保持，没有新增模型前向、训练或测试选模。三个模型的请求、目标／输入指纹、mask及hand／split身份逐项一致后，才生成下面的三列表。

每模型保留全部**2792**请求，其中**2760**至少有一个有效角度，**32**为全无效请求。跨split的请求权重是gesture1508、user1076、both208，分母仍2792；全无效请求没有被丢掉。误差和与所有有效角度计数仍与原完整轨迹报告逐项核对。

## 主视图：有效角时间pooled

单位度。累加弧度绝对误差和，再除以有效角度×时间计数；与固定官方helper的pooled归约方式对应。这里读取原验收BF16预测，不把官方另一个逐用户group helper的默认FP32执行当成同一次推理。

| 组 | S | M | L |
| --- | ---: | ---: | ---: |
| Distal | 11.095341 | 11.157554 | 11.555395 |
| Wrist deviation | 11.346086 | 11.362513 | 11.118596 |
| Thumb | 13.958216 | 13.949219 | 14.138894 |
| Index | 15.075410 | 14.926107 | 15.068880 |
| Proximal | 15.494917 | 15.292226 | 15.333670 |
| Ring | 17.833665 | 17.661909 | 17.473049 |
| Middle finger | 17.619145 | 17.349687 | 17.651143 |
| Wrist flexion | 17.070757 | 17.256015 | 16.857555 |
| Pinky | 19.366682 | 19.265244 | 19.831519 |
| Mid-phalanx | 24.997319 | 24.779728 | 25.108052 |

[主视图CSV](table12-native-valid-angle-pooled.csv)保存各组角索引、完整弧度误差和、有效角度分母及PDF原显示值差。十组有重叠，不能相加。

## PDF caption的明确解释：跨split请求数加权

每个split先合并左右手并计算该组pooled MAE，再计算 `sum(N_split * MAE_split) / 2792`。N包含所有请求和全无效请求；若某个非空split的组无有效观察，保留其权重并输出undefined。当前三个模型十组均有有效观察。该定义是对caption的明确解释，没有声称取得作者未发布的Table12归约脚本。

| 组 | S | M | L |
| --- | ---: | ---: | ---: |
| Distal | 11.096602 | 11.157584 | 11.552421 |
| Wrist deviation | 11.324392 | 11.341466 | 11.093035 |
| Thumb | 13.937795 | 13.931307 | 14.119748 |
| Index | 15.068147 | 14.922517 | 15.064761 |
| Proximal | 15.480353 | 15.282671 | 15.321049 |
| Ring | 17.818797 | 17.647939 | 17.457069 |
| Middle finger | 17.614168 | 17.345943 | 17.644203 |
| Wrist flexion | 17.046767 | 17.230109 | 16.827901 |
| Pinky | 19.351720 | 19.253375 | 19.811047 |
| Mid-phalanx | 24.975195 | 24.757939 | 25.082943 |

[请求数加权CSV](table12-caption-interpretation-request-weighted.csv)保存加权分子、2792分母、缺失非空split及PDF显示值差；[逐手／split／组270行CSV](split-hand-group-counts.csv)保存每格请求／全无效请求和有效角度统计。

## 原论文Table12参照

以下30个显示值已逐项对照用户提供PDF物理第24页。该表没有SD列，Wrist deviation对应官方角21（wrist_RU），Wrist flexion对应角20（wrist_FE）。

| 组 | S | M | L |
| --- | ---: | ---: | ---: |
| Distal | 10.2 | 10.1 | 10.0 |
| Wrist deviation | 10.8 | 10.1 | 10.0 |
| Thumb | 12.8 | 12.3 | 12.3 |
| Index | 13.4 | 13.3 | 13.3 |
| Proximal | 13.7 | 13.3 | 13.4 |
| Ring | 15.2 | 14.7 | 14.6 |
| Middle finger | 15.2 | 14.8 | 14.8 |
| Wrist flexion | 15.7 | 14.9 | 14.6 |
| Pinky | 17.3 | 16.7 | 17.0 |
| Mid-phalanx | 21.6 | 20.7 | 20.7 |

当前整体高于原表，不能称为原表数值已复现。当前旧版41段、作者混合来源、单rank／多seed、自然人身份与源标签、上游episode预过滤和作者聚合等价仍未证明；两种当前权重的差异也不能解释为修复这些来源差异。当前BF16、12,000输入／11,490步对齐轨迹；该表不使用6000点中心或6255对齐中心来替代完整轨迹。

## 实际来源与独立完成证据

[总验收凭据](postexit-acceptance-20261102.json)绑定固定源胶囊、三个原报告／fit／BEST、全原始包和来源代际。三个CPU读取阶段与三个独立标准库复算阶段均退出0；12个worker／guardian加两父拥有者共14个身份物理关闭，另一个退出后导出读取器也已关闭。独立复算不调用主权重函数，检查所有公共格、分母、来源不变及真实producer退出后，拥有控制器才提升analysis状态。

- [S原始聚合及完成链](small/analysis.json) · [M](middle/analysis.json) · [L](large/analysis.json) · [实际工作流](workflow-result.json)
- [全部表格JSON与PDF SHA](table12-native-two-weight-views.json) · [完整轨迹原报告与200轮曲线](../native-emgformer-trajectory/README.md)
- [生产读取CLI](../../scripts/paper/collect_native_joint_analysis.py) · [纯归约](../../egoemg_repro/native_joint_analysis.py) · [实际控制器归档](controller-source.py)（需原private请求胶囊，非独立通用入口）
- [51项新组件检查／一次审查修复](review-fix-engineering-tests-20261102.json)：真实BF16包、来源／actor替换、公开用户别名映射、全无效权重、独立标准库复算与隐私schema均覆盖。旧组件套件与已结束模型没有重跑。

原始每窗口统计、用户／样本映射及输入路径保留私有；公共数字仅为聚合和哈希。旧legacy关节表和既有结果保持原字节。本项完成可得旧版三规模原生关节分析，不能关闭EMG2Pose Table13、标注／腕角生成或整篇论文目标。
