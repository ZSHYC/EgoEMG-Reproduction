"""Render verified retained-checkpoint Tables12/14/15 and Figure12 evidence.

The pinned official plotter writes private intermediate PDFs unchanged; Poppler
exports public PNG/SVG. Full-range companions expose points its2.5degree axes
clip. No raw sample records or machine paths are copied into public outputs.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

REVISION='f7cbd048b7092139852270491d3859cd472e0311'
PROTOCOLS=('official_unmasked','masked_pooled')
FAMILIES=('Single-hand','Symmetric bimanual','Asymmetric bimanual')
COLORS={'Single-hand':'#1f77b4','Symmetric bimanual':'#8c564b','Asymmetric bimanual':'#9467bd'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def flat(row,protocol):
    value=dict(n_requested=row['n_requested'],n_valid=row['n_valid'])
    for model in ('vision','emg','fusion'):
        metric=row[protocol][model];value[model+'_mae_deg']=metric['mae_deg']
        if protocol=='masked_pooled':
            value[model+'_absolute_error_sum_rad']=metric['absolute_error_sum_rad']
            value[model+'_valid_angle_count']=metric['valid_angle_count']
        else:
            value[model+'_sample_error_sum_rad']=metric['sample_error_sum_rad']
            value[model+'_sample_count']=metric['sample_count']
    value['delta_vision_fusion_deg']=row['official_gain_vision_fusion_deg' if protocol=='official_unmasked' else 'masked_gain_vision_fusion_deg']
    return value


def companion(rows,path,*,view,protocol,valid_samples,precision_label):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.6,5.4))
    maximum=math.ceil(max(max(r['vision_mae_deg'],r['fusion_mae_deg']) for r in rows))+1
    ax.plot([0,maximum],[0,maximum],'k--',linewidth=.8,alpha=.45)
    for family in FAMILIES:
        selected=[r for r in rows if r['family']==family]
        ax.scatter([r['vision_mae_deg'] for r in selected],[r['fusion_mae_deg'] for r in selected],
            s=[max(18,min(130,math.sqrt(r['n'])*4.5)) for r in selected],c=COLORS[family],
            edgecolors='white',linewidth=.5,alpha=.88,label=family)
    # Explicit companion: show every negative-gain class and the native-clipped
    # points, rather than altering the pinned plotter's annotation/axis loops.
    for row in rows:
        if row['fusion_mae_deg']>row['vision_mae_deg'] or min(row['vision_mae_deg'],row['fusion_mae_deg'])<2.5:
            ax.annotate(row['name'],(row['vision_mae_deg'],row['fusion_mae_deg']),xytext=(6,5),
                textcoords='offset points',fontsize=8,arrowprops=dict(arrowstyle='->',lw=.5))
    condition,crops=view.split('/')
    population=('All test splits' if condition=='all_test' else 'Gesture label group')+', '+('all requests' if crops=='all_requested' else 'crop-present')
    formula=('Official formula' if protocol=='official_unmasked' else 'Masked aligned center')+', '+precision_label
    title=f'Retained legacy: {population}\n{formula}; N={sum(r["n"] for r in rows):,}, valid={valid_samples:,}, G={len(rows)}'
    ax.set(xlim=(0,maximum),ylim=(0,maximum),xlabel='Vision-only MAE (degrees)',ylabel='Fusion MAE (degrees)')
    ax.set_title(title,fontsize=10)
    ax.set_aspect('equal');ax.legend(loc='lower right',fontsize=8);fig.tight_layout()
    fig.savefig(path.with_suffix('.png'),dpi=180,metadata={'Software':'EgoEMG-Reproduction'})
    fig.savefig(path.with_suffix('.svg'),metadata={'Date':None,'Creator':'EgoEMG-Reproduction'})
    plt.close(fig)


def render(args):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.hashsalt']='egoemg-retained-gesture-analysis-v1'
    os.environ['SOURCE_DATE_EPOCH']='0'
    upstream=args.upstream.resolve()
    assert subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()==REVISION
    assert not subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip()
    data=json.loads(args.input.read_text())
    assert data['format']=='egoemg_verified_retained_paper_analysis_v1' and data['official_revision']==REVISION
    assert data['execution']['actual_exit_code']==0 and data['execution']['consumers_independently_nonlive']==14
    assert data['execution']['independent_numeric_and_target_pair_checks'] and not data['training_performed'] and not data['full_paper_complete']
    assert set(data['models'])=={'emg-s','emg-m','emg-l','rn18','fusion-rn18-s'}
    assert all(m['complete_test_pass'] and m['baseline_verification']['agreement'] for m in data['models'].values())
    precision_label='/'.join(sorted({m['precision'] for m in data['models'].values()})).replace('bfloat16','bf16')
    assert not any(s in json.dumps(data) for s in ('/home/','/data/','private_subject','private_center_row'))
    output=args.output;output.mkdir(parents=True);args.pdf_workdir.mkdir(parents=True)
    source=upstream/'scripts/paper/generate_per_gesture_appendix.py';source_sha=sha(source)
    spec=importlib.util.spec_from_file_location('original_gesture_appendix',source)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    write_json(output/'source-analysis.json',data)
    write_json(output/'joint-groups.json',data['joint_tables'])
    write_json(output/'gesture-tables.json',data['gesture_tables'])
    joint_rows=[]
    order=('proximal','mid','distal','thumb','index','middle','ring','pinky','wrist_FE','wrist_RU')
    for group in order:
        row=dict(group=group,indices=','.join(map(str,data['joint_tables']['emg-s']['groups'][group]['indices'])))
        for label in ('emg-s','emg-m','emg-l'):
            metric=data['joint_tables'][label]['groups'][group]
            for field in ('mae_deg','absolute_error_sum_rad','valid_angle_count'):row[label+'_'+field]=metric[field]
        joint_rows.append(row)
    write_csv(output/'table12-legacy-pooled-joint-groups.csv',joint_rows)
    summaries=[];native_checks=[]
    for view,table in data['gesture_tables']['views'].items():
        assert len(table['rows'])==59 and {x['id'] for x in table['rows']}==set(range(60))-{50}
        for protocol in PROTOCOLS:
            name=protocol+'-'+view.replace('/','-');gain='official_gain_vision_fusion_deg' if protocol=='official_unmasked' else 'masked_gain_vision_fusion_deg'
            ordered=sorted(table['rows'],key=lambda row:(FAMILIES.index(row['family']),
                row[gain] is None,-row[gain] if row[gain] is not None else 0,row['id']))
            write_csv(output/('table15-'+name+'.csv'),[dict(id=row['id'],name=row['name'],family=row['family'],**flat(row,protocol)) for row in ordered])
            write_csv(output/('table14-'+name+'.csv'),[dict(family=row['family'],observed_gestures=row['observed_gestures'],**flat(row,protocol)) for row in table['families']])
            rows=[dict(gesture_class=row['id'],name=row['name'],family=row['family'],n=row['n_requested'],
                vision_mae_deg=row[protocol]['vision']['mae_deg'],emg_mae_deg=row[protocol]['emg']['mae_deg'],
                fusion_mae_deg=row[protocol]['fusion']['mae_deg']) for row in ordered if row[protocol]['vision']['mae_deg'] is not None]
            assert rows
            work=args.pdf_workdir/name;(work/'tables').mkdir(parents=True)
            if protocol=='official_unmasked':
                stats=native.compute_family_stats(rows)
                for family in table['families']:
                    if not family['n_requested']:continue
                    expected=family[protocol];actual=stats[family['family']]
                    assert actual['n_samples']==family['n_requested']
                    for field,model in (('vis_avg','vision'),('emg_avg','emg'),('fus_avg','fusion')):
                        assert abs(actual[field]-expected[model]['mae_deg'])<1e-10
                if set(stats)==set(FAMILIES):
                    native.generate_family_table(stats,work)
                    shutil.copyfile(work/'tables/per_gesture_family.tex',output/('native-table14-'+name+'.tex'))
                native.generate_full_table(rows,work)
                shutil.copyfile(work/'tables/per_gesture_full.tex',output/('native-table15-'+name+'.tex'))
            native.generate_scatter(rows,work)
            pdf=work/'figures/per_gesture_scatter.pdf';prefix=output/('figure12-'+name)
            subprocess.run(['pdftoppm','-r','180','-singlefile','-png',str(pdf),str(prefix)],check=True,capture_output=True)
            subprocess.run(['pdftocairo','-svg',str(pdf),str(prefix.with_suffix('.svg'))],check=True,capture_output=True)
            clipped=[row['gesture_class'] for row in rows if min(row['vision_mae_deg'],row['fusion_mae_deg'])<2.5]
            companion(rows,output/('full-range-'+name),view=view,protocol=protocol,valid_samples=table['counts']['valid_samples'],precision_label=precision_label)
            observed=len(rows);improved=sum(row['vision_mae_deg']>row['fusion_mae_deg'] for row in rows)
            summaries.append(dict(view=view,protocol=protocol,**table['counts'],observed_classes=observed,improved_classes=improved,
                mae_deg={m:table['overall'][protocol][m]['mae_deg'] for m in ('vision','emg','fusion')},native_axis_clipped_class_ids=clipped))
            native_checks.append(dict(view=view,protocol=protocol,pdf_sha256=sha(pdf),
                original_plotter_used=True,full_range_companion=True,native_axis_clipped_class_ids=clipped,
                presentation_annotation_policy='label negative gains and native-clipped points; crowded original top4 names remain in CSV/native reference'))
    write_json(output/'view-summary.json',summaries)
    all_view=data['gesture_tables']['views']['all_test/all_requested'];main=all_view['overall'];crop=data['gesture_tables']['views']['all_test/crop_present']
    lines=['# 逐关节与逐手势分析：旧版完整实验的派生评价','',
        '2026-10-28完成五个冻结检查点的完整测试集推理及退出后独立复核。本目录支撑Table12/14/15和Figure12的方法与旧版实验分析；没有新增训练，也不代表PDF原表或当前官方0.02配方已复现。','',
        '原始S/M/L、RN18、RN18+S各2,792个手中心，全部中心指标均通过原1e-4度/计数精确相等门槛；三EMG完整轨迹还核对了整体、左右手、split、组件与逐用户原结果。七个实际受管理阶段139.31秒exit0，14个worker/guardian及外层拥有者独立确认退出，随后纯CPU/无Torch重读并独立核对所有类/族群/角度统计和三模型真值配对。','',
        '## 人口和统计口径','',
        f"全部请求2,792，排除未分配动作18，raw50排除0，得到59类的{all_view['counts']['eligible']:,}个请求；其中{all_view['counts']['valid_samples']:,}个至少有一个有效角度，84个所有角度均无效。有图人口为{crop['counts']['requested']:,}个请求，排除18个未分配后{crop['counts']['eligible']:,}个。它们不是PDF的4,278个手样本。",'',
        '- `official_unmasked`按固定官方逐手势代码：Standalone EMG取原始token中点，视觉/融合取中心；22角均值不使用mask，随后等样本平均，转角度系数57.3。它在全部请求视图保留84个全角无效请求，只用于如实说明代码口径，不能当作有效监督精度。','- `masked_pooled`保留既有legacy协议：EMG按510上下文裁剪并插值后取输入中心，逐角mask有效性，按误差和/有效角数汇总，转角度系数180/pi。表中N_requested、N_valid和valid_angle_count各自保留。','- 五次推理保留原运行的bfloat16、matmul high，以及S/M/L各batch500、RN18 batch900、融合batch200，使既有baseline可以独立核对。官方公式视图复用取点、无mask、均值和57.3规则，精度仍为本次冻结legacy推理；未运行原脚本默认FP32的完整评价流程。','- 四个预先定义视图为all_test/gesture_only乘以all_requested/crop_present；activity只记录与配对，不额外过滤。Gesture标签组实际覆盖24类；名称不能证明等同论文的10类heldout定义。','- 所有表从未舍入底表生成；59类定义行保留，零覆盖的MAE为null/CSV空白。族群按固定非连续ID集合。官方公式的族群加权另与原`compute_family_stats`核对，差异低于1e-10度。','',
        '## Table14/15与Figure12','',
        '| 视图 | 公式 | 参与统计请求 | 至少一角有效 | 覆盖类 | 融合改善类 | Vision | EMG | Fusion |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in summaries:
        v=row['mae_deg'];lines.append(f"| {row['view']} | {row['protocol']} | {row['eligible']} | {row['valid_samples']} | {row['observed_classes']} | {row['improved_classes']} | {v['vision']:.6f} | {v['emg']:.6f} | {v['fusion']:.6f} |")
    lines += ['', '在本次已声明旧版队列、seed42与原验证选模下，全测试59类中58类融合误差低于RN18；这是本项目实测，不能替代论文54/59或证明多seed稳定性。全测试mask口径的MAE为Vision '+f"{main['masked_pooled']['vision']['mae_deg']:.6f}° / EMG {main['masked_pooled']['emg']['mae_deg']:.6f}° / Fusion {main['masked_pooled']['fusion']['mae_deg']:.6f}°。",'',
        '![清晰展示图：全测试、官方公式](full-range-official_unmasked-all_test-all_requested.png)','',
        '![清晰展示图：有图子集、有效角度mask](full-range-masked_pooled-all_test-crop_present.png)','',
        '逐类CSV遵循官方族群顺序，并在每族内按未舍入Vision−Fusion降序排列，空类置后；族群CSV、原始JSON和相同数据的PNG/SVG同时保存。原脚本生成的LaTeX片段只用于官方公式，宏与表格环境需由论文工程提供。','',
        '## 原始绘图行为与补图','',
        '名为`figure12-*`的原生参考PNG/SVG由固定官方`generate_scatter`原样输出的PDF经Poppler转换；PDF仅作为私有中间文件。原生参考保留2.5度坐标下界和固定偏移的Top4标注；后者在全测试视图会出现原脚本自身的文字重叠。清晰展示使用`full-range-*`，保留相同数据、族群色及sqrt(N)截断到18–130的点面积，从0度显示全部点，标出退化类和原范围外的点，Top4名字可查未舍入CSV。Gesture-only的PinkyHook落在原下界外，清晰图完整显示。原脚本的负收益标注循环在首个非负收益处break，清晰图与表格仍完整列出退化类。原始脚本与原生参考均未修改。','',
        '![Gesture-only全范围补图，显示原坐标范围外的点](full-range-official_unmasked-gesture_only-all_requested.png)','',
        '## Table12：三个EMG完整轨迹的逐组误差','',
        '| 组 | S | M | L |','| --- | ---: | ---: | ---: |']
    for name in order:lines.append('| '+name+' | '+' | '.join(f"{data['joint_tables'][label]['groups'][name]['mae_deg']:.6f}" for label in ('emg-s','emg-m','emg-l'))+' |')
    lines += ['', '单位为度。对应CSV/JSON同时给出每组弧度误差和、有效角时间计数及22个独立角统计；proximal/mid/distal与五指分组有重叠，不能相加。这里按legacy有效角时间pooled统计，不冒称原表分组Avg或跨用户SD已经等价。','',
        '## 来源与限制','',
        '固定官方f7cbd048b7092139852270491d3859cd472e0311。历史训练代码以Git7b805d4中的train/visual精确字节隔离加载；首次包因使用后来扩展DINO的当前文件而被严格预检拒绝，尚未进入GPU。失败和全部记录保留，旧检查点/数据/来源校验未改写或放宽。实际重新打包后五次推理完成；读取器一次本地路径拼接失败只重做只读读取，没有重复GPU工作。','',
        '仍使用旧版数据、0.01旧训练目标与训练区间标准化；User/Both标签与训练有用户键交集，当前unified来源、0.02新训练、严格PDF标签版本均未因此补齐。独立原始标注、其他模型、遮挡/定性图与全文其余项继续保持未完成。','',
        '- [全部实际来源、指标与完成证据](source-analysis.json) · [四视图摘要](view-summary.json) · [逐手势全部统计](gesture-tables.json) · [逐角/关节组](joint-groups.json)','- [Table12 CSV](table12-legacy-pooled-joint-groups.csv) · [官方公式Table14](table14-official_unmasked-all_test-all_requested.csv) · [官方公式Table15](table15-official_unmasked-all_test-all_requested.csv)','- [生产者与产物摘要](producer-manifest.json)；原始逐样本记录和身份映射保持私有。']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    assert sha(source)==source_sha
    artifacts={p.relative_to(output).as_posix():sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output/'producer-manifest.json',dict(format='egoemg_joint_gesture_render_v1',input_sha256=sha(args.input),
        renderer_sha256=sha(__file__),official_plotter_sha256=source_sha,official_revision=REVISION,
        native_pdf_intermediates=native_checks,artifacts_sha256=artifacts,
        scientific_boundary='retained legacy selected-checkpoint analysis; not full-paper completion'))
    print(json.dumps(dict(files=len(artifacts)+1,native_scatter_pdfs=8,full_range_companions=sum(x['full_range_companion'] for x in native_checks),
        manifest_sha256=sha(output/'producer-manifest.json'),full_paper_complete=False)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input','output','upstream','pdf-workdir'):p.add_argument('--'+name,type=Path,required=True)
    render(p.parse_args())


if __name__=='__main__':main()
