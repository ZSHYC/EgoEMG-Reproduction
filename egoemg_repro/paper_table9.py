"""Read-only per-hand trajectory Table9 from immutable completed reports.

All reported means are independently reconstructed from valid-angle-time sums
and counts. Different historical recipes and test populations remain explicit;
these derived cells do not establish paper numerical or aggregation equivalence.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re

from .classic_emg_cohort import _safe,_json,_StatGuard

FORMAT='retained_complete_five_method_table9_v1'
SPLITS=('gesture','user','both')
HANDS=('left','right')
SOURCE_REPORTS=(
 ('EMGFormer-S','results/completed/emgformer-s/metrics.json','1a263923298e3a033e43deee334e3d40266be77a439517db0cb1077a5e18c8ef'),
 ('EMGFormer-M','results/completed/emgformer-m/metrics.json','a85f9a11818684d4189af0c24d5bd7f7222059a9aa41360732c0743f5b648884'),
 ('EMGFormer-L','results/completed/emgformer-l/metrics.json','c4e6bf1a00141040e4a0db3e649fea1425ffa912269bb45d7f93cb81a92eb6ee'),
 ('vemg2pose','results/original-classic/vemg2pose/metrics.json','ae6ca5322b2369568002733d8a10d4ab9f75d568a9dc02158ee56730a1a80fbe'),
 ('NeuroPose','results/original-classic/neuropose/metrics.json','b4b3e4cb8c96236e0369648bab4b99579b6cec4dedda2f53cc246a0d0db8e6a9'),
)
PDF_REFERENCE={
 'EMGFormer-S':{'left':[12.4,14.7,16.5],'right':[13.2,16.6,18.4]},
 'EMGFormer-M':{'left':[11.5,15.5,17.0],'right':[12.1,16.6,17.8]},
 'EMGFormer-L':{'left':[11.4,15.3,17.6],'right':[12.0,16.0,17.7]},
 'vemg2pose':{'left':[14.7,15.7,17.2],'right':[15.2,17.0,17.5]},
 'NeuroPose':{'left':[15.9,15.4,15.7],'right':[15.7,16.0,16.9]},
}
PDF_SHA256='124318dce75d9af00ab4d8427acf01315cf83ae1992082b783d5a488c2474725'


def _hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

IMPORTED_SHA=_sha(__file__)
BINDING=_hash([SOURCE_REPORTS,PDF_REFERENCE,PDF_SHA256])


def _sources():
    if _sha(__file__)!=IMPORTED_SHA or _hash([SOURCE_REPORTS,PDF_REFERENCE,PDF_SHA256])!=BINDING:
        raise ValueError('Table9 imported source or immutable report/reference binding changed')
    return {n:_sha(Path(__file__).resolve().parents[1]/n)for n in ('egoemg_repro/paper_table9.py','egoemg_repro/classic_emg_cohort.py')}


def _record(value):
    if type(value)is not dict:raise ValueError('actual angle-time sufficient statistic required')
    error,count,mae=(value[k]for k in ('absolute_error_sum_rad','valid_angle_count','mae_deg'))
    if(type(error)not in (int,float)or not math.isfinite(error)or error<0 or type(count)is not int or count<0
        or count==0 and (error!=0 or mae is not None)):
        raise ValueError('finite nonnegative error and exact integer count required')
    expected=float(error)/count*180/math.pi if count else None
    if count and (type(mae)not in (int,float)or not math.isfinite(mae)or not math.isclose(float(mae),expected,rel_tol=2e-12,abs_tol=1e-10)):
        raise ValueError('saved mean differs from exact angle-time error/count')
    return dict(absolute_error_sum_rad=float(error),valid_angle_count=count,mae_deg=expected)


def _margin(cells,target):
    target=_record(target)
    if(sum(c['valid_angle_count']for c in cells)!=target['valid_angle_count']
        or not math.isclose(math.fsum(c['absolute_error_sum_rad']for c in cells),target['absolute_error_sum_rad'],rel_tol=2e-12,abs_tol=1e-8)):
        raise ValueError('trajectory hand/split/overall sufficient statistics do not conserve')


def trajectory_cells(statistics):
    """Six cells; this standalone arithmetic grants no completed-model permit."""
    try:
        s=statistics
        if(s['mode']!='trajectory'or set(s['by_hand_split'])!=set(HANDS)or set(s['by_hand'])!=set(HANDS)
            or set(s['by_split'])!=set(SPLITS)or any(set(s['by_hand_split'][h])!=set(SPLITS)for h in HANDS)):
            raise ValueError('exact bilateral three-split trajectory statistics required')
        cells=[dict(hand=h,split=split,**_record(s['by_hand_split'][h][split]))for h in HANDS for split in SPLITS]
        for h in HANDS:_margin([c for c in cells if c['hand']==h],s['by_hand'][h])
        for split in SPLITS:_margin([c for c in cells if c['split']==split],s['by_split'][split])
        _margin(cells,s['overall']);return cells
    except(KeyError,TypeError,AttributeError)as exc:raise ValueError('complete trajectory cells and all original margins required')from exc


def _selected(value,legacy,method):
    if legacy:
        if(value['completion_status']!='training_complete'or value['benchmark_eligible']is not True
            or value['test']['bounded']is not False or value['model_kind']!='emg'
            or value['training']['validation_history_complete']is not True):raise ValueError('completed selected legacy trajectory model required')
        key=value['identities']['selected_checkpoint_file_sha256']
        selected=dict(selected_checkpoint_file_sha256=key,selected_checkpoint_step=value['training']['selected_checkpoint_step'],
            protocol_identities=copy.deepcopy(value['identities']))
    elif method=='vemg2pose':
        if(value['scientific_eligible']is not True or value['variant']!='vemg2pose'
            or value['selection']['stop_reason']!='early_stopping'):raise ValueError('completed native vemg2pose selected report required')
        key=value['selection']['selected_model_sha256'];selected=copy.deepcopy(value['selection'])
    else:
        if(value['scientific_eligible']is not True or value['variant']!='neuropose'or value['scope']!='full'):
            raise ValueError('completed native NeuroPose selected report required')
        key=value['selected_best']['checkpoint_sha256'];selected=copy.deepcopy(value['selected_best'])
    if type(key)is not str or re.fullmatch('[0-9a-f]{64}',key)is None:raise ValueError('actual selected weight identity required')
    return selected


def build_table9(repo):
    repo=_safe(repo);sources=_sources();code_root=Path(__file__).resolve().parents[1]
    paths=[_safe(repo/n)for _,n,_ in SOURCE_REPORTS]
    guard=_StatGuard([*paths,*[code_root/n for n in sources]])
    models=[];cells=[];gaps=[]
    for index,((method,name,digest),path)in enumerate(zip(SOURCE_REPORTS,paths)):
        if _sha(path)!=digest:raise ValueError('pinned complete Table9 source report bytes differ: '+method)
        value=_json(path);guard.check();legacy=index<3;selected=_selected(value,legacy,method)
        s=value['test']['sampling']['trajectory']if legacy else value['statistics']['trajectory']
        primitive=trajectory_cells(s)
        samples=value['test']['requested_hand_windows']if legacy else value['statistics']['samples']
        length=11490 if legacy else value['statistics']['alignment']['output_length']
        if(type(samples)is not int or samples!=(2792 if legacy else 4306)or length!=(11490 if legacy else 6000 if index==3 else 7790)):
            raise ValueError('complete original method-specific test population/alignment differs')
        protocol='retained legacy EMGFormer'if legacy else'original native '+method+' on legacy release'
        models.append(dict(method=method,source_report=name,source_report_sha256=digest,source_protocol=protocol,
            requested_hand_windows=samples,input_length=12000 if legacy else 7790,output_length=length,
            selected_weights_identity=selected,trajectory_overall=_record(s['overall'])))
        for c in primitive:
            expected=PDF_REFERENCE[method][c['hand']][SPLITS.index(c['split'])]
            cells.append(dict(method=method,source_protocol=protocol,model_test_hand_windows=samples,**c,paper_mae_deg=expected,
                difference_from_paper_deg=c['mae_deg']-expected if c['mae_deg']is not None else None))
        for split in SPLITS:
            l,r=[next(c for c in primitive if c['hand']==h and c['split']==split)for h in HANDS]
            gaps.append(dict(method=method,split=split,left_mae_deg=l['mae_deg'],right_mae_deg=r['mae_deg'],
                absolute_gap_deg=abs(r['mae_deg']-l['mae_deg'])if l['mae_deg']is not None and r['mae_deg']is not None else None))
    guard.check()
    result=dict(format=FORMAT,analysis_complete=True,full_paper_complete=False,scientific_settings_added=0,
        paper_numerical_equivalence_established=False,paper_cell_aggregation_equivalence_established=False,
        all_methods_same_test_population=False,
        aggregation='float64 radian absolute-error sum / exact valid angle-time count; degrees=180/pi; no cell averaging or invented SD',
        source_boundary='three retained legacy EMGFormer protocols and two completed original native classic protocols; different population, window, objective, normalization and masks remain separate; no controlled ranking claim',
        native_emgformer_full_trajectory_replaced=False,models=models,cells=cells,hand_gaps=gaps,
        max_absolute_hand_gap_deg=max(g['absolute_gap_deg']for g in gaps if g['absolute_gap_deg']is not None),
        paper_reference=dict(pdf_sha256=PDF_SHA256,printed_page=23,table=9,visually_inspected=True,
            cell_weighting_specified=False,reported_left_right_consistency_bound_deg=2.,mae_deg=copy.deepcopy(PDF_REFERENCE)),
        source_files_sha256=sources)
    result['sha256']=_hash(result);return result


ARTIFACT_FILES=('metrics.json','cells.csv','hand-gaps.csv','README.md')


def _json_bytes(value):return (json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()


def _csv_bytes(rows):
    import csv
    import io
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows);return stream.getvalue().encode()


def _markdown(report):
    lines=['# Table 9：保留完整结果的逐手轨迹明细','',
        '以下30个单元由五份已完成、验证选模后的报告中的弧度绝对误差和／有效角时间计数独立重算；未新增训练或模型推理。','',
        '**协议分列**：EMGFormer-S/M/L 为旧版保留配方，2,792 个测试手窗口，12000 输入／11490 输出；vemg2pose 与 NeuroPose 为原生配方、4,306 个测试手窗口，7790 输入，分别6000／7790 输出。两组并非同一受控测试人口。',
        '各行沿用各自原模型的标签 mask、归一化、损失和时间对齐；未把中心评价填入轨迹表。','',
        '| 方法 | 手 | Gesture (°) | User (°) | Both (°) | 测试手窗口总数 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    def show(value,sign=False):return '—'if value is None else format(value,'+.6f'if sign else'.6f')
    for model in report['models']:
        for hand in HANDS:
            cells=[c for c in report['cells']if c['method']==model['method']and c['hand']==hand]
            lines.append('| '+ ' | '.join([model['method'],hand,*[show(c['mae_deg'])for c in cells],f"{model['requested_hand_windows']:,}"])+' |')
    lines+=['','## 相对 PDF 显示数值的差（本次减 PDF，°）','',
        'PDF 第23页 Table 9 为单纯 MAE，没有 SD 列；其逐手单元的用户／样本加权方式未明确。下列差值仅对照显示数字，不证明论文人口、聚合方式或配方等价。','',
        '| 方法 | 手 | Gesture Δ | User Δ | Both Δ |','| --- | --- | ---: | ---: | ---: |']
    for model in report['models']:
        for hand in HANDS:
            cells=[c for c in report['cells']if c['method']==model['method']and c['hand']==hand]
            lines.append('| '+' | '.join([model['method'],hand,*[show(c['difference_from_paper_deg'],True)for c in cells]])+' |')
    lines+=['',f"15组模型／划分的最大左右差为 **{report['max_absolute_hand_gap_deg']:.6f}°**。这是已有报告的描述性对账，不是独立的共享权重／镜像策略因果试验。",'',
        '- [完整未舍入单元、来源与 PDF 对照](metrics.json)','- [30单元的误差和／有效角时间计数／MAE／差值](cells.csv)',
        '- [15组左右手差值](hand-gaps.csv)','- [产物完整性记录](completion.json)','',
        '本分析未替换正在等待的原生 EMGFormer 全轨迹评价，不增加科学实验设置。整篇论文复现尚未完成。','']
    return '\n'.join(lines).encode()


def _artifact_bytes(report):
    return {'metrics.json':_json_bytes(report),'cells.csv':_csv_bytes(report['cells']),
        'hand-gaps.csv':_csv_bytes(report['hand_gaps']),'README.md':_markdown(report)}


def _completion(report,files):
    result=dict(format='source_pinned_table9_artifact_completion_v1',report_sha256=report['sha256'],
        files_sha256={n:hashlib.sha256(v).hexdigest()for n,v in files.items()},
        source_reports_sha256={m['source_report']:m['source_report_sha256']for m in report['models']},
        analysis_complete=True,full_paper_complete=False)
    result['sha256']=_hash(result);return result


def _output_location(repo,output,*,exists):
    repo=_safe(repo);out=_safe(output,exists=exists)
    for _,name,_ in SOURCE_REPORTS:
        protected=(repo/name).parent
        if out==protected or out in protected.parents or protected in out.parents:
            raise ValueError('Table9 output overlaps a frozen source report directory')
    return out


def export_table9(repo,output):
    import os
    repo=_safe(repo);out=_output_location(repo,output,exists=False)
    if out.exists():raise ValueError('fresh immutable Table9 output required')
    report=build_table9(repo);files=_artifact_bytes(report);out.mkdir(mode=0o700)
    for name,raw in files.items():
        with(out/name).open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
    if build_table9(repo)!=report:raise ValueError('Table9 original input/source generation changed during export')
    with(out/'completion.json').open('xb')as f:f.write(_json_bytes(_completion(report,files)));f.flush();os.fsync(f.fileno())
    return verify_table9(repo,out)


def verify_table9(repo,output):
    repo=_safe(repo);out=_output_location(repo,output,exists=True)
    names=(*ARTIFACT_FILES,'completion.json');paths=[_safe(out/n)for n in names];guard=_StatGuard(paths)
    report=build_table9(repo);files=_artifact_bytes(report)
    expected={**files,'completion.json':_json_bytes(_completion(report,files))}
    if set(p.name for p in out.iterdir())!=set(names)or any((out/n).read_bytes()!=raw for n,raw in expected.items()):
        raise ValueError('Table9 output bytes differ from independent pinned-source reconstruction')
    guard.check();return report


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('export','verify'))
    parser.add_argument('--repo',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();value=(export_table9 if args.action=='export'else verify_table9)(args.repo,args.output)
    print(json.dumps(dict(format=value['format'],sha256=value['sha256'],cells=len(value['cells']),full_paper_complete=False)))


if __name__=='__main__':main()
