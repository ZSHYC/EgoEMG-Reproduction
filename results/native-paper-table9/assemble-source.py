"""Reconstruct the frozen native Table9 artifact; no model or raw tensor IO.

Run with --repo PATH --output FRESH_DIRECTORY. The current folder initially
contains this archived producer only; existing generated files are preserved.
"""
from pathlib import Path
import argparse,copy,csv,hashlib,io,json,math,shutil,sys

# These are the exact already accepted source bytes, not caller-selected inputs.
PINS = {'egoemg_repro/classic_emg_cohort.py': '72267d660f55e9f5fbbd2e1e17edf07627ef01d2ba09e92fca1086c50a16dbd3',
 'egoemg_repro/paper_table9.py': '101e7732ee4e7230200b8fa5c071378df80b8f47bc851121704bca834b825aed',
 'results/native-emgformer-trajectory/all-sizes-complete-20261102.json': '5c7dcff61cb5adf2b708d150cb89d891eee71e72eb8ea475cec2b64838025bb7',
 'results/native-emgformer-trajectory/large/completion.json': 'a48d2a4ce0e177bafeaf0421e88ed031d5c2d07cc88bbe2f174849c119cc757d',
 'results/native-emgformer-trajectory/large/metrics.json': '0a87b22f8a939e07edece7711814f06be95d86bbd6d20d11ef14bc47c708b241',
 'results/native-emgformer-trajectory/large/metrics.md': 'ed72782e0451f9419a594a7f1af5bd1c24ff0efc36c7fe86be8a197811fecf43',
 'results/native-emgformer-trajectory/large/metrics.pending.json': 'e6824017718f87ffc8c0bdcb9844079b96dd02d91bbc2e0c385715131e886d4c',
 'results/native-emgformer-trajectory/large/postexit-acceptance.json': '946bc7121ac5eac94372cc6249b04474b83633049092c7d58241e6bf784a9b84',
 'results/native-emgformer-trajectory/large/verification.json': '2c6244403937c7692abdd4b9fc443b83950d7c809b004903de782ce16f27d702',
 'results/native-emgformer-trajectory/large/workflow-result.json': '6b1ac6fb20ecd85ef08129bb87ee7691d43e66430e3e496b4ed49e7f721467bd',
 'results/native-emgformer-trajectory/middle/completion.json': 'd6fb12d377ada99289ff95e53b0b70165b9fc9539df1b21f131d13a5ceb8878b',
 'results/native-emgformer-trajectory/middle/metrics.json': '4848d1606d1e90f2af28351fdd7243f56a46921d193bc30ed55907185c290c7c',
 'results/native-emgformer-trajectory/middle/metrics.md': '55517812875ede01b2f72eb92be28cfa538b7b0545b1a3fc298b0933f4becb52',
 'results/native-emgformer-trajectory/middle/metrics.pending.json': '919e0143e0cb07f8e008f03ac0073973a88cb5c8bf19307b0d29999e7d09a82e',
 'results/native-emgformer-trajectory/middle/postexit-acceptance.json': '667d29bf4cf29a46c52e69e248ff807ff9fa39e6e7f9e5b0fd22df4abdf96b3b',
 'results/native-emgformer-trajectory/middle/verification.json': 'a50b163cacdc636ff3aa772f430e1ab36cc5f07dd547f1e9c5eea105d31d893a',
 'results/native-emgformer-trajectory/middle/workflow-result.json': 'f8ed3d1c9b7492b35a2687f8e398a49bae027999abc22c9e2f904177aa130444',
 'results/native-emgformer-trajectory/small/completion.json': '6f63776e5cd4d75843dcd1ca028cb0182c4bf0cf3565ca2e980daa41c2080c57',
 'results/native-emgformer-trajectory/small/metrics.json': '9d4df56557c25dc838a7e6cdb020866f667574d9c431697166a1ea1c355917c9',
 'results/native-emgformer-trajectory/small/metrics.md': 'b5aa667d89aed3924dc761ae6ecc9325a0fd3311dc1d43f1b6de270f409d46a5',
 'results/native-emgformer-trajectory/small/metrics.pending.json': 'ebcaadfdc1cc7ac571e999ffcfc3e655b5d9982df58163b091a62c01a066dd97',
 'results/native-emgformer-trajectory/small/postexit-acceptance.json': '764a2172bd8ac8437c059dad735322b979cdd185d493d8c72d42960b125bde3d',
 'results/native-emgformer-trajectory/small/verification.json': '9b4d459849f27ed9a27078acb87c73dcff5b48c0112712283af507967207d4cf',
 'results/native-emgformer-trajectory/small/workflow-result.json': 'a0924d46ffcc4065dc7c4ebb431be8cd2cf9980966f22787805236983900fc53',
 'results/native-joint-analysis/large/analysis.json': 'a6a0e2e2f9a55318150ebac05bd674278a6eed18ca74b5ba6c0ed8177802cf14',
 'results/native-joint-analysis/middle/analysis.json': '0ac2a294f185da68c0d18448928aae8a59c14b70e41b0eaf3cfbc622e68015f3',
 'results/native-joint-analysis/small/analysis.json': '9c00af900c3256ed0bf80cc63115515524c9f36daa39efae024a687f5c4f68e5',
 'results/original-classic/neuropose/metrics.json': 'b4b3e4cb8c96236e0369648bab4b99579b6cec4dedda2f53cc246a0d0db8e6a9',
 'results/original-classic/vemg2pose/metrics.json': 'ae6ca5322b2369568002733d8a10d4ab9f75d568a9dc02158ee56730a1a80fbe',
 'results/paper-table9/README.md': '80b49ddb668c51e81c3cb795cb45e7ba398220f5008405a08f02adf9b3e1e2dc',
 'results/paper-table9/cells.csv': '2f01b09ff6130336e444681722da0e35333245089608ded996ae2097a592b47f',
 'results/paper-table9/completion.json': 'b670338282ca1c634fc219ee6f8599630aadd747f4a8d5a57194ae9b89704548',
 'results/paper-table9/hand-gaps.csv': '3953529c03bda541667c90bdd6455ad9903cb091be3566f50cb7f8cbb38045c5',
 'results/paper-table9/metrics.json': '7a2a8c021e48d19c73df7fb83edf54e3a046160e9d7235d04d4b4e5bc4ce8176'}
HANDS=('left','right');SPLITS=('gesture','user','both')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def seal(v):return dict(v,sha256=digest(v))
def stamp(p):
 assert p.is_file()and not any(a.is_symlink()for a in(p,*p.parents)),'regular nonsymlink source required'
 s=p.stat();return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_uid,s.st_gid,s.st_nlink)
def json_bytes(v):return(json.dumps(v,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
def csv_bytes(rows):
 stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows);return stream.getvalue().encode()
def read(p):return json.loads(p.read_text())
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
 repo=args.repo.resolve(strict=True);out=args.output.absolute();producer=Path(__file__).resolve()
 assert not any(p.is_symlink()for p in(out,*out.parents)),'output symlink forbidden'
 paths={name:repo/name for name in PINS};stamps={name:stamp(p)for name,p in paths.items()}
 assert all(sha(paths[name])==expected for name,expected in PINS.items()),'accepted source bytes changed'
 def check():assert all(stamp(paths[name])==before for name,before in stamps.items()),'source generation changed'
 check();sys.path.insert(0,str(repo))
 from egoemg_repro.paper_table9 import trajectory_cells
 assert 'torch'not in sys.modules and'pytorch_lightning'not in sys.modules,'no learned runtime imports allowed'
 prior=read(paths['results/paper-table9/metrics.json']);old_completion=read(paths['results/paper-table9/completion.json'])
 assert prior['sha256']==digest({k:v for k,v in prior.items()if k!='sha256'})and prior['analysis_complete']
 assert old_completion['report_sha256']==prior['sha256']and old_completion['analysis_complete']
 for name,h in old_completion['files_sha256'].items():assert sha(repo/'results/paper-table9'/name)==h
 allclosed=read(paths['results/native-emgformer-trajectory/all-sizes-complete-20261102.json'])
 assert allclosed['closure_actor_count']==130 and allclosed['all130_original_size_loader_and_additional_reader_actors_physically_closed']and allclosed['source_protocol_complete_on_available_legacy_cohort']
 assert not allclosed['full_paper_complete']and not allclosed['paper_numerical_equivalence_established']
 models=[];cells=[];gaps=[];populations=[]
 for size,label,epoch in [('small','S',169),('middle','M',124),('large','L',142)]:
  name='results/native-emgformer-trajectory/'+size+'/metrics.json';r=read(paths[name]);proof=read(paths['results/native-emgformer-trajectory/'+size+'/postexit-acceptance.json'])
  assert r['sha256']==digest({k:v for k,v in r.items()if k!='sha256'})==proof['actual_report_sha256']
  assert r['scientific_eligible']and r['trajectory_pipeline_complete']and not r['full_paper_complete']and not r['paper_numerical_equivalence_established']
  assert r['training']['scope']=='full'and r['training']['completed_epochs']==200 and r['training']['optimizer_executions']==25000 and r['training']['full_training_terminated']
  assert r['training']['selected_best']==proof['selected_best']and r['fit_result_sha256']==proof['actual_fit_result_sha256']and proof['selected_completed_epoch']==epoch
  assert proof['scientific_eligible']and proof['all24_original_loader_workers_closed']and len(proof['all16_original_size_actors_closed'])==16 and len(proof['all2_independent_reader_actors_closed'])==2
  assert sha(paths['results/native-emgformer-trajectory/'+size+'/postexit-acceptance.json'])==allclosed['per_size_local_proof_file_sha256'][size]
  assert all(sha(repo/'results/native-emgformer-trajectory'/size/file)==h for file,h in proof['original_report_files_sha256'].items())
  test=r['test'];alignment=test['alignment'];assert test['consumed_samples']==test['unique_samples']==2792 and test['duplicate_consumptions']==0
  assert alignment==dict(window_length=12000,left_context=510,right_context=0,raw_prediction_time_steps=230,aligned_prediction_time_steps=11490,aligned_center_index=5745,source_aligned_center_index=6255,original_point_center_index=6000)
  j=read(paths['results/native-joint-analysis/'+size+'/analysis.json']);assert j['source_report_sha256']==r['sha256']and j['source_run_scientific_eligible']and j['analysis_reader_complete']
  assert j['views']['n_requested']==2792 and j['views']['n_with_any_valid_angle']==2760
  populations.append(j['request_population_sha256']);method='EMGFormer-'+label;protocol='original native200 EMGFormer; legacy release;0.02; BF16'
  primitive=trajectory_cells(test['trajectory']);assert sum(c['valid_angle_count']for c in primitive)==684769448
  models.append(dict(method=method,source_report=name,source_report_sha256=PINS[name],source_protocol=protocol,requested_hand_windows=2792,input_length=12000,output_length=11490,selected_weights_identity=dict(fit_result_sha256=r['fit_result_sha256'],selected_best=r['training']['selected_best'],selected_completed_epoch=epoch),trajectory_overall=copy.deepcopy(test['trajectory']['overall']),source_postexit_proof_sha256=PINS['results/native-emgformer-trajectory/'+size+'/postexit-acceptance.json'],request_population_sha256=j['request_population_sha256'],request_counts_by_hand_split={h:{s:dict(n_requested=j['views']['by_hand_split'][h][s]['n_requested'],n_with_any_valid_angle=j['views']['by_hand_split'][h][s]['n_with_any_valid_angle'])for s in SPLITS}for h in HANDS}))
  for c in primitive:
   ref=prior['paper_reference']['mae_deg'][method][c['hand']][SPLITS.index(c['split'])]
   cells.append(dict(method=method,source_protocol=protocol,model_test_hand_windows=2792,**c,paper_mae_deg=ref,difference_from_paper_deg=c['mae_deg']-ref if c['mae_deg']is not None else None))
 for method in ('vemg2pose','NeuroPose'):
  model=copy.deepcopy(next(x for x in prior['models']if x['method']==method));assert sha(repo/model['source_report'])==model['source_report_sha256']==old_completion['source_reports_sha256'][model['source_report']]
  r=read(repo/model['source_report']);assert r['scientific_eligible']and not r['full_paper_complete']and r['statistics']['samples']==4306
  primitive=trajectory_cells(r['statistics']['trajectory']);selected=[copy.deepcopy(c)for c in prior['cells']if c['method']==method]
  assert len(selected)==len(primitive)==6
  for c,p in zip(selected,primitive):assert all(c[k]==p[k]for k in('hand','split','absolute_error_sum_rad','valid_angle_count','mae_deg'))
  models.append(model);cells.extend(selected)
 assert len(set(populations))==1 and len(models)==5 and len(cells)==30
 for model in models:
  for split in SPLITS:
   l,r=[next(c for c in cells if c['method']==model['method']and c['hand']==h and c['split']==split)for h in HANDS]
   gaps.append(dict(method=model['method'],split=split,left_mae_deg=l['mae_deg'],right_mae_deg=r['mae_deg'],absolute_gap_deg=abs(r['mae_deg']-l['mae_deg'])if l['mae_deg']is not None and r['mae_deg']is not None else None))
 above=[g for g in gaps if g['absolute_gap_deg']is not None and g['absolute_gap_deg']>prior['paper_reference']['reported_left_right_consistency_bound_deg']]
 report=seal(dict(conditions_exceeding_reported_pdf_lr_bound=above,paper_reported_lr_bound_met_by_current_all_cells=not above,paper_lr_consistency_prose_printed_page=22,format='source_pinned_native_SML_plus_retained_native_classic_table9_v1',analysis_complete=True,scientific_settings_added=0,new_learned_forwards=0,new_optimizer_updates=0,full_paper_complete=False,paper_numerical_equivalence_established=False,paper_cell_aggregation_equivalence_established=False,all_methods_same_test_population=False,native_SML_request_target_mask_population_equal=True,native_SML_request_population_sha256=populations[0],legacy_table9_artifacts_modified=False,classic12_cells_unchanged=True,aggregation='sum radian absolute error / exact valid angle-time count; degrees180/pi; no cell averaging or invented SD',source_boundary='native200 S/M/L2792 windows and completed original native classic4306 windows; different grids, context, normalization/masks and precision remain explicit',models=models,cells=cells,hand_gaps=gaps,max_absolute_hand_gap_deg=max(g['absolute_gap_deg']for g in gaps if g['absolute_gap_deg']is not None),paper_reference=copy.deepcopy(prior['paper_reference']),source_files_sha256=PINS,producer_source_sha256=sha(producer)))
 lines=['# Table9：原生S/M/L与已完成classic的逐手轨迹对照','','三个原生200轮／25,000更新模型的完整轨迹报告已结束，本表将其18个逐手／split单元与两份已完成classic的12个原单元分列；没有新增推理、训练或测试选模。旧legacy Table9及其全部产物保持原字节。','','S/M/L使用原中心验证BEST169／124／142，2,792手窗口、12,000输入／11,490输出、0.02配方与BF16；三个模型的完整请求／目标／mask人口指纹相同。每模型32个全无效请求仍在原2792队列中；本表按有效角时间计数归约。vemg2pose／NeuroPose沿用原已完成报告的4,306手窗口、7,790输入及6,000／7,790输出，不能混称五方法同一受控人口。','','| 方法 | 手 | Gesture (°) | User (°) | Both (°) | 模型测试手窗口总数 |','| --- | --- | ---: | ---: | ---: | ---: |']
 for m in models:
  for h in HANDS:
   row=[next(c for c in cells if c['method']==m['method']and c['hand']==h and c['split']==s)for s in SPLITS]
   lines.append('| '+' | '.join([m['method'],h,*[format(c['mae_deg'],'.6f')for c in row],str(m['requested_hand_windows'])])+' |')
 lines+=['','## 对照、分母与边界','','[30个未舍入单元CSV](cells.csv)给出每格弧度误差和、有效角时间计数、PDF原显示值及描述性差；[15组左右差CSV](hand-gaps.csv)保留两个手均值。六格分别与逐手、split和原整体误差和／计数守恒。','',f"五方法15个条件的最大左右差为{report['max_absolute_hand_gap_deg']:.6f}°；当前3组超过论文D.1物理第22页所写的2°范围：S的User／Both及L的User；原文的逐手一致性结论尚未在本数据与pooled口径上复现。这是描述性差值，不是共享权重或物理镜像的因果试验。",'','PDF物理第23页Table9不列SD，逐手单元的用户／样本加权方式未明确。PDF SHA及30个显示值沿用已逐项视觉核验并锁定来源的旧表，完整参照在JSON；本表有效角时间pooled不证明原表聚合或人口等价。','','源报告来自可得旧版41段。当前完整PDF人口、精确作者混合来源、多rank／多seed、生成标签及上游整episode预过滤的限制保持；组间不同上下文、窗口网格、mask与归一化也保留。完整轨迹与6000原中心／6255对齐中心分开；本表没有用中心填轨迹或重选测试权重。','','## 来源与重建','','- [完整报告、选定权重、原生请求计数与所有源SHA](metrics.json) · [产物完成记录](completion.json)','- [原生完整轨迹与200轮曲线](../native-emgformer-trajectory/README.md) · [原生关节双权重／人口核验](../native-joint-analysis/README.md)','- [旧legacy Table9（保持原字节）](../paper-table9/README.md)','- [固定源重建脚本](assemble-source.py)：复用未改动的trajectory_cells归约及既有NumPy依赖，不载入Torch；在源码仓库中运行 `python results/native-paper-table9/assemble-source.py --repo . --output /tmp/new-native-table9` 可生成相同产物字节。','','原生S/M/L来源均有真实七阶段、完整输入／原始包独立读取与各自进程关闭凭据；两父队列和共130个原拥有者／DataLoader／额外读取身份已独立关闭。源绑定、算术和表格完成不构成整篇论文完成。','']
 files={'metrics.json':json_bytes(report),'cells.csv':csv_bytes(cells),'hand-gaps.csv':csv_bytes(gaps),'README.md':'\n'.join(lines).encode(),'assemble-source.py':producer.read_bytes()}
 completion=seal(dict(format='actual_native_table9_source_bound_artifact_completion_v1',report_sha256=report['sha256'],source_reports_sha256={m['source_report']:m['source_report_sha256']for m in models},files_sha256={name:hashlib.sha256(body).hexdigest()for name,body in files.items()},analysis_complete=True,source_or_training_completion_created_by_arithmetic=False,new_learned_forwards=0,new_optimizer_updates=0,full_paper_complete=False))
 files['completion.json']=json_bytes(completion);check()
 if out.exists():assert out.resolve()==producer.parent and list(out.iterdir())==[producer],'fresh output required'
 else:out.mkdir(mode=0o755,parents=True)
 for name,body in files.items():
  p=out/name
  if p.resolve()==producer:assert p.read_bytes()==body;continue
  with p.open('xb')as f:f.write(body)
 check();assert set(p.name for p in out.iterdir())==set(files)
 assert all((out/name).read_bytes()==body for name,body in files.items());assert 'torch'not in sys.modules and'pytorch_lightning'not in sys.modules
 print(json.dumps(dict(cells=30,hand_gaps=15,max_absolute_hand_gap_deg=report['max_absolute_hand_gap_deg'],all_native_SML_source_protocols_complete=True,full_paper_complete=False)))
if __name__=='__main__':main()
