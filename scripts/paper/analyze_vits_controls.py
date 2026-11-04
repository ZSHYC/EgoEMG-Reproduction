"""Exploratory clustered contrasts from the full, replay-verified raw errors."""
from pathlib import Path
import argparse,csv,hashlib,io,json,math
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('results/visual-missing-emg'))
parser.add_argument('--verify',action='store_true',help='compare existing numeric artifacts without writing')
args=parser.parse_args()
ROOT=Path('results/visual-missing-emg');OUT=args.output;D=ROOT/'vits-controls-full';raw=D/'raw-control-errors.csv'
if not args.verify:OUT.mkdir(parents=True,exist_ok=True)
proof=json.loads((ROOT/'vits-six-condition-complete-20261104.json').read_text())
assert hashlib.sha256(raw.read_bytes()).hexdigest()==proof['raw_control_errors_sha256']
groups_path=ROOT/'legacy-subject-groups-20261103.json';join=json.loads(groups_path.read_text())
rows=list(csv.DictReader(raw.open()));old=list(csv.DictReader(Path('results/native-dense-occlusion/vits-full/raw_paired_errors.csv').open()))
assert len(rows)==len(old)==17168
for i,(r,b)in enumerate(zip(rows,old)):
 assert all(r[n]==b[n]for n in('sequence','split','hand','ep_idx','center_idx','window_start','window_end'))
 assert float(r['mae_fusion_rad_c0'])==float(b['mae_fusion_rad'])and float(r['mae_vision_rad_c0'])==float(b['mae_vision_rad'])
flags=__import__('base64').b64decode(json.loads((ROOT/'vits-availability-extraction-20261103.json').read_text())['ordered_label_crop_flags_b64'])
assert bytes(int(r[n])for r in rows for n in('label_valid','natural_crop_valid'))==flags
subjects=np.array([join['anonymous_episode_subject_group'][r['ep_idx']]for r in rows]);valid=np.array([r['label_valid']=='1'for r in rows]);crop=np.array([r['natural_crop_valid']=='1'for r in rows])
fusion=np.array([[float(r[f'mae_fusion_rad_c{c}'])for r in rows]for c in range(6)],dtype=np.float64)
vision=np.array([[float(r[f'mae_vision_rad_c{c}'])for r in rows]for c in range(6)],dtype=np.float64)
metrics=json.loads((D/'metrics.json').read_text())
for g,mask in [('all',valid),('natural_crop_missing',valid&~crop),('natural_crop_present',valid&crop)]:
 for c in range(6):
  r=metrics['groups'][g]['conditions'][c]
  assert r['valid_label_requests']==int(mask.sum())
  for role,a in [('fusion',fusion),('vision',vision)]:
   assert math.isclose(math.fsum(a[c,mask])/int(mask.sum())*57.3,r[role+'_mae_deg'],abs_tol=2e-13,rel_tol=0)
selection={'all_valid':valid,'natural_crop_missing_valid':valid&~crop,'natural_crop_present_valid':valid&crop}
contrasts={'EMG_zero_penalty_native_vision':fusion[2]-fusion[0],
 'EMG_zero_penalty_missing_vision':fusion[3]-fusion[1],
 'EMG_roll_penalty_native_vision':fusion[4]-fusion[0],
 'EMG_roll_penalty_missing_vision':fusion[5]-fusion[1],
 'fusion_gain_over_standalone_missing_vision':vision[1]-fusion[1]}
summary={};per=[];rng_seed=20261103
for name,mask in selection.items():
 summary[name]={};ids=np.unique(subjects[mask]);n=int(mask.sum())
 for contrast,delta in contrasts.items():
  sums=np.array([math.fsum(delta[mask&(subjects==s)])*57.3 for s in ids]);counts=np.array([int((mask&(subjects==s)).sum())for s in ids]);means=sums/counts
  rng=np.random.default_rng(rng_seed);draw=rng.integers(0,len(ids),size=(10000,len(ids)))
  estimates=sums[draw].sum(axis=1)/counts[draw].sum(axis=1);interval=np.percentile(estimates,[2.5,97.5],method='linear')
  independent=math.fsum(delta[mask])/n*57.3;assert math.isclose(independent,math.fsum(sums)/n,abs_tol=2e-13,rel_tol=0)
  summary[name][contrast]=dict(valid_requests=n,subject_recording_groups=len(ids),request_weighted_mean_deg=independent,equal_subject_mean_deg=float(means.mean()),
   groups_positive=int((means>0).sum()),groups_negative=int((means<0).sum()),groups_equal=int((means==0).sum()),
   requests_positive=int((delta[mask]>0).sum()),requests_negative=int((delta[mask]<0).sum()),requests_equal=int((delta[mask]==0).sum()),
   exploratory_cluster_bootstrap_percentile95_deg=interval.tolist(),bootstrap_whole_group_draws=10000,bootstrap_seed=rng_seed)
  for s,total,count,mean in zip(ids,sums,counts,means):per.append(dict(selection=name,contrast=contrast,anonymous_subject_group=int(s),valid_requests=int(count),sum_gain_deg=float(total),mean_gain_deg=float(mean)))
result=dict(format='complete_ViTS_fixed_checkpoint_EMG_intervention_clustered_contrasts_v1',raw_control_errors_sha256=proof['raw_control_errors_sha256'],
 full_postexit_proof_sha256=hashlib.sha256((ROOT/'vits-six-condition-complete-20261104.json').read_bytes()).hexdigest(),subject_join_receipt_sha256=hashlib.sha256(groups_path.read_bytes()).hexdigest(),
 all17168_original_baseline_keys_and_unrounded_errors_exact=True,all_original_label_and_crop_flags_exact=True,independent_standard_library_fsum_means_match_complete_report=True,
 positive_contrast_meaning='zero or roll has greater error than aligned EMG in the same frozen fusion checkpoint; final contrast compares separate standalone vision model',
 summaries=summary,limitations=['Exploratory percentile intervals, no multiple comparison correction, no significance claim; whole legacy subject/recording groups resampled.',
 '41 episode_subject labels, one recording each; physical independence and PDF population equivalence not established.',
 'Normalized zero is an intervention on the trained input, potentially out of distribution, not a separately trained vision-only ablation.',
 'Half-window circular roll preserves same-window activity and adds a seam; it does not remove all EMG pose information.',
 'Original dense center windows retain disclosed training-input overlap; these are legacy-pipeline sensitivity results.',
 'Artificial full loss, natural crop absence, and geometric occlusion are distinct; these numbers do not close the full paper.'],full_paper_complete=False)
out=OUT/'vits-controls-subject-analysis-20261104.json'
text=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n'
buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=list(per[0]));writer.writeheader();writer.writerows(per)
per_path=OUT/'vits-controls-per-subject-20261104.csv'
if args.verify:
 assert out.read_text()==text and per_path.read_bytes()==buffer.getvalue().encode()
 print('verified all full-population joins, fsum means, clustered contrasts and numeric artifact bytes')
 raise SystemExit(0)
assert not out.exists()and not per_path.exists()
out.write_text(text);per_path.write_bytes(buffer.getvalue().encode())
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,2,figsize=(12.5,4.7),layout='constrained')
for ax,(selection_name,title)in zip(axs,[('all_valid','Induced full-image loss: 16,650 valid requests'),('natural_crop_missing_valid','Natural crop absence: 179 valid requests')]):
 names=['EMG_zero_penalty_missing_vision','EMG_roll_penalty_missing_vision'];values=[summary[selection_name][n]for n in names]
 for i,x in enumerate(values):
  lo,hi=x['exploratory_cluster_bootstrap_percentile95_deg'];m=x['request_weighted_mean_deg'];ax.plot([lo,hi],[i,i],color='#275d8d',lw=3);ax.scatter(m,i,color='#275d8d',s=45,zorder=3)
  ax.scatter(x['equal_subject_mean_deg'],i,marker='x',s=65,color='#bd5519',zorder=4)
 ax.axvline(0,color='#777',lw=1);ax.set_yticks([0,1],['Zero EMG − real EMG','Rolled EMG − real EMG']);ax.set_ylim(1.6,-.6);ax.set_title(title,fontsize=10);ax.set_xlabel('MAE penalty (°); positive favors real EMG',fontsize=10);ax.grid(axis='x',alpha=.2)
fig.suptitle('Same frozen ViT-S + EMGFormer-S; exploratory whole-subject bootstrap\nBlue: request-weighted mean and 95% interval; orange ×: equal-subject mean',fontsize=11)
fig.savefig(OUT/'vits-controls-missing-contrasts-20261104.png',dpi=160);plt.close(fig)
for name in summary:
 print(name,json.dumps(summary[name]))
