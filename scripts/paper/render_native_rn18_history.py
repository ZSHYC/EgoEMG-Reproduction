"""Rebuild native RN18 curves from the source-bound public150-epoch history.

Run with the existing plotting environment. No model, training or data asset is
loaded; original report bytes and history digest must match their completion proof.
"""
from pathlib import Path
import csv,hashlib,json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]


def main():
    folder=ROOT/'results/original-resnet/resnet18'
    proof_path=ROOT/'results/native-rn18-full-complete-20261102.json'
    proof=json.loads(proof_path.read_text());report=json.loads((folder/'metrics.json').read_text())
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    for name,digest in proof['original_report_files_sha256'].items():assert sha(folder/name)==digest
    assert sha(folder/'training-history.csv')==proof['public_training_history_sha256']
    with(folder/'training-history.csv').open()as stream:history=list(csv.DictReader(stream))
    assert len(history)==150
    epochs=[int(row['completed_epoch'])for row in history]
    assert epochs==list(range(1,151))
    validated=[row for row in history if row['val_mae_rad']]
    assert [int(row['completed_epoch'])for row in validated]==list(range(2,151,2))
    best=proof['actual_selected_best']['completed_epochs']
    assert int(min(validated,key=lambda row:float(row['val_mae_rad']))['completed_epoch'])==best
    assert float(history[best-1]['val_mae_rad'])==proof['actual_selected_best']['score']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':'egoemg-native-rn18-20261102','svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(11.5,3.8),constrained_layout=True)
    axes[0].plot(epochs,[float(row['train_loss'])for row in history],color='#24599c',lw=1.6)
    axes[0].set(xlabel='Completed epoch',ylabel='Native training objective',title='Training: all 150 epochs')
    x=[int(row['completed_epoch'])for row in validated];y=[float(row['val_mae_rad'])*180/math.pi for row in validated]
    axes[1].plot(x,y,color='#17765c',lw=1.3,marker='o',markersize=2.5,label='Original validation checks')
    axes[1].axvline(best,color='#b94d25',lw=1,ls='--',label=f'Validation-selected epoch {best}')
    axes[1].scatter([best],[proof['actual_selected_best']['score']*180/math.pi],color='#b94d25',s=35,zorder=5)
    axes[1].set(xlabel='Completed epoch',ylabel='Native validation MAE (degrees)',title='Validation: original every-two-epoch cadence')
    axes[1].legend(frameon=False,fontsize=8)
    for axis in axes:
        axis.set_xlim(1,150);axis.grid(alpha=.18);axis.spines[['right','top']].set_visible(False)
    fig.suptitle('ResNet-18: native recipe, legacy EgoEMG release',fontsize=12)
    fig.savefig(folder/'learning-curves.png',dpi=170,metadata={'Software':'EgoEMG-Reproduction'})
    fig.savefig(folder/'learning-curves.svg',metadata={'Creator':'EgoEMG-Reproduction','Date':None})
    svg=folder/'learning-curves.svg'
    svg.write_text('\n'.join(line.rstrip()for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    for name,digest in proof['original_report_files_sha256'].items():assert sha(folder/name)==digest
    assert sha(folder/'training-history.csv')==proof['public_training_history_sha256']
    curve=dict(format='native_rn18_complete_training_history_plot_v1',source_report_sha256=proof['actual_report_sha256'],source_history_sha256=sha(folder/'training-history.csv'),actual_epochs=150,actual_validation_checks=75,selected_completed_epoch=best,missing_odd_epoch_validation_not_filled=True,test_scores_not_used_for_plot_or_selection=True,files_sha256={name:sha(folder/name)for name in('learning-curves.png','learning-curves.svg')},matplotlib_version=matplotlib.__version__,full_paper_complete=False)
    curve['renderer_path']=Path(__file__).relative_to(ROOT).as_posix()
    curve['renderer_source_sha256']=sha(Path(__file__))
    (folder/'learning-curves.json').write_text(json.dumps(curve,sort_keys=True,indent=2)+'\n')


if __name__=="__main__":
    main()
