"""Measure pinned original EgoEMG EMGFormer factories; render Figure4/Table7."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

REVISION='f7cbd048b7092139852270491d3859cd472e0311'
FORMAT='egoemg_original_emgformer_architecture_audit_v1'

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def model_digest(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def source_identity(root):
    assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==REVISION
    assert not subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=all'],text=True).strip()
    norm_files=subprocess.check_output(['git','-C',str(root),'ls-files','assets/per_dataset_norm_stats*.json'],text=True).splitlines()
    files=sorted([*root.glob('egoemg/**/*.py'),*root.glob('config/**/*.yaml'),*[root/n for n in norm_files]])
    hashes={p.relative_to(root).as_posix():sha(p) for p in files}
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),hashes

def probe(args):
    import torch
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    assert not torch.cuda.is_initialized()
    root=args.upstream.resolve();bundle,files=source_identity(root);sys.path.insert(0,str(root))
    from egoemg._train_utils import setup_runtime
    from egoemg.train import make_lightning_module
    rows=[]
    for size,expected in [('small',3460134),('middle',6617382),('large',16320806)]:
        experiment='emgformer/egoemg_emgformer_'+size
        with initialize_config_dir(config_dir=str(root/'config'),version_base='1.1'):
            cfg=compose(config_name='base',overrides=['experiment='+experiment])
        assert cfg.checkpoint is None and cfg.pretrained_checkpoint is None
        assert cfg.datamodule.window_length==cfg.datamodule.val_test_window_length==12000
        assert cfg.module.head.activation=='relu' and cfg.module.decoder.pos_encoding=='rope'
        setup_runtime(cfg);module=make_lightning_module(cfg).cpu().eval();model=module.model
        parameters=sum(p.numel() for p in model.parameters());assert parameters==expected
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_parameters==parameters
        initial_model_digest=model_digest(model)
        assert (model.left_context,model.right_context)==(510,0)
        assert model.featurizer.layers[0].conv[0].in_channels==8
        selected=['featurizer.layers.0','featurizer.layers.1',
            'featurizer.layers.2.layers.conv1dblock','featurizer.layers.2.layers.tds_block',
            'featurizer.layers.3.layers.conv1dblock','featurizer.layers.3.layers.tds_block',
            'featurizer','decoder.input_proj','decoder','head']
        members=dict(model.named_modules());trace={};hooks=[]
        def hook(name):
            def observed(_module,_inputs,out):trace[name]=list(out.shape)
            return observed
        for name in selected:hooks.append(members[name].register_forward_hook(hook(name)))
        forwards=[]
        for length in (12000,7790):
            trace.clear();target=torch.arange(22*length,dtype=torch.float32).reshape(1,22,length)/100000
            mask=(torch.arange(length).reshape(1,length)%3!=0)
            batch=dict(emg=torch.zeros(1,8,length),joint_angles=target,label_valid_mask=mask)
            with torch.inference_mode():pred,aligned,actual_mask=module(batch)
            expected_tokens=230 if length==12000 else 146
            assert trace['head']==[1,22,expected_tokens] and model.featurizer.get_output_length(length)==expected_tokens
            assert pred.shape==aligned.shape==(1,22,length-510) and actual_mask.shape==(1,length-510)
            assert pred.dtype==aligned.dtype==torch.float32 and actual_mask.dtype==torch.bool
            assert torch.equal(aligned,target[...,510:]) and torch.equal(actual_mask,mask[...,510:])
            assert torch.isfinite(pred).all()
            forwards.append(dict(input_points=length,official_recipe_length=length==12000,input_shape=[1,8,length],
                module_trace=dict(trace),prediction_shape=list(pred.shape),target_slice_exact=True,mask_slice_exact=True,
                prediction_dtype=str(pred.dtype),target_dtype=str(aligned.dtype),mask_dtype=str(actual_mask.dtype),
                predictions_finite=True,input_center_index=length//2,aligned_center_index=length//2-510))
        for h in hooks:h.remove()
        assert model_digest(model)==initial_model_digest
        se=[dict(name=n,mode=m.mode) for n,m in model.named_modules() if type(m).__name__=='SqueezeExcite']
        assert se and all(x['mode']=='global' for x in se)
        norm_name=Path(str(cfg.datamodule.per_dataset_norm_stats_path)).name
        tracked_norm=subprocess.check_output(['git','-C',str(root),'ls-files','assets/'+norm_name],text=True).strip()=='assets/'+norm_name
        assert tracked_norm and (root/'assets'/norm_name).is_file()
        rows.append(dict(size=size,experiment=experiment,model_class=type(model).__module__+'.'+type(model).__name__,parameters=parameters,
            trainable_parameters=trainable_parameters,
            initialized_model_sha256=initial_model_digest,model_state_unchanged_by_forwards=True,
            original_factory_used=True,seed_before_factory=int(cfg.seed),module_config=OmegaConf.to_container(cfg.module,resolve=True),
            window_points=12000,train_stride=int(cfg.datamodule.stride),evaluation_stride=int(cfg.datamodule.val_test_stride),
            input_layout=str(cfg.egoemg_emg_layout),data_emg_preference=str(cfg.egoemg_emg_field_preference),
            normalization_mode=str(cfg.datamodule.norm_mode),declared_normalization_filename=norm_name,
            normalization_alias_in_pinned_upstream=(root/'assets'/norm_name).exists(),
            normalization_alias_is_tracked=tracked_norm,normalization_alias_sha256=sha(root/'assets'/norm_name),
            head_layers=[type(m).__name__ for m in model.head.net],decoder_input_projection=type(model.decoder.input_proj).__name__,
            transformer_norm_first=bool(model.decoder.encoder.layers[0].norm_first),global_se_modules=se,
            train_transforms=OmegaConf.to_container(cfg.transforms,resolve=True),raw_loss_weights=OmegaConf.to_container(cfg.loss_weights,resolve=True),
            optimizer_config=OmegaConf.to_container(cfg.optimizer,resolve=True),
            scheduler_config=OmegaConf.to_container(cfg.lr_scheduler,resolve=True),
            callback_configs=OmegaConf.to_container(cfg.callbacks,resolve=True),
            trainer=dict(max_epochs=int(cfg.trainer.max_epochs),precision=str(cfg.trainer.precision),batch_size=int(cfg.batch_size)),
            left_context=510,right_context=0,forwards=forwards))
        del module,model
    assert source_identity(root)==(bundle,files) and not torch.cuda.is_initialized()
    result=dict(format=FORMAT,source_revision=REVISION,source_bundle_sha256=bundle,source_files_sha256=files,models=rows,
        python=platform.python_version(),dependencies={n:importlib.metadata.version(n) for n in ('torch','pytorch-lightning','hydra-core','omegaconf')},
        script_sha256=sha(__file__),actual_original_cpu_forwards=6,random_initialization=True,datasets_loaded=False,
        checkpoints_loaded=False,optimizer_updates=0,cuda_initialized=False,scientific_complete=False,full_paper_complete=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError('architecture audit output already exists')
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(parameters={r['size']:r['parameters'] for r in rows},actual_original_cpu_forwards=6,source_clean=True,cuda_initialized=False)))

def render(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    d=json.loads(args.input.read_text());assert d['format']==FORMAT and d['actual_original_cpu_forwards']==6
    models=d['models'];s=models[0];trace=s['forwards'][0]['module_trace']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':'egoemg-emgformer-architecture-v1'})
    fig,ax=plt.subplots(figsize=(16,9));ax.set(xlim=(0,16),ylim=(0,9));ax.axis('off')
    def box(x,y,w,h,title,detail,color='#e6f0f7'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12',facecolor=color,edgecolor='#476276',linewidth=1))
        ax.text(x+w/2,y+h-.28,title,ha='center',va='top',fontsize=11,fontweight='bold')
        ax.text(x+w/2,y+h/2-.16,detail,ha='center',va='center',fontsize=9.5,linespacing=1.5)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color='#476276',linewidth=1.4))
    ax.text(8,8.75,'Figure4 / Table7: measured pinned-code EMGFormer architecture',ha='center',fontsize=17,fontweight='bold')
    ax.text(8,8.3,'Official EgoEMG S/M/L recipes: 8 target-hand channels, 12,000 points; all shapes measured on original factories',ha='center',fontsize=11)
    nodes=[(.35,'Input','B x 8 x 12,000\n6.000 s at 2 kHz'),(3.4,'Conv block1','8 -> 256; k11 / s5\nB x 256 x '+str(trace['featurizer.layers.0'][-1])),
        (6.45,'Conv block2','256 -> 256; k5 / s2\nB x 256 x '+str(trace['featurizer.layers.1'][-1])),
        (9.5,'TDS stage1','k9/s5: 1,197 -> 238\nTDS k5: 238 -> 234\nFC + global SE'),
        (12.55,'TDS stage2','k3/s1: 234 -> 232\nTDS k3: 232 -> 230\nFC + global SE')]
    for x,title,detail in nodes:box(x,6.1,2.65,1.65,title,detail)
    for i in range(4):arrow((nodes[i][0]+2.8,6.9),(nodes[i+1][0]-.17,6.9))
    box(.45,3.3,3.2,1.8,'Declared data contract','target_hand layout\nper-dataset normalization\nToTensor; Rotation (train only)','#f7eedc')
    box(4.25,3.3,3.55,1.8,'Transformer','256 -> D input projection\nRoPE on Q/K; noncausal\npre-LN attention + ReLU FFN\nL encoder layers','#e2eddf')
    box(8.45,3.3,3.15,1.8,'Token regression head','Linear D -> 512\nReLU; dropout0.1\nLinear512 -> 22\nB x 22 x 230','#efe5f4')
    box(12.25,3.3,3.2,1.8,'Original alignment','target/mask crop: [510:]\nlinear prediction interpolation\nB x 22 x 11,490\ncenter: 6,000 -> 5,490','#e6f0f7')
    arrow((13.85,5.96),(6.05,5.25));arrow((7.95,4.2),(8.3,4.2));arrow((11.75,4.2),(12.1,4.2))
    ax.text(.4,2.73,'Measured original model parameters and decoder settings',fontsize=12,fontweight='bold')
    columns=['Model','Layers','D','Heads','FFN','Input projection','Exact parameters']
    values=[]
    for r in models:
        c=r['module_config']['decoder'];values.append([r['size'].upper(),str(c['num_layers']),str(c['model_dim']),str(c['num_heads']),str(c['ffn_dim']),r['decoder_input_projection'],f"{r['parameters']:,}"])
    table=ax.table(cellText=values,colLabels=columns,colWidths=[.12,.1,.1,.1,.1,.21,.19],bbox=[.02,.135,.96,.15],cellLoc='center')
    table.auto_set_font_size(False);table.set_fontsize(10)
    for (i,j),cell in table.get_celld().items():cell.set_edgecolor('#c2ced7');cell.set_facecolor('#e6f0f7' if i==0 else '#ffffff')
    assert all(r['normalization_alias_in_pinned_upstream'] and r['normalization_alias_is_tracked'] for r in models)
    note='Pinned code: tds_slim has 2 stages and a ReLU head. Diagnostic 7,790-point input: 146 tokens and 7,280 aligned points.\nGlobal SE and noncausal attention use the whole window; the 511-point local convolution receptive field is not an online latency.\nCPU random-init shape audit only; no datasets, checkpoints, optimizer updates or benchmark accuracy.\nThe declared normalization asset is tracked and present; data loading and normalization were not executed.'
    ax.text(.35,.13,note,va='bottom',fontsize=8.5,linespacing=1.45)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'),dpi=180,metadata={'Software':'EgoEMG-Reproduction'})
    fig.savefig(args.output.with_suffix('.svg'),metadata={'Date':None,'Creator':'EgoEMG-Reproduction'})
    plt.close(fig);print(json.dumps(dict(png_sha256=sha(args.output.with_suffix('.png')),svg_sha256=sha(args.output.with_suffix('.svg')))))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('probe');a.add_argument('--upstream',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('render');a.add_argument('--input',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    args=p.parse_args();probe(args) if args.command=='probe' else render(args)

if __name__=='__main__':main()
