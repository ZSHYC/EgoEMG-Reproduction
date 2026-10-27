"""Figure3(b): explicit paper schematic beside actual anonymous release labels.

Scan reads only shared labels, hashes their complete bytes and checks the
previous frame audit. Render needs only the published anonymous aggregate.
"""
import argparse
import hashlib
import hmac
import json
from pathlib import Path
import secrets

import numpy as np

REVISION='f7cbd048b7092139852270491d3859cd472e0311'
PAPER_SHA='124318dce75d9af00ab4d8427acf01315cf83ae1992082b783d5a488c2474725'
SPLITS=('train','user','gesture','both')
FORMAT='egoemg_anonymous_released_split_matrix_v1'

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def stamp(path):
    p=Path(path)
    if p.is_symlink():raise ValueError('ordinary source files required')
    s=p.stat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path.name+' already exists')
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')

def scan(args):
    data=args.data.resolve();output=args.output.resolve();private=args.private_output.resolve()
    if any(p==data or data in p.parents for p in (output,private)):
        raise ValueError('analysis outputs must be outside original data')
    audit=json.loads(args.frame_audit.read_text());vocabulary=json.loads(args.vocabulary.read_text())
    assert vocabulary['sources']['official']['revision']==REVISION
    names=[g['name'] for g in vocabulary['rows']];assert [g['id'] for g in vocabulary['rows']]==list(range(60))
    manifest=json.loads((data/'manifest.json').read_text());n=manifest['total_rows']
    assert n==audit['total_synchronized_rows'] and audit['no_hand_duplication']
    assert tuple(manifest['frame_split_labels'])==tuple(audit['split_labels'])==SPLITS
    fields={'label_gesture_class':np.dtype('<i4'),'label_gesture_active':np.dtype('bool'),'frame_split_id':np.dtype('i1')}
    paths=[data/'manifest.json',data/'metadata.npz',*[data/(f+'.dat') for f in fields]]
    stamps={p:stamp(p) for p in paths};metadata_sha=sha(data/'metadata.npz')
    with np.load(data/'metadata.npz',allow_pickle=False) as md:
        start,end,length,subject=[md[k].copy() for k in ('episode_start_idx','episode_end_idx','episode_length','episode_subject_id')]
        ids=md['subjects_subject_id'].copy()
    assert len(start)==len(end)==len(length)==len(subject)==manifest['num_episodes']
    assert start[0]==0 and end[-1]==n and np.array_equal(start[1:],end[:-1]) and np.array_equal(end-start,length)
    assert len(set(map(int,ids)))==len(ids) and set(map(int,subject))==set(map(int,ids))
    users={int(uid):np.zeros((61,4,2),dtype=np.int64) for uid in ids}
    hashes={f:hashlib.sha256() for f in fields};arrays={}
    for f,dtype in fields.items():
        spec=manifest['fields'][f]
        assert spec['filename']==f+'.dat' and np.dtype(spec['dtype'])==dtype and spec['shape']==[n]
        assert (data/(f+'.dat')).stat().st_size==n*dtype.itemsize
        arrays[f]=np.memmap(data/(f+'.dat'),dtype=dtype,shape=(n,),mode='r')
    for lo,hi,uid in zip(start,end,subject):
        for a in range(int(lo),int(hi),262144):
            b=min(a+262144,int(hi));parts={f:x[a:b] for f,x in arrays.items()}
            for f,x in parts.items():hashes[f].update(memoryview(x))
            c=parts['label_gesture_class'];s=parts['frame_split_id'];active=parts['label_gesture_active']
            assert np.all((c>=-1)&(c<60)) and np.all((s>=0)&(s<4))
            encoded=((c.astype(np.int64)+1)*4+s)*2+active
            users[int(uid)]+=np.bincount(encoded,minlength=61*4*2).reshape(61,4,2)
            assert all(stamp(p)==s for p,s in stamps.items()),'source changed during scan'
    total=sum(users.values(),np.zeros((61,4,2),dtype=np.int64))
    expected=np.zeros_like(total);seen=set()
    for r in audit['counts']:
        k=(r['class_id']+1,SPLITS.index(r['split']),int(r['active']));assert k not in seen;seen.add(k)
        expected[k]=r['synchronized_rows']
    assert np.array_equal(total,expected) and int(total.sum())==n
    field_proofs={}
    for f,digest in hashes.items():
        actual=digest.hexdigest();assert actual==audit['fields'][f]['sha256'],f+' differs from full-frame audit'
        field_proofs[f]=dict(dtype=fields[f].str,shape=[n],bytes=n*fields[f].itemsize,sha256=actual,matches_recorded_transfer=True)
        arrays[f]._mmap.close()
    assert all(stamp(p)==s for p,s in stamps.items()) and sha(data/'metadata.npz')==metadata_sha
    private.mkdir(mode=0o700,parents=True,exist_ok=False)
    key=secrets.token_bytes(32)
    def order(uid):
        by_split=users[uid][1:].sum(axis=(0,2));train=bool(by_split[0]);held=bool(by_split[1] or by_split[3])
        return not train,held if train else False,hmac.new(key,str(uid).encode(),hashlib.sha256).hexdigest()
    rows=[];mapping=[]
    for i,uid in enumerate(sorted(users,key=order),1):
        alias=f'P{i:02d}';matrix=users[uid]
        rows.append(dict(alias=alias,counts=matrix[1:].sum(axis=2).tolist(),unassigned_frames=int(matrix[0].sum())))
        mapping.append(dict(alias=alias,source_subject_id=uid))
    write(private/'alias-map.json',dict(alias_key_hex=key.hex(),mapping=mapping));(private/'alias-map.json').chmod(0o600)
    counts=np.asarray([r['counts'] for r in rows],dtype=np.int64);by_user=counts.sum(axis=1)>0
    result=dict(format=FORMAT,scope='actual old-release shared-frame labels; not model windows or independent train/val/test roles',
        schematic_scope='paper conceptual6-user/10-gesture holdout; not measured release membership',
        source_revision=REVISION,paper_sha256=PAPER_SHA,paper_printed_page=5,paper_figure='3(b)',
        split_labels=list(SPLITS),gesture_names=names,rows=rows,total_synchronized_frames=n,
        defined_frames=int(counts.sum()),unassigned_frames=sum(r['unassigned_frames'] for r in rows),
        participant_metadata_keys=len(rows),observed_gestures=int((counts.sum(axis=(0,2))>0).sum()),
        by_split_defined_frames={s:int(counts[:,:,j].sum()) for j,s in enumerate(SPLITS)},
        users_by_split={s:int(by_user[:,j].sum()) for j,s in enumerate(SPLITS)},
        users_in_train_and_user=int((by_user[:,0]&by_user[:,1]).sum()),
        users_in_train_and_both=int((by_user[:,0]&by_user[:,3]).sum()),
        multilabel_cells=int(((counts>0).sum(axis=2)>1).sum()),
        row_order='train-containing rows first, mixed held labels last within train, then private keyed order',
        alias_scope='this figure only; private mapping not published; rendering uses these fixed anonymous rows',
        source_fields=field_proofs,metadata_sha256=metadata_sha,manifest_sha256=sha(data/'manifest.json'),
        frame_audit_sha256=sha(args.frame_audit),vocabulary_sha256=sha(args.vocabulary),script_sha256=sha(__file__),
        full_payload_rehash_performed=True,full_frame_counts_match_previous_audit=True,no_hand_duplication=True,
        full_paper_complete=False,publisher_authenticity='local transfer identity, not publisher authentication')
    assert result['defined_frames']+result['unassigned_frames']==n
    write(output,result)
    write(private/'source-bindings.json',dict(stamps={str(p):list(s) for p,s in stamps.items()},public_output_sha256=sha(output)))
    print(json.dumps({k:result[k] for k in ('participant_metadata_keys','observed_gestures','defined_frames','unassigned_frames','multilabel_cells','users_in_train_and_user','users_in_train_and_both')}))

def render(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Patch
    d=json.loads(args.input.read_text());assert d['format']==FORMAT and tuple(d['split_labels'])==SPLITS
    counts=np.asarray([r['counts'] for r in d['rows']],dtype=np.int64)
    assert counts.shape==(d['participant_metadata_keys'],60,4) and np.all(counts>=0)
    assert int(counts.sum())==d['defined_frames'] and d['defined_frames']+d['unassigned_frames']==d['total_synchronized_frames']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.hashsalt':'egoemg-release-split-matrix-v1'})
    colors=['#eeeeee','#f7c993','#bad5ed','#cce2b5'];display=['Train','User','Gesture','Both']
    fig=plt.figure(figsize=(16,11));ax=fig.add_axes([.35,.12,.62,.79]);schematic=fig.add_axes([.045,.65,.235,.25])
    for x,y,w,h,j in [(0,0,35,50,0),(35,0,6,50,1),(0,50,35,10,2),(35,50,6,10,3)]:
        schematic.add_patch(Rectangle((x,y),w,h,facecolor=colors[j],edgecolor='#7b8791',linewidth=.7))
        schematic.text(x+w/2,y+h/2,display[j],ha='center',va='center',fontsize=9 if w>6 else 8,rotation=0 if w>6 else 90)
    schematic.set(xlim=(0,41),ylim=(0,60),xlabel='Participants (conceptual)',ylabel='Gestures (conceptual)',title='Paper protocol schematic')
    schematic.set_xticks([]);schematic.set_yticks([])
    for u in range(len(counts)):
        for g in range(60):
            labels=np.flatnonzero(counts[u,g]>0)
            if not len(labels):continue
            for lane,j in enumerate(labels):
                ax.add_patch(Rectangle((u+lane/len(labels),g),1/len(labels),1,facecolor=colors[j],edgecolor='none'))
    ax.set(xlim=(0,len(counts)),ylim=(60,0),xlabel='Anonymous participant keys (scoped to this figure)',ylabel='Official gesture vocabulary (all60classes)',title='Measured legacy release: shared-frame label presence')
    ax.set_xticks(np.arange(len(counts))+.5,labels=[r['alias'] for r in d['rows']],rotation=90,fontsize=8)
    ax.set_yticks(np.arange(60)+.5,labels=d['gesture_names'],fontsize=7)
    ax.set_xticks(np.arange(len(counts)+1),minor=True);ax.set_yticks(np.arange(61),minor=True)
    ax.grid(which='minor',color='#ffffff',linewidth=.35);ax.tick_params(which='both',length=0)
    fig.legend(handles=[Patch(facecolor=colors[j],edgecolor='#7b8791',label=display[j]) for j in (0,2,1,3)],
        loc='lower center',bbox_to_anchor=(.66,.025),ncol=4,frameon=False)
    notes=(f"Actual release audit\n\n{len(counts)} anonymous participant keys\n60 official gesture classes (raw retained)\n{d['total_synchronized_frames']:,} shared frames\n{d['defined_frames']:,} defined-gesture frames\n{d['unassigned_frames']:,} unassigned frames (not drawn)\n\nTrain and User: {d['users_in_train_and_user']} shared key(s)\nTrain and Both: {d['users_in_train_and_both']} shared key(s)\n{d['multilabel_cells']} cells contain multiple labels\n\nA striped cell preserves every observed split.\nWhite space means no defined frames.\n\nFrame labels do not define the independent\ntraining/validation holdout used by models.\nThis legacy release is not proven equivalent\nto the PDF cohort.")
    fig.text(.045,.59,notes,ha='left',va='top',fontsize=10,linespacing=1.45)
    fig.suptitle('Figure3(b) reproduction: intended protocol and actual released labels',fontsize=15,y=.975)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'),dpi=180,metadata={'Software':'EgoEMG-Reproduction'})
    fig.savefig(args.output.with_suffix('.svg'),metadata={'Date':None,'Creator':'EgoEMG-Reproduction'})
    plt.close(fig)
    print(json.dumps(dict(png_sha256=sha(args.output.with_suffix('.png')),svg_sha256=sha(args.output.with_suffix('.svg')))))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('scan')
    for n in ('data','frame-audit','vocabulary','private-output','output'):s.add_argument('--'+n,type=Path,required=True)
    r=sub.add_parser('render');r.add_argument('--input',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    a=p.parse_args();scan(a) if a.command=='scan' else render(a)

if __name__=='__main__':main()
