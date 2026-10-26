"""Actual server readonly cohort/key census; no model, optimizer, or scientific qualification."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from egoemg_repro.data import LegacyStore,union_intervals
from egoemg_repro.vision_literal import compose_vision,vision_protocol
from egoemg_repro.vision_literal_data import VisionLiteralDataset,build_vision_partitions
from egoemg_repro.vision_literal_data import _crop_path

data,crops,upstream=map(Path,sys.argv[1:4]);started=time.monotonic()
assert torch.cuda.is_initialized() is False
config=compose_vision(upstream);protocol=vision_protocol(upstream,config);store=LegacyStore(data)
partitions=build_vision_partitions(store);records={};datasets={};sample_count=0
used=['frame_split_id','generated_label_valid',
    *['generated_joint_angles_'+h for h in ('left','right')],
    *['mocap_'+h+'_wrist_'+a for h in ('left','right') for a in ('pitch','yaw')]]
frame_field=next(f for f in ('image_head_frame_index','image_webcam_frame_index') if f in store.fields);used.append(frame_field)
guard_paths=[data/'manifest.json',data/'metadata.npz',*[data/store.fields[f]['filename'] for f in used]]
for episode in {w['episode'] for w in partitions['windows']}:
    path=_crop_path(crops,store.episodes[episode]['id']);guard_paths.append(path/'data.mdb' if path.is_dir() else path)
def stamp(path):
    s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
before={p:stamp(p) for p in guard_paths}
for role in ('train','val','test'):
    ds=VisionLiteralDataset(data,partitions,crops,upstream=upstream,config=config,role=role);datasets[role]=ds
    n=len(ds.windows);assert len(ds)==partitions['coverage'][role]['consumed_hand_samples']
    if n:
        np.random.seed(42)
        for index in sorted({0,n-1,n,2*n-1,2*n,4*n-1}):
            value=ds[index];center=value['center'];hand=value['target_hand'];h=value['target_hand_index']
            expected=np.empty((22,1),np.float32);expected[:20,0]=np.asarray(store.field('generated_joint_angles_'+hand)[center],np.float32)
            expected[20,0]=np.deg2rad(float(store.field('mocap_'+hand+'_wrist_pitch')[center]))
            expected[21,0]=np.deg2rad(float(store.field('mocap_'+hand+'_wrist_yaw')[center]))
            assert value['joint_angles'].dtype==np.float32 and value['joint_angles'].tobytes()==expected.tobytes()
            assert bool(value['label_valid_mask'][0])==bool(store.field('generated_label_valid')[center,h])
            assert int(value['vision_frame_indices'][0])==int(store.field(ds.frame_mapping_field)[center])
            assert value['vision_img'].shape==(3,256,256) and np.isfinite(value['vision_img']).all()
            assert value['emg'].shape==(16,0);sample_count+=1
    result=dict(anchor_windows=n,unique_hand_samples=2*n,consumed_hand_samples=len(ds),
        possible_center_rows=0,possible_center_target_nonfinite_entries=0,possible_center_label_true_hand_rows=0,
        negative_frame_index_rows=0,unique_episode_frame_hand_keys=0,present_unique_keys=0,missing_unique_keys=0)
    for episode in sorted({w['episode'] for w in ds.windows}):
        intervals=union_intervals((w['center'],w['center']+max(1,w['jitter_upper']))
            for w in ds.windows if w['episode']==episode)
        frames=set()
        for lo,hi in intervals:
            result['possible_center_rows']+=hi-lo
            for a in range(lo,hi,262144):
                if time.monotonic()-started>580:raise TimeoutError('readonly census exceeded explicit bound')
                b=min(hi,a+262144);ids=np.asarray(store.field(ds.frame_mapping_field)[a:b])
                result['negative_frame_index_rows']+=int(np.count_nonzero(ids<0));frames.update(map(int,np.unique(ids)))
                if len(frames)>1000000:raise ValueError('bounded per-episode frame census exceeded')
                result['possible_center_label_true_hand_rows']+=int(np.count_nonzero(store.field('generated_label_valid')[a:b]))
                for hand in ('left','right'):
                    for field in ('generated_joint_angles_'+hand,'mocap_'+hand+'_wrist_pitch','mocap_'+hand+'_wrist_yaw'):
                        result['possible_center_target_nonfinite_entries']+=int(np.count_nonzero(~np.isfinite(store.field(field)[a:b])))
        for h in (0,1):
            with ds.views[h]._get_pec_txn(store.episodes[episode]['id']) as txn:
                cursor=txn.cursor();code='L' if h==0 else 'R'
                for frame in frames:
                    present=cursor.set_key(f'{frame:08d}_{code}'.encode())
                    result['present_unique_keys' if present else 'missing_unique_keys']+=int(True)
            result['unique_episode_frame_hand_keys']+=len(frames)
    assert result['present_unique_keys']+result['missing_unique_keys']==result['unique_episode_frame_hand_keys']
    records[role]=result
    print(role+' readonly census finished',file=sys.stderr,flush=True)
assert torch.cuda.is_initialized() is False
from egoemg_repro import vision_literal,vision_literal_data
assert all(stamp(path)==state for path,state in before.items()),'actual data/crop namespace changed during census'
value=dict(format='original_vision_legacy_readonly_census_v1',scope='actual metadata/jitter-position targets/crop-key census plus bounded original point reads',
    scientific_eligible=False,production_ready=False,full_paper_complete=False,model_constructed=False,optimizer_executions=0,
    cuda_initialized=False,variant='vit_small',stat_guard_members=len(before),source_stat_guard_stable=True,
    data_payload_content_rehashed=False,protocol_sha256=protocol['sha256'],partition_sha256=partitions['provenance']['partition_sha256'],
    sources_sha256={str(Path(m.__file__).name):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (vision_literal,vision_literal_data)},
    declared_current_training_corpus=protocol['declared_training_corpus'],available_data='released legacy EgoEMG only; original merged source not admitted',
    frame_mapping_field=datasets['train'].frame_mapping_field,source_subject_keys=len({str(e['subject']) for e in store.episodes}),
    episode_records=len(store.episodes),discarded_unsafe_jitter_anchors=partitions['discarded']['vision_jitter_unsafe_anchor'],
    roles=records,point_samples_read_and_raw_target_checked=sample_count,
    crop_scope='unique (episode,source-video-frame,hand) key presence across ALL possible jitter centers; keys are not sample/angle denominators; JPEG decode is bounded samples only',
    target_scope='all possible distinct center rows, unweighted across anchors; not an observed training epoch or performance metric',
    elapsed_seconds=time.monotonic()-started)
print(json.dumps(value,sort_keys=True,indent=2))
