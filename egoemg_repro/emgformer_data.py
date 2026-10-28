"""Original EMGFormer reader semantics over explicitly adapted legacy rows.

Native merged-data normalization fails for the pinned asset's missing namespace.
The legacy namespace/field substitutions below are explicit and grant no corpus,
fit, GPU or scientific permit.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .classic_emg import _activate, _file_hash, _hash
from .data import LegacyStore, union_intervals
from .emg_initializer_data import _schema, _view
from .emgformer_literal import _validated, emgformer_protocol, emgformer_normalization, NORM_ASSET
from .partitions import build_partitions

W=12000
STRIDE=1200
ROLES=('train','val','test')
ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
LOCAL=('egoemg_repro/emgformer_data.py','egoemg_repro/emgformer_literal.py',
       'egoemg_repro/emg_initializer_data.py','egoemg_repro/emg_initializer_literal.py',
       'egoemg_repro/classic_emg_data.py','egoemg_repro/classic_emg.py',
       'egoemg_repro/partitions.py','egoemg_repro/data.py')


def emgformer_data_contract(upstream,config):
    root,size,value=_validated(upstream,config)
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported EMGFormer reader source changed')
    protocol=emgformer_protocol(root,config);_activate(root)
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    from egoemg.datamodule import make_data_module
    dm=make_data_module(config)
    namespace={v['dataset_name']for role in ROLES for v in value['dataset'][role]}
    if namespace!={'egoemg_unified'}:raise ValueError('original EMGFormer dataset namespace differs')
    errors={}
    for hand in ('left','right'):
        declared=SimpleNamespace(norm_mode=dm.norm_mode,norm_stats_path=str(root/NORM_ASSET),
            dataset_name='egoemg_unified',emg_field_preference=value['egoemg_emg_field_preference'],
            emg_layout=value['egoemg_emg_layout'],target_hand=hand)
        try:EgoEmgMemmapDataset._load_norm_stats(declared)
        except KeyError:errors[hand]='KeyError'
        else:errors[hand]=None
    center={role:{bool(v.get('center_target_only',False))for v in value['dataset'][role]}for role in ROLES}
    if any(len(v)!=1 for v in center.values()):raise ValueError('native per-hand role semantics differ')
    result=dict(format='original_egoemg_emgformer_legacy_reader_contract_v1',size=size,
        runtime_protocol_sha256=protocol['sha256'],declared_normalization_namespace='egoemg_unified',
        canonical_namespace_available=all(e is None for e in errors.values()),
        canonical_normalization_error_type='KeyError'if set(errors.values())=={'KeyError'}else None,
        canonical_normalization_errors_by_hand=errors,normalization_namespace='egoemg',
        normalization=emgformer_normalization(root,config),
        normalization_substitution='explicit legacy egoemg namespace in the same pinned declared alias asset',
        physical_legacy_emg_fields={h:f'emg_{h}_filtered'for h in ('left','right')},
        requested_field=value['egoemg_emg_field_preference'],
        field_substitution='reader-only filtered_paper alias of released filtered; raw manifest/files unchanged',
        center_target_only_by_role={r:next(iter(v))for r,v in center.items()},
        training_target_shape=[22,W],evaluation_target_shape=[22,1],
        batch_size=dm.batch_size,declared_workers=dm.num_workers,
        repeat_by_role={r:dm.dataset_repeat for r in ROLES},
        original_dataset_declarations=value['dataset'],
        loader=dict(eval_workers=dm.eval_num_workers,prefetch_factor=dm.prefetch_factor,
            persistent_workers=dm.persistent_workers,pin_memory=dm.pin_memory,
            drop_last=False,generator=None,session_shuffle=dm.session_shuffle,
            session_block_batches=dm.session_block_batches,
            sampler='native RandomSampler for concatenated hands; no session_index_groups interface'),
        mask_policy='original generated time/center label mask broadcast to22; stored wrist-valid flag ignored',
        available_corpus='released legacy EgoEMG only; not merged EgoEMG/ShowEE/Incre',
        local_sources_sha256={n:_file_hash(ROOT/n)for n in LOCAL},
        upstream_sources_sha256=protocol['sources_sha256'],
        data_admitted=False,scientific_eligible=False,production_ready=False,full_paper_complete=False)
    result['sha256']=_hash(result);return result


def build_emgformer_partitions(store,val_fraction=.12):
    if type(val_fraction)not in (int,float)or not math.isfinite(val_fraction)or not 0<=val_fraction<1:
        raise ValueError('finite validation fraction in [0,1) required')
    result=build_partitions(store,window_length=W,train_stride=STRIDE,eval_stride=W,val_fraction=val_fraction)
    forbidden=union_intervals([(v['start'],v['end'])for v in result['validation_intervals']]+
        [(w['start'],w['end'])for w in result['windows']if w['role']=='test'])
    ends=[b for _,b in forbidden];splits=store.field('frame_split_id')
    train_id=next(i for i,n in store.split_labels.items()if n=='train');windows=[];discarded=0
    for w in result['windows']:
        upper=min(STRIDE,store.episodes[w['episode']]['end']-w['end'])if w['role']=='train'else 0
        maximum=max(0,upper-1);end=w['end']+maximum;i=bisect_right(ends,w['start'])
        if w['role']=='train'and ((i<len(forbidden)and forbidden[i][0]<end)or not np.all(splits[w['end']:end]==train_id)):
            discarded+=1;continue
        windows.append(dict(w,jitter_upper=upper,jitter_max_offset=maximum,
            footprint_start=w['start'],footprint_end=end))
    result['windows']=windows;result['discarded']['native_jitter_unsafe_anchor']=discarded
    result['settings'].update(jitter=True,repeat_by_role={r:1 for r in ROLES},
        index_order='left-role-windows then right-role-windows',center_target_only_by_role=dict(train=False,val=True,test=True))
    for role in ROLES:
        chosen=[w for w in windows if w['role']==role]
        spans=union_intervals((w['footprint_start'],w['footprint_end'])for w in chosen)
        result['coverage'][role]=dict(windows=len(chosen),hand_samples=2*len(chosen),consumed_hand_samples=2*len(chosen),
            unique_possible_rows=sum(b-a for a,b in spans),episodes=len({w['episode']for w in chosen}),
            by_split=dict(Counter(w['split']for w in chosen)))
    provenance=result['provenance'];del provenance['partition_sha256']
    provenance.update(protocol='native-egoemg-emgformer-on-explicit-legacy-adapter-v1',
        normalization='explicit same-asset egoemg per-hand vectors; declared unified namespace missing',
        evaluation='original12000 episode-grid test centers; train-only temporal validation; native center targets',
        training='original exclusive jitter range retained; unsafe anchors removed without clipping')
    provenance['partition_sha256']=_hash(result);return result


class EmgformerDataset:
    """Native per-hand IO plus provenance; construction is not numeric admission."""
    def __init__(self,data,partitions,*,upstream,config,normalization,role):
        if role not in ROLES:raise ValueError('EMGFormer role must be train, val or test')
        root,size,_=_validated(upstream,config);contract=emgformer_data_contract(root,config)
        if _hash(normalization)!=_hash(contract['normalization']):raise ValueError('EMGFormer normalization binding differs')
        store=LegacyStore(data);_schema(store)
        expected=build_emgformer_partitions(store,partitions['settings']['val_fraction'])
        if _hash(partitions)!=_hash(expected):raise ValueError('EMGFormer partition binding differs')
        self.partitions=copy.deepcopy(expected);self.role=role;self.repeat=1;self.size=size
        self.windows=[w for w in expected['windows']if w['role']==role]
        if not self.windows:raise ValueError('EMGFormer requested role must be nonempty')
        self.store=store;self.upstream=root;self.config=copy.deepcopy(config)
        self.contract=contract;self.normalization=copy.deepcopy(normalization)
        self.views=[_view(store,self.windows,h,role,root,config)for h in ('left','right')]
        split_ids={name:i for i,name in store.split_labels.items()}
        for view in self.views:
            view.center_target_only=view._center_target_only=contract['center_target_only_by_role'][role]
            view._window_split_id=np.asarray([split_ids[w['split']]for w in self.windows],np.int32)
        self._partition_seal=_hash(self.partitions)

    def __getstate__(self):
        # Reopen process-local read-only views instead of pickling memmap payloads.
        return dict(data=self.store.root,partitions=self.partitions,upstream=self.upstream,
            config=self.config,normalization=self.normalization,role=self.role)

    def __setstate__(self,state):
        self.__init__(**state)

    def __len__(self):return 2*len(self.windows)

    def __getitem__(self,index):
        if type(index)is not int or not 0<=index<len(self):raise IndexError(index)
        hand,wi=divmod(index,len(self.windows));w=self.windows[wi];result=self.views[hand][wi]
        start=int(result['window_start_idx'])if self.role=='train'else w['start']
        end=int(result['window_end_idx'])if self.role=='train'else w['end']
        result.update(role=self.role,split=w['split'],episode=w['episode'],hand=('left','right')[hand],
            start=start,end=end,center=start+W//2,anchor_start=w['start'],anchor_end=w['end'],anchor_center=w['center'],
            jitter_offset=start-w['start'],emgformer_window_index=wi,emgformer_sample_index=index,
            emgformer_unique_sample_index=index,emgformer_repeat_index=0)
        return result


def native_emgformer_loader(dataset,*,cpu_engineering=False):
    if type(dataset)is not EmgformerDataset or type(cpu_engineering)is not bool:
        raise ValueError('native EMGFormer dataset and explicit engineering boolean required')
    if (_hash(dataset.partitions)!=dataset._partition_seal or
            dataset.windows!=[w for w in dataset.partitions['windows']if w['role']==dataset.role]or
            _hash(dataset.contract)!=_hash(emgformer_data_contract(dataset.upstream,dataset.config))or
            _hash(dataset.normalization)!=_hash(dataset.contract['normalization'])):
        raise ValueError('EMGFormer loader reader contract changed')
    _activate(dataset.upstream)
    from egoemg.datamodule import make_data_module
    dm=make_data_module(dataset.config)
    if cpu_engineering:dm.num_workers=dm.eval_num_workers=0
    loader=dm._make_loader(dataset,shuffle=dataset.role=='train',
        num_workers=dm.num_workers if dataset.role=='train'else dm.eval_num_workers)
    loader.egoemg_repro_scope='cpu_engineering'if cpu_engineering else 'native_loader_construction_only'
    return loader
