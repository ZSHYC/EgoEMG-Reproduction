"""Source-native ordinary-grid Figure7 candidates, never an inference permit.

Seeded subsampling is reset separately for every original split/hand group.
Only exact standard-grid keys can later consume an existing prediction packet.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import weakref

import numpy as np

from .classic_emg import _hash
from .classic_emg_cohort import _safe
from .data import LegacyStore, file_sha256
from .native_ik_inputs import _Guard
from .occlusion_dense_data import COLUMNS, SPLITS, HANDS, _recipe, _providers, _native_view, _typed_sha

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'results/native-dense-occlusion/full-index-admission-20261101.json'
REFERENCE_SHA = 'b57de34dcf8c18fa94cb6390c1a92cf90c619f7b1e95dfc77ce3f78ca0417318'
IMPORTED_SHA = file_sha256(__file__)
_ADMITTED = weakref.WeakKeyDictionary()


def _arguments(max_samples_per_split, seed):
    if type(max_samples_per_split) is not int or max_samples_per_split < 0:
        raise ValueError('nonnegative integer max_samples_per_split required;0 means all')
    if type(seed) is not int or seed < 0:
        raise ValueError('nonnegative integer qualitative seed required')


def _paths(upstream):
    result = _providers(upstream)
    result['upstream/scripts/eval/find_fusion_wins.py'] = _safe(upstream/'scripts/eval/find_fusion_wins.py')
    result['egoemg_repro/qualitative_data.py'] = _safe(__file__)
    result['egoemg_repro/native_ik_inputs.py'] = _safe(ROOT/'egoemg_repro/native_ik_inputs.py')
    return result


def qualitative_protocol(upstream, config, *, recipe, max_samples_per_split=500, seed=42):
    _arguments(max_samples_per_split, seed)
    root, value, experiment = _recipe(upstream, config, recipe)
    paths = _paths(root); guard = _Guard(paths.values(), [])
    hashes = {n:file_sha256(p) for n,p in paths.items()}; guard.check()
    if hashes['egoemg_repro/qualitative_data.py'] != IMPORTED_SHA:
        raise ValueError('imported qualitative selection source changed')
    source = root/'scripts/eval/find_fusion_wins.py'
    tree = ast.parse(source.read_text()); defaults = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'add_argument' and node.args and isinstance(node.args[0], ast.Constant)):
            for keyword in node.keywords:
                if keyword.arg == 'default' and node.args[0].value in ('--seed','--max-samples-per-split','--splits','--hands','--batch-size'):
                    defaults[node.args[0].value] = ast.literal_eval(keyword.value)
    if defaults != {'--seed':42,'--max-samples-per-split':500,'--splits':list(SPLITS),
                    '--hands':list(HANDS),'--batch-size':64}:
        raise ValueError('original qualitative parser defaults differ')
    dm=value['datamodule']; window=dm['val_test_window_length']; stride=dm['val_test_stride']
    expected=7790 if recipe=='vits_s' else 12000
    if type(window) is not int or type(stride) is not int or (window,stride)!=(expected,expected):
        raise ValueError('original ordinary qualitative window/stride required')
    # Initial composition precedes the first file guard. Revalidate the clean
    # pinned checkout afterward so that this admission gap cannot bless drift.
    final_root,final_value,final_experiment=_recipe(root,config,recipe)
    if final_root!=root or _hash(final_value)!=_hash(value) or final_experiment!=experiment:
        raise ValueError('original qualitative source/config changed during admission')
    guard.check()
    result=dict(format='original_qualitative_candidate_protocol_v1', recipe=recipe,
        upstream_revision='f7cbd048b7092139852270491d3859cd472e0311', upstream_experiment=experiment,
        source_script='scripts/eval/find_fusion_wins.py', window_length=window, stride=stride,
        max_samples_per_split=max_samples_per_split, seed=seed, original_defaults=defaults,
        rng_reset_each_split_and_hand=True, sampled_native_indices_sorted=True,
        selection='native episode-start grid, center split only', eval_center_stride=None,
        splits=list(SPLITS), hands=list(HANDS), repeat=1, jitter=False,
        full_window_split_filter=False, source_files_sha256=hashes, config_sha256=_hash(value),
        model_or_geometry_permit=False, figure7_reproduced=False, full_paper_complete=False)
    result['sha256']=_hash(result); return result


@dataclass(frozen=True, eq=False)
class QualitativeSelection:
    def check(self):
        if type(self) is not QualitativeSelection or self not in _ADMITTED:
            raise ValueError('registered original qualitative selection required')
        state=_ADMITTED[self]; state['guard'].check()
        if (file_sha256(__file__) != IMPORTED_SHA or _hash(state['identity']) != state['identity_hash']
            or any(a.flags.writeable or _typed_sha(a)!=state['identity']['arrays_sha256'][n]
                   for n,a in state['points'].items())):
            raise ValueError('original qualitative source/identity/point binding changed')

    @property
    def identity(self):
        self.check(); return copy.deepcopy(_ADMITTED[self]['identity'])

    @property
    def points(self):
        self.check(); points={n:a.copy() for n,a in _ADMITTED[self]['points'].items()}
        for a in points.values():a.flags.writeable=False
        return points


def admit_qualitative_selection(data, *, upstream, config, recipe,
                                max_samples_per_split=500, seed=42, engineering=False):
    _arguments(max_samples_per_split,seed)
    if type(engineering) is not bool: raise ValueError('explicit qualitative engineering boolean required')
    protocol=qualitative_protocol(upstream,config,recipe=recipe,max_samples_per_split=max_samples_per_split,seed=seed)
    data,upstream=_safe(data),_safe(upstream); paths=_paths(upstream)
    inputs={'manifest':_safe(data/'manifest.json'),'metadata':_safe(data/'metadata.npz')}
    initial=_Guard([*paths.values(),*inputs.values(),*([_safe(REFERENCE)] if not engineering else [])],[])
    manifest=json.loads(inputs['manifest'].read_text()); n=manifest.get('total_rows')
    spec=manifest.get('fields',{}).get('frame_split_id',{})
    if (type(n) is not int or n<1 or spec.get('shape')!=[n]
        or np.dtype(spec.get('dtype','float64')).kind not in 'iu'):
        raise ValueError('exact integer one-dimensional split schema/shape required')
    inputs['frame_split_id']=_safe(data/spec.get('filename','frame_split_id.dat'))
    guard=_Guard([*initial.stamps,inputs['frame_split_id']],[]); initial.check()
    hashes={name:file_sha256(path) for name,path in inputs.items()}; guard.check()
    if {name:file_sha256(path) for name,path in paths.items()}!=protocol['source_files_sha256']:
        raise ValueError('qualitative provider bytes changed before selection')
    store=LegacyStore(data)
    if (list(store.split_labels.values())!=['train',*SPLITS]
        or list(store.metadata.get('splits_split',[]))!=['train',*SPLITS]
        or list(store.metadata.get('splits_split_id',[]))!=[0,1,2,3]):
        raise ValueError('exact original split labels and identifiers required')
    split=store.field('frame_split_id')
    if (not isinstance(split,np.memmap) or split.flags.writeable or split.mode!='r' or split.offset!=0
        or split.shape!=(n,) or not split.flags.c_contiguous or Path(split.filename)!=inputs['frame_split_id']
        or inputs['frame_split_id'].stat().st_size!=n*split.dtype.itemsize or np.any((split<0)|(split>3))):
        raise ValueError('complete readonly original split mapping required')
    if not engineering:
        if file_sha256(REFERENCE)!=REFERENCE_SHA:
            raise ValueError('exact published complete index reference required')
        reference=json.loads(REFERENCE.read_text())['recipes'][recipe]
        if (n!=reference['total_rows'] or len(store.episodes)!=reference['episode_records']
            or hashes!=reference['input_files_sha256']):
            raise ValueError('full registered released qualitative index source differs')
    columns={name:[] for name in COLUMNS};groups=[];crossings=0;seen=set()
    for group,role in enumerate(SPLITS):
        sid=group+1
        for hand_index,hand in enumerate(HANDS):
            guard.check()
            view=_native_view(store,upstream,protocol['window_length'],protocol['stride'],role,hand)
            count=len(view)
            rng=np.random.default_rng(seed)  # The original loop resets for EACH group.
            indices=(sorted(rng.choice(count,size=max_samples_per_split,replace=False).tolist())
                     if max_samples_per_split>0 and count>max_samples_per_split else list(range(count)))
            groups.append(dict(split=role,hand=hand,native_points=count,selected_points=len(indices),
                               selected_native_indices_sha256=_typed_sha(np.asarray(indices,np.int64))))
            for native_index in indices:
                episode,center=view._resolve_index_to_center(native_index)
                start=center-protocol['window_length']//2;end=start+protocol['window_length']
                if (int(split[center])!=sid or (episode,center,hand_index) in seen
                    or not store.episodes[episode]['start']<=start<end<=store.episodes[episode]['end']):
                    raise ValueError('exact unique native qualitative hand/window key required')
                seen.add((episode,center,hand_index));crossings+=int(np.any(split[start:end]!=sid))
                values=(len(columns['sequence']),episode,start,end,center,sid,hand_index,native_index,group)
                for name,value in zip(COLUMNS,values):columns[name].append(value)
            guard.check()
    points={name:np.asarray(values,dtype=COLUMNS[name]) for name,values in columns.items()}
    for a in points.values():a.flags.writeable=False
    if hashes!={name:file_sha256(path) for name,path in inputs.items()}:
        raise ValueError('complete original qualitative index input bytes changed')
    if protocol['source_files_sha256']!={name:file_sha256(path) for name,path in paths.items()}:
        raise ValueError('qualitative source bytes changed during selection')
    guard.check();initial.check()
    torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():raise ValueError('qualitative index must remain CPU-only')
    identity=dict(format='original_qualitative_candidate_selection_v1',protocol=protocol,
        engineering=engineering,full_registered_index_source=not engineering,
        complete_index_reference_sha256=REFERENCE_SHA if not engineering else None,
        input_files_sha256=hashes,total_rows=n,episode_records=len(store.episodes),groups=groups,
        native_hand_points=sum(g['native_points'] for g in groups),selected_hand_points=len(seen),
        arrays_sha256={name:_typed_sha(a) for name,a in points.items()},
        windows_crossing_center_split=crossings,independent_unseen_input_footprints_proven=False,
        category_labels_assigned=False,learned_pose_models_constructed=0,optimizer_updates=0,
        native_geometry_executed=False,cuda_initialized=False,model_or_geometry_permit=False,
        figure7_reproduced=False,scientific_eligible=False,full_paper_complete=False)
    selected=QualitativeSelection()
    _ADMITTED[selected]=dict(guard=guard,identity=identity,identity_hash=_hash(identity),points=points)
    selected.check();return selected
