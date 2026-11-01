"""Current, byte-bound original tracking/stale/per-hand marker flags.

Only provider names are adapted for the released legacy files. No generated
label, crop, mesh-validity or prediction filter replaces the original flags.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import weakref

import numpy as np

from .classic_emg import _hash
from .classic_emg_cohort import _safe
from .data import file_sha256
from .native_ik_inputs import _Guard
from .occlusion_dense_inference import native_batch_ranges
from .occlusion_dense_predictions import KEY_FIELDS
from .qualitative_run import QualitativeEvidence

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'results/native-dense-occlusion/geometry-input-admission-20261101.json'
REFERENCE_SHA='d89f6298787e4eaae8a12207703f9a0b18f9959f90e8f03b69bf3dbbaaebbfdb'
IMPORTED_SHA=file_sha256(__file__)
FIELDS=('mocap_webcam_tracked','image_webcam_stale','mocap_left_valid','mocap_right_valid')
ALIASES=dict(mocap_head_valid='mocap_webcam_tracked',image_head_stale='image_webcam_stale')
_ADMITTED=weakref.WeakKeyDictionary()


def extract_qualitative_flags(keys,fields):
    """Pure numeric extraction; no source admission and no population drop."""
    keys=copy.deepcopy(list(keys));native_batch_ranges(keys)
    if not isinstance(fields,dict)or set(fields)!=set(FIELDS):
        raise ValueError('all four original boolean flag fields required; no generated-label substitute')
    tracked=fields['mocap_webcam_tracked'];n=len(tracked)
    for name,a in fields.items():
        if (not isinstance(a,np.ndarray)or a.dtype!=np.bool_
            or a.shape!=((n,21)if name.endswith('_valid')else(n,))):
            raise ValueError('exact full boolean frame/21-marker source shapes required')
    rows=[]
    for k in keys:
        c=k['center_idx']
        if type(c)is not int or not 0<=c<n:raise ValueError('actual original center inside full flag rows required')
        rows.append(dict({name:k[name]for name in KEY_FIELDS},mocap_tracked=bool(tracked[c]),
            image_stale=bool(fields['image_webcam_stale'][c]),
            markers_valid=bool(fields['mocap_'+k['hand']+'_valid'][c].any())))
    return rows


@dataclass(frozen=True,eq=False)
class QualitativeFlags:
    def check(self):
        if type(self)is not QualitativeFlags or self not in _ADMITTED:
            raise ValueError('registered original qualitative flag source required')
        s=_ADMITTED[self];s['evidence'].check();s['guard'].check()
        if (file_sha256(__file__)!=IMPORTED_SHA or _hash(s['identity'])!=s['identity_hash']
            or _hash(s['records'])!=s['records_hash']):raise ValueError('original flag identity or records changed')
        for name,a in s['arrays'].items():
            if (type(a)is not np.memmap or a.mode!='r' or a.flags.writeable or a.offset!=0
                or a.dtype!=np.bool_ or not a.flags.c_contiguous or _safe(a.filename)!=s['paths'][name]
                or a.shape!=((s['n'],21)if name.endswith('_valid')else(s['n'],))):
                raise ValueError('exact readonly original boolean source mapping required')

    @property
    def identity(self):self.check();return copy.deepcopy(_ADMITTED[self]['identity'])

    def records(self):self.check();return [json.loads(s)for s in _ADMITTED[self]['records']]


def admit_qualitative_flags(evidence,*,data,geometry):
    if type(evidence)is not QualitativeEvidence:raise ValueError('registered actual ordinary source required')
    evidence.check();data,geometry=_safe(data),_safe(geometry)
    reference=_safe(REFERENCE);manifest=_safe(data/'manifest.json')
    sources=[_safe(__file__),reference,manifest]
    initial=_Guard(sources,[])
    if file_sha256(reference)!=REFERENCE_SHA or file_sha256(__file__)!=IMPORTED_SHA:
        raise ValueError('fixed complete original boolean geometry reference required')
    accepted=json.loads(reference.read_text())['identity'];meta=json.loads(manifest.read_text());n=accepted['total_rows']
    if (n!=66161725 or meta['total_rows']!=n
        or file_sha256(manifest)!=evidence.identity['complete_payload_sha256']['metadata:manifest']):
        raise ValueError('same complete released manifest/row population required')
    paths={};arrays={}
    for name in FIELDS:
        spec=meta['fields'][name];shape=[n,21]if name.endswith('_valid')else[n]
        if np.dtype(spec['dtype'])!=np.bool_ or spec['shape']!=shape:
            raise ValueError('actual original manifest flag dtype/shape differs')
        path=_safe((data if name.startswith('image_')else geometry)/spec['filename'])
        if path.stat().st_size!=int(np.prod(shape)):raise ValueError('complete flag payload byte length required')
        paths[name]=path
    guard=_Guard([*sources,*paths.values()],[]);before={}
    for name,path in paths.items():
        guard.check();before[name]=file_sha256(path);guard.check()
        if before[name]!=accepted['raw_input_files_sha256']['field:'+name]:
            raise ValueError('same fully admitted original boolean bytes required')
        arrays[name]=np.memmap(path,mode='r',dtype=np.bool_,shape=tuple(meta['fields'][name]['shape']))
    rows=extract_qualitative_flags(evidence.keys(),arrays)
    after={}
    for name,path in paths.items():guard.check();after[name]=file_sha256(path);guard.check()
    if before!=after:raise ValueError('complete original boolean files changed during extraction')
    counts=dict(selected=len(rows),source_filter_pass=sum(r['mocap_tracked']and not r['image_stale']and r['markers_valid']for r in rows),
        untracked=sum(not r['mocap_tracked']for r in rows),image_stale=sum(r['image_stale']for r in rows),no_valid_hand_marker=sum(not r['markers_valid']for r in rows))
    identity=dict(format='complete_actual_ordinary_qualitative_boolean_flags_v1',
        source_evidence_identity_sha256=_hash(evidence.identity),source_key_sha256=_hash(evidence.keys()),
        reference_sha256=REFERENCE_SHA,manifest_sha256=file_sha256(manifest),
        full_boolean_files_sha256=before,full_boolean_source_bytes=sum(p.stat().st_size for p in paths.values()),
        total_rows=n,selected_hand_points=len(rows),source_aliases=ALIASES,
        counts=counts,all_selected_rows_retained=True,filter_applied_to_prediction_population=False,
        source_filter='tracked and not stale and any valid marker for actual hand',
        publisher_authenticated_bulk_sha256_available=False,physical_camera_equivalence_proven=False,
        source_files_sha256={'qualitative_flags.py':IMPORTED_SHA},
        model_or_geometry_permit=False,figure7_reproduced=False,scientific_eligible=False,full_paper_complete=False)
    records=tuple(json.dumps(r,sort_keys=True,separators=(',',':'))for r in rows)
    initial.check();guard.check();evidence.check()
    result=QualitativeFlags();_ADMITTED[result]=dict(evidence=evidence,guard=guard,identity=identity,
        identity_hash=_hash(identity),arrays=arrays,paths=paths,n=n,records=records,records_hash=_hash(records))
    result.check();return result
