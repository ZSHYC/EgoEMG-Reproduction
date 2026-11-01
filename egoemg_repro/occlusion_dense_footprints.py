"""Describe dense heldout-center input intersections; never admit a model.

Unique temporal rows are counted once across the two matched hands. A separate
request count retains both hands. The original half-open interval primitives
are reused; callers must bind/reconstruct the actual producer partitions.
"""
from bisect import bisect_right

import numpy as np

from .classic_emg import _hash
from .data import union_intervals
from .fusion_initializer_handoff import _bounds,_intersection
from .occlusion_dense_data import COLUMNS


def _overlaps(spans,ends,lo,hi):
    i=bisect_right(ends,lo)
    return i<len(spans)and spans[i][0]<hi


def summarize_dense_footprints(points,partitions):
    """Pure aggregate diagnostics, not proof of source/data/weight provenance."""
    if not isinstance(points,dict)or set(points)!=set(COLUMNS):raise ValueError('complete typed dense point columns required')
    n=len(points['sequence'])
    for name,dtype in COLUMNS.items():
        a=points[name]
        if not isinstance(a,np.ndarray)or a.dtype!=np.dtype(dtype)or a.shape!=(n,):
            raise ValueError('dense footprint column dtype/shape differs')
    if not np.array_equal(points['sequence'],np.arange(n,dtype=np.int64)):
        raise ValueError('unique ordered dense sequence required')
    temporal={};keys=set()
    for j in range(n):
        ep,center,hand=(int(points[k][j])for k in('episode','center','hand_index'))
        lo,hi=(int(points[k][j])for k in('start','end'))
        if ep<0 or hand not in(0,1)or not 0<=lo<=center<hi or(ep,center,hand)in keys:
            raise ValueError('unique bounded hand/episode/center required')
        keys.add((ep,center,hand));pair=temporal.setdefault((ep,center),{})
        pair[hand]=(lo,hi)
    if any(set(v)!={0,1}or v[0]!=v[1]for v in temporal.values()):
        raise ValueError('complete two-hand paired window bounds required')
    dense=union_intervals(v[0]for v in temporal.values());producer={r:[]for r in('train','val','test')}
    try:
        for w in partitions['windows']:
            if w['role']not in producer:raise ValueError('unknown producer role')
            producer[w['role']].append(_bounds(w['footprint_start'],w['footprint_end']))
    except(KeyError,TypeError)as exc:raise ValueError('complete producer role/footprint bounds required')from exc
    roles={}
    for role,raw in producer.items():
        spans=union_intervals(raw);ends=[b for _,b in spans]
        roles[role]=dict(producer_input_union_rows=sum(b-a for a,b in spans),intersection_rows=_intersection(dense,spans),
            overlapping_hand_requests=sum(_overlaps(spans,ends,int(a),int(b))for a,b in zip(points['start'],points['end'])),
            centers_in_producer_rows=sum(_overlaps(spans,ends,int(c),int(c)+1)for c in points['center']))
    return dict(hand_requests=n,unique_temporal_windows=len(temporal),dense_input_union_rows=sum(b-a for a,b in dense),
        roles=roles,producer_partition_content_sha256=_hash(partitions),
        any_training_input_overlap=roles['train']['intersection_rows']>0,
        population_policy='all supplied paired native keys; unique temporal input rows once, hand requests separately; half-open bounds',
        source_data_and_checkpoint_admitted=False,model_or_geometry_permit=False,scientific_eligible=False,full_paper_complete=False)
