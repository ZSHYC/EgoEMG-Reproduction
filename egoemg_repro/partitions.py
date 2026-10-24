"""Deterministic temporal validation guarded against fixed official test windows."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import LegacyStore, file_sha256, union_intervals


LIMITATIONS = [
    'Legacy prefiltered temporal validation: official filtered EMG was processed over whole episodes; disjoint row footprints do not remove upstream filter dependence.',
    'Generated labels are the public release labels; public label-generation code differs from the paper description. No labels are regenerated here.',
    'Validation is a deterministic contiguous temporal holdout within official training runs, not a new unseen-subject benchmark.',
    'Short available training runs with fewer than two full windows contribute training only; validation fraction is approximate.',
]


def _subtract(intervals, excluded):
    """Subtract sorted disjoint half-open test footprints from official train runs."""
    out = []
    j = 0
    for start,end in intervals:
        while j < len(excluded) and excluded[j][1] <= start:
            j += 1
        cursor = start
        k = j
        while k < len(excluded) and excluded[k][0] < end:
            lo,hi = excluded[k]
            if lo > cursor:
                out.append((cursor,min(lo,end)))
            cursor = max(cursor,hi)
            if cursor >= end:
                break
            k += 1
        if cursor < end:
            out.append((cursor,end))
    return out


def _contained(start, end, intervals, starts):
    index = bisect_right(starts,start)-1
    return index >= 0 and end <= intervals[index][1]


def build_partitions(store, window_length=12000, train_stride=1200, eval_stride=12000, val_fraction=.12):
    """Preserve episode-grid official test centers, then reserve train-run tails.

    Validation uses complete windows at each sufficiently long residual run's end.
    Training windows stay on the episode-anchored training grid and must fit wholly
    inside retained official-train intervals. Both hands share this single manifest.
    """
    if window_length <= 0 or train_stride <= 0 or eval_stride <= 0:
        raise ValueError('Window length and strides must be positive')
    if not 0 <= val_fraction < 1:
        raise ValueError('val_fraction must be in [0,1)')
    splits = store.field('frame_split_id')
    train_id = next(i for i,name in store.split_labels.items() if name=='train')
    windows, validations = [], []
    discarded = Counter()
    official_counts = Counter()
    candidates = Counter()
    W = window_length
    for ep in store.episodes:
        lo,hi = ep['start'],ep['end']
        values = np.asarray(splits[lo:hi])
        ids,counts = np.unique(values,return_counts=True)
        for split_id,count in zip(ids,counts):
            if int(split_id) not in store.split_labels:
                raise ValueError(f'Unknown frame split ID {split_id} in episode {ep["index"]}')
            official_counts[store.split_labels[int(split_id)]] += int(count)
        def record(start,role,split):
            return dict(episode=ep['index'],start=int(start),end=int(start+W),center=int(start+W//2),role=role,split=split)
        tests = []
        # Shared eval center grid is independent of model window length.
        for center in range(lo+eval_stride//2,hi,eval_stride):
            start = center-W//2
            if start < lo or start+W > hi:
                discarded['eval_episode_boundary'] += 1
                continue
            name = store.split_labels[int(splits[center])]
            if name != 'train':
                tests.append(record(start,'test',name))
        windows.extend(tests)
        forbidden = union_intervals((w['start'],w['end']) for w in tests)
        is_train = values == train_id
        edges = np.flatnonzero(np.diff(np.r_[False,is_train,False]))
        runs = [(lo+int(a),lo+int(b)) for a,b in zip(edges[::2],edges[1::2])]
        residual = _subtract(runs,forbidden)
        train_intervals = []
        for start,end in residual:
            n_full = (end-start)//W
            if val_fraction > 0 and n_full >= 2:
                n_val = min(n_full-1,max(1,int(n_full*val_fraction+.5)))
                val_start = end-n_val*W
                validations.append(dict(episode=ep['index'],start=val_start,end=end))
                for pos in range(val_start,end-W+1,eval_stride):
                    windows.append(record(pos,'val','train'))
                train_intervals.append((start,val_start))
            else:
                train_intervals.append((start,end))
                if n_full < 2:
                    discarded['short_runs_without_validation'] += 1
        starts = [start for start,end in train_intervals]
        for start in range(lo,hi-W+1,train_stride):
            candidates['training_grid'] += 1
            if _contained(start,start+W,train_intervals,starts):
                windows.append(record(start,'train','train'))
            else:
                discarded['training_grid_outside_retained_train'] += 1
        if hi-lo < W:
            discarded['short_episodes'] += 1
    windows.sort(key=lambda w:(w['episode'],w['start'],w['role']))
    coverage = {}
    for role in ('train','val','test'):
        selected = [w for w in windows if w['role']==role]
        intervals = union_intervals((w['start'],w['end']) for w in selected)
        coverage[role] = dict(windows=len(selected),hand_samples=2*len(selected),unique_rows=sum(b-a for a,b in intervals),
                              episodes=len({w['episode'] for w in selected}),by_split=dict(Counter(w['split'] for w in selected)))
    # The row footprint invariant is checked without enumerating millions of rows.
    role_intervals = {role:union_intervals((w['start'],w['end']) for w in windows if w['role']==role) for role in coverage}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        original = sum(y-x for x,y in role_intervals[a])
        remaining = sum(y-x for x,y in _subtract(role_intervals[a],role_intervals[b]))
        if original != remaining:
            raise AssertionError(f'Overlapping {a}/{b} row footprints')
    spec = store.fields['frame_split_id']
    settings = dict(window_length=W,train_stride=train_stride,eval_stride=eval_stride,val_fraction=val_fraction,jitter=False)
    result = dict(schema_version=1,settings=settings,episodes=store.episodes,windows=windows,validation_intervals=validations,
                  coverage=coverage,official_split_rows=dict(official_counts),discarded=dict(discarded),candidates=dict(candidates),
                  provenance=dict(manifest_sha256=file_sha256(store.manifest_path),metadata_sha256=file_sha256(store.metadata_path),
                                  frame_split_sha256=file_sha256(store.root/spec.get('filename','frame_split_id.dat')),
                                  format_version=store.manifest.get('format_version'),
                                  protocol='legacy-prefiltered-temporal-validation',limitations=LIMITATIONS,
                                  interval_convention='half-open [start,end)',normalization='train-only unique row union per hand'))
    canonical = json.dumps(result,sort_keys=True,separators=(',',':')).encode()
    result['provenance']['partition_sha256'] = hashlib.sha256(canonical).hexdigest()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--window-length',type=int,default=12000)
    parser.add_argument('--train-stride',type=int,default=1200)
    parser.add_argument('--eval-stride',type=int,default=12000)
    parser.add_argument('--val-fraction',type=float,default=.12)
    args = parser.parse_args(argv)
    manifest = build_partitions(LegacyStore(args.data),args.window_length,args.train_stride,args.eval_stride,args.val_fraction)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),coverage=manifest['coverage'],discarded=manifest['discarded']),indent=2))


if __name__ == '__main__':
    main()
