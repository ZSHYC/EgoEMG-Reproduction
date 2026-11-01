#!/usr/bin/env python3
"""Native ordinary Figure7 frozen inference and source-bound raw verification."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('infer','verify','compare'))
    p.add_argument('--evidence',default=str(ROOT/'results/native-qualitative'))
    p.add_argument('--paired-root',required=True)
    p.add_argument('--upstream',required=True)
    p.add_argument('--data');p.add_argument('--crops');p.add_argument('--fit')
    p.add_argument('--pass-root');p.add_argument('--replay-of');p.add_argument('--output',required=True)
    a=p.parse_args()
    from egoemg_repro.qualitative_run import (load_qualitative_evidence,admit_qualitative_run_inputs,
        run_qualitative_pass,read_qualitative_pass,compare_qualitative_passes)
    from egoemg_repro.occlusion_dense_run import _seal,_write_json
    from egoemg_repro.occlusion_dense_data import _typed_sha
    from egoemg_repro.classic_emg import _hash
    from egoemg_repro.emg2pose_memory_process import process_identity
    e=load_qualitative_evidence(evidence=a.evidence,paired_root=a.paired_root,upstream=a.upstream)
    if a.mode=='infer':
        if not all((a.data,a.crops,a.fit)):p.error('infer requires --data, --crops and --fit')
        inputs=admit_qualitative_run_inputs(e,data=a.data,crops=a.crops)
        run_qualitative_pass(inputs,a.fit,a.output,replay_of=a.replay_of)
    elif a.mode=='verify':
        if not a.pass_root:p.error('verify requires --pass-root')
        v=read_qualitative_pass(a.pass_root,evidence=e)
        result=_seal(dict(format='fresh_actual_ordinary_prediction_readback_v1',actor=process_identity(),
            source_evidence_identity_sha256=_hash(e.identity),producing_actor=v['result']['actor'],
            identity_sha256=v['identity']['sha256'],result_file_sha256=v['_result_file_sha256'],
            points=v['result']['points'],batches=v['result']['batches'],
            complete_arrays_sha256={n:_typed_sha(x)for n,x in v['arrays'].items()},
            crop_missing=v['result']['crop_missing'],label_invalid=v['result']['label_invalid'],
            native_frozen_raw_packets_verified=True,figure7_reproduced=False,scientific_eligible=False,full_paper_complete=False))
        v['_guard'].check();v['_packet_guard'].check();e.check();_write_json(Path(a.output),result)
        v['_guard'].check();v['_packet_guard'].check();e.check()
    else:
        if not(a.pass_root and a.replay_of):p.error('compare requires --pass-root and --replay-of')
        result=compare_qualitative_passes(a.pass_root,a.replay_of,evidence=e)
        _write_json(Path(a.output),_seal(dict(result,actor=process_identity(),source_evidence_identity_sha256=_hash(e.identity))))


if __name__=='__main__':main()
