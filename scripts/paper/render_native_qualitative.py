#!/usr/bin/env python3
"""Export an actual ordinary-grid RGB/pose candidate pool for visual review."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',default=str(ROOT/'results/native-qualitative'))
    for name in ('paired-root','upstream','data','crops','geometry','first','second','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--limit',type=int,default=120)
    args=parser.parse_args()
    from egoemg_repro.qualitative_run import load_qualitative_evidence,admit_qualitative_run_inputs
    from egoemg_repro.qualitative_flags import admit_qualitative_flags
    from egoemg_repro.qualitative_render import export_qualitative_candidates
    evidence=load_qualitative_evidence(evidence=args.evidence,paired_root=args.paired_root,upstream=args.upstream,
        producer_root=Path(args.first).parent.parent)
    inputs=admit_qualitative_run_inputs(evidence,data=args.data,crops=args.crops)
    flags=admit_qualitative_flags(evidence,data=args.data,geometry=args.geometry)
    result=export_qualitative_candidates(inputs,args.first,args.second,flags=flags,output=args.output,limit=args.limit)
    print(json.dumps(dict(rendered_candidates=result['rendered_candidates'],counts=result['counts'],
        condition_examples_selected=False,figure7_reproduced=False,full_paper_complete=False)))


if __name__=='__main__':main()
