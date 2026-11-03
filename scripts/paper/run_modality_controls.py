"""Owned fixed-checkpoint EMG/visual-loss controls on complete native grids."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('workflow','predict','accept-qualification','report','verify-report'))
    parser.add_argument('--request',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--scope',choices=('qualification','complete'))
    parser.add_argument('--qualification',type=Path)
    parser.add_argument('--replay-of',type=Path)
    parser.add_argument('--first',type=Path)
    parser.add_argument('--second',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    from egoemg_repro import modality_control_run as run
    request=json.loads(args.request.read_text())
    if args.command=='workflow':
        value=run.run_control_workflow(request,args.output)
    else:
        evidence=run.load_control_evidence(request)
        if args.command=='predict':
            if args.scope is None:parser.error('predict requires explicit --scope')
            inputs=run.admit_control_inputs(evidence)
            value=run.run_control_pass(inputs,args.output,scope=args.scope,
                qualification=args.qualification,replay_of=args.replay_of)
        else:
            if args.first is None or args.second is None:parser.error('two distinct complete producing passes required')
            if args.command=='accept-qualification':
                value=run.accept_control_qualification(args.first,args.second,evidence=evidence,output=args.output)
            elif args.command=='report':
                value=run.build_control_report(args.first,args.second,evidence=evidence,output=args.output)
            else:
                if args.report is None:parser.error('verify-report requires --report')
                value=run.verify_control_report(args.first,args.second,args.report,evidence=evidence,output=args.output)
    print(json.dumps(value,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
