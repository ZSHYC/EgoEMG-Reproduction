"""Selected original S/M/L FP32 shared-center evaluation; no training."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
from egoemg_repro.center_reference_eval import run_reference_pass,read_reference_pass

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run');run.add_argument('--source-fit',required=True,type=Path)
    run.add_argument('--admitted-inputs',required=True,type=Path);run.add_argument('--output',required=True,type=Path)
    run.add_argument('--kind',required=True,choices=('native','replay'));run.add_argument('--reference',type=Path)
    run.add_argument('--scope',required=True,choices=('engineering','full'))
    run.add_argument('--device',required=True,choices=('cpu','cuda:0'));run.add_argument('--gpu-uuid')
    read=sub.add_parser('read');read.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.command=='run':
        value=run_reference_pass(args.source_fit,args.admitted_inputs,args.output,kind=args.kind,
            reference=args.reference,scope=args.scope,device=args.device,gpu_uuid=args.gpu_uuid)
    else:value=read_reference_pass(args.output)
    print(json.dumps({k:value[k]for k in('size','scope','kind','samples','neural_forwards','optimizer_updates','scientific_eligible','full_paper_complete')}))

if __name__=='__main__':main()
