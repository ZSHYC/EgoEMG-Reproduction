"""Guarded offline CPU entry point for native full dense prediction/report."""
from pathlib import Path
import argparse
import json
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['workflow', 'predict', 'accept-qualification', 'report', 'verify-report'])
    for name in ('evidence', 'upstream', 'index-root', 'paired-root', 'geometry-root', 'output'):
        parser.add_argument('--'+name, required=True)
    for name in ('data', 'crops', 'fit', 'first', 'second', 'qualification', 'replay-of', 'report'):
        parser.add_argument('--'+name)
    parser.add_argument('--scope', choices=['qualification', 'complete'])
    args = parser.parse_args()
    # DataLoader tensor sharing uses local AF_UNIX sockets. Deny only network
    # connections, preserving the original4worker loader and owned-stage IPC.
    connect = socket.socket.connect
    def offline(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError('dense evaluation is offline')
        return connect(self, address)
    socket.socket.connect = offline
    if args.action == 'workflow':
        from egoemg_repro.rn18_occlusion_run import run_dense_workflow
        names = ('evidence', 'upstream', 'index_root', 'paired_root', 'geometry_root', 'data', 'crops', 'fit')
        result = run_dense_workflow({n: getattr(args, n) for n in names}, args.output)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return
    import torch
    torch.set_num_threads(2)
    def no_optimizer(*a, **kw): raise RuntimeError('frozen prediction cannot construct optimizer')
    torch.optim.Optimizer.__init__ = no_optimizer
    if args.action != 'predict':
        def no_model(*a, **kw): raise RuntimeError('dense report must not construct learned models')
        torch.nn.Module.__init__ = no_model
    from egoemg_repro.rn18_occlusion_run import (load_dense_evidence, admit_dense_run_inputs,
        run_dense_pass, accept_dense_qualification, build_dense_report, verify_dense_report)
    evidence = load_dense_evidence(evidence=args.evidence, upstream=args.upstream,
        index_root=args.index_root, paired_root=args.paired_root, geometry_root=args.geometry_root)
    if args.action == 'predict':
        inputs = admit_dense_run_inputs(evidence, data=args.data, crops=args.crops)
        result = run_dense_pass(inputs, args.fit, args.output, scope=args.scope,
            qualification=args.qualification, replay_of=args.replay_of)
    elif args.action == 'accept-qualification':
        result = accept_dense_qualification(args.first, args.second, evidence=evidence, output=args.output)
    elif args.action == 'report':
        result = build_dense_report(args.first, args.second, evidence=evidence, output=args.output)
    else:
        result = verify_dense_report(args.first, args.second, args.report, evidence=evidence, output=args.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
