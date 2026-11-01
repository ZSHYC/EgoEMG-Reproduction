"""Guarded offline entry point for native full-window EMGFormer evaluation."""
from pathlib import Path
import argparse
import json
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['workflow', 'pass', 'freeze', 'report', 'verify'])
    parser.add_argument('--output', required=True)
    for name in ('fit', 'data', 'upstream', 'role', 'kind', 'reference', 'profile', 'first', 'second', 'test-replay', 'report'):
        parser.add_argument('--'+name)
    parser.add_argument('--engineering', action='store_true')
    args = parser.parse_args()
    connect = socket.socket.connect
    def offline(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6): raise RuntimeError('trajectory evaluation is offline')
        return connect(self, address)
    socket.socket.connect = offline
    from egoemg_repro import emgformer_trajectory_eval as evaluation
    if args.action == 'workflow':
        result = evaluation.run_trajectory_workflow({n: getattr(args, n) for n in ('fit', 'data', 'upstream', 'engineering')}, args.output)
    else:
        import torch
        torch.set_num_threads(2)
        def no_optimizer(*a, **kw): raise RuntimeError('trajectory evaluation cannot construct optimizer')
        torch.optim.Optimizer.__init__ = no_optimizer
        if args.action != 'pass':
            def no_model(*a, **kw): raise RuntimeError('trajectory report cannot construct learned model')
            torch.nn.Module.__init__ = no_model
        if args.action == 'pass':
            from egoemg_repro.emgformer_literal import compose_emgformer
            from egoemg_repro.emgformer_trajectory_data import admit_trajectory_inputs
            fit, identity, _ = evaluation._fit(args.fit, evaluation._Read(), require_full=not args.engineering)
            inputs = admit_trajectory_inputs(args.data, upstream=args.upstream,
                config=compose_emgformer(args.upstream, identity['policy']['size']), normalization=identity['cohort']['normalization'],
                expected_point_identity=identity['cohort'])
            result = evaluation.run_trajectory_pass(inputs, args.fit, args.output, role=args.role, kind=args.kind,
                reference=args.reference, profile=args.profile, require_full=not args.engineering)
        elif args.action == 'freeze': result = evaluation.freeze_trajectory_validation(args.first, args.second, args.output)
        elif args.action == 'report': result = evaluation.publish_trajectory_report(args.test_replay, args.output)
        else:
            result = evaluation.verify_trajectory_report(args.test_replay, args.report,
                receipt=Path(args.output)/'verification.json')
    print(json.dumps(dict(action=args.action, sha256=result['sha256']), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
