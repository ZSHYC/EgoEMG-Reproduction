"""Only metadata preflight is available; no inference or evaluation command exists."""
import argparse
import json
import os
from pathlib import Path
import sys

from .preflight import _safe,preflight_manifest,read_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    pre=sub.add_parser('preflight');pre.add_argument('--manifest',required=True);pre.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    owned=None;owned_inode=None
    try:
        output=_safe(args.output)
        if output.exists():raise ValueError('output must be new')
        result=preflight_manifest(read_json(args.manifest))
        output.mkdir()
        owned=output/'preflight.json'
        descriptor=os.open(owned,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        opened=os.fstat(descriptor);owned_inode=(opened.st_dev,opened.st_ino)
        with os.fdopen(descriptor,'w') as stream:
            json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        directory=os.open(output,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    except (ValueError,OSError):
        if owned_inode is not None:
            try:
                current=owned.lstat()
                if (current.st_dev,current.st_ino)==owned_inode:owned.unlink()
            except OSError:pass
        print('legacy preflight refused',file=sys.stderr)
        return 1
    print(json.dumps(dict(status='preflight_only',scope='no_inference',benchmark_eligible=False)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
