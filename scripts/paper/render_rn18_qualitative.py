#!/usr/bin/env python3
"""Render the accepted ordinary RN18 cache; extras are explicit diagnostics."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in('cache-root','upstream','data','crops','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--limit',type=int,default=120)
    p.add_argument('--extra-ranks',type=int,nargs='*',default=[])
    a=p.parse_args()
    from egoemg_repro.rn18_qualitative_render import open_rn18_cache,export_rn18_candidates
    cache=open_rn18_cache(a.cache_root)
    v=export_rn18_candidates(cache,upstream=a.upstream,data=a.data,crops=a.crops,
        output=a.output,limit=a.limit,extra_ranks=tuple(a.extra_ranks))
    print(json.dumps({n:v[n]for n in('rendered_candidates','default_rank_candidates','diagnostic_extra_candidates','figure7_reproduced')}))

if __name__=='__main__':main()
