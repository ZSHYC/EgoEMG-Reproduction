"""Restored provider files must not authenticate stale code already in memory."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from test_emg_initializer_data import source
from test_emg_initializer_literal import UPSTREAM

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('provider',['reader','binding'])
def test_stale_import_then_source_restore_is_not_a_numeric_admission(tmp_path,provider):
    pytest.importorskip('torch');pytest.importorskip('pytorch_lightning')
    data=tmp_path/'data';data.mkdir();source(data,optional_labels=True)
    checkout=tmp_path/'checkout';shutil.copytree(ROOT/'egoemg_repro',checkout/'egoemg_repro',ignore=shutil.ignore_patterns('__pycache__'))
    code=r'''from pathlib import Path
import ast,importlib,sys
root=Path(sys.argv[1]);data=Path(sys.argv[2]);upstream=Path(sys.argv[3]);which=sys.argv[4]
name='emg_initializer_data'if which=='reader'else'emg_initializer_cohort'
p=root/'egoemg_repro'/(name+'.py');raw=p.read_bytes();text=raw.decode();tree=ast.parse(text)
if which=='reader':
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='EmgInitializerDataset')
 node=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='__getitem__')
 line='        raise RuntimeError("unapproved stale reader executed")\n'
else:
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_reader_binding')
 line='    stale_reader_binding_version = 1\n'
lines=text.splitlines(keepends=True);at=node.body[0].lineno-1
if isinstance(node.body[0],ast.Expr)and isinstance(node.body[0].value,ast.Constant)and isinstance(node.body[0].value.value,str):at=node.body[0].end_lineno
lines.insert(at,line);p.write_text(''.join(lines));importlib.import_module('egoemg_repro.'+name);p.write_bytes(raw)
from egoemg_repro.rn18_initializer_cohort import admit_rn18_initializer_cohort
from egoemg_repro.emg_initializer_literal import compose_emg_initializer,emg_initializer_normalization
try:admit_rn18_initializer_cohort(data,upstream=upstream,config=compose_emg_initializer(upstream),normalization=emg_initializer_normalization(upstream),require_full=False)
except ValueError as exc:
 assert 'import' in str(exc)or 'provider' in str(exc),str(exc)
else:raise AssertionError('restored disk bytes authenticated stale imported provider')
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(checkout)+os.pathsep+os.environ.get('PYTHONPATH',''))
    p=subprocess.run([sys.executable,'-B','-c',code,str(checkout),str(data),str(UPSTREAM),provider],cwd=checkout,env=env,capture_output=True,text=True,timeout=90)
    assert p.returncode==0,p.stderr[-5000:]
