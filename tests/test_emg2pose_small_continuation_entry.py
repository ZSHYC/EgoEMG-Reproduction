"""DataLoader spawn must not execute the resume driver's admissions or training."""
from pathlib import Path
import subprocess
import sys

import pytest

ENTRY=Path(__file__).resolve().parents[1]/'scripts/paper/run_emg2pose_small_continuation.py'


@pytest.mark.parametrize('name',['__mp_main__','imported_resume_driver'])
def test_child_import_performs_no_data_gpu_or_training_admission(tmp_path,name):
    assert ENTRY.is_file(),'spawn-safe native continuation entry is missing'
    # This is the exact runpy mode used by multiprocessing.spawn. No request or
    # data is supplied; a child must define functions and return without IO.
    p=subprocess.run([sys.executable,'-B','-c',
        "import runpy,sys;runpy.run_path(sys.argv[1],run_name=sys.argv[2]);assert 'torch'not in sys.modules;print('child-import-ok')",
        str(ENTRY),name],cwd=tmp_path,capture_output=True,text=True,timeout=15)
    assert p.returncode==0,p.stderr
    assert p.stdout.strip()=='child-import-ok'
    assert list(tmp_path.iterdir())==[]
