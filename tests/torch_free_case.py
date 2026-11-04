"""Run a Torch-free boundary in its own interpreter if collection loaded Torch."""
import os
import subprocess
import sys


def run_torch_free_case(request):
    if 'torch' not in sys.modules:
        return False
    flag = 'EGOEMG_TORCH_FREE_TEST_CHILD'
    node = request.node.nodeid
    if os.environ.get(flag) == node:
        raise AssertionError('the dedicated Torch-free test interpreter imported Torch')
    env = dict(os.environ, **{flag: node, 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
        'CUDA_VISIBLE_DEVICES': ''})
    completed = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', node],
        cwd=request.config.rootpath, env=env, capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert '1 passed' in completed.stdout, 'the Torch-free boundary must execute, not skip'
    return True
