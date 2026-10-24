#!/usr/bin/env bash
# Create a dedicated runtime; never activate or modify another project's env.
set -euo pipefail
repro_prefix="${1:?Usage: setup_environment.sh ENV_PREFIX}"
repro_conda="${EGOEMG_CONDA_BIN:-$(command -v conda || true)}"
if [[ -z "$repro_conda" || ! -x "$repro_conda" ]]; then
  printf 'Set EGOEMG_CONDA_BIN to an existing conda executable.\n' >&2
  exit 2
fi
python3 - "$repro_prefix" <<'PY'
import pathlib, shutil, sys
path = pathlib.Path(sys.argv[1]).expanduser().resolve()
parent = path.parent
while not parent.exists():
    parent = parent.parent
free = shutil.disk_usage(parent).free
if free < 25 * 1024**3:
    raise SystemExit('Need at least 25 GiB free for setup, including the 15 GiB reserve.')
PY
if [[ ! -x "$repro_prefix/bin/python" ]]; then
  "$repro_conda" create --prefix "$repro_prefix" --override-channels -c conda-forge python=3.11.12 pip=24.0 -y
fi
mkdir -p "$repro_prefix/../wheels"
if compgen -G "$repro_prefix/../wheels/*.whl" >/dev/null; then
  "$repro_prefix/bin/python" -m pip install --no-index --no-deps "$repro_prefix/../wheels/"*.whl
fi
"$repro_prefix/bin/python" -m pip install --no-cache-dir \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu118 \
  --find-links "$repro_prefix/../wheels" \
  torch==2.2.2+cu118 torchvision==0.17.2+cu118 numpy==1.26.4 Pillow==11.0.0
"$repro_prefix/bin/python" -m pip install --no-cache-dir \
  numpy==1.26.4 scipy==1.13.1 h5py==3.11.0 pandas==2.2.2 \
  pytorch-lightning==2.5.1.post0 hydra-core==1.3.2 \
  opencv-python-headless==4.10.0.84 tensorboard==2.18.0 \
  gdown==5.2.0 pytest==8.2.2 lmdb==1.7.3 Pillow==11.0.0 \
  timm==1.0.15 joblib==1.4.2 pyyaml==6.0.2
"$repro_prefix/bin/python" -m pip check
"$repro_prefix/bin/python" - <<'PY'
import sys, numpy, torch, torchvision, hydra, pytorch_lightning
print('Python:', sys.version)
print('NumPy:', numpy.__version__)
print('Torch:', torch.__version__, 'CUDA build:', torch.version.cuda)
print('Torchvision:', torchvision.__version__)
assert torch.cuda.is_available(), 'CUDA must be available on the designated GPU server'
with torch.autocast('cuda', dtype=torch.bfloat16):
    x = torch.randn(64, 64, device='cuda')
    y = x @ x
assert torch.isfinite(y).all()
print('GPU:', torch.cuda.get_device_name(0), 'bf16 matrix multiplication: passed')
PY
