#!/usr/bin/env bash
set -euo pipefail
repro_destination="${1:-upstream}"
repro_revision=f7cbd048b7092139852270491d3859cd472e0311
if [[ ! -d "$repro_destination/.git" ]]; then
  git clone --no-checkout https://github.com/zhenqis123/EgoEMG.git "$repro_destination"
  git -C "$repro_destination" checkout --detach "$repro_revision"
fi
if [[ -n "$(git -C "$repro_destination" status --porcelain)" ]]; then
  printf 'Refusing to change a dirty upstream checkout.\n' >&2
  exit 1
fi
git -C "$repro_destination" checkout --detach "$repro_revision"
test "$(git -C "$repro_destination" rev-parse HEAD)" = "$repro_revision"
