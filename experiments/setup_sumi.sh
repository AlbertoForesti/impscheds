#!/usr/bin/env bash
#####################################################################
# Fetch the Sumi evaluation code (Ye et al., 2026) at the commit used in
# the paper and create its uv environment:
#
#   third_party/sumi        https://github.com/tohoku-nlp/sumi @ SUMI_COMMIT
#   third_party/sumi/.venv  (or $UV_PROJECT_ENVIRONMENT if set)
#
# sumi_schedules/ only imports the upstream `sumi_eval` package and never
# modifies it. The environment needs a CUDA GPU at run time.
#####################################################################
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SUMI_REPO=https://github.com/tohoku-nlp/sumi
SUMI_COMMIT=e6a306146613a36b49b64097b0052c4301dfc3a5

if [ ! -d third_party/sumi/.git ]; then
  git clone "$SUMI_REPO" third_party/sumi
fi
git -C third_party/sumi checkout --detach "$SUMI_COMMIT"

cd third_party/sumi
uv sync
# used by the Sumi model code but not declared in its pyproject
uv pip install --python "${UV_PROJECT_ENVIRONMENT:-.venv}/bin/python" einops
