#!/usr/bin/env bash
# Install / upgrade the acr CLI from uv-algo-compute-orc (researchers).
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

echo "Installing acr-client from UVeye/uv-algo-compute-orc (main)..."
uv tool install --force \
  --from "git+ssh://git@github.com/UVeye/uv-algo-compute-orc.git@main#subdirectory=packages/acr-client" \
  acr-client

echo ""
acr --help | head -5
echo ""
echo "Next: acr init   (see docs/ACR.md)"
echo "Then: ./scripts/acr_submit.sh status"
