#!/usr/bin/env bash
set -euo pipefail

echo "=== Running tests ==="
uv run pytest tests/

echo ""
echo "=== Running tests with coverage ==="
uv run pytest --cov=src tests/

echo ""
echo "=== Running gremlins ==="
uv run pytest --gremlins --gremlin-parallel
