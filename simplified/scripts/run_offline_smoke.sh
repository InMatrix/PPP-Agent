#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

simplified/.venv/bin/ppp-simple doctor
simplified/.venv/bin/ppp-simple inspect
simplified/.venv/bin/ppp-simple prepare
simplified/.venv/bin/ppp-simple run --mode offline
