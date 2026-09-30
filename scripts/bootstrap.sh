#!/usr/bin/env bash
set -euo pipefail

REPO="https://github.com/Snowflake-Labs/dare-bench.git"
COMMIT="01447145304c67b861a004ada6d86f29640de61a"
VENDOR="vendor/DARE-Bench"

mkdir -p vendor
if [ ! -d "$VENDOR/.git" ]; then
  git clone "$REPO" "$VENDOR"
fi

git -C "$VENDOR" fetch --all --tags
git -C "$VENDOR" checkout "$COMMIT"

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-analysis.txt -r requirements-agent.txt

echo "Bootstrap complete."
echo "Pinned DARE-Bench commit: $COMMIT"
