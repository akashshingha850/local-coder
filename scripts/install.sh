#!/usr/bin/env bash
# One-shot setup for a fresh machine. Idempotent - safe to re-run.
set -euo pipefail
LC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LC_ROOT"

echo "==> checking for uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "==> creating python 3.12 virtualenv"
[ -d .venv ] || uv venv --python 3.12 .venv

echo "==> installing vLLM, LiteLLM gateway, HuggingFace CLI (~8 GB, several minutes)"
uv pip install --python .venv/bin/python vllm litellm[proxy] huggingface_hub hf_transfer

echo "==> done"
./bin/lc doctor
echo
echo "Next:"
echo "  ./bin/lc pull     download model weights"
echo "  ./bin/lc up       start the stack"
echo "  ./bin/lc vscode   configure the VSCode extension"
