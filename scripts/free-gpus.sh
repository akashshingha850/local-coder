#!/usr/bin/env bash
# Ollama and vLLM both want the whole GPU. This hands the cards to vLLM.
# Needs sudo because ollama runs as a system service.
set -euo pipefail

echo "==> stopping ollama (weights stay on disk; re-enable any time)"
sudo systemctl stop ollama
sudo systemctl disable ollama

echo "==> current VRAM"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

cat <<'NOTE'

Ollama is now stopped and will not return at boot.
To bring it back:   sudo systemctl enable --now ollama

Note: GPU 1 also drives your desktop (Xorg, GNOME), which holds ~1.5 GB.
That is accounted for in the memory fractions in config/profiles.sh.
NOTE
