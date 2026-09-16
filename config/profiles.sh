#!/usr/bin/env bash
# ── Model profiles ───────────────────────────────────────────────────────────
# Each profile pins three roles onto the two RTX 6000 Ada cards (48 GB each):
#
#   CHAT   the agent brain - tool calling, refactors, multi-file edits
#   FIM    fill-in-middle autocomplete - must be fast, small, base (not chat)
#   EMBED  embeddings used to index your repo for @codebase retrieval
#
# GPU assignment is explicit so chat and autocomplete never fight for VRAM:
# a long agent run on GPU 0 leaves your keystroke completions on GPU 1 instant.
#
# Add your own profile by copying a block and changing the model ids.

load_profile() {
  local p="$1"

  # Defaults shared by all profiles
  CHAT_TP=1; CHAT_GPUS="0"
  FIM_GPUS="1"; EMBED_GPUS="1"
  FIM_MODEL="Qwen/Qwen2.5-Coder-1.5B"
  FIM_LEN=16384; FIM_UTIL=0.15
  EMBED_MODEL="BAAI/bge-m3"
  EMBED_LEN=8192; EMBED_UTIL=0.10
  CHAT_EXTRA=""

  case "$p" in

    # ── fast (default) ───────────────────────────────────────────────────────
    # Mixture-of-experts: 30B total but only ~3B active per token, so it
    # generates like a small model while reasoning like a large one.
    #
    # NOTE: FP8 here is a MEMORY win, not a compute one. vLLM reports that this
    # GPU has no native FP8 support and falls back to Marlin weight-only for the
    # dense layers (weights stored FP8, dequantised to BF16 to multiply); only
    # the MoE experts run true fp8_w8a8 via Triton. FP8 is what makes 30B fit on
    # one card - it is not what makes it fast.
    #
    # CHAT_UTIL: vLLM's CUDA-graph memory profiling (default since 0.21) makes
    # 0.92 behave as 0.9066. 0.95 reclaims that and ~3 GiB more of KV cache.
    fast)
      CHAT_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
      CHAT_LEN=131072
      CHAT_UTIL=0.95
      CHAT_PARSER="qwen3_coder"
      CHAT_DESC="Qwen3-Coder 30B-A3B FP8 (MoE, 128K ctx, GPU 0)"
      ;;

    # ── quality ──────────────────────────────────────────────────────────────
    # Dense 32B. Slower per token than the MoE but often sharper on tight,
    # single-file code generation. AWQ 4-bit keeps it to ~19 GB.
    quality)
      CHAT_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
      CHAT_LEN=32768
      CHAT_UTIL=0.90
      CHAT_PARSER="hermes"
      CHAT_DESC="Qwen2.5-Coder 32B AWQ (dense, 32K ctx, GPU 0)"
      ;;

    # ── smart ────────────────────────────────────────────────────────────────
    # Qwen3-Coder-Next: 80B total, still only ~3B active per token, so it costs
    # roughly what 'fast' costs per token while scoring 70.6 vs 51.6 on
    # SWE-bench Verified. Hybrid architecture - 36 of 48 layers are Gated
    # DeltaNet (linear attention), only 12 need a KV cache - which makes the
    # cache ~4x cheaper per token than 'fast' (24 KB vs ~96 KB measured).
    #
    # Fit: 80.4 GB of FP8 weights, so it needs BOTH cards (40.2 GB each) and
    # leaves no room for FIM or embeddings, exactly like max-ctx. Budget at
    # 256K ctx: 40.2 weights + 3.2 KV + 1.1 activation + 0.4 graphs = 44.9 GB
    # of 49.1 per card, plus ~1.8 GB of desktop on GPU 1.
    #
    # TP=2 runs over PCIe Gen4 x16 (no NVLink on Ada; P2P verified OK), which
    # costs perhaps 20-30% of single-card decode speed. Expect the long-context
    # case to still beat 'fast', which does full attention over every token.
    #
    # CHAT_UTIL is 0.92, not 0.95: vLLM applies one utilisation fraction to every
    # card, but GPU 1 also drives the desktop, so it has ~1.8 GiB less free than
    # GPU 0. The fraction must therefore be sized for the *worst* card - 0.95 asks
    # for 44.97 GiB where GPU 1 has only 44.88 free, and startup aborts. The cost
    # is that the same ~1.8 GiB goes unused on GPU 0.
    smart)
      CHAT_MODEL="Qwen/Qwen3-Coder-Next-FP8"
      CHAT_LEN=262144
      CHAT_UTIL=0.92
      CHAT_PARSER="qwen3_coder"
      CHAT_TP=2; CHAT_GPUS="0,1"
      CHAT_DESC="Qwen3-Coder-Next 80B-A3B FP8 (tensor-parallel x2, 256K ctx)"
      FIM_MODEL=""      # both cards are consumed by the chat model
      EMBED_MODEL=""
      ;;

    # ── max-ctx ──────────────────────────────────────────────────────────────
    # Full BF16 weights split across BOTH cards via tensor parallelism, for
    # whole-repository context. Note: this consumes every GPU, so FIM and
    # embeddings are disabled - autocomplete falls back to the chat model.
    max-ctx)
      CHAT_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
      CHAT_LEN=262144
      CHAT_UTIL=0.92
      CHAT_PARSER="qwen3_coder"
      CHAT_TP=2; CHAT_GPUS="0,1"
      CHAT_DESC="Qwen3-Coder 30B-A3B BF16 (tensor-parallel x2, 256K ctx)"
      FIM_MODEL=""      # no VRAM left
      EMBED_MODEL=""
      ;;

    *)
      echo "unknown profile: $p" >&2
      echo "available: fast quality smart max-ctx" >&2
      return 1
      ;;
  esac

  export CHAT_MODEL CHAT_LEN CHAT_UTIL CHAT_PARSER CHAT_TP CHAT_GPUS CHAT_DESC CHAT_EXTRA
  export FIM_MODEL FIM_LEN FIM_UTIL FIM_GPUS
  export EMBED_MODEL EMBED_LEN EMBED_UTIL EMBED_GPUS
}

list_profiles() { echo "fast quality smart max-ctx"; }
