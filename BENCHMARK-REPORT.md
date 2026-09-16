# Benchmark report: `fast` vs `smart`

**Date:** 2026-09-13 · **Host:** 2× RTX 6000 Ada (49,140 MiB each), AMD platform,
driver 580.105.08 · **Engine:** vLLM 0.29.0, torch 2.13.0+cu130

Both profiles were measured with the same harness, same prompts, same order,
through the gateway on `:4000`. Medians of 3 runs unless stated.

| | `fast` | `smart` |
|---|---|---|
| Model | Qwen3-Coder-30B-A3B-Instruct-FP8 | Qwen3-Coder-Next-FP8 |
| Total / active params | 30B / ~3B | **80B / ~3B** |
| SWE-bench Verified (published) | 51.6 | **70.6** |
| Context | 131,072 | **262,144** |
| GPUs | GPU 0 only | both, TP=2 |
| Autocomplete + embeddings | yes | **no** |

---

## Headline: the 80B is not slower

The expected trade was accuracy for speed, since `smart` doubles the parameter
count and splits across two cards over PCIe with no NVLink. That trade did not
materialise.

| Metric | `fast` | `smart` | Δ |
|---|---|---|---|
| Single-stream generation | 162.6 tok/s | **168.3 tok/s** | **+3.5%** |
| Time to first token | **18.6 ms** | 32.0 ms | +72% |
| Aggregate @ 4 streams | **391.4 tok/s** | 380.9 tok/s | −2.7% |
| Aggregate @ 8 streams | **680.8 tok/s** | 642.9 tok/s | −5.6% |
| Prefill @ 34K context | 7,960 tok/s | **17,000 tok/s** | **+114%** |
| KV cache | 151,216 tok / 13.85 GiB | **308,878 tok / 3.61 GiB** | **7.8x cheaper/token** |
| Cold start | 47 s | 80 s | — |
| Tool calling | 2/2 correct | 2/2 correct | — |

Both profiles emitted correct parallel tool calls (`read_file`, `list_dir`) with
well-formed arguments, which is the capability that matters most for agent use.

### Why it is faster despite tensor parallelism

Both models activate ~3B parameters per token, so per-token compute is
comparable — the extra 50B sits in experts that are not selected. TP=2 splits
that work across two cards, and the PCIe all-reduce cost is small relative to
the saving, so decode comes out marginally ahead.

The prefill result is the more important one. `smart` is a hybrid: 36 of its 48
layers are Gated DeltaNet (linear attention) and only 12 are full attention.
`fast` runs quadratic attention on every layer. At 8.5K context the two are
indistinguishable (~14,000 tok/s prefill each); at 34K, `fast` has degraded to
7,960 tok/s while `smart` holds 17,000. **The gap widens with context**, which is
precisely where coding agents operate.

### The KV cache result changes the concurrency story

`smart` stores 2x the context in **a quarter of the memory** — 12.3 KB/token
against 96 KB/token. This was the single biggest constraint on `fast`: its
13.85 GiB cache allows only 1.15x concurrency at full context. `smart` reaches
1.18x at *double* the context, and would allow far more at typical agent context
lengths.

### Where `fast` still wins

- **Latency.** 18.6 ms TTFT against 32.0 ms. TP=2 adds a synchronisation per
  step, and that shows up at the head of the stream. For interactive chat it is
  imperceptible; for anything that wants sub-20ms it is not.
- **Throughput under load.** ~6% better aggregate at 8 concurrent streams.
- **It leaves a GPU free**, which is what makes autocomplete and embeddings
  possible at all.

---

## The blocker: tensor parallelism did not work on this machine

`smart` did not start on the first three attempts. Neither did `max-ctx`, which
has shipped in this repo since the beginning and had **never once been run**.
Three distinct faults, in order:

### 1. NCCL hangs on the first all-reduce — AMD IOMMU

**Symptom:** no error. Both worker processes spin at 100% CPU indefinitely, the
log freezes immediately after `vLLM is using nccl==2.29.7`, and `lc up` gives up
after its 600s timeout.

**Cause:** AMD-Vi is active (`/sys/class/iommu/ivhd*`) with no passthrough on the
kernel cmdline, which blocks the GPU-to-GPU DMA that NCCL's peer-to-peer
transport depends on.

**Confirmed** with a 2-process torch all-reduce, which isolates it from vLLM in
seconds rather than a ten-minute model load:

```
default            → hangs (killed at 75s)
NCCL_P2P_DISABLE=1 → ALLREDUCE_OK sum=2.0
```

**A trap worth recording:** `nvidia-smi topo -p2p r` reports `OK` for both
directions on this machine. It reports the *capability*, not whether DMA
actually survives the IOMMU. An earlier audit trusted that output and concluded
TP=2 was viable. It was not.

**Fix applied:** `NCCL_P2P_DISABLE=1` in the unit environment when `CHAT_TP > 1`.

### 2. vLLM's custom all-reduce hangs on the same wall

**Symptom:** gets further — weights load, VRAM fills to ~40 GB/card — then hangs
during warmup, logging `No available shared memory broadcast block found in 60
seconds` on a loop while both workers spin at 100% CPU.

**Cause:** vLLM has its own custom all-reduce over CUDA IPC that bypasses NCCL
entirely, so `NCCL_P2P_DISABLE` does not reach it.

**Fix applied:** `--disable-custom-all-reduce` when `CHAT_TP > 1`.

### 3. Utilisation must be sized for the *worst* card

**Symptom:** clean, fast failure —

```
ValueError: Free memory on device cuda:1 (44.88/47.34 GiB) on startup is less
than desired GPU memory utilization (0.95, 44.97 GiB)
```

Short by 0.09 GiB. vLLM applies one utilisation fraction to every card, but
GPU 1 also drives the desktop (~1.8 GiB), so it has that much less free.

**Fix applied:** `CHAT_UTIL=0.92` for `smart`. The cost is that the same ~1.8 GiB
goes unused on GPU 0.

### The better fix for 1 and 2

Both workarounds stage transfers through host memory, which costs bandwidth. The
real fix is `iommu=pt` (passthrough) on the kernel cmdline, which restores true
P2P and needs a reboot. **The numbers in this report were measured with P2P
disabled** — so `smart` may well be faster still than what is recorded here.

---

## Verdict

Adopt `smart` for agentic work. It scores ~19 points higher on SWE-bench
Verified, generates marginally faster, prefills long contexts twice as fast, and
stores 2x the context in a quarter of the memory. The cost is real but bounded:
**no autocomplete and no `@codebase` embeddings while it is active**, and ~13 ms
more latency before the first token.

That makes them complementary rather than competing:

- **`fast`** — the daily driver. Inline completion, `@codebase` retrieval, lowest
  latency, one GPU.
- **`smart`** — multi-file refactors, long-context reasoning, anything where the
  19-point accuracy gap decides whether the agent succeeds.

Switch with `lc profile smart && lc up`; expect ~80 s to load.

### Open items

| Item | Note |
|---|---|
| `iommu=pt` reboot | Would restore real P2P and likely improve every `smart` number here |
| Untuned MoE kernel | Still warns `Using default MoE config. Performance might be sub-optimal!` — affects both profiles |
| FP8 is weight-only | Neither profile gets native FP8 compute on Ada; the dense layers run Marlin weight-only, only the MoE experts run true `fp8_w8a8` |
| No autocomplete under `smart` | A small FIM model cannot be co-resident. Running it on CPU is untested |
| Accuracy not independently measured | The 51.6 / 70.6 figures are published benchmarks, not reproduced here |

---

## Method

Harness: one script, run unchanged against both profiles through the gateway.
Single-stream and TTFT are medians of 3; concurrency uses threads with a shared
start; long-context feeds a synthetic repo excerpt and measures wall time, with
prefill derived by subtracting decode at the measured per-token rate.

Raw output is in the session scratchpad (`res_fast.json`, `res_smart.json`). The
harness is not yet committed — see open items.
