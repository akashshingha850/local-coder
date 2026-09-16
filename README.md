# local-coder

A coding agent that runs entirely on this machine's two RTX 6000 Ada cards and
plugs into VSCode. No code, prompt, or file leaves the box.

## Quick start

```bash
./scripts/free-gpus.sh      # hand the GPUs to vLLM (stops ollama, needs sudo)
./bin/lc doctor             # verify hardware, toolchain, disk
./bin/lc pull               # download weights (~35 GB for the default profile)
./bin/lc up                 # start everything
./bin/lc test               # prove chat, tool calling, autocomplete, embeddings work
./bin/lc vscode             # write the VSCode extension config
code --install-extension Continue.continue
```

Reload VSCode, open the Continue sidebar, pick `local-chat`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and the measured
layer-by-layer audit, and [BENCHMARK-REPORT.md](BENCHMARK-REPORT.md) for a
head-to-head of the `fast` and `smart` profiles.

## How it is put together

Four processes. VSCode only ever talks to the gateway on port 4000, which
speaks the OpenAI API, so any tool that can point at a custom base URL works.

```
VSCode (Continue)
      │
      ▼
  gateway :4000  ── LiteLLM: one URL, one key, request logging
      ├──────────► chat  :8001   GPU 0   the agent brain
      ├──────────► fim   :8002   GPU 1   inline autocomplete
      └──────────► embed :8003   GPU 1   @codebase retrieval
```

### Why the models are split across the two cards

The agent and the autocomplete serve opposite needs. The agent runs for
minutes and saturates a GPU; autocomplete must answer between keystrokes.
Sharing one card would make every completion queue behind whatever the agent
is doing. Pinning them to separate cards keeps completions instant during long
agent runs.

GPU 1 also drives the desktop, which costs about 1.5 GB. The big model
therefore lives on GPU 0, which is clean.

## Profiles

Switch with `lc profile <name>`, then `lc pull && lc up`.

| Profile | Chat model | Context | Notes |
|---|---|---|---|
| `fast` (default) | Qwen3-Coder 30B-A3B FP8 | 128K | Mixture of experts: 30B total, ~3B active per token, so it generates at small-model speed. |
| `quality` | Qwen2.5-Coder 32B AWQ | 32K | Dense 4-bit. Slower per token, often sharper on tight single-file work. |
| `smart` | Qwen3-Coder-Next 80B-A3B FP8 | 256K | 80B total, still ~3B active, so it costs about what `fast` costs per token while scoring far higher on agentic benchmarks (70.6 vs 51.6 SWE-bench Verified). Splits across both cards, so autocomplete and embeddings are disabled. |
| `max-ctx` | Qwen3-Coder 30B-A3B BF16 | 256K | Full weights split across both cards. Uses all VRAM, so autocomplete and embeddings are disabled. |

A note on FP8: it is a **memory** win here, not a compute one. vLLM reports that
these cards have no native FP8 support and falls back to Marlin weight-only for
the dense layers; only the MoE experts run true `fp8_w8a8`. FP8 is what makes a
30B or 80B model fit, not what makes it fast.

Add your own by copying a block in `config/profiles.sh`.

## Commands

| Command | Does |
|---|---|
| `lc doctor` | Check GPUs, toolchain, disk, VRAM conflicts |
| `lc pull` | Download weights for the active profile |
| `lc up` | Start all services and wait until they answer |
| `lc down` | Stop everything, free VRAM |
| `lc status` | Service states and live GPU usage |
| `lc test` | End-to-end smoke test |
| `lc logs chat` | Follow a service log |
| `lc profile` | Show or switch profile |
| `lc enable-linger` | Keep services alive after logout |

## Operational notes

Services are systemd user units that restart on failure. They stop when your
last login session ends unless you run `lc enable-linger` once.

Models stay resident in VRAM so there is no cold start after the first load.
Initial load takes roughly 40 seconds. `lc down` frees the cards when you want
them for training.

Weights live in `models/` and logs in `logs/`, both git-ignored.

## Measured on this machine

Profile `fast`, single stream, nothing else loaded.

| Metric | Value |
|---|---|
| Chat generation, 1 stream | 162 tokens/sec |
| Chat generation, 8 streams | 673 tokens/sec aggregate |
| Autocomplete latency | 159 ms median |
| Embedding, 32 chunks | 78 ms |
| KV cache | 151,216 tokens |
| Gateway overhead | ~0.5% vs talking to vLLM directly |
| VRAM, GPU 0 | 46.3 / 49.1 GB |
| VRAM, GPU 1 | 12.6 / 49.1 GB |

The KV cache holds about 151K tokens. At the full 131K context that is one
request at a time; at shorter contexts vLLM batches them continuously — eight
parallel requests reach 673 tok/s aggregate, 4.2x the single-stream rate. If
you would rather have more parallelism than maximum context, lower `CHAT_LEN`
in `config/profiles.sh`.

## Troubleshooting

**A service will not start.** Read the log: `lc logs chat`. The units restart
on failure, so a crash loop shows up as repeating tracebacks.

**`FileNotFoundError: ninja` or nvcc errors.** FlashInfer tries to compile
kernels at startup. Two things guard against this, both already set in the
generated units: the venv and bundled CUDA toolchain are put on `PATH`, since
systemd otherwise hands services a minimal one; and `VLLM_USE_FLASHINFER_SAMPLER=0`
selects vLLM's native sampler, because the CUDA headers FlashInfer ships
disagree with the CUDA 13 compiler in the venv. Performance is unaffected.

**Out of memory on startup.** Something else holds VRAM. Run `lc status` to
see, then `scripts/free-gpus.sh`. Ollama is the usual culprit: it looks idle
until a request loads 42 GB of weights underneath you.

**Autocomplete produces nonsense.** The fill-in-middle template is pinned in
`~/.continue/config.yaml`. If you swap the autocomplete model for a non-Qwen
one, change the `template` line to match that model's FIM tokens.

**A client reports a server error or retries forever.** Check the API key
first. The gateway returns HTTP 500 for a missing key and 400 for a wrong one,
rather than 401, so an auth problem looks like an outage to any client that
retries on 5xx. The key is in `config/local-coder.env`; `lc vscode` rewrites
Continue's copy of it.

**A tensor-parallel profile (`smart`, `max-ctx`) hangs on startup.** Both
workers sit at 100% CPU and `lc up` eventually times out. This machine's IOMMU
blocks GPU-to-GPU P2P; `lc` already sets `NCCL_P2P_DISABLE=1` and
`--disable-custom-all-reduce` for `CHAT_TP > 1`, which works around it at some
cost in bandwidth. The proper fix is `iommu=pt` on the kernel cmdline, which
needs a reboot. See ARCHITECTURE.md §11.

**Config changes do nothing.** `lc gen` rewrites the units and gateway config
but does not restart anything. Use `lc restart`.

## Using it from something other than Continue

The gateway speaks both major wire formats, so most tools work unmodified.

**OpenAI-compatible clients** (Continue, Aider, Cursor, anything with a custom
base URL):

```bash
export OPENAI_BASE_URL=http://127.0.0.1:4000/v1
export OPENAI_API_KEY=<the key in config/local-coder.env>
```

**Anthropic-protocol clients** (Claude Code, OpenCode, OpenClaw) — the gateway
serves `/v1/messages` and translates to the local model, streaming included:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:4000
export ANTHROPIC_API_KEY=<the key in config/local-coder.env>
```

Model ids are `local-chat`, `qwen2.5-coder-1.5b`, and `local-embed`.

Measured overhead of going through the gateway rather than straight to vLLM is
about 0.5%, so there is no reason to bypass it.
