# local-coder — architecture

A private coding-assistant backend: three model servers and a gateway on two
RTX 6000 Ada cards, exposed as one OpenAI-compatible endpoint on `127.0.0.1:4000`.

The whole project is ~700 lines of bash. It writes systemd units and a gateway
config; it does not implement inference, routing, or an editor UI. Everything
below explains where that boundary is and why it sits there.

---

## 1. Why this exists when Ollama, Opilot, Claude Code, OpenCode and OpenClaw exist

The short answer: those tools are not alternatives to each other, and only one
of them overlaps with this project at all.

### The stack has three layers

```
  layer 3   agent / editor UI      Continue, Claude Code, OpenCode,
            "what the human uses"  OpenClaw, Opilot, Cursor, Aider
                    │  speaks OpenAI or Anthropic HTTP
  layer 2   routing / gateway      LiteLLM   ← local-coder owns this
            "one URL, one key"
                    │  speaks OpenAI HTTP
  layer 1   inference runtime      vLLM      ← local-coder owns this
            "weights → tokens"     (alt: Ollama, llama.cpp, TGI, SGLang)
                    │
  layer 0   hardware               2× RTX 6000 Ada, 48 GB each
```

- **Claude Code, OpenCode, OpenClaw** are layer 3. They are agent harnesses:
  loop, tools, permissions, context management. Every one of them needs a model
  endpoint to point at. They do not serve models.
- **Opilot** is layer 3 too — a VS Code extension that surfaces models inside
  Copilot Chat. Its own docs are explicit that it serves nothing: it requires a
  working Ollama install and talks to it.
- **Ollama** is layer 1, and it is the *only* real overlap. Choosing vLLM over
  Ollama is the one substantive decision this project makes.

So the question "why build this when Claude Code exists" has the same shape as
"why run a database when you already have an ORM." This project is the thing
those tools point at. It does not compete with them — it makes them work on
hardware you own, with weights you downloaded, with no network egress.

That is not hypothetical. The gateway already answers Anthropic's wire format
as well as OpenAI's, verified against the running stack:

```bash
curl http://127.0.0.1:4000/v1/messages \
  -H "x-api-key: $LC_API_KEY" -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"local-chat","max_tokens":32,"messages":[{"role":"user","content":"say hi"}]}'
# → 200, {"type":"message","role":"assistant","content":[{"type":"text",...}]}
```

LiteLLM translates that into an OpenAI call against local vLLM. Any layer-3
tool that honours `ANTHROPIC_BASE_URL` or `OPENAI_BASE_URL` can therefore run
against this box without code changes.

### The one real decision: vLLM instead of Ollama

Ollama is excellent and much easier to run. It loses here for four specific
reasons tied to this hardware and this workload:

| | Ollama | this project (vLLM) |
|---|---|---|
| Concurrent models | one active model per request cycle; serving a second swaps weights | three servers, three processes, resident simultaneously |
| GPU placement | scheduler's choice | explicit pinning: agent on GPU 0, autocomplete + embeddings on GPU 1 |
| Quantisation on Ada | GGUF | FP8 weights — 30B MoE at ~160 tok/s (see §9 for the caveat) |
| Batching / KV | per-request | PagedAttention + continuous batching; 151,216-token KV pool, 673 tok/s at N=8 |

The decisive one is the first two together, and it is measured, not assumed:
during a 600-token generation GPU 0 ran at 100% / 231 W / 2700 MHz while GPU 1
never left P8 (210 MHz, 21 W). The isolation is real. A coding assistant runs two jobs
with opposite latency profiles: an agent that saturates a card for minutes, and
autocomplete that must answer between keystrokes. On a single shared runtime
every completion queues behind whatever the agent is doing. Pinning them to
separate cards is the entire point of the layout, and it is not something the
layer-3 tools or a single-model runtime can do for you.

Ollama also actively conflicts with this stack — it looks idle while holding
tens of GB of VRAM — which is why `scripts/free-gpus.sh` exists and why
`lc doctor` checks for it.

### When you should not use this

Be honest about the trade. Use Ollama and stop reading if: you have one GPU or
none; you want models that only ship as GGUF; you want zero setup; or you only
ever have one request in flight. The complexity here buys concurrency and
throughput on two big cards. With one card and one user, it buys nothing.

---

## 2. Runtime topology

Four processes, all systemd **user** units, all bound to `127.0.0.1`.

```
   editor / agent  (Continue, Claude Code, OpenCode, …)
          │  OpenAI /v1/*  or  Anthropic /v1/messages
          ▼
   ┌──────────────────────────────────────────────┐
   │  lc-gateway   :4000    LiteLLM               │   no GPU
   │  one base URL, one key, protocol translation │
   └───┬───────────────┬──────────────┬───────────┘
       │ local-chat    │ qwen2.5-…    │ local-embed
       ▼               ▼              ▼
   lc-chat :8001   lc-fim :8002   lc-embed :8003
   vLLM            vLLM           vLLM (pooling)
   GPU 0           GPU 1          GPU 1
   30B MoE FP8     1.5B FIM       bge-m3
   util 0.92       util 0.22      util 0.10
```

GPU 1 also drives the desktop (~1.5 GB), which is why the big model lives alone
on the clean card and why the GPU-1 memory fractions sum well under 1.0.

Isolation is by `CUDA_VISIBLE_DEVICES` on each unit, not by vLLM flags — the
process simply cannot see the other card.

---

## 3. Components

### `bin/lc` — the only entry point (405 lines)

A dispatcher over one idea: **config is generated, never hand-edited.**
`config/local-coder.env` + `config/profiles.sh` are the sources of truth; every
runtime artefact is derived from them and is disposable.

| Generated | From | Lifetime |
|---|---|---|
| `~/.config/systemd/user/lc-*.service` | `gen_units()` | rewritten each `lc up` / `lc gen` |
| `config/generated/litellm.yaml` | `gen_litellm()` | same |
| `config/generated/tool-parsers.txt` | `resolve_parser()` | cached per vLLM version |
| `~/.continue/config.yaml` | `scripts/write-vscode-config.sh` | rewritten on `lc vscode`, previous copy backed up |

Consequences worth knowing:

- Editing a generated unit is pointless — the next `lc up` overwrites it.
- `lc gen` regenerates but deliberately does **not** restart. Use `lc restart`.
- Nothing generated is in git (`config/generated/` is ignored), so a clone plus
  `install.sh` reproduces the machine state exactly.

### `config/profiles.sh` — the model plan

`load_profile <name>` exports ~15 variables describing all three roles: model
id, context length, memory fraction, GPU list, tensor-parallel width, tool-call
parser. Three profiles ship:

| Profile | Chat model | Ctx | Layout |
|---|---|---|---|
| `fast` (default) | Qwen3-Coder 30B-A3B FP8 | 131072 | GPU 0; FIM + embed on GPU 1 |
| `quality` | Qwen2.5-Coder 32B AWQ | 32768 | GPU 0; FIM + embed on GPU 1 |
| `smart` | Qwen3-Coder-Next 80B-A3B FP8 | 262144 | TP=2 across both cards; FIM + embed off |
| `max-ctx` | Qwen3-Coder 30B-A3B BF16 | 262144 | TP=2 across both cards |

A profile can disable roles by setting `FIM_MODEL=""` / `EMBED_MODEL=""` — which
`max-ctx` does, because full BF16 weights across both cards leave no room. That
empty string propagates everywhere through `active_services()`: the unit file is
removed, the model is dropped from `litellm.yaml`, `lc test` skips its checks,
and `write-vscode-config.sh` omits the block. **One variable, four consistent
downstream effects** — this is the main structural invariant in the codebase.
Anything new keyed off a role must respect it.

### The gateway (LiteLLM)

The gateway exists so that the client contract is stable while the backend
moves. It provides, in order of importance:

1. **One URL and one key** for three servers on three ports. Switching profiles
   changes which vLLM processes exist; the editor config never changes.
2. **Protocol translation** — OpenAI *and* Anthropic message formats, verified
   above.
3. `drop_params: true`, which silently discards parameters vLLM does not accept.
   Clients send a wide variety of sampling fields; without this, one unknown
   field is a 400 instead of a completion.
4. `request_timeout: 1200`, sized for long agent turns rather than chat.
5. Request logging across all models in one place.

It holds no GPU and starts instantly — but it will answer with errors until the
vLLM processes behind it are loaded, which is why `lc up` waits on each backend.

### Weight management

`lc pull` runs `hf download` with `HF_HOME` pointed at `models/` (git-ignored,
25–70 GB per profile) and `HF_XET_HIGH_PERFORMANCE=1`. Weights are never fetched
at serve time; `lc up` on a cold cache fails at model load instead of silently
downloading 35 GB.

---

## 4. Lifecycle

```
scripts/install.sh     uv venv (py3.12) + vllm, litellm[proxy], huggingface_hub
        │
   lc doctor           GPUs, toolchain, systemd, linger, disk, ollama conflict
        │
   lc pull             weights for the active profile → models/
        │
   lc up               gen_units → systemctl enable --now ×N → wait_ready ×N
        │
   lc test             models · chat · tool-calling · FIM · embeddings
        │
   lc vscode           write ~/.continue/config.yaml
```

`lc up` is idempotent and is the normal way to apply any config change.

**Readiness** is polled, not assumed: `wait_ready()` probes `/health`, falling
back to an authenticated `/v1/models`, every 3s up to `LC_STARTUP_TIMEOUT`
(600s). Models load in parallel; the first load takes ~40s, after which weights
stay resident and there is no cold start.

**`lc test` is the real contract.** It checks tool-calling explicitly, because
that is the capability that separates a chat endpoint from an agent backend and
the one most likely to break on a model or vLLM upgrade.

---

## 5. Failure and recovery model

Design stance: **assume startup is fragile, assume steady state is not.**

- `Restart=on-failure`, `RestartSec=10` on every unit. A crash loop surfaces as
  repeating tracebacks in `logs/<svc>.log` rather than a silent dead service.
- `KillSignal=SIGINT` so vLLM shuts down cleanly and releases VRAM.
- Logs append to `logs/<svc>.log` (git-ignored); `lc logs <svc>` tails them.
- **`PATH` is set explicitly in every unit.** systemd hands services a minimal
  `PATH` that excludes the venv. FlashInfer JIT-compiles kernels at startup and
  shells out to `ninja` and `nvcc`, so the venv bin dir and the bundled CUDA
  toolchain (located at generation time by finding `nvcc` under `.venv/lib`) are
  both injected.
- **`VLLM_USE_FLASHINFER_SAMPLER=0`.** The CUDA headers FlashInfer ships
  disagree with the CUDA 13 compiler in the venv. vLLM's native sampler is
  equivalent here and needs no compiler, keeping startup hermetic.
- **Tool-parser name drift is handled.** `resolve_parser()` reads the parser
  list out of `vllm serve --help=all`, caches it against the vLLM version (the
  call takes ~60s), and falls back to `hermes` if the profile's preferred parser
  is gone. A vLLM upgrade that renames a parser degrades instead of crashing.
- **Session lifetime is a real trap.** User units die with your last login
  session. `lc enable-linger` (one-time, needs sudo) makes them survive logout
  and start at boot; `lc doctor` warns when it is off.

---

## 6. Trust boundary

- Every service binds `127.0.0.1` by default (`LC_BIND_HOST`). Nothing listens
  off-box unless you change it to `0.0.0.0`.
- The gateway is the only authenticated hop: `master_key` in `litellm.yaml`,
  generated on first `lc up` into `config/local-coder.env` if still `CHANGEME`.
  The vLLM backends take `api_key: none` — they are reachable only from
  localhost and sit behind the gateway.
- `config/local-coder.env` contains the key and **is tracked in git**. It is the
  one file to review before pushing this repo anywhere.
- No telemetry: `continue.telemetryEnabled: false` in the workspace settings, no
  outbound calls anywhere in the stack after weights are downloaded.

---

## 7. Extending it

**Add a profile** — copy a block in `config/profiles.sh`, set the model ids and
memory fractions, then `lc profile <name> && lc pull && lc up`. If it disables a
role, set the model variable to `""` and the rest follows automatically.

**Use a different client** — point it at the gateway:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:4000/v1
export OPENAI_API_KEY=<key from config/local-coder.env>
# or, for Anthropic-protocol clients:
export ANTHROPIC_BASE_URL=http://127.0.0.1:4000
```

Model ids: `local-chat`, `qwen2.5-coder-1.5b`, `local-embed`.

**Add a service** — write a `_unit` call in `gen_units()`, add it to `SERVICES`,
`port_for()` and `active_services()`, and register it in `gen_litellm()` if it
should be routable. Four places, all in `bin/lc`.

### Deliberately out of scope

No agent loop, no editor UI, no model training, no multi-user auth, no
autoscaling, no container runtime. Layer 3 is other people's software and layer
1 is vLLM's; this project is the ~700 lines that make them meet on this
hardware. Anything that grows an agent loop here belongs in a separate repo
pointed at port 4000.

---

## 8. Known gaps

| Gap | Detail |
|---|---|
| `LC_IDLE_TIMEOUT` is dead config | Documented in `config/local-coder.env`, read by nothing. Models are always resident. Either wire it up or delete it. |
| Only Continue is wired | `lc vscode` writes Continue's config. Claude Code / OpenCode / OpenClaw work against the gateway — Anthropic streaming verified end to end in §10 — but have no generated config or smoke test. |
| API key is committed | `config/local-coder.env` is tracked. Fine for a single-machine repo, wrong the moment it is pushed. |
| `lc test` is single-stream | Concurrency has now been measured by hand (§9) — 673 tok/s aggregate at N=8 — but nothing in `lc test` exercises it, so a regression would go unnoticed. |
| Single-machine assumptions | Two-GPU layout, desktop on GPU 1, and Ada-native FP8 are baked into the profiles. Another box needs new profiles, not new code. |

---

## 9. Layer 0 / layer 1: verified, then tuned

Everything here was measured against the running stack. Re-run after any
profile or vLLM change.

### Verified sound

| Claim | Evidence |
|---|---|
| GPU isolation works | 600-token generation: GPU 0 at 100% / 231 W / 2700 MHz, GPU 1 pinned at P8 / 210 MHz / 21 W throughout |
| PCIe is not a bottleneck | Links report Gen 1 at idle (normal ASPM downtraining); GPU 0 trains to **Gen 4 x16** under load |
| ~~TP=2 is viable~~ **WRONG — see §11** | `nvidia-smi topo -p2p r` reports `OK`, but that is the *capability*, not whether DMA survives the IOMMU. Tensor parallelism hung until worked around |
| Agent-friendly engine defaults | `enable_prefix_caching=True`, `enable_chunked_prefill=True`, `cudagraph_mode=FULL_AND_PIECEWISE`, `FLASH_ATTN` backend |
| FlashInfer guard works | The ninja/CUDA-13 header crash appears only in pre-guard log entries; zero errors since |
| Power headroom | Peak 231 W against a 300 W limit; `SW Power Cap: Not Active`, `HW Slowdown: Not Active` |
| **Continuous batching works** | Aggregate throughput 160 → 437 → 673 tok/s at N = 1, 4, 8 concurrent (4.2x at N=8) |

### FP8 is *not* native here

The original docs claimed FP8 is native on Ada. vLLM disagrees at startup:

```
WARNING [marlin_utils_fp8.py:112] Your GPU does not have native support for FP8
computation but FP8 quantization is being used. Weight-only FP8 compression
will be used leveraging the Marlin kernel.
```

What actually runs: dense linears use `MarlinFP8ScaledMMLinearKernel` — weights
*stored* FP8, dequantised to BF16 to multiply, so a memory win with no compute
win. Only the MoE experts run true `fp8_w8a8` via Triton. **FP8 is what makes
30B fit on one card; it is not what makes it fast.** Corrected in `profiles.sh`.

### Applied

| Change | Before → after | Effect |
|---|---|---|
| `CHAT_UTIL` 0.92 → 0.95 | KV 12.42 → **13.85 GiB**, 135,696 → **151,216 tokens** | Full-context concurrency 1.04x → **1.15x**. vLLM's CUDA-graph profiling (default since 0.21) silently made 0.92 behave as 0.9066; this reclaims it |
| `OMP_NUM_THREADS` 8 → 1 | vLLM's spin-wait-contention warning gone | No measurable throughput change (162 tok/s either way). Removes a documented risk, not a bottleneck |
| `--generation-config vllm` | Model's `generation_config.json` no longer applied | Stops `repetition_penalty 1.05` / `temperature 0.7` silently reaching clients that send no sampling params — notably anything on `/v1/messages`, where it degrades tool-call output |
| `FIM_UTIL` 0.22 → 0.15 | FIM KV 168,272 → **80,736 tokens** | Was 10.3x concurrency for a server capped at `--max-num-seqs 8`. Still 4.9x |
| New `smart` profile | — | See §3; 80B total at the same 3B active |

Verified after restart: `lc test` passes all five stages (chat, tool calling,
FIM, embeddings), throughput unchanged at ~162 tok/s.

### Remaining, by value

| # | Finding | Evidence | Fix |
|---|---|---|---|
| 1 | **GPU 1 is ~75% idle** under `fast` | 12.6 of 49.1 GB used | Inherent to the profile — `smart` uses both cards instead |
| 2 | **FP8 KV cache unused** | `kv_cache_dtype=auto` (BF16); this build supports `fp8_e4m3` | `--kv-cache-dtype fp8_e4m3` roughly doubles the KV pool. Test tool-calling quality before adopting |
| 3 | **No tuned MoE kernel config** | `Using default MoE config. Performance might be sub-optimal! Config file not found at …device_name=NVIDIA_RTX_6000_Ada_Generation…json` | Run vLLM's `benchmark_moe.py --tune` once and commit the JSON. This is the MoE path — the bulk of the compute |
| 4 | **Speculative decoding unused** | — | n-gram or EAGLE drafting; coding workloads are unusually redundant, which is the case where it pays |
| 5 | **API key is committed** | `config/local-coder.env` is tracked in git | Fine for one machine, wrong the moment it is pushed |

### Hardware policy, noted deliberately

- **ECC is Disabled** on both cards. That buys ~6% of VRAM and a little
  throughput, and gives up error detection across ~46 GB of resident weights.
  At 0.95 utilisation it could not be enabled now without re-tuning — a real
  trade, not an oversight.
- **Persistence mode is Disabled.** Moot while vLLM holds a CUDA context
  permanently; it only affects the gap after `lc down`.

---

## 10. Layer 2: the gateway

Audited the same way. The gateway is the client contract, so its failure modes
are the ones editors and agents actually see.

### Verified sound

| Property | Measurement |
|---|---|
| **Overhead is nil** | 300-token generation direct to vLLM `:8001` = 162.9 tok/s; through the gateway `:4000` = 162.4 tok/s. ~0.5%, within noise |
| OpenAI streaming | SSE streams correctly (32 chunks on a 40-token request) |
| **Anthropic streaming** | `/v1/messages` emits a well-formed event sequence: `message_start` → `content_block_start` → 27 × `content_block_delta` → `content_block_stop` → `message_delta` → `message_stop` |
| Tool calling survives the hop | `lc test` stage 3 returns `tool_calls` through the gateway |
| Unknown model | Rejected with HTTP 400 |
| Backends are unauthenticated but unreachable | `:8001` accepts an unauthenticated request; it binds `127.0.0.1` only, and the gateway is the authenticated hop |

The Anthropic result is the significant one: **Claude Code, OpenCode and any
other Anthropic-protocol client can drive this box today**, streaming included,
with no translation layer of your own.

### Finding: auth failures return misleading status codes

| Request | Returns | Should be |
|---|---|---|
| No API key | **HTTP 500** `{"error":{"message":"Internal server error"}}` | 401 |
| Wrong API key | **HTTP 400** | 401 |
| Correct key | 200 | 200 |

Auth is enforced — nothing leaks — but the codes lie about *why*. This matters
because agent harnesses branch on status: a 500 reads as "server is broken,
retry with backoff", so a client with a bad key will hammer the gateway through
its whole retry budget and then report a server fault instead of "check your
key." A misconfigured editor will look like an outage.

Worth a LiteLLM-side fix or a note in the troubleshooting section; it is a
usability bug, not a security one.

### Not yet exercised

- No rate limiting or per-client keys — one shared `master_key` for everything.
- No fallbacks or retry policy configured between models.
- Request logging is on by default but has no configured sink, so it lives only
  in `logs/gateway.log`.

---

## 11. Layer 1 addendum: tensor parallelism is broken by default here

Discovered while benchmarking the `smart` profile — see
[BENCHMARK-REPORT.md](BENCHMARK-REPORT.md) for the full write-up. Both TP=2
profiles (`smart`, and `max-ctx`, which had never been run since the repo was
created) fail to start on this machine without three workarounds.

**AMD-Vi (IOMMU) is active with no passthrough**, which blocks the GPU-to-GPU
DMA that peer-to-peer transports rely on. The failure mode is silence, not an
error: both workers spin at 100% CPU forever and `lc up` times out.

| Fault | Symptom | Fix, applied in `gen_units()` when `CHAT_TP > 1` |
|---|---|---|
| NCCL P2P | Log freezes after `vLLM is using nccl`; 100% CPU on both workers | `NCCL_P2P_DISABLE=1` |
| vLLM custom all-reduce (bypasses NCCL over CUDA IPC) | Weights load, then `No available shared memory broadcast block found in 60 seconds` on a loop | `--disable-custom-all-reduce` |
| Asymmetric free VRAM | `Free memory on device cuda:1 (44.88/47.34 GiB) … less than desired (0.95, 44.97 GiB)` | Size `CHAT_UTIL` for the card that also drives the desktop |

**Diagnostic worth keeping.** A two-process all-reduce isolates this in seconds
instead of a ten-minute model load:

```bash
torchrun --nproc_per_node=2 nccl_test.py          # hangs
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 nccl_test.py   # ALLREDUCE_OK
```

**The proper fix is `iommu=pt` on the kernel cmdline**, which restores real P2P
and needs a reboot. Both workarounds stage through host memory instead, so every
`smart` measurement in the benchmark report is a floor, not a ceiling.

**Lesson for this repo:** a shipped profile that has never been run is not a
feature. `max-ctx` was documented, listed in `lc profile`, and broken from the
start. Nothing in `lc test` or `lc doctor` exercises a TP path.

---

## Reference: measured on this machine

Profile `fast`, single stream, nothing else loaded.

| Metric | Value |
|---|---|
| Chat generation, 1 stream | 162 tok/s |
| Chat generation, 8 streams | 673 tok/s aggregate |
| Gateway overhead | ~0.5% vs direct vLLM |
| Autocomplete latency | 159 ms median |
| Embedding, 32 chunks | 78 ms |
| KV cache | 151,216 tokens (13.85 GiB) |
| VRAM GPU 0 / GPU 1 | 44.8 / 12.4 GB of 49.1 each |
