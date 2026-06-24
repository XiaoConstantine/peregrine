# Peregrine

Minimal Zig + Metal inference server for `mlx-community/Qwen3.5-9B-4bit` on
Apple Silicon.

## Quick Start

```sh
./scripts/run-qwen35
```

The wrapper builds `ReleaseFast`, finds the checkpoint from
`$PEREGRINE_QWEN35_REPO`, the Hugging Face cache, or `/tmp/qwen3.5-9b-q4`, then
starts the local OpenAI-compatible server.

## Model Download

Peregrine does not download model files itself. Use the Hugging Face CLI once,
then let `scripts/run-qwen35` find the local directory:

```sh
hf download mlx-community/Qwen3.5-9B-4bit --local-dir /tmp/qwen3.5-9b-q4
```

If you store the checkpoint somewhere else, pass it explicitly:

```sh
./scripts/run-qwen35 serve --repo /path/to/Qwen3.5-9B-4bit
```

Or export it for repeated runs:

```sh
export PEREGRINE_QWEN35_REPO=/path/to/Qwen3.5-9B-4bit
./scripts/run-qwen35
```

Equivalent explicit form:

```sh
./scripts/run-qwen35 serve --host 127.0.0.1 --port 8080
```

Default serving profile:

```text
mode:                 serve
profile:              agent-optimized
max total tokens:     65536
default output:       512 tokens
max output:           4096 tokens
prefix cache:         24576 prompt tokens
prefill chunk:        1600 prompt tokens
prefill group:        10 chunks
socket timeout:       30 seconds
```

Override limits when needed:

```sh
./scripts/run-qwen35 serve \
  --max-total-tokens 16384 \
  --default-new-tokens 512 \
  --max-new-tokens 1024 \
  --prefix-cache-tokens 4096
```

Use `--dry-run` to print the exact `zig-out/bin/peregrine` command without
starting the server.

## MTP Speculative Decode

Peregrine runs a Qwen3.5 MTP sidecar for speculative decode: the sidecar
drafts one token and the verifier accepts it plus a bonus token, yielding up
to 2 tokens per verifier pass (1.5–1.9x decode throughput on warm agent
prompts).

MTP is **auto-enabled** when the sidecar is present — `scripts/run-qwen35`
detects it from `$PEREGRINE_QWEN35_MTP_REPO`, the Hugging Face cache, or
`/tmp/qwen3.5-9b-mtp-q4`. Download it once:

```sh
hf download mlx-community/Qwen3.5-9B-MTP-4bit --local-dir /tmp/qwen3.5-9b-mtp-q4
./scripts/run-qwen35 serve
```

To force it on with an explicit path, or off when present:

```sh
./scripts/run-qwen35 serve --mtp-dir /path/to/compatible-qwen3.5-9b-mtp-q4
./scripts/run-qwen35 serve --no-mtp
```

The sidecar must match the target model fingerprint. When MTP is disabled
(or a cached prefix lacks real hidden states), serving falls back to normal
greedy decode.

With MTP, the prefix cache also stores the target's normalized prompt
hidden states alongside the KV cache, and the persisted prefix-state file
includes them (format version 3; older files are rebuilt on first start).
The drafter is seeded from the cached prefix hiddens, so a warm prefix avoids
replaying the prompt through the target and drafter on every request. A
cached prefix without real hidden states (e.g. a file written by a non-MTP
run, or a prefix cached before MTP captured rows) disables MTP for that
request and falls back to normal decode until the prefix is re-prewarmed
under MTP. The hidden buffer adds a bounded ~192 MiB
(`prefix-cache-tokens × 4096 × 2 bytes`) to the prefix cache; the existing
`agent-optimized` defaults are already MTP-optimal.

## API Surface

Metadata:

```text
GET  /health
GET  /v1/me
GET  /v1/models
GET  /v1/models/{id}
GET  /v1/prefix/status
```

Serving:

```text
POST /v1/chat/completions
POST /v1/prefix/warmup
```

Metrics and dashboard:

```text
GET  /dashboard
GET  /v1/metrics
GET  /v1/metrics/history
```

`/v1/chat/completions` accepts omitted `model`, `qwen3.5-9b-4bit`,
`qwen3.5-9b-q4`, and `mlx-community/Qwen3.5-9B-4bit`. Other model ids are
rejected. It supports streaming SSE, non-streaming JSON, OpenAI text content
parts, function tools as Qwen prompt context, stop sequences, usage chunks when
`stream_options.include_usage` is true, and greedy generation controls.
Peregrine also always treats Qwen's `<|im_end|>` marker as an internal stop so
clients do not need to send it.

Supported output-token aliases are `max_new_tokens`, `max_tokens`,
`max_completion_tokens`, and `max_output_tokens`.

`response_format` is still accepted for text-only compatibility:
`json_object` and `json_schema` add a Qwen system instruction, and non-streaming
responses are post-validated after generation.

## Metrics Dashboard

Peregrine ships an embedded, dependency-free metrics dashboard at
`GET /dashboard`. Open `http://127.0.0.1:8080/dashboard` in a browser while the
server is running. The page polls two JSON endpoints every second and renders
summary cards, sparkline charts, and a recent-request table — no external
assets, no npm build, no CDN.

`GET /v1/metrics` returns a cumulative snapshot: request totals, token totals
(prompt/generated/reuse/computed), recent averages (TTFT, decode tok/s, prefix
hit ratio, MTP acceptance, tokens/step), and a live prefix-cache occupancy
snapshot. When the decode lock is busy, live cache fields are `null` and
`busy: true`.

`GET /v1/metrics/history` returns the last 256 completed request records
(oldest-first) with per-request timings, throughput, MTP acceptance, and a
cache-occupancy snapshot. Records store only numeric counts and timings —
never prompts, completions, or decoded text.

Structured metrics are always collected (independent of
`PEREGRINE_TRACE_PREFILL`), so the dashboard has data even when the trace env
var is not set. The trace logs remain a separate profiling surface.

## Pi Client Configuration

Pi reads local providers from `~/.pi/agent/models.json`. Add a `peregrine`
provider like this:

```json
{
  "providers": {
    "peregrine": {
      "name": "Peregrine",
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "authHeader": false,
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "supportsUsageInStreaming": true,
        "thinkingFormat": "qwen",
        "maxTokensField": "max_tokens"
      },
      "models": [
        {
          "id": "qwen3.5-9b-q4",
          "name": "Qwen3.5 9B Q4 (Peregrine)",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 64000,
          "maxTokens": 512,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

Then select it in `~/.pi/agent/settings.json`:

```json
{
  "defaultProvider": "peregrine",
  "defaultModel": "qwen3.5-9b-q4",
  "defaultThinkingLevel": "off"
}
```

Merge those settings into the existing file instead of deleting unrelated Pi
fields such as packages, extensions, theme, or changelog state.

The Pi status line should then show:

```text
(peregrine) qwen3.5-9b-q4
```

If Pi runs outside the same host, start Peregrine on an externally reachable
interface:

```sh
./scripts/run-qwen35 serve --host 0.0.0.0 --port 8080
```

Then configure Pi with:

```text
baseUrl: http://<mac-lan-ip>:8080/v1
```

For the fastest first real Pi turn, save one captured Pi chat payload to
`/tmp/pi-peregrine-request.json` before starting Peregrine. The wrapper will
auto-prewarm that static prefix.

## Pi Prefix Warmup

If `/tmp/pi-peregrine-request.json` exists, `./scripts/run-qwen35 serve` uses it
automatically as a startup warmup request. Set
`PEREGRINE_PI_PREWARM_REQUEST=/path/to/request.json` to choose another captured
Pi/OpenAI payload, or pass `--no-auto-prewarm` to start cold.

Explicit captured-request warmup:

```sh
./scripts/run-qwen35 serve --prewarm-request-file /tmp/pi-peregrine-request.json
```

Peregrine renders and caches the stable prefix before the latest user content.
The first real request then computes only the short user suffix plus decode.

After startup prewarm completes, the warmed prefix state is persisted to
`~/.cache/peregrine/qwen35-9b-q4-prefix-state.bin` (atomic write, one bounded
file). The next server start reloads it in seconds instead of recomputing the
~minute-long raw cold prefill of the 16k Pi prefix. Override the location with
`--prefix-state-file FILE` or disable persistence with `--no-prefix-state`;
files that fail validation (format version, model fingerprint, exact size) are
ignored and the server falls back to a normal cold prefill.

Raw prompt warmup:

```sh
curl -sS http://127.0.0.1:8080/v1/prefix/warmup \
  -H 'content-type: application/json' \
  --data '{"prompt":"<exact rendered Qwen/Pi prefix>"}'
```

Raw prefix warmups also accept an `input` alias, including a string or singleton
string array. Chat-style warmups use `messages`.

Warm a captured Pi request after startup:

```sh
jq '. + {"warmup_mode":"before_last_user_content"}' /tmp/pi-peregrine-request.json \
  | curl -sS http://127.0.0.1:8080/v1/prefix/warmup \
      -H 'content-type: application/json' \
      --data-binary @-
```

Inspect cache state:

```sh
curl -sS http://127.0.0.1:8080/v1/prefix/status
```

The status response reports `cached_tokens`, `cached_logits`, hit/miss counts,
prefill chunk settings, full-cache dtype, prepared-RHS status, and whether the
raw cold first-content route is ready.

## Validation

Fast local validation:

```sh
zig build test
zig build ci
zig build -Doptimize=ReleaseFast
```
