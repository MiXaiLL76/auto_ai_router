# OpenTelemetry

Auto AI Router has built-in [OpenTelemetry](https://opentelemetry.io/) (OTEL) support and can export all three signals — **traces**, **logs**, and **metrics** — to any OTLP-compatible collector (OpenTelemetry Collector, Grafana Alloy, Tempo, Loki, Jaeger, etc.).

OTEL is **disabled by default**. When disabled, no SDK components are initialized and the router behaves exactly as before: pretty stdout logs, no tracing, and pull-based Prometheus metrics only.

## Configuration

All settings live under the `otel` section of the config:

```yaml
otel:
  enabled: false                  # Master switch (default: false)
  endpoint: "localhost:4317"      # OTLP collector endpoint
  protocol: "grpc"                # grpc | http (http/protobuf)
  insecure: true                  # Disable TLS for the exporter connection (default: true)
  service_name: "auto-ai-router"  # Reported as the service.name resource attribute and on spend events
  logs_enabled: true              # Ship slog records via OTLP in addition to stdout
  traces_enabled: true            # Server/client spans + traceparent propagation
  metric_export_interval: 60s     # OTLP metric push interval
  trace_sample_ratio: 1.0         # Head sampling ratio [0.0-1.0], parent-based
  trust_incoming_traceparent: true
  # headers:                      # Optional headers for OTLP requests
  #   Authorization: "os.environ/OTEL_AUTH_HEADER"
```

| Field                        | Default                    | Description                                                                                               |
| ---------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------- |
| `enabled`                    | `false`                    | Master switch. When `false`, no OTEL components are initialized.                                          |
| `endpoint`                   | `localhost:4317` / `:4318` | OTLP collector. `host:port` for grpc; `host:port` or full URL for http. Default port depends on protocol. |
| `protocol`                   | `grpc`                     | OTLP transport: `grpc` or `http` (http/protobuf).                                                         |
| `insecure`                   | `true`                     | Disable TLS for the exporter connection (typical for an in-cluster collector).                            |
| `service_name`               | `auto-ai-router`           | Reported as the `service.name` resource attribute, and as `service_name` on every Kafka spend event.      |
| `headers`                    | —                          | Extra headers added to every OTLP export request (e.g. auth tokens). Supports `os.environ/VAR_NAME`.      |
| `logs_enabled`               | `true`                     | Ship `slog` records via OTLP in addition to stdout.                                                       |
| `traces_enabled`             | `true`                     | Create server/client spans and propagate trace context to upstreams.                                      |
| `metric_export_interval`     | `60s`                      | Period between OTLP metric pushes.                                                                        |
| `trace_sample_ratio`         | `1.0`                      | Head sampling ratio in `[0.0, 1.0]`. Parent-based (sampled upstream decisions are respected).             |
| `trust_incoming_traceparent` | `true`                     | Adopt the caller's W3C `traceparent` so the router's spans nest under it.                                 |

All fields support environment variable resolution via `os.environ/VAR_NAME`, so the endpoint, headers, and toggles can be supplied at deploy time.

The default endpoint is chosen from the protocol: `localhost:4317` for grpc and `localhost:4318` for http. For http, the standard OTLP signal paths (`/v1/traces`, `/v1/logs`, `/v1/metrics`) are appended automatically when the endpoint has no explicit path.

## Quick start

Point the router at a local collector and enable OTEL:

```yaml
otel:
  enabled: true
  endpoint: "otel-collector:4317"
  protocol: "grpc"
  insecure: true
```

On startup the router logs the active configuration:

```
OpenTelemetry initialized  endpoint=otel-collector:4317 protocol=grpc logs_enabled=true traces_enabled=true metrics_enabled=true
```

!!! note "Failures degrade gracefully"
OTEL is observability, not core functionality. If the SDK fails to initialize (bad endpoint, unreachable collector at boot), the router logs the error and **continues running without it** instead of failing startup.

## Signals

### Traces

When `traces_enabled` is true, the router instruments inbound and outbound HTTP automatically:

- **Server spans** are created for every API request. Health checks (`health_check_path`), readiness probes (`/health/readiness`), and metric scrapes (`/metrics`) are excluded to avoid trace noise. Spans are named `METHOD /path` (e.g. `POST /v1/chat/completions`).
- **Client spans** are created for every outbound request to a provider or chained router, and the `traceparent` header is injected so the trace continues across hops.

Each server span is annotated with routing attributes:

| Attribute              | Example             | Description                         |
| ---------------------- | ------------------- | ----------------------------------- |
| `gen_ai.request.model` | `gpt-4o`            | Model requested by the client       |
| `aar.real_model`       | `gpt-4o-2024-08-06` | Resolved underlying model           |
| `aar.credential`       | `openai_main`       | Credential selected by the balancer |
| `aar.provider`         | `openai`            | Provider type                       |
| `aar.streaming`        | `true`              | Whether the response is streamed    |
| `aar.request_id`       | `req_abc123`        | Internal request identifier         |

**Sampling** is head-based and parent-based: `trace_sample_ratio` controls the fraction of new root traces that are sampled, while decisions inherited from an upstream `traceparent` are always honored.

**Incoming trace context** is governed by `trust_incoming_traceparent`:

- `true` (default) — the server adopts an incoming W3C `traceparent` and nests its span under it. Use this when a trusted hop sits in front (for example a LiteLLM proxy with `forward_traceparent_to_llm_provider: true`), so the router's spans join the caller's trace.
- `false` — client-supplied trace context is ignored and every request starts a fresh root span. Use this for standalone or public-facing deployments. Outgoing `traceparent` propagation to upstreams is unaffected either way.

### Logs

When `logs_enabled` is true, `slog` records are shipped via OTLP in addition to stdout. The `logging_level` (`info`/`debug`/`error`) is applied to the OTLP destination just like stdout.

Logs emitted inside an active span automatically carry that span's `trace_id` and `span_id`, giving you **log↔trace correlation** in backends like Grafana (jump from a trace to its logs and back).

To ship logs **only** via OTLP and drop the stdout stream, set `stdout_logs_enabled: false` under `server`:

```yaml
server:
  stdout_logs_enabled: false   # Ship logs only via OTEL
```

If you disable stdout logs while OTLP log export is not active, the router keeps stdout logging and warns, so you never silently lose logs.

### Metrics

When `otel.enabled` is true and metrics are being collected, the router's existing Prometheus registry is **bridged** into the OTEL pipeline and pushed to the collector over OTLP every `metric_export_interval`. No application instrumentation changes — the same metrics described in [Prometheus](prometheus.md) are exported.

This OTLP **push** path is independent of the pull-based `/metrics` endpoint:

- `monitoring.prometheus_enabled: true` exposes the `/metrics` endpoint for scraping.
- `otel.enabled: true` pushes the same metrics to the OTLP collector.

You can run either, both, or neither. Metric collection turns on whenever **either** path is enabled, so you can push metrics via OTLP without exposing `/metrics`.

## Resource attributes

Every exported span, log, and metric carries:

- `service.name` — from `otel.service_name` (default `auto-ai-router`)
- `service.version` — the router build version

plus the defaults provided by the OTEL SDK (host, process, SDK info).

## Authentication

For collectors that require auth (e.g. a hosted OTLP endpoint), add headers and keep secrets in the environment:

```yaml
otel:
  enabled: true
  endpoint: "https://otlp.example.com"
  protocol: "http"
  insecure: false
  headers:
    Authorization: "os.environ/OTEL_AUTH_HEADER"
```

## Diagnostics

Export activity is logged to stdout (never back through the OTLP pipeline, which would loop):

- **DEBUG** — per-batch export successes with batch size and duration (visible with `logging_level: debug`).
- **WARN** — export failures and internal SDK errors (queue overflows, async export problems).

If traces or metrics aren't reaching your backend, set `logging_level: debug` and watch for `OTLP ... export succeeded` / `failed` lines.
