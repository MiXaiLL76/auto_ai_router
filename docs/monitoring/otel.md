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
- **Tunnels through a proxy**: for a credential with a `proxy_url`, the `CONNECT` that opens a tunnel carries the `traceparent` of the client span it is opened for (W3C trace context only, never baggage), so a proxy that traces can add a span of its own to the request's trace. A pooled tunnel is opened once: the requests that reuse it send no `CONNECT`.

Each server span is annotated with routing attributes:

| Attribute              | Example             | Description                         |
| ---------------------- | ------------------- | ----------------------------------- |
| `gen_ai.request.model` | `gpt-4o`            | Model requested by the client       |
| `aar.real_model`       | `gpt-4o-2024-08-06` | Resolved underlying model           |
| `aar.credential`       | `openai_main`       | Credential selected by the balancer |
| `aar.provider`         | `openai`            | Provider type                       |
| `aar.streaming`        | `true`              | Whether the response is streamed    |
| `aar.request_id`       | `req_abc123`        | Internal request identifier         |

Each client span is one upstream request, and says which attempt of the client's request it is and how it left:

| Attribute               | Example      | Description                                                                                                                                                          |
| ----------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `aar.credential`        | `openai_eu`  | Credential the request was sent to                                                                                                                                   |
| `aar.attempt.number`    | `2`          | Which attempt of the client's request this is, from 1                                                                                                                |
| `aar.attempt.kind`      | `retry`      | `first`, `retry` (the next credential after one failed) or `fallback` (a fallback credential)                                                                        |
| `aar.attempt.reason`    | `rate_limit` | Why the attempt before it failed: `rate_limit`, `server_error`, `auth_error`, `network_error` or `payment_error`. Absent on the first                                |
| `aar.connection.reused` | `true`       | Whether the request was sent on a pooled connection. Absent when it never got one                                                                                    |
| `aar.exit`              | `de-ber-01`  | Exit the connection leaves through: the name the farthest proxy gave itself in the `Via` of its reply to `CONNECT`. Absent without a proxy, or when it names nothing |
| `aar.exit_rtt_ms`       | `52`         | Round trip to that exit when the connection was opened, in milliseconds: the `CONNECT` written to its reply, less `aar.exit_connect_ms`                              |
| `aar.exit_connect_ms`   | `98`         | The exit's connect to the provider when the connection was opened, as its reply to `CONNECT` says it in `Server-Timing`. Both absent without that header             |

Beneath a client span, each step of the attempt that took time is a span of its own, with the same attributes:

| Span               | From, to                                                          |
| ------------------ | ----------------------------------------------------------------- |
| `upstream.connect` | DNS and the TCP connect, to the proxy when the credential has one |
| `upstream.tunnel`  | The connection made, the proxy's reply to `CONNECT`               |
| `upstream.tls`     | The TLS handshake with the provider                               |
| `upstream.ttfb`    | The request written, the first byte of the response               |

A request on a pooled connection has only `upstream.ttfb`: the connection was opened by an earlier attempt, whose span has the other steps. The router's own requests (a price list, a token, a health check) are not attempts: their client span says how it left and has no steps. The exit is a property of the connection, so a request on a pooled tunnel names it too, and carries the two times the tunnel's opening measured: the round trip to the exit and the exit's connect to the provider. A proxy names itself with a `Via` header on its reply to `CONNECT` (`Via: HTTP/1.1 exit-name`) and gives the time it took to connect to the provider in `Server-Timing` (`Server-Timing: connect;dur=98`, in milliseconds); SOCKS proxies have no such reply.

The Kafka spend event carries the summary of all this (see [Kafka spend log](../litellm-integration/kafka_spend_log.md)): the trace's id as `trace_id`, the number of attempts as `upstream_attempts`, the serving credential's exit as `exit` and the two legs of the way through it as `exit_rtt_ms` and `exit_connect_ms`, whether the request had its connection opened as `conn_reused`, and how the request's time divided as `retry_ms`, `conn_wait_ms` and `ttfb_ms`. A spend row says where a request's time went, and leads to the trace that says why.

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
