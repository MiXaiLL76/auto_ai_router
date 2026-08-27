# Build stage
FROM golang:1.27-alpine AS builder

# CGO disabled explicitly: this builder has no C toolchain and the app has no
# cgo dependencies, so the build already behaves this way implicitly today.
# Pinning it guarantees a fully static binary (pure-Go DNS resolver, portable
# across libc flavors) even if a future base image or dependency adds gcc.
# Also required for the final stage: distroless/static has no libc at all.
ENV CGO_ENABLED=0
# GOAMD64=v3 targets Haswell+ (AVX2, BMI2, ...) — every mainstream cloud
# instance since ~2015. Lets the compiler/stdlib (encoding/json, compress/*,
# crypto) use newer instructions in their hand-tuned asm paths. Do not raise
# this if a deployment target ever includes pre-2015 hardware — the binary
# will hit SIGILL there instead of falling back.
ENV GOAMD64=v3

# Install build dependencies
RUN apk add --no-cache git make

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
ARG VERSION=dev
ARG COMMIT=unknown
RUN go build -ldflags="-s -w -X main.Version=${VERSION} -X main.Commit=${COMMIT}" -o auto_ai_router ./cmd/server

# distroless/static has no shell/wget/curl, so HEALTHCHECK below execs this
# tiny binary directly instead — see cmd/healthcheck.
RUN go build -ldflags="-s -w" -o healthcheck ./cmd/healthcheck

# Final stage — distroless/static: no shell, no package manager, no libc,
# just ca-certificates + tzdata + a nonroot user (uid/gid 65532) baked in.
# That's why there's no apk/addgroup/chown here anymore: none of those
# tools exist in this image, and the ownership/user setup they used to do
# is already done for us by the :nonroot tag.
FROM gcr.io/distroless/static-debian12:nonroot

WORKDIR /app

# COPY --chown isn't needed: the copied binaries land as root:root 0755, and
# every uid has read+execute on them — the removed `chown -R appuser:appuser`
# was never what made them runnable. The /app directory itself is a different
# story: WORKDIR above creates it owned by 65532, so writes under /app work
# for the default user only (see the uid note above).
COPY --from=builder /app/auto_ai_router .
COPY --from=builder /app/healthcheck .

# Go runtime tuning defaults — overridable via env in the actual deployment
# manifest (e.g. if a pod's memory limit differs from the value assumed here).
#
# GOGC=300: let the heap triple (default is double, GOGC=100) before a GC
# cycle runs. Fewer/larger GC cycles trade RAM for CPU, which pays off here
# because pods are CPU-bound (GOMAXPROCS auto-tracks limits.cpu via cgroup
# quota since Go 1.25) while memory has headroom between requests.memory and
# limits.memory.
ENV GOGC=300
# GOMEMLIMIT=1700MiB: soft cap for the Go runtime's own memory use, tuned for
# a 2Gi container memory limit (~85%, leaving room for goroutine stacks and
# other non-heap RSS). As live heap approaches this, GC runs more aggressively
# to stay under it — turning a would-be abrupt OOMKill into graceful, bounded
# GC pressure instead. This is what makes GOGC=300 safe to run.
# ENV GOMEMLIMIT=1700MiB

# Expose port (adjust if needed)
EXPOSE 8080
# pprof port — off by default (monitoring.pprof_enabled: false). Documented
# here for clarity only; never publish it (-p / k8s Service) to anything but
# an internal debug path (kubectl port-forward).
EXPOSE 6060

ENV HEALTHCHECK_PORT=8080

# Health check — exec form, not "CMD wget ... || exit 1": there's no shell
# to interpret "||" here, and ./healthcheck already returns 0/1 on its own.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["/app/healthcheck"]

# Run the application
CMD ["/app/auto_ai_router"]
