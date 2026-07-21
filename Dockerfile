# Build stage
FROM golang:1.26-alpine AS builder

# CGO disabled explicitly: this builder has no C toolchain and the app has no
# cgo dependencies, so the build already behaves this way implicitly today.
# Pinning it guarantees a fully static binary (pure-Go DNS resolver, portable
# across libc flavors) even if a future base image or dependency adds gcc.
ENV CGO_ENABLED=0

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

# Final stage
FROM alpine:latest

# Install ca-certificates for HTTPS
RUN apk add --no-cache ca-certificates wget curl

# Create non-root user
RUN addgroup -g 1000 appuser && \
    adduser -D -u 1000 -G appuser appuser

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/auto_ai_router .

# Change ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

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
ENV GOMEMLIMIT=1700MiB

# Expose port (adjust if needed)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Run the application
CMD ["./auto_ai_router"]
