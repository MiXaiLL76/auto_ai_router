# Installation

## Prerequisites

- **Go 1.24+** (for building from source)
- **Docker** (alternative)

## Build from Source

```bash
git clone https://github.com/MiXaiLL76/auto_ai_router.git
cd auto_ai_router
go build -o auto_ai_router ./cmd/server/
```

Run with a config file:

```bash
./auto_ai_router -config config.yaml
```

## Docker

### Build locally

```bash
docker build -t auto-ai-router:latest .
docker run -p 8080:8080 -v $(pwd)/config.yaml:/app/config.yaml auto-ai-router:latest
```

### Pull from GHCR

```bash
docker pull ghcr.io/mixaill76/auto_ai_router:latest
docker run -p 8080:8080 -v $(pwd)/config.yaml:/app/config.yaml ghcr.io/mixaill76/auto_ai_router:latest
```

## Docker Compose

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  auto-ai-router:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: auto-ai-router
    ports:
      - "8080:8080"
    volumes:
      - ./config.yaml:/app/config.yaml:ro
      - ./logs:/app/logs
    environment:
      - LOG_LEVEL=debug
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:8080/health"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 5s
    networks:
      - app-network

networks:
  app-network:
    driver: bridge
```

Then run:

```bash
docker-compose up -d
```

## Verify

Check that the router is running:

```bash
curl http://localhost:8080/health
```
