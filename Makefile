.PHONY: build run clean test fmt vet lint format help install-deps docker-build docs-install docs-serve docs-build docs-deploy

# Build variables
BINARY_NAME=auto_ai_router
BUILD_DIR=.
CMD_DIR=./cmd/server
GO=go
GOFLAGS=-v
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
COMMIT ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
LDFLAGS=-ldflags="-s -w -X main.Version=$(VERSION) -X main.Commit=$(COMMIT)"
INTERNAL_PKGS=./internal/...

# Docker variables
DOCKER_IMAGE=auto-ai-router
DOCKER_TAG?=latest
DOCKER_REGISTRY?=ghcr.io/mixaill76

# Default target
all: build

## help: Display this help message
help:
	@echo "Available targets:"
	@echo "  build                - Build the binary"
	@echo "  build-opt            - Build optimized binary (smaller size)"
	@echo "  run                  - Build and run the application"
	@echo "  clean                - Remove build artifacts"
	@echo "  test                 - Run all tests"
	@echo "  test-coverage        - Run tests with coverage report"
	@echo "  test-coverage-html   - Generate HTML coverage report"
	@echo "  test-race            - Run tests with race detector"
	@echo "  test-pkg PKG=<name>  - Run tests for specific package"
	@echo "  test-check-coverage  - Check if coverage meets 80% threshold"
	@echo "  fmt                  - Format code"
	@echo "  vet                  - Run go vet"
	@echo "  lint                 - Run golangci-lint (requires installation)"
	@echo "  format               - Format code and run pre-commit checks"
	@echo "  install-deps         - Install/update dependencies"
	@echo "  mod-tidy             - Tidy go.mod"
	@echo ""
	@echo "Docker targets:"
	@echo "  docker-build  - Build Docker image"
	@echo ""
	@echo "Docs targets:"
	@echo "  docs-install - Install Zensical"
	@echo "  docs-serve   - Serve docs locally (http://127.0.0.1:8000)"
	@echo "  docs-build   - Build static site to site/"
	@echo "  docs-deploy  - Deploy to GitHub Pages"

## build: Build the application
build:
	@echo "Building $(BINARY_NAME)..."
	export PATH=/usr/local/go/bin:$$PATH && $(GO) build $(GOFLAGS) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME) $(CMD_DIR)
	@echo "Build complete: $(BUILD_DIR)/$(BINARY_NAME)"

## build-opt: Build optimized binary
build-opt:
	@echo "Building optimized $(BINARY_NAME)..."
	export PATH=/usr/local/go/bin:$$PATH && $(GO) build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME) $(CMD_DIR)
	@echo "Optimized build complete: $(BUILD_DIR)/$(BINARY_NAME)"

## run: Build and run the application
run: build
	@echo "Starting $(BINARY_NAME)..."
	./$(BINARY_NAME)

## clean: Remove build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(BUILD_DIR)/$(BINARY_NAME)
	@echo "Clean complete"

## test: Run all tests
test:
	@echo "Running tests..."
	export PATH=/usr/local/go/bin:$$PATH && $(GO) test -v $(INTERNAL_PKGS)
	@echo "Tests complete"

## test-coverage: Run tests with coverage report
test-coverage:
	@echo "Running tests with coverage..."
	@export PATH=/usr/local/go/bin:$$PATH; \
	export GOCACHE=$${GOCACHE:-/tmp/go-build}; \
	COVERPKG="$$( $(GO) list ./internal/... | paste -sd "," - )"; \
	$(GO) test -coverpkg="$$COVERPKG" -coverprofile=coverage.out $(INTERNAL_PKGS)
	@echo ""
	@echo "Coverage by package:"
	export PATH=/usr/local/go/bin:$$PATH && $(GO) tool cover -func=coverage.out | grep -E "github.com"
	@echo ""
	@echo "Total coverage:"
	export PATH=/usr/local/go/bin:$$PATH && $(GO) tool cover -func=coverage.out | grep total
	@echo ""
	@echo "To view HTML coverage report, run: make test-coverage-html"

## test-coverage-html: Generate HTML coverage report
test-coverage-html: test-coverage
	@echo "Generating HTML coverage report..."
	export PATH=/usr/local/go/bin:$$PATH && $(GO) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

## test-race: Run tests with race detector
test-race:
	@echo "Running tests with race detector..."
	export PATH=/usr/local/go/bin:$$PATH && $(GO) test -race $(INTERNAL_PKGS)
	@echo "Race detection tests complete"

## test-pkg: Run tests for specific package (usage: make test-pkg PKG=config)
test-pkg:
	@echo "Running tests for package $(PKG)..."
	@export PATH=/usr/local/go/bin:$$PATH; \
	export GOCACHE=$${GOCACHE:-/tmp/go-build}; \
	COVERPKG="$$( $(GO) list ./internal/... | paste -sd "," - )"; \
	$(GO) test -v -coverpkg="$$COVERPKG" -cover ./internal/$(PKG)/...

## test-check-coverage: Check if coverage meets threshold
test-check-coverage:
	@echo "Checking coverage threshold..."
	@export PATH=/usr/local/go/bin:$$PATH; \
	export GOCACHE=$${GOCACHE:-/tmp/go-build}; \
	COVERPKG="$$( $(GO) list ./internal/... | paste -sd "," - )"; \
	$(GO) test -coverpkg="$$COVERPKG" -coverprofile=coverage.out $(INTERNAL_PKGS) > /dev/null
	@export PATH=/usr/local/go/bin:$$PATH && $(GO) tool cover -func=coverage.out | grep total | awk '{print "Total coverage: " $$3}'
	@export PATH=/usr/local/go/bin:$$PATH && $(GO) tool cover -func=coverage.out | grep total | awk '{gsub(/%/,"",$$3); if ($$3+0 < 80) {print "❌ Coverage is below 80%"; exit 1} else {print "✅ Coverage is above 80%"}}'

## fmt: Format code
fmt:
	@echo "Formatting code..."
	export PATH=/usr/local/go/bin:$$PATH && $(GO) fmt ./...
	@echo "Format complete"

## vet: Run go vet
vet:
	@echo "Running go vet..."
	export PATH=/usr/local/go/bin:$$PATH && $(GO) vet ./...
	@echo "Vet complete"

## lint: Run golangci-lint
lint:
	@echo "Running golangci-lint..."
	@export PATH=/usr/local/go/bin:$$(go env GOPATH)/bin:$$PATH && which golangci-lint > /dev/null || (echo "golangci-lint not installed. Install from https://golangci-lint.run/usage/install/" && exit 1)
	export PATH=/usr/local/go/bin:$$(go env GOPATH)/bin:$$PATH && golangci-lint run ./...
	@echo "Lint complete"

## format: Format code and run pre-commit checks
format:
	@echo "Running formatters..."
	export PATH=/usr/local/go/bin:$$PATH && $(GO) fmt ./...
	@echo "Running go vet..."
	export PATH=/usr/local/go/bin:$$PATH && $(GO) vet ./...
	@echo "Running pre-commit hooks..."
	@which pre-commit > /dev/null || (echo "pre-commit not installed. Install with: pip install pre-commit" && exit 1)
	pre-commit run --all-files
	@echo "Format complete"

## install-deps: Install/update dependencies
install-deps:
	@echo "Installing dependencies..."
	export PATH=/usr/local/go/bin:$$PATH && $(GO) get -u ./...
	@echo "Dependencies installed"

## mod-tidy: Tidy go.mod
mod-tidy:
	@echo "Tidying go.mod..."
	export PATH=/usr/local/go/bin:$$PATH && $(GO) mod tidy
	@echo "go.mod tidied"

## docker-build: Build Docker image
docker-build:
	@echo "Building Docker image $(DOCKER_IMAGE):$(DOCKER_TAG)..."
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "Docker image built successfully"

## docs-install: Install Zensical
docs-install:
	pip install zensical

## docs-serve: Serve docs locally with live reload
docs-serve:
	zensical serve

## docs-build: Build static site to site/
docs-build:
	zensical build --clean

## docs-deploy: Deploy to GitHub Pages (use GitHub Actions instead)
docs-deploy:
	zensical build --clean
