.PHONY: build run clean test fmt vet lint help install-deps

# Build variables
BINARY_NAME=auto_ai_router
BUILD_DIR=.
CMD_DIR=./cmd/server
GO=go
GOFLAGS=-v
LDFLAGS=-ldflags="-s -w"

# Default target
all: build

## help: Display this help message
help:
	@echo "Available targets:"
	@echo "  build         - Build the binary"
	@echo "  build-opt     - Build optimized binary (smaller size)"
	@echo "  run           - Build and run the application"
	@echo "  clean         - Remove build artifacts"
	@echo "  test          - Run tests"
	@echo "  fmt           - Format code"
	@echo "  vet           - Run go vet"
	@echo "  lint          - Run golangci-lint (requires installation)"
	@echo "  install-deps  - Install/update dependencies"
	@echo "  mod-tidy      - Tidy go.mod"

## build: Build the application
build:
	@echo "Building $(BINARY_NAME)..."
	export PATH=/usr/local/go/bin:$$PATH && $(GO) build $(GOFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME) $(CMD_DIR)
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

## test: Run tests
test:
	@echo "Running tests..."
	export PATH=/usr/local/go/bin:$$PATH && $(GO) test -v -race -coverprofile=coverage.out ./...
	@echo "Tests complete"

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
	@which golangci-lint > /dev/null || (echo "golangci-lint not installed. Install from https://golangci-lint.run/usage/install/" && exit 1)
	golangci-lint run ./...
	@echo "Lint complete"

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
