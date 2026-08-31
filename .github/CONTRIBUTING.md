# Contributing to Auto AI Router

Thanks for your interest in the project — contributions of all kinds are welcome.

Auto AI Router is a Go-based AI proxy that forwards requests to OpenAI, Vertex AI and
Anthropic, with round-robin credential balancing (fail2ban + rate limiting), LiteLLM DB
integration for spend logging and auth, and provider responses normalized to the OpenAI
format. Development targets the Go toolchain pinned in `go.mod` (currently Go 1.27).
Common tasks are driven through the `Makefile`: `make build`, `make test`,
`make test-race`, `make lint` (pinned `golangci-lint`), and `make format` before you
commit. New code is expected to keep package test coverage at or above the 80% threshold
(`make test-check-coverage`), so add tests alongside behavior changes. If you touch the
Kafka/ClickHouse spend pipeline, `make kafka-up` brings up a local stack for manual
verification.

A few conventions specific to this codebase. When working on the provider converters
(`internal/converter/vertex/`, `internal/converter/openai/`, `internal/converter/anthropic/`),
check the actual SDK source rather than relying on memory of the API surface — the
Google genai, OpenAI and Anthropic SDKs are the ground truth for wire shapes (header vs
body fields, struct names, casing). Watch for the recurring bug classes we keep hitting:
never advance a balancer/round-robin position counter for filtered-out items (only when
the selected item is returned); always nil-check `TokenUsage` before reading its fields;
prefer `atomic.Bool` over `bool` + mutex for simple flags; don't `defer cancel()` inside
a loop; and be careful with inner `:=` shadowing. Finally, if you are working on a
long-lived feature branch, rebase or diff against `origin/main` before opening or
finalizing a PR — several past PRs broke on silent semantic drift (renamed
fields/functions on one side only) that produced no git conflict markers but failed the
build after merge. Always run `go build ./...` and the full test suite after merging main.
