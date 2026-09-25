package ratelimit

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockNetError implements net.Error for testing isTransientError.
type mockNetError struct {
	timeout bool
}

func (e *mockNetError) Error() string   { return fmt.Sprintf("mock net error (timeout=%v)", e.timeout) }
func (e *mockNetError) Timeout() bool   { return e.timeout }
func (e *mockNetError) Temporary() bool { return false }

// redisBackendForTest creates a real RedisBackend from VALKEY_ADDR env var,
// or skips the test if the variable is not set.
func redisBackendForTest(t *testing.T, prefix string) *RedisBackend {
	t.Helper()
	addr := os.Getenv("VALKEY_ADDR")
	if addr == "" {
		t.Skip("VALKEY_ADDR not set, skipping Redis integration test")
	}
	cfg := config.RedisConfig{
		InitAddresses:     []string{addr},
		ForceSingleClient: true,
		KeyPrefix:         prefix,
		KeyTTL:            60,
	}
	b, err := NewRedisBackend(cfg)
	require.NoError(t, err)
	t.Cleanup(func() { b.Close() })
	return b
}

// ── isTransientError ────────────────────────────────────────────────────────

func TestIsTransientError_nil(t *testing.T) {
	assert.False(t, isTransientError(nil))
}

func TestIsTransientError_EOF(t *testing.T) {
	assert.True(t, isTransientError(io.EOF))
}

func TestIsTransientError_contextCanceled(t *testing.T) {
	assert.False(t, isTransientError(context.Canceled))
}

func TestIsTransientError_deadlineExceeded(t *testing.T) {
	assert.False(t, isTransientError(context.DeadlineExceeded))
}

func TestIsTransientError_nonTimeoutNetError(t *testing.T) {
	// non-timeout net.Error → connection setup failure → safe to retry
	err := &mockNetError{timeout: false}
	assert.True(t, isTransientError(err))
}

func TestIsTransientError_timeoutNetError(t *testing.T) {
	// timeout net.Error → command may have already executed → not safe to retry
	err := &mockNetError{timeout: true}
	assert.False(t, isTransientError(err))
}

func TestIsTransientError_wrappedEOF(t *testing.T) {
	wrapped := fmt.Errorf("outer: %w", io.EOF)
	assert.True(t, isTransientError(wrapped))
}

func TestIsTransientError_wrappedContextCanceled(t *testing.T) {
	wrapped := fmt.Errorf("outer: %w", context.Canceled)
	assert.False(t, isTransientError(wrapped))
}

func TestIsTransientError_genericError(t *testing.T) {
	assert.False(t, isTransientError(errors.New("some other error")))
}

// ── cmdCtx ──────────────────────────────────────────────────────────────────

func TestCmdCtx_ZeroTimeout_ReturnsParent(t *testing.T) {
	b := &RedisBackend{commandTimeout: 0}
	parent := context.Background()
	ctx, cancel := b.cmdCtx(parent)
	defer cancel()

	_, hasDeadline := ctx.Deadline()
	assert.False(t, hasDeadline, "zero commandTimeout should not add a deadline")
}

func TestCmdCtx_WithTimeout_AddsDeadline(t *testing.T) {
	b := &RedisBackend{commandTimeout: 500 * time.Millisecond}
	ctx, cancel := b.cmdCtx(context.Background())
	defer cancel()

	deadline, ok := ctx.Deadline()
	assert.True(t, ok, "should have a deadline")
	remaining := time.Until(deadline)
	assert.Greater(t, remaining, time.Duration(0))
	assert.LessOrEqual(t, remaining, 500*time.Millisecond)
}

func TestCmdCtx_ParentDeadlineTighter_ReturnsParent(t *testing.T) {
	b := &RedisBackend{commandTimeout: 5 * time.Second}
	// parent deadline is 100ms — much tighter than commandTimeout
	parent, parentCancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer parentCancel()

	ctx, cancel := b.cmdCtx(parent)
	defer cancel()

	parentDeadline, _ := parent.Deadline()
	ctxDeadline, ok := ctx.Deadline()
	assert.True(t, ok)
	// The ctx deadline should equal the parent's (not commandTimeout's)
	assert.Equal(t, parentDeadline, ctxDeadline)
}

// ── key helpers ─────────────────────────────────────────────────────────────

func TestRedisBackend_WithSharedKeyPrefix(t *testing.T) {
	b := &RedisBackend{keyPrefix: "ru01", keyTTL: 60, commandTimeout: 2 * time.Second}
	c := b.WithSharedKeyPrefix("air-balancer:")

	assert.Equal(t, "air-balancer:rpm:{c:mycred}:m:mycred:mymodel", c.rpmKey("m:mycred:mymodel"))
	assert.Equal(t, "air-balancer:", c.KeyPrefix())
	assert.Equal(t, 60, c.keyTTL)
	assert.Equal(t, 2*time.Second, c.commandTimeout)
	// The original backend keeps its namespace and still deletes keys.
	assert.Equal(t, "ru01rpm:{c:mycred}:m:mycred:mymodel", b.rpmKey("m:mycred:mymodel"))
	assert.False(t, b.keepKeysOnDelete)
	// Shared counters are never deleted: with a nil client a real DEL would panic.
	assert.NotPanics(t, func() { c.deleteKey(context.Background(), "m:mycred:mymodel") })
}

// Two deployments with different key_prefix but the same balancer prefix must
// see each other's traffic for the same credential/model, and removing the
// model in one deployment must not wipe the other's usage.
func TestRedisBackend_Integration_WithSharedKeyPrefix(t *testing.T) {
	shared := fmt.Sprintf("test:balancer:%d:", time.Now().UnixNano())
	deployA := redisBackendForTest(t, "test:ru01:").WithSharedKeyPrefix(shared)
	deployB := redisBackendForTest(t, "test:ru02:").WithSharedKeyPrefix(shared)
	ctx := context.Background()
	key := "m:grant:claude-opus-4.6"
	cleanup := redisBackendForTest(t, shared)
	t.Cleanup(func() { cleanup.deleteKey(ctx, key) })

	require.True(t, deployA.tryAllowRPM(ctx, key, 4))
	require.True(t, deployA.tryAllowRPM(ctx, key, 4))
	assert.Equal(t, 2, deployB.currentRPM(ctx, key))

	require.True(t, deployB.tryAllowRPM(ctx, key, 4))
	require.True(t, deployB.tryAllowRPM(ctx, key, 4))
	assert.False(t, deployB.tryAllowRPM(ctx, key, 4), "joint limit of 4 must be reached across both deployments")

	deployA.deleteKey(ctx, key)
	assert.Equal(t, 4, deployB.currentRPM(ctx, key), "model removal in A must not reset B's usage")
}

func TestRedisBackend_KeyFunctions(t *testing.T) {
	b := &RedisBackend{keyPrefix: "rl:"}
	// Credential key: wrapped in hash tag for slot consistency.
	assert.Equal(t, "rl:rpm:{c:mycred}", b.rpmKey("c:mycred"))
	assert.Equal(t, "rl:tpm:{c:mycred}", b.tpmKey("c:mycred"))
	// Model key: inherits credential's hash tag.
	assert.Equal(t, "rl:rpm:{c:mycred}:m:mycred:mymodel", b.rpmKey("m:mycred:mymodel"))
	assert.Equal(t, "rl:tpm:{c:mycred}:m:mycred:mymodel", b.tpmKey("m:mycred:mymodel"))
	// Generic key: wrapped as-is.
	assert.Equal(t, "rl:rpm:{other}", b.rpmKey("other"))
}

func TestRedisBackend_KeyFunctions_EmptyPrefix(t *testing.T) {
	b := &RedisBackend{keyPrefix: ""}
	assert.Equal(t, "rpm:{c:mycred}", b.rpmKey("c:mycred"))
	assert.Equal(t, "tpm:{c:mycred}", b.tpmKey("c:mycred"))
}

// ── constructors ────────────────────────────────────────────────────────────

func TestNewRedisBackendFromClientWithTTL(t *testing.T) {
	// nil client is fine for construction only
	b := NewRedisBackendFromClientWithTTL(nil, "prefix:", 90)
	assert.Equal(t, "prefix:", b.keyPrefix)
	assert.Equal(t, 90, b.keyTTL)
	assert.Equal(t, defaultCommandTimeout, b.commandTimeout)
}

func TestNewRedisBackendFromClient_DefaultTTL(t *testing.T) {
	b := NewRedisBackendFromClient(nil, "test:")
	assert.Equal(t, "test:", b.keyPrefix)
	assert.Equal(t, 120, b.keyTTL)
}

// ── setCurrentUsage no-op ───────────────────────────────────────────────────

func TestRedisBackend_SetCurrentUsage_IsNoOp(t *testing.T) {
	b := &RedisBackend{}
	// Should not panic; does nothing.
	b.setCurrentUsage(context.Background(), "any-key", 100, 5000)
}

// ── integration tests (require VALKEY_ADDR) ─────────────────────────────────

func TestRedisBackend_Integration_Ping(t *testing.T) {
	b := redisBackendForTest(t, "test:ping:")
	err := b.Ping(context.Background())
	assert.NoError(t, err)
}

func TestRedisBackend_Integration_Client(t *testing.T) {
	b := redisBackendForTest(t, "test:client:")
	assert.NotNil(t, b.Client())
}

func TestRedisBackend_Integration_TryAllowRPM_UnderLimit(t *testing.T) {
	b := redisBackendForTest(t, "test:rpm1:")
	ctx := context.Background()

	// limit=5, make 5 requests — all should pass
	for i := 0; i < 5; i++ {
		assert.True(t, b.tryAllowRPM(ctx, "cred1", 5), "request %d should be allowed", i+1)
	}
}

func TestRedisBackend_Integration_TryAllowRPM_OverLimit(t *testing.T) {
	b := redisBackendForTest(t, "test:rpm2:")
	ctx := context.Background()

	// fill up to limit=3
	for i := 0; i < 3; i++ {
		b.tryAllowRPM(ctx, "cred1", 3)
	}
	// 4th must be rejected
	assert.False(t, b.tryAllowRPM(ctx, "cred1", 3))
}

func TestRedisBackend_Integration_TryAllowRPM_Unlimited(t *testing.T) {
	b := redisBackendForTest(t, "test:rpm3:")
	ctx := context.Background()

	for i := 0; i < 200; i++ {
		assert.True(t, b.tryAllowRPM(ctx, "cred1", -1))
	}
}

func TestRedisBackend_Integration_CanAllowRPM_NoRecord(t *testing.T) {
	b := redisBackendForTest(t, "test:canrpm:")
	ctx := context.Background()

	// canAllowRPM should not record — RPM counter stays 0
	assert.True(t, b.canAllowRPM(ctx, "cred1", 2))
	assert.Equal(t, 0, b.currentRPM(ctx, "cred1"))

	// Actually record one
	b.tryAllowRPM(ctx, "cred1", 2)
	assert.Equal(t, 1, b.currentRPM(ctx, "cred1"))

	// canAllow still true (1 < 2)
	assert.True(t, b.canAllowRPM(ctx, "cred1", 2))
	// Record second
	b.tryAllowRPM(ctx, "cred1", 2)
	// Now at limit — canAllow false
	assert.False(t, b.canAllowRPM(ctx, "cred1", 2))
}

func TestRedisBackend_Integration_ConsumeTokens_And_CurrentTPM(t *testing.T) {
	b := redisBackendForTest(t, "test:tpm1:")
	ctx := context.Background()

	assert.Equal(t, 0, b.currentTPM(ctx, "cred1"))

	b.consumeTokens(ctx, "cred1", 500)
	assert.Equal(t, 500, b.currentTPM(ctx, "cred1"))

	b.consumeTokens(ctx, "cred1", 300)
	assert.Equal(t, 800, b.currentTPM(ctx, "cred1"))
}

func TestRedisBackend_Integration_CanAllowTPM_Unlimited(t *testing.T) {
	b := redisBackendForTest(t, "test:tpm2:")
	ctx := context.Background()

	// limit=-1 means unlimited
	b.consumeTokens(ctx, "cred1", 999999)
	assert.True(t, b.canAllowTPM(ctx, "cred1", -1))
}

func TestRedisBackend_Integration_CanAllowTPM_AtLimit(t *testing.T) {
	b := redisBackendForTest(t, "test:tpm3:")
	ctx := context.Background()

	b.consumeTokens(ctx, "cred1", 1000)
	assert.False(t, b.canAllowTPM(ctx, "cred1", 1000))
}

// Integration tests use real key prefixes matching production ("c:" for credentials,
// "m:credname:model" for models) so that hashTag() generates consistent slot assignments
// and multi-key EVAL does not panic.

func TestRedisBackend_Integration_TryAllowAll_AllPass(t *testing.T) {
	b := redisBackendForTest(t, "test:all1:")
	ctx := context.Background()

	// Use production-format keys: "c:" prefix for creds, "m:cred:model" for models.
	allowed := b.tryAllowAll(ctx, "c:cred1", 10, 10000, "m:cred1:model1", 5, 5000)
	assert.True(t, allowed)
	// tryAllowAll and currentRPM use the same rpmKey() — counters must be visible.
	assert.Equal(t, 1, b.currentRPM(ctx, "c:cred1"), "cred RPM must be recorded")
	assert.Equal(t, 1, b.currentRPM(ctx, "m:cred1:model1"), "model RPM must be recorded")
}

func TestRedisBackend_Integration_TryAllowAll_CredRPMExhausted(t *testing.T) {
	b := redisBackendForTest(t, "test:all2:")
	ctx := context.Background()

	// Exhaust cred RPM using the same key that tryAllowAll will check.
	b.tryAllowRPM(ctx, "c:cred1", 2)
	b.tryAllowRPM(ctx, "c:cred1", 2)

	allowed := b.tryAllowAll(ctx, "c:cred1", 2, -1, "m:cred1:model1", 10, -1)
	assert.False(t, allowed)
	// Model RPM must not have been recorded (check failed before record).
	assert.Equal(t, 0, b.currentRPM(ctx, "m:cred1:model1"))
}

func TestRedisBackend_Integration_TryAllowAll_ModelRPMExhausted(t *testing.T) {
	b := redisBackendForTest(t, "test:all3:")
	ctx := context.Background()

	// Exhaust model RPM using the same key that tryAllowAll will check.
	b.tryAllowRPM(ctx, "m:cred1:model1", 2)
	b.tryAllowRPM(ctx, "m:cred1:model1", 2)

	allowed := b.tryAllowAll(ctx, "c:cred1", 10, -1, "m:cred1:model1", 2, -1)
	assert.False(t, allowed)
	// Cred RPM must NOT have been recorded (atomic check-and-record).
	assert.Equal(t, 0, b.currentRPM(ctx, "c:cred1"))
}

func TestRedisBackend_Integration_TryAllowAll_NoModel(t *testing.T) {
	b := redisBackendForTest(t, "test:all4:")
	ctx := context.Background()

	// Empty modelKey means no model checks.
	allowed := b.tryAllowAll(ctx, "c:cred1", 10, -1, "", -1, -1)
	assert.True(t, allowed)
	assert.Equal(t, 1, b.currentRPM(ctx, "c:cred1"))
}

func TestRedisBackend_Integration_TryAllowAll_CredTPMExhausted(t *testing.T) {
	b := redisBackendForTest(t, "test:all5:")
	ctx := context.Background()

	// Fill cred TPM; consumeTokens and tryAllowAll share the same tpmKey().
	b.consumeTokens(ctx, "c:cred1", 1000)

	allowed := b.tryAllowAll(ctx, "c:cred1", 10, 1000, "", -1, -1)
	assert.False(t, allowed, "cred TPM at limit must block")
	// RPM must not have been recorded.
	assert.Equal(t, 0, b.currentRPM(ctx, "c:cred1"))
}

func TestRedisBackend_Integration_TryAllowAll_ConsumeTokensVisible(t *testing.T) {
	b := redisBackendForTest(t, "test:all6:")
	ctx := context.Background()

	// consumeTokens and tryAllowAll must read/write the same TPM key.
	b.consumeTokens(ctx, "c:cred1", 500)
	assert.Equal(t, 500, b.currentTPM(ctx, "c:cred1"))

	// 500 < 1000 limit → allowed.
	assert.True(t, b.tryAllowAll(ctx, "c:cred1", 10, 1000, "", -1, -1))

	// Consume remaining budget.
	b.consumeTokens(ctx, "c:cred1", 500)
	assert.Equal(t, 1000, b.currentTPM(ctx, "c:cred1"))

	// At limit → rejected.
	assert.False(t, b.tryAllowAll(ctx, "c:cred1", 10, 1000, "", -1, -1))
}

func TestRedisBackend_Integration_CurrentRPM_Empty(t *testing.T) {
	b := redisBackendForTest(t, "test:curr1:")
	ctx := context.Background()
	assert.Equal(t, 0, b.currentRPM(ctx, "nonexistent"))
}

func TestRedisBackend_Integration_CurrentTPM_Empty(t *testing.T) {
	b := redisBackendForTest(t, "test:curr2:")
	ctx := context.Background()
	assert.Equal(t, 0, b.currentTPM(ctx, "nonexistent"))
}

// TestNewRedisBackend_Integration tests constructor with a real connection.
func TestNewRedisBackend_Integration(t *testing.T) {
	addr := os.Getenv("VALKEY_ADDR")
	if addr == "" {
		t.Skip("VALKEY_ADDR not set, skipping Redis integration test")
	}
	cfg := config.RedisConfig{
		InitAddresses:     []string{addr},
		ForceSingleClient: true,
		KeyPrefix:         "test:ctor:",
	}
	b, err := NewRedisBackend(cfg)
	require.NoError(t, err)
	defer b.Close()

	// defaults applied: KeyTTL=120, commandTimeout=defaultCommandTimeout
	assert.Equal(t, 120, b.keyTTL)
	assert.Equal(t, defaultCommandTimeout, b.commandTimeout)
}

func TestNewRedisBackend_Integration_CustomTTL(t *testing.T) {
	addr := os.Getenv("VALKEY_ADDR")
	if addr == "" {
		t.Skip("VALKEY_ADDR not set, skipping Redis integration test")
	}
	cfg := config.RedisConfig{
		InitAddresses:     []string{addr},
		ForceSingleClient: true,
		KeyTTL:            300,
		CommandTimeout:    2 * time.Second,
	}
	b, err := NewRedisBackend(cfg)
	require.NoError(t, err)
	defer b.Close()

	assert.Equal(t, 300, b.keyTTL)
	assert.Equal(t, 2*time.Second, b.commandTimeout)
}
