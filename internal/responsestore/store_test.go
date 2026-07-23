package responsestore

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/stretchr/testify/require"
)

// createTestResponse creates a minimal Response for testing
func createTestResponse(id string) *responses.Response {
	return &responses.Response{
		ID:        id,
		CreatedAt: time.Now().Unix(),
		Model:     "gpt-4",
		Status:    "completed",
		Output: []responses.OutputItem{
			{
				Type: "message",
				ID:   "msg-" + id,
				Content: []responses.OutputContent{
					{
						Type: "output_text",
						Text: "Hello, world!",
					},
				},
			},
		},
		Usage: &responses.Usage{
			InputTokens:  10,
			OutputTokens: 5,
			TotalTokens:  15,
		},
	}
}

func TestNew(t *testing.T) {
	// Test that New() creates a store without error
	store, err := New()
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	if store == nil {
		t.Fatal("store is nil")
	}
}

func TestBboltStore_SaveAndGetResponse(t *testing.T) {
	store, err := New()
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	ctx := context.Background()
	apiKeyHash := "test-api-key-hash"
	resp := createTestResponse("test-response-id")

	// Save response
	err = store.SaveResponse(ctx, apiKeyHash, resp, nil, 0, nil, "cred-1")
	if err != nil {
		t.Fatalf("failed to save response: %v", err)
	}

	// Get response
	got, err := store.GetResponse(ctx, "test-response-id", apiKeyHash)
	if err != nil {
		t.Fatalf("failed to get response: %v", err)
	}

	require.NotNil(t, got, "response is nil")

	if got.ID != resp.ID {
		t.Errorf("expected ID %s, got %s", resp.ID, got.ID)
	}

	if got.Model != resp.Model {
		t.Errorf("expected Model %s, got %s", resp.Model, got.Model)
	}
}

func TestBboltStore_GetResponse_NotFound(t *testing.T) {
	store, err := New()
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	ctx := context.Background()

	_, err = store.GetResponse(ctx, "nonexistent-id", "some-hash")
	if err == nil {
		t.Error("expected error for nonexistent response")
	}
}

func TestBboltStore_GetResponse_Unauthorized(t *testing.T) {
	store, err := New()
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	ctx := context.Background()
	apiKeyHash := "correct-hash"
	resp := createTestResponse("auth-test-id")

	err = store.SaveResponse(ctx, apiKeyHash, resp, nil, 0, nil, "cred-1")
	if err != nil {
		t.Fatalf("failed to save response: %v", err)
	}

	// Try to get with wrong hash
	_, err = store.GetResponse(ctx, "auth-test-id", "wrong-hash")
	if err == nil {
		t.Error("expected error for unauthorized access")
	}
}

func TestBboltStore_GetResponseByID(t *testing.T) {
	store, err := New()
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	ctx := context.Background()
	apiKeyHash := "test-hash"
	resp := createTestResponse("by-id-test")

	err = store.SaveResponse(ctx, apiKeyHash, resp, nil, 0, nil, "cred-1")
	if err != nil {
		t.Fatalf("failed to save response: %v", err)
	}

	// Get by ID without ownership check
	got, err := store.GetResponseByID(ctx, "by-id-test")
	if err != nil {
		t.Fatalf("failed to get response by ID: %v", err)
	}

	if got.ID != "by-id-test" {
		t.Errorf("expected ID by-id-test, got %s", got.ID)
	}
}

func TestBboltStore_GetEntry(t *testing.T) {
	store, err := New()
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	ctx := context.Background()
	apiKeyHash := "entry-test-hash"
	resp := createTestResponse("entry-test-id")
	metadata := map[string]string{"key": "value"}
	accumulatedInput := json.RawMessage(`{"test": true}`)

	err = store.SaveResponse(ctx, apiKeyHash, resp, metadata, 0, accumulatedInput, "cred-1")
	if err != nil {
		t.Fatalf("failed to save response: %v", err)
	}

	// Get full entry
	entry, err := store.GetEntry(ctx, "entry-test-id", apiKeyHash)
	if err != nil {
		t.Fatalf("failed to get entry: %v", err)
	}

	require.NotNil(t, entry, "entry is nil")

	if entry.ResponseID != "entry-test-id" {
		t.Errorf("expected ResponseID entry-test-id, got %s", entry.ResponseID)
	}

	if entry.APIKeyHash != apiKeyHash {
		t.Errorf("expected APIKeyHash %s, got %s", apiKeyHash, entry.APIKeyHash)
	}
	if entry.CredentialName != "cred-1" {
		t.Errorf("expected CredentialName cred-1, got %s", entry.CredentialName)
	}

	if entry.Metadata == nil || entry.Metadata["key"] != "value" {
		t.Error("metadata not stored correctly")
	}

	// Compare as JSON (accounting for formatting differences)
	var expected, actual map[string]interface{}
	if err := json.Unmarshal(accumulatedInput, &expected); err != nil {
		t.Fatalf("failed to unmarshal expected: %v", err)
	}
	if err := json.Unmarshal(entry.AccumulatedInput, &actual); err != nil {
		t.Fatalf("failed to unmarshal actual: %v", err)
	}

	// Compare as generic maps
	expectedJSON, _ := json.Marshal(expected)
	actualJSON, _ := json.Marshal(actual)
	if string(expectedJSON) != string(actualJSON) {
		t.Errorf("expected accumulatedInput %s, got %s", expectedJSON, actualJSON)
	}
}

func TestBboltStore_SaveResponse_NilResponse(t *testing.T) {
	store, err := New()
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	ctx := context.Background()

	err = store.SaveResponse(ctx, "hash", nil, nil, 0, nil, "cred-1")
	if err == nil {
		t.Error("expected error for nil response")
	}
}

func TestBboltStore_GetResponseByID_NotFound(t *testing.T) {
	store, err := New()
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	ctx := context.Background()

	_, err = store.GetResponseByID(ctx, "nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent response")
	}
}

func TestStoredEntry_JSON(t *testing.T) {
	// Test that StoredEntry marshals and unmarshals correctly
	entry := StoredEntry{
		ResponseID:   "test-id",
		APIKeyHash:   "hash123",
		Model:        "gpt-4",
		CreatedAt:    1234567890,
		ExpiresAt:    0,
		ResponseJSON: createTestResponse("inner-id"),
		Metadata:     map[string]string{"foo": "bar"},
	}

	data, err := json.Marshal(entry)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	var unmarshaled StoredEntry
	err = json.Unmarshal(data, &unmarshaled)
	if err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if unmarshaled.ResponseID != entry.ResponseID {
		t.Errorf("expected ResponseID %s, got %s", entry.ResponseID, unmarshaled.ResponseID)
	}

	if unmarshaled.APIKeyHash != entry.APIKeyHash {
		t.Errorf("expected APIKeyHash %s, got %s", entry.APIKeyHash, unmarshaled.APIKeyHash)
	}
}

func TestBboltStore_Expires(t *testing.T) {
	store, err := New()
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	ctx := context.Background()
	apiKeyHash := "expire-test-hash"
	resp := createTestResponse("expire-test-id")

	// Save with TTL of 2 seconds (to account for second-level precision)
	err = store.SaveResponse(ctx, apiKeyHash, resp, nil, 2, nil, "cred-1")
	if err != nil {
		t.Fatalf("failed to save response: %v", err)
	}

	// Get immediately - should work
	got, err := store.GetResponse(ctx, "expire-test-id", apiKeyHash)
	if err != nil {
		t.Fatalf("failed to get response before expiry: %v", err)
	}
	if got == nil {
		t.Fatal("response is nil before expiry")
	}

	// Wait for expiry - need to wait more than TTL seconds
	// Use 2.5 seconds to ensure the second counter has advanced
	time.Sleep(2500 * time.Millisecond)

	// Get after expiry - should fail
	_, err = store.GetResponse(ctx, "expire-test-id", apiKeyHash)
	if err == nil {
		t.Error("expected error after expiry")
	}
}

// TestBboltStore_CleanupExpired tests the CleanupExpired function
// Note: This test may be flaky due to second-level precision in expiry times
func TestBboltStore_CleanupExpired(t *testing.T) {
	store, err := New()
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	ctx := context.Background()
	apiKeyHash := "cleanup-test-hash"

	// Save a response with TTL=0 (no expiry)
	resp := createTestResponse("cleanup-test-id")
	err = store.SaveResponse(ctx, apiKeyHash, resp, nil, 0, nil, "cred-1")
	if err != nil {
		t.Fatalf("failed to save response: %v", err)
	}

	// Run cleanup - should not fail for non-expired entries
	err = store.CleanupExpired(ctx)
	if err != nil {
		t.Fatalf("cleanup failed: %v", err)
	}

	// Response should still exist
	got, err := store.GetResponse(ctx, "cleanup-test-id", apiKeyHash)
	if err != nil {
		t.Fatalf("response should exist after cleanup: %v", err)
	}
	if got == nil {
		t.Error("response is nil after cleanup")
	}
}

func TestBboltStore_MultipleResponses(t *testing.T) {
	store, err := New()
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	ctx := context.Background()
	apiKeyHash := "multi-test-hash"

	// Save multiple responses
	for i := 0; i < 10; i++ {
		resp := createTestResponse("multi-" + string(rune('0'+i)))
		err = store.SaveResponse(ctx, apiKeyHash, resp, nil, 0, nil, "cred-1")
		if err != nil {
			t.Fatalf("failed to save response %d: %v", i, err)
		}
	}

	// Get all responses
	for i := 0; i < 10; i++ {
		id := "multi-" + string(rune('0'+i))
		got, err := store.GetResponse(ctx, id, apiKeyHash)
		if err != nil {
			t.Fatalf("failed to get response %s: %v", id, err)
		}
		if got.ID != id {
			t.Errorf("expected ID %s, got %s", id, got.ID)
		}
	}
}
