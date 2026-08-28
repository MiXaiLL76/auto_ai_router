// Package responsestore persists Responses API state in bbolt or Redis.
package responsestore

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/valkey-io/valkey-go"

	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
)

// redisStore implements Store using Valkey/Redis.
// Each response is stored as a JSON string under key "{prefix}response:{id}".
// TTL is applied natively via Redis EXPIRE, so CleanupExpired is a no-op.
type redisStore struct {
	client    valkey.Client
	keyPrefix string
}

// NewRedis returns a Redis-backed Store.
// client must already be connected (e.g. obtained via ratelimit.NewValkeyClient).
// keyPrefix is prepended to every key (e.g. "rl:" → keys look like "rl:response:{id}").
func NewRedis(client valkey.Client, keyPrefix string) Store {
	return &redisStore{client: client, keyPrefix: keyPrefix}
}

func (s *redisStore) responseKey(id string) string {
	return s.keyPrefix + "response:" + id
}

func (s *redisStore) Close() error {
	// The client is shared; caller is responsible for closing it.
	return nil
}

func (s *redisStore) SaveResponse(
	ctx context.Context,
	apiKeyHash string,
	resp *responses.Response,
	metadata map[string]string,
	ttlSeconds int,
	accumulatedInput json.RawMessage,
	credentialName string,
) error {
	if resp == nil {
		return fmt.Errorf("responsestore: response is nil")
	}

	entry := StoredEntry{
		ResponseID:       resp.ID,
		APIKeyHash:       apiKeyHash,
		CredentialName:   credentialName,
		Model:            resp.Model,
		CreatedAt:        resp.CreatedAt,
		ExpiresAt:        0,
		ResponseJSON:     resp,
		Metadata:         metadata,
		AccumulatedInput: accumulatedInput,
	}
	if ttlSeconds > 0 {
		entry.ExpiresAt = time.Now().Unix() + int64(ttlSeconds)
	}

	data, err := json.Marshal(entry)
	if err != nil {
		return fmt.Errorf("responsestore: marshal failed: %w", err)
	}

	key := s.responseKey(resp.ID)

	var cmd valkey.Completed
	if ttlSeconds > 0 {
		cmd = s.client.B().Set().Key(key).Value(string(data)).Ex(time.Duration(ttlSeconds) * time.Second).Build()
	} else {
		cmd = s.client.B().Set().Key(key).Value(string(data)).Build()
	}

	if err := s.client.Do(ctx, cmd).Error(); err != nil {
		return fmt.Errorf("responsestore: redis SET failed: %w", err)
	}
	return nil
}

func (s *redisStore) GetResponse(ctx context.Context, responseID, apiKeyHash string) (*responses.Response, error) {
	entry, err := s.getEntry(ctx, responseID)
	if err != nil {
		return nil, err
	}
	if entry.APIKeyHash != apiKeyHash {
		return nil, fmt.Errorf("responsestore: unauthorized")
	}
	return entry.ResponseJSON, nil
}

func (s *redisStore) GetEntry(ctx context.Context, responseID, apiKeyHash string) (*StoredEntry, error) {
	entry, err := s.getEntry(ctx, responseID)
	if err != nil {
		return nil, err
	}
	if entry.APIKeyHash != apiKeyHash {
		return nil, fmt.Errorf("responsestore: unauthorized")
	}
	return entry, nil
}

func (s *redisStore) GetResponseByID(ctx context.Context, responseID string) (*responses.Response, error) {
	entry, err := s.getEntry(ctx, responseID)
	if err != nil {
		return nil, err
	}
	return entry.ResponseJSON, nil
}

func (s *redisStore) getEntry(ctx context.Context, responseID string) (*StoredEntry, error) {
	data, err := s.client.Do(ctx, s.client.B().Get().Key(s.responseKey(responseID)).Build()).AsBytes()
	if err != nil {
		if valkey.IsValkeyNil(err) {
			return nil, fmt.Errorf("responsestore: not found: %s", responseID)
		}
		return nil, fmt.Errorf("responsestore: redis GET failed: %w", err)
	}

	var entry StoredEntry
	if err := json.Unmarshal(data, &entry); err != nil {
		return nil, fmt.Errorf("responsestore: unmarshal failed: %w", err)
	}

	// ExpiresAt check is a safety net for entries stored without Redis TTL (ttl==0).
	// Entries with ttl>0 are expired by Redis automatically before GET returns them.
	if entry.ExpiresAt > 0 && time.Now().Unix() > entry.ExpiresAt {
		return nil, fmt.Errorf("responsestore: expired: %s", responseID)
	}
	return &entry, nil
}

// CleanupExpired is a no-op for the Redis backend.
// Redis expires keys natively via the TTL set in SaveResponse.
func (s *redisStore) CleanupExpired(_ context.Context) error { return nil }
