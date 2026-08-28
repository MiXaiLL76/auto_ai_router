package responsestore

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	bolt "go.etcd.io/bbolt"

	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
)

// StoredEntry is the value persisted per response (shared by all backends).
type StoredEntry struct {
	ResponseID       string              `json:"response_id"`
	APIKeyHash       string              `json:"api_key_hash"`
	CredentialName   string              `json:"credential_name,omitempty"`
	Model            string              `json:"model"`
	CreatedAt        int64               `json:"created_at"`
	ExpiresAt        int64               `json:"expires_at"` // 0 = never expires
	ResponseJSON     *responses.Response `json:"response_json"`
	Metadata         map[string]string   `json:"metadata"`
	AccumulatedInput json.RawMessage     `json:"accumulated_input,omitempty"` // full input context for multi-turn
}

// Store is the pluggable response storage interface.
// Implementations: bboltStore (local), redisStore (distributed).
type Store interface {
	// SaveResponse persists a response. ttlSeconds==0 means no expiry.
	// accumulatedInput is the full normalised input array; may be nil.
	SaveResponse(
		ctx context.Context,
		apiKeyHash string,
		resp *responses.Response,
		metadata map[string]string,
		ttlSeconds int,
		accumulatedInput json.RawMessage,
		credentialName string,
	) error

	// GetResponse returns the response if it exists, is not expired, and the
	// apiKeyHash matches the one used at save time.
	GetResponse(ctx context.Context, responseID, apiKeyHash string) (*responses.Response, error)

	// GetEntry returns the full StoredEntry including AccumulatedInput.
	// Same ownership and expiry checks as GetResponse.
	GetEntry(ctx context.Context, responseID, apiKeyHash string) (*StoredEntry, error)

	// GetResponseByID returns a stored response by ID without an ownership
	// check (intended for master-key requests).
	GetResponseByID(ctx context.Context, responseID string) (*responses.Response, error)

	// CleanupExpired removes all entries whose TTL has elapsed.
	// Redis-backed stores may treat this as a no-op (TTL handled natively).
	CleanupExpired(ctx context.Context) error

	// Close releases any resources held by the store.
	Close() error
}

// ── bbolt backend ─────────────────────────────────────────────────────────────

var responseBucket = []byte("responses")

type bboltStore struct {
	db *bolt.DB
}

// New opens (or creates) the bbolt database and returns a Store.
// Database location: /data/auto_ai_router/responses.db if /data/auto_ai_router exists,
// otherwise /tmp/auto_ai_router/responses.db.
func New() (Store, error) {
	dir := "/data/auto_ai_router"
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		dir = "/tmp/auto_ai_router"
	}
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return nil, fmt.Errorf("responsestore: failed to create dir %s: %w", dir, err)
	}
	dbPath := filepath.Join(dir, "responses.db")
	db, err := bolt.Open(dbPath, 0o600, &bolt.Options{Timeout: 5 * time.Second})
	if err != nil {
		return nil, fmt.Errorf("responsestore: failed to open db %s: %w", dbPath, err)
	}
	if err := db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists(responseBucket)
		return err
	}); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("responsestore: failed to create bucket: %w", err)
	}
	return &bboltStore{db: db}, nil
}

func (s *bboltStore) Close() error { return s.db.Close() }

func (s *bboltStore) SaveResponse(
	_ context.Context,
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
	return s.db.Update(func(tx *bolt.Tx) error {
		return tx.Bucket(responseBucket).Put([]byte(resp.ID), data)
	})
}

func (s *bboltStore) GetResponse(_ context.Context, responseID, apiKeyHash string) (*responses.Response, error) {
	entry, err := s.getEntry(responseID)
	if err != nil {
		return nil, err
	}
	if entry.APIKeyHash != apiKeyHash {
		return nil, fmt.Errorf("responsestore: unauthorized")
	}
	return entry.ResponseJSON, nil
}

func (s *bboltStore) GetEntry(_ context.Context, responseID, apiKeyHash string) (*StoredEntry, error) {
	entry, err := s.getEntry(responseID)
	if err != nil {
		return nil, err
	}
	if entry.APIKeyHash != apiKeyHash {
		return nil, fmt.Errorf("responsestore: unauthorized")
	}
	return entry, nil
}

func (s *bboltStore) GetResponseByID(_ context.Context, responseID string) (*responses.Response, error) {
	entry, err := s.getEntry(responseID)
	if err != nil {
		return nil, err
	}
	return entry.ResponseJSON, nil
}

func (s *bboltStore) getEntry(responseID string) (*StoredEntry, error) {
	var entry StoredEntry
	if err := s.db.View(func(tx *bolt.Tx) error {
		v := tx.Bucket(responseBucket).Get([]byte(responseID))
		if v == nil {
			return fmt.Errorf("responsestore: not found: %s", responseID)
		}
		return json.Unmarshal(v, &entry)
	}); err != nil {
		return nil, err
	}
	if entry.ExpiresAt > 0 && time.Now().Unix() >= entry.ExpiresAt {
		return nil, fmt.Errorf("responsestore: expired: %s", responseID)
	}
	return &entry, nil
}

func (s *bboltStore) CleanupExpired(_ context.Context) error {
	now := time.Now().Unix()
	return s.db.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket(responseBucket)
		var toDelete [][]byte
		_ = b.ForEach(func(k, v []byte) error {
			var entry StoredEntry
			if json.Unmarshal(v, &entry) == nil && entry.ExpiresAt > 0 && now >= entry.ExpiresAt {
				toDelete = append(toDelete, append([]byte(nil), k...))
			}
			return nil
		})
		for _, k := range toDelete {
			if err := b.Delete(k); err != nil {
				return err
			}
		}
		return nil
	})
}
