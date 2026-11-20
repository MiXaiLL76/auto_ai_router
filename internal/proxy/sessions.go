package proxy

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/valkey-io/valkey-go"
)

// sessionBindingStore keeps session -> credential bindings for session-sticky
// routing. The in-memory SessionStore is local to one router replica; the
// Redis-backed store shares bindings between all replicas of a deployment, so a
// follow-up turn that lands on another pod still reaches the same credential
// (and the provider's prompt cache on it).
type sessionBindingStore interface {
	Get(sessionID, modelID string) (string, bool)
	Set(sessionID, modelID, credentialName string)
	Delete(sessionID, modelID string)
	Len() int
	StartCleanup(ctx context.Context, interval time.Duration)
}

type sessionKey struct {
	sessionID string
	modelID   string
}

type SessionEntry struct {
	CredentialName string
	LastAccess     time.Time
}

// SessionStore is the process-local binding store.
//
// The TTL is refreshed by Set only, i.e. when a credential served a turn of the
// session. Get does not extend it, so a binding whose credential is only being
// waited for (rate limited) expires one TTL after it last served.
type SessionStore struct {
	mu      sync.Mutex
	entries map[sessionKey]*SessionEntry
	ttl     time.Duration
	now     func() time.Time
}

func NewSessionStore(ttl time.Duration) *SessionStore {
	return &SessionStore{
		entries: make(map[sessionKey]*SessionEntry),
		ttl:     ttl,
		now:     time.Now,
	}
}

func (s *SessionStore) Get(sessionID, modelID string) (string, bool) {
	key := sessionKey{sessionID: sessionID, modelID: modelID}

	s.mu.Lock()
	defer s.mu.Unlock()

	entry, ok := s.entries[key]
	if !ok {
		return "", false
	}

	if s.now().Sub(entry.LastAccess) > s.ttl {
		delete(s.entries, key)
		return "", false
	}

	return entry.CredentialName, true
}

func (s *SessionStore) Set(sessionID, modelID, credentialName string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.entries[sessionKey{sessionID: sessionID, modelID: modelID}] = &SessionEntry{
		CredentialName: credentialName,
		LastAccess:     s.now(),
	}
}

func (s *SessionStore) Delete(sessionID, modelID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.entries, sessionKey{sessionID: sessionID, modelID: modelID})
}

func (s *SessionStore) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(s.now())
	return len(s.entries)
}

func (s *SessionStore) StartCleanup(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.mu.Lock()
			s.cleanupExpiredLocked(s.now())
			s.mu.Unlock()
		}
	}
}

func (s *SessionStore) cleanupExpiredLocked(now time.Time) {
	for key, entry := range s.entries {
		if now.Sub(entry.LastAccess) > s.ttl {
			delete(s.entries, key)
		}
	}
}

const (
	// redisSessionCommandTimeout bounds one binding command on the request path.
	redisSessionCommandTimeout = 200 * time.Millisecond
	// redisSessionRetryAfter is how long the store stays on the replica-local copy
	// after a Redis error, so an outage costs one timeout per window, not one per request.
	redisSessionRetryAfter = 5 * time.Second
)

// sessionKV is the part of Redis the binding store needs; valkeySessionKV is the
// production implementation, tests substitute a fake.
type sessionKV interface {
	Get(ctx context.Context, key string) (value string, found bool, err error)
	Set(ctx context.Context, key, value string, ttl time.Duration) error
	Del(ctx context.Context, key string) error
}

type valkeySessionKV struct{ client valkey.Client }

func (kv valkeySessionKV) Get(ctx context.Context, key string) (string, bool, error) {
	value, err := kv.client.Do(ctx, kv.client.B().Get().Key(key).Build()).ToString()
	if err != nil {
		if valkey.IsValkeyNil(err) {
			return "", false, nil
		}
		return "", false, err
	}
	return value, value != "", nil
}

func (kv valkeySessionKV) Set(ctx context.Context, key, value string, ttl time.Duration) error {
	return kv.client.Do(ctx, kv.client.B().Set().Key(key).Value(value).Px(ttl).Build()).Error()
}

func (kv valkeySessionKV) Del(ctx context.Context, key string) error {
	return kv.client.Do(ctx, kv.client.B().Del().Key(key).Build()).Error()
}

// redisSessionStore shares bindings through Redis/Valkey (keys
// "{prefix}session:{sha256(model, session)}", native PX TTL) and mirrors every
// write into a local SessionStore. After a Redis error it serves from the local
// copy for redisSessionRetryAfter before trying Redis again.
type redisSessionStore struct {
	kv        sessionKV
	keyPrefix string
	ttl       time.Duration
	local     *SessionStore
	logger    *slog.Logger
	now       func() time.Time
	retryAt   atomic.Int64 // unix nanos; Redis is skipped until then
	degraded  atomic.Bool  // true between the first error and the next success
}

func newRedisSessionStore(client valkey.Client, keyPrefix string, ttl time.Duration, logger *slog.Logger) *redisSessionStore {
	return newRedisSessionStoreWithKV(valkeySessionKV{client: client}, keyPrefix, ttl, logger)
}

func newRedisSessionStoreWithKV(kv sessionKV, keyPrefix string, ttl time.Duration, logger *slog.Logger) *redisSessionStore {
	return &redisSessionStore{
		kv:        kv,
		keyPrefix: keyPrefix,
		ttl:       ttl,
		local:     NewSessionStore(ttl),
		logger:    logger,
		now:       time.Now,
	}
}

// key hashes the client-supplied session ID so key length and charset do not
// depend on what clients send.
func (s *redisSessionStore) key(sessionID, modelID string) string {
	sum := sha256.Sum256([]byte(modelID + "\x00" + sessionID))
	return s.keyPrefix + "session:" + hex.EncodeToString(sum[:16])
}

func (s *redisSessionStore) redisAvailable() bool {
	return s.now().UnixNano() >= s.retryAt.Load()
}

func (s *redisSessionStore) fail(op string, err error) {
	s.retryAt.Store(s.now().Add(redisSessionRetryAfter).UnixNano())
	if s.degraded.CompareAndSwap(false, true) && s.logger != nil {
		s.logger.Warn("Session-sticky routing: Redis unavailable, using replica-local bindings",
			"operation", op, "error", err, "retry_after", redisSessionRetryAfter)
	}
}

func (s *redisSessionStore) succeed() {
	if s.degraded.CompareAndSwap(true, false) && s.logger != nil {
		s.logger.Info("Session-sticky routing: Redis reachable again, bindings are shared")
	}
}

func (s *redisSessionStore) Get(sessionID, modelID string) (string, bool) {
	if !s.redisAvailable() {
		return s.local.Get(sessionID, modelID)
	}
	ctx, cancel := context.WithTimeout(context.Background(), redisSessionCommandTimeout)
	defer cancel()
	cred, found, err := s.kv.Get(ctx, s.key(sessionID, modelID))
	if err != nil {
		s.fail("get", err)
		return s.local.Get(sessionID, modelID)
	}
	s.succeed()
	return cred, found
}

func (s *redisSessionStore) Set(sessionID, modelID, credentialName string) {
	s.local.Set(sessionID, modelID, credentialName)
	if !s.redisAvailable() {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), redisSessionCommandTimeout)
	defer cancel()
	if err := s.kv.Set(ctx, s.key(sessionID, modelID), credentialName, s.ttl); err != nil {
		s.fail("set", err)
		return
	}
	s.succeed()
}

func (s *redisSessionStore) Delete(sessionID, modelID string) {
	s.local.Delete(sessionID, modelID)
	if !s.redisAvailable() {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), redisSessionCommandTimeout)
	defer cancel()
	if err := s.kv.Del(ctx, s.key(sessionID, modelID)); err != nil {
		s.fail("delete", err)
		return
	}
	s.succeed()
}

// Len reports the replica-local mirror only; the shared keyspace is not scanned.
// It is used by tests and is not a count of all bindings of the deployment.
func (s *redisSessionStore) Len() int { return s.local.Len() }

func (s *redisSessionStore) StartCleanup(ctx context.Context, interval time.Duration) {
	s.local.StartCleanup(ctx, interval)
}
