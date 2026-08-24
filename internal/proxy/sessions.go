package proxy

import (
	"context"
	"sync"
	"time"
)

type sessionKey struct {
	sessionID string
	modelID   string
}

type SessionEntry struct {
	CredentialName string
	LastAccess     time.Time
}

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

	now := s.now()
	if now.Sub(entry.LastAccess) > s.ttl {
		delete(s.entries, key)
		return "", false
	}

	entry.LastAccess = now
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
