package fail2ban

import (
	"sync"
	"time"
)

type Fail2Ban struct {
	mu          sync.RWMutex
	maxAttempts int
	banDuration time.Duration // 0 means permanent ban
	errorCodes  map[int]bool
	failures    map[string]int
	banned      map[string]bool
	banTime     map[string]time.Time
	lastError   map[string]time.Time
}

func New(maxAttempts int, banDuration time.Duration, errorCodes []int) *Fail2Ban {
	errorCodesMap := make(map[int]bool)
	for _, code := range errorCodes {
		errorCodesMap[code] = true
	}

	return &Fail2Ban{
		maxAttempts: maxAttempts,
		banDuration: banDuration,
		errorCodes:  errorCodesMap,
		failures:    make(map[string]int),
		banned:      make(map[string]bool),
		banTime:     make(map[string]time.Time),
		lastError:   make(map[string]time.Time),
	}
}

func (f *Fail2Ban) RecordResponse(credentialName string, statusCode int) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if f.banned[credentialName] {
		return
	}

	if statusCode == 200 {
		f.failures[credentialName] = 0
		return
	}

	if !f.errorCodes[statusCode] {
		return
	}

	f.failures[credentialName]++
	f.lastError[credentialName] = time.Now()

	if f.failures[credentialName] >= f.maxAttempts {
		f.banned[credentialName] = true
		f.banTime[credentialName] = time.Now()
	}
}

func (f *Fail2Ban) IsBanned(credentialName string) bool {
	// First check with read lock
	f.mu.RLock()
	if !f.banned[credentialName] {
		f.mu.RUnlock()
		return false
	}

	// Permanent ban (banDuration = 0)
	if f.banDuration == 0 {
		f.mu.RUnlock()
		return true
	}

	// Check if ban has expired
	banTime, exists := f.banTime[credentialName]
	if !exists {
		f.mu.RUnlock()
		return false
	}

	expired := time.Since(banTime) >= f.banDuration
	f.mu.RUnlock()

	// If ban expired, upgrade to write lock and unban
	if expired {
		f.mu.Lock()
		defer f.mu.Unlock()
		// Double-check after acquiring write lock
		if f.banned[credentialName] {
			banTime, exists := f.banTime[credentialName]
			if exists && time.Since(banTime) >= f.banDuration {
				delete(f.banned, credentialName)
				delete(f.banTime, credentialName)
				f.failures[credentialName] = 0
				return false
			}
		}
	}

	return !expired
}

func (f *Fail2Ban) GetFailureCount(credentialName string) int {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.failures[credentialName]
}

func (f *Fail2Ban) Unban(credentialName string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	delete(f.banned, credentialName)
	f.failures[credentialName] = 0
}

func (f *Fail2Ban) GetBannedCredentials() []string {
	f.mu.RLock()
	defer f.mu.RUnlock()

	var banned []string
	for name := range f.banned {
		banned = append(banned, name)
	}
	return banned
}
