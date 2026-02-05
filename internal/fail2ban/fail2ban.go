package fail2ban

import (
	"strconv"
	"sync"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/monitoring"
)

// ErrorCodeRule defines per-error-code ban rules
type ErrorCodeRule struct {
	Code        int
	MaxAttempts int
	BanDuration time.Duration // 0 means permanent ban
}

// banInfo stores information about a ban
type banInfo struct {
	banTime     time.Time
	banDuration time.Duration // 0 = permanent
	errorCode   int
}

type Fail2Ban struct {
	mu             sync.RWMutex
	maxAttempts    int
	banDuration    time.Duration // 0 means permanent ban
	errorCodes     map[int]bool
	errorCodeRules map[int]*ErrorCodeRule // Per-code rules
	failures       map[string]map[int]int // credential -> code -> count
	banned         map[string]*banInfo
	lastError      map[string]time.Time
}

func New(maxAttempts int, banDuration time.Duration, errorCodes []int) *Fail2Ban {
	errorCodesMap := make(map[int]bool)
	for _, code := range errorCodes {
		errorCodesMap[code] = true
	}

	return &Fail2Ban{
		maxAttempts:    maxAttempts,
		banDuration:    banDuration,
		errorCodes:     errorCodesMap,
		errorCodeRules: make(map[int]*ErrorCodeRule),
		failures:       make(map[string]map[int]int),
		banned:         make(map[string]*banInfo),
		lastError:      make(map[string]time.Time),
	}
}

// NewWithRules creates a Fail2Ban instance with per-error-code rules
func NewWithRules(maxAttempts int, banDuration time.Duration, errorCodes []int, rules []ErrorCodeRule) *Fail2Ban {
	f := New(maxAttempts, banDuration, errorCodes)

	// Apply per-code rules
	for i := range rules {
		f.errorCodeRules[rules[i].Code] = &rules[i]
	}

	return f
}

// getRule returns the rule for an error code, or the default rule
func (f *Fail2Ban) getRule(statusCode int) *ErrorCodeRule {
	if rule, exists := f.errorCodeRules[statusCode]; exists {
		return rule
	}

	// Return default rule
	return &ErrorCodeRule{
		Code:        statusCode,
		MaxAttempts: f.maxAttempts,
		BanDuration: f.banDuration,
	}
}

func (f *Fail2Ban) RecordResponse(credentialName string, statusCode int) {
	f.mu.Lock()
	defer f.mu.Unlock()

	// Check if already banned and still within ban duration
	if ban, exists := f.banned[credentialName]; exists {
		// Check for auto-unban of expired temporary bans
		if ban.banDuration > 0 && time.Since(ban.banTime) > ban.banDuration {
			// Ban has expired, remove it
			delete(f.banned, credentialName)
			// Reset all failure counters for this credential
			delete(f.failures, credentialName)
			// Record unban event
			monitoring.CredentialUnbanEvents.WithLabelValues(credentialName).Inc()
		} else {
			// Still banned
			return
		}
	}

	// Success resets all counters
	if statusCode == 200 {
		delete(f.failures, credentialName)
		return
	}

	// Only track configured error codes (if list is not empty)
	if len(f.errorCodes) > 0 && !f.errorCodes[statusCode] {
		return
	}

	// Get rule for this error code
	rule := f.getRule(statusCode)

	// Initialize failure map for this credential if needed
	if f.failures[credentialName] == nil {
		f.failures[credentialName] = make(map[int]int)
	}

	// Increment failure count for this specific error code
	f.failures[credentialName][statusCode]++
	f.lastError[credentialName] = time.Now().UTC()

	// Check if we've hit the max attempts for this error code
	if f.failures[credentialName][statusCode] >= rule.MaxAttempts {
		f.banned[credentialName] = &banInfo{
			banTime:     time.Now().UTC(),
			banDuration: rule.BanDuration,
			errorCode:   statusCode,
		}
		// Record ban event
		monitoring.CredentialBanEvents.WithLabelValues(credentialName, strconv.Itoa(statusCode)).Inc()
	}
}

func (f *Fail2Ban) IsBanned(credentialName string) bool {
	// First check with read lock
	f.mu.RLock()
	ban, exists := f.banned[credentialName]
	if !exists {
		f.mu.RUnlock()
		return false
	}

	// Permanent ban (banDuration = 0)
	if ban.banDuration == 0 {
		f.mu.RUnlock()
		return true
	}

	// Check if temporary ban has expired - store elapsed time to avoid timing issues
	elapsed := time.Since(ban.banTime)
	expired := elapsed > ban.banDuration
	f.mu.RUnlock()

	// If ban expired, upgrade to write lock and unban
	if expired {
		f.mu.Lock()
		defer f.mu.Unlock()
		// Double-check after acquiring write lock
		if ban, exists := f.banned[credentialName]; exists && ban.banDuration > 0 {
			if time.Since(ban.banTime) > ban.banDuration {
				delete(f.banned, credentialName)
				delete(f.failures, credentialName)
				return false
			}
		}
	}

	return !expired
}

func (f *Fail2Ban) GetFailureCount(credentialName string) int {
	f.mu.RLock()
	defer f.mu.RUnlock()

	codes := f.failures[credentialName]
	if codes == nil {
		return 0
	}

	// Return total failure count across all error codes
	total := 0
	for _, count := range codes {
		total += count
	}
	return total
}

func (f *Fail2Ban) Unban(credentialName string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if _, exists := f.banned[credentialName]; exists {
		delete(f.banned, credentialName)
		delete(f.failures, credentialName)
		// Record unban event only if credential was actually banned
		monitoring.CredentialUnbanEvents.WithLabelValues(credentialName).Inc()
	}
}

func (f *Fail2Ban) GetBannedCredentials() []string {
	f.mu.RLock()
	defer f.mu.RUnlock()

	banned := make([]string, 0, len(f.banned))
	for name := range f.banned {
		banned = append(banned, name)
	}
	return banned
}

// GetBannedCount returns the count of banned credentials without allocating a slice
func (f *Fail2Ban) GetBannedCount() int {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return len(f.banned)
}
