package fail2ban

import (
	"io"
	"log/slog"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/utils"
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
	reason      string
}

// BanPair represents a banned credential+model pair with ban details
type BanPair struct {
	Credential      string
	Model           string
	ErrorCode       int
	ErrorCodeCounts map[int]int
	BanTime         time.Time
	BanDuration     time.Duration
	BanUntil        time.Time
	Reason          string
}

type Fail2Ban struct {
	mu             sync.RWMutex
	maxAttempts    int
	banDuration    time.Duration // 0 means permanent ban
	errorCodes     map[int]bool
	errorCodeRules map[int]*ErrorCodeRule // Per-code rules
	failures       map[string]map[int]int // banKey -> code -> count
	banned         map[string]*banInfo    // banKey -> banInfo
	lastError      map[string]time.Time   // banKey -> last error time
	logger         *slog.Logger
}

// SetLogger sets the logger used for ban/unban events.
// Without it ban events are only visible as Prometheus metrics.
func (f *Fail2Ban) SetLogger(logger *slog.Logger) {
	if logger != nil {
		f.logger = logger
	}
}

// banKey creates a composite key from credential name and model ID.
// Format: "credentialName|modelID"
func banKey(credentialName, modelID string) string {
	return credentialName + "|" + modelID
}

// parseBanKey splits a composite ban key back into credential and model.
func parseBanKey(key string) (credential, model string) {
	parts := strings.SplitN(key, "|", 2)
	if len(parts) == 2 {
		return parts[0], parts[1]
	}
	return key, ""
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
		logger:         slog.New(slog.NewTextHandler(io.Discard, nil)),
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

func (f *Fail2Ban) RecordResponse(credentialName, modelID string, statusCode int) {
	key := banKey(credentialName, modelID)

	f.mu.Lock()
	defer f.mu.Unlock()

	// Check if already banned and still within ban duration
	if ban, exists := f.banned[key]; exists {
		// Check for auto-unban of expired temporary bans
		if ban.banDuration > 0 && time.Since(ban.banTime) > ban.banDuration {
			// Ban has expired, remove it
			delete(f.banned, key)
			// Reset all failure counters for this pair
			delete(f.failures, key)
			// Record unban event
			monitoring.CredentialUnbanEvents.WithLabelValues(credentialName, modelID).Inc()
			f.logger.Info("Credential unbanned (ban expired)",
				"credential", credentialName,
				"model", modelID,
				"ban_duration", ban.banDuration)
		} else {
			// Still banned
			return
		}
	}

	// Success resets all counters for this specific cred+model pair
	if statusCode >= 200 && statusCode < 300 {
		delete(f.failures, key)
		return
	}

	// Only track configured error codes (if list is not empty)
	if len(f.errorCodes) > 0 && !f.errorCodes[statusCode] {
		return
	}

	// Get rule for this error code
	rule := f.getRule(statusCode)

	// Initialize failure map for this pair if needed
	if f.failures[key] == nil {
		f.failures[key] = make(map[int]int)
	}

	// Increment failure count for this specific error code
	f.failures[key][statusCode]++
	f.lastError[key] = utils.NowUTC()

	// Check if we've hit the max attempts for this error code
	if f.failures[key][statusCode] >= rule.MaxAttempts {
		f.banned[key] = &banInfo{
			banTime:     utils.NowUTC(),
			banDuration: rule.BanDuration,
			errorCode:   statusCode,
		}
		// Record ban event
		monitoring.CredentialBanEvents.WithLabelValues(credentialName, modelID, strconv.Itoa(statusCode)).Inc()
		// Losing a credential directly affects routing capacity and is the first
		// thing to check when debugging "No credentials available" — log at ERROR.
		f.logger.Error("Credential banned by fail2ban",
			"error_code", statusCode,
			"credential", credentialName,
			"model", modelID,
			"failures", f.failures[key][statusCode],
			"max_attempts", rule.MaxAttempts,
			"ban_duration", rule.BanDuration,
			"permanent", rule.BanDuration == 0)
	}
}

// BanUntil immediately bans a credential+model pair until the absolute deadline.
// It bypasses attempt thresholds and configured error-code filters. An existing
// permanent or later-expiring ban is never shortened.
func (f *Fail2Ban) BanUntil(credentialName, modelID string, statusCode int, until time.Time, reason string) {
	now := utils.NowUTC()
	until = until.UTC()
	if !until.After(now) {
		return
	}

	key := banKey(credentialName, modelID)

	f.mu.Lock()
	defer f.mu.Unlock()

	if current, exists := f.banned[key]; exists {
		if current.banDuration == 0 {
			return
		}
		currentUntil := current.banTime.Add(current.banDuration)
		if !currentUntil.Before(until) {
			return
		}
	}

	if f.failures[key] == nil {
		f.failures[key] = make(map[int]int)
	}
	f.failures[key][statusCode]++
	f.lastError[key] = now

	duration := until.Sub(now)
	f.banned[key] = &banInfo{
		banTime:     now,
		banDuration: duration,
		errorCode:   statusCode,
		reason:      reason,
	}

	monitoring.CredentialBanEvents.WithLabelValues(credentialName, modelID, strconv.Itoa(statusCode)).Inc()
	f.logger.Error("Credential and model banned until provider quota retry",
		"error_code", statusCode,
		"credential", credentialName,
		"model", modelID,
		"provider_error", reason,
		"ban_until", until,
		"ban_duration", duration)
}

func (f *Fail2Ban) IsBanned(credentialName, modelID string) bool {
	key := banKey(credentialName, modelID)

	// First check with read lock
	f.mu.RLock()
	ban, exists := f.banned[key]
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
		// Re-check after acquiring write lock — a new ban may have been added in the gap
		ban, exists = f.banned[key]
		if !exists {
			return false
		}
		if ban.banDuration == 0 {
			return true
		}
		if time.Since(ban.banTime) > ban.banDuration {
			delete(f.banned, key)
			delete(f.failures, key)
			monitoring.CredentialUnbanEvents.WithLabelValues(credentialName, modelID).Inc()
			return false
		}
		// Ban is still active (new ban was added during lock upgrade)
		return true
	}

	return true
}

func (f *Fail2Ban) GetFailureCount(credentialName, modelID string) int {
	key := banKey(credentialName, modelID)

	f.mu.RLock()
	defer f.mu.RUnlock()

	codes := f.failures[key]
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

func (f *Fail2Ban) Unban(credentialName, modelID string) {
	key := banKey(credentialName, modelID)

	f.mu.Lock()
	defer f.mu.Unlock()
	if _, exists := f.banned[key]; exists {
		delete(f.banned, key)
		delete(f.failures, key)
		// Record unban event only if pair was actually banned
		monitoring.CredentialUnbanEvents.WithLabelValues(credentialName, modelID).Inc()
		f.logger.Info("Credential unbanned manually",
			"credential", credentialName, "model", modelID)
	}
}

// UnbanCredential unbans ALL models for a given credential
func (f *Fail2Ban) UnbanCredential(credentialName string) {
	f.mu.Lock()
	defer f.mu.Unlock()

	prefix := credentialName + "|"
	for key := range f.banned {
		if strings.HasPrefix(key, prefix) {
			_, model := parseBanKey(key)
			delete(f.banned, key)
			delete(f.failures, key)
			monitoring.CredentialUnbanEvents.WithLabelValues(credentialName, model).Inc()
			f.logger.Info("Credential unbanned manually",
				"credential", credentialName, "model", model)
		}
	}
}

// HasAnyBan returns true if any model on the given credential is currently banned
func (f *Fail2Ban) HasAnyBan(credentialName string) bool {
	f.mu.RLock()
	defer f.mu.RUnlock()

	prefix := credentialName + "|"
	for key, ban := range f.banned {
		if strings.HasPrefix(key, prefix) {
			// Check if this ban is still active
			if ban.banDuration == 0 || time.Since(ban.banTime) <= ban.banDuration {
				return true
			}
		}
	}
	return false
}

// GetBannedModelsForCredential returns model IDs that are currently banned for a credential
func (f *Fail2Ban) GetBannedModelsForCredential(credentialName string) []string {
	f.mu.RLock()
	defer f.mu.RUnlock()

	prefix := credentialName + "|"
	var models []string
	for key, ban := range f.banned {
		if strings.HasPrefix(key, prefix) {
			if ban.banDuration == 0 || time.Since(ban.banTime) <= ban.banDuration {
				_, model := parseBanKey(key)
				models = append(models, model)
			}
		}
	}
	return models
}

// GetBannedPairs returns all currently banned credential+model pairs
func (f *Fail2Ban) GetBannedPairs() []BanPair {
	f.mu.RLock()
	defer f.mu.RUnlock()

	pairs := make([]BanPair, 0, len(f.banned))
	for key, ban := range f.banned {
		credential, model := parseBanKey(key)
		counts := make(map[int]int)
		if codeCounts, ok := f.failures[key]; ok {
			for code, count := range codeCounts {
				counts[code] = count
			}
		}
		pairs = append(pairs, BanPair{
			Credential:      credential,
			Model:           model,
			ErrorCode:       ban.errorCode,
			ErrorCodeCounts: counts,
			BanTime:         ban.banTime,
			BanDuration:     ban.banDuration,
			BanUntil:        banUntil(ban),
			Reason:          ban.reason,
		})
	}
	return pairs
}

func banUntil(ban *banInfo) time.Time {
	if ban == nil || ban.banDuration == 0 {
		return time.Time{}
	}
	return ban.banTime.Add(ban.banDuration).UTC()
}

// GetBannedCount returns the count of currently active (non-expired) banned credential+model pairs
func (f *Fail2Ban) GetBannedCount() int {
	f.mu.RLock()
	defer f.mu.RUnlock()

	count := 0
	for _, ban := range f.banned {
		if ban.banDuration == 0 || time.Since(ban.banTime) <= ban.banDuration {
			count++
		}
	}
	return count
}
