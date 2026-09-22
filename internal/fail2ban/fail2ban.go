// Package fail2ban tracks abusive clients and bans them for a cooldown period.
package fail2ban

import (
	"io"
	"log/slog"
	"sort"
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

const (
	// WildcardModel is the model part of a ban key that bans every model of a
	// credential, including models the router only learns about later (e.g. the
	// ones a proxy/AIR credential discovers from its upstream /health).
	WildcardModel = "*"

	// OriginFail2Ban marks bans created automatically from upstream errors.
	OriginFail2Ban = "fail2ban"
	// OriginAdmin marks bans created by an operator through the admin API.
	OriginAdmin = "admin"

	// adminErrorCode is the service error code recorded for admin bans; it is
	// not an HTTP status and only shows up in metrics and /api/bans.
	adminErrorCode = 0
)

// banInfo stores information about a ban
type banInfo struct {
	banTime     time.Time
	banDuration time.Duration // 0 = permanent
	errorCode   int
	reason      string
	origin      string // OriginFail2Ban (default) or OriginAdmin
}

func (b *banInfo) originOrDefault() string {
	if b.origin == "" {
		return OriginFail2Ban
	}
	return b.origin
}

// active reports whether the ban has not expired yet. Permanent bans never expire.
func (b *banInfo) active() bool {
	return b.banDuration == 0 || time.Since(b.banTime) <= b.banDuration
}

// BanPair represents a banned credential+model pair with ban details
type BanPair struct {
	Credential      string
	Model           string
	ErrorCode       int
	ErrorCodeCounts map[int]int
	BanTime         time.Time
	BanDuration     time.Duration
	BanUntil        time.Time // zero for a permanent ban
	Reason          string
	Origin          string // OriginFail2Ban or OriginAdmin
}

type Fail2Ban struct {
	mu             sync.RWMutex
	maxAttempts    int
	banDuration    time.Duration // 0 means permanent ban
	errorCodes     map[int]bool
	errorCodeRules map[int]*ErrorCodeRule // Per-code rules

	// Per-credential overrides. A credential name present here fully replaces
	// the global errorCodes/errorCodeRules for that credential; a credential
	// with no entry keeps using the global settings above. This exists
	// because "what counts as a failure" is not universal: a reseller/
	// aggregator upstream may signal a blocked account with a plain 400
	// instead of 429/5xx, and banning on 400 globally would also ban any
	// credential+model pair on an ordinary bad client request.
	credentialErrorCodes     map[string]map[int]bool
	credentialErrorCodeRules map[string]map[int]*ErrorCodeRule

	failures  map[string]map[int]int // banKey -> code -> count
	banned    map[string]*banInfo    // banKey -> banInfo
	lastError map[string]time.Time   // banKey -> last error time
	logger    *slog.Logger
}

// CredentialOverride replaces the global error_codes/error_code_rules for one
// specific credential (matched by the credentialName passed to
// RecordResponse). See SetCredentialOverrides.
type CredentialOverride struct {
	ErrorCodes     []int
	ErrorCodeRules []ErrorCodeRule
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

// SetCredentialOverrides installs per-credential error-code overrides,
// replacing any previously installed ones. Call it once during startup,
// before serving traffic. A credential with no entry in overrides keeps
// using the global error_codes/error_code_rules unchanged.
func (f *Fail2Ban) SetCredentialOverrides(overrides map[string]CredentialOverride) {
	credErrorCodes := make(map[string]map[int]bool, len(overrides))
	credErrorCodeRules := make(map[string]map[int]*ErrorCodeRule, len(overrides))
	for name, ov := range overrides {
		if len(ov.ErrorCodes) > 0 {
			codes := make(map[int]bool, len(ov.ErrorCodes))
			for _, code := range ov.ErrorCodes {
				codes[code] = true
			}
			credErrorCodes[name] = codes
		}
		if len(ov.ErrorCodeRules) > 0 {
			rules := make(map[int]*ErrorCodeRule, len(ov.ErrorCodeRules))
			for i := range ov.ErrorCodeRules {
				rule := ov.ErrorCodeRules[i]
				rules[rule.Code] = &rule
			}
			credErrorCodeRules[name] = rules
		}
	}

	f.mu.Lock()
	defer f.mu.Unlock()
	f.credentialErrorCodes = credErrorCodes
	f.credentialErrorCodeRules = credErrorCodeRules
}

// getRule returns the rule for an error code, preferring a per-credential
// override, then the global per-code rule, then the default rule.
func (f *Fail2Ban) getRule(credentialName string, statusCode int) *ErrorCodeRule {
	if rules, ok := f.credentialErrorCodeRules[credentialName]; ok {
		if rule, exists := rules[statusCode]; exists {
			return rule
		}
	}

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

	// Only track configured error codes (if list is not empty). A
	// per-credential override, if present, fully replaces the global set.
	errorCodes := f.errorCodes
	if credCodes, ok := f.credentialErrorCodes[credentialName]; ok {
		errorCodes = credCodes
	}
	if len(errorCodes) > 0 && !errorCodes[statusCode] {
		return
	}

	// Get rule for this error code
	rule := f.getRule(credentialName, statusCode)

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
		origin:      OriginFail2Ban,
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

// IsBanned reports whether credentialName is banned for modelID, either by a
// ban on that exact pair or by a credential-wide wildcard ban (WildcardModel).
func (f *Fail2Ban) IsBanned(credentialName, modelID string) bool {
	if f.isBannedKey(credentialName, modelID) {
		return true
	}
	return modelID != WildcardModel && f.isBannedKey(credentialName, WildcardModel)
}

func (f *Fail2Ban) isBannedKey(credentialName, modelID string) bool {
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
		f.removeBanLocked(key, credentialName, modelID)
	}
}

// UnbanCredential unbans ALL models for a given credential
func (f *Fail2Ban) UnbanCredential(credentialName string) {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.removeCredentialBansLocked(credentialName)
}

// removeBanLocked drops one ban and its failure counters. Caller holds f.mu.
func (f *Fail2Ban) removeBanLocked(key, credentialName, modelID string) {
	delete(f.banned, key)
	delete(f.failures, key)
	monitoring.CredentialUnbanEvents.WithLabelValues(credentialName, modelID).Inc()
	f.logger.Info("Credential unbanned manually",
		"credential", credentialName, "model", modelID)
}

// removeCredentialBansLocked drops every ban of a credential, wildcard
// included, and returns how many of them were still active. Caller holds f.mu.
func (f *Fail2Ban) removeCredentialBansLocked(credentialName string) int {
	prefix := credentialName + "|"
	removed := 0
	for key, ban := range f.banned {
		if strings.HasPrefix(key, prefix) {
			_, model := parseBanKey(key)
			if ban.active() {
				removed++
			}
			f.removeBanLocked(key, credentialName, model)
		}
	}
	return removed
}

// AdminBan bans credentialName for modelID on behalf of an operator. An empty
// modelID or WildcardModel bans every model of the credential. A ttl of 0
// bans until AdminUnban (or a restart — bans live in memory only).
//
// Unlike BanUntil it replaces any existing ban for the same key, shorter or
// longer, so an operator can always override an automatic ban. Failure
// counters are left untouched.
func (f *Fail2Ban) AdminBan(credentialName, modelID string, ttl time.Duration, reason string) BanPair {
	if modelID == "" {
		modelID = WildcardModel
	}
	reason = adminReason(reason)
	key := banKey(credentialName, modelID)
	now := utils.NowUTC()

	f.mu.Lock()
	defer f.mu.Unlock()

	ban := &banInfo{
		banTime:     now,
		banDuration: ttl,
		errorCode:   adminErrorCode,
		reason:      reason,
		origin:      OriginAdmin,
	}
	f.banned[key] = ban

	monitoring.CredentialBanEvents.WithLabelValues(credentialName, modelID, strconv.Itoa(adminErrorCode)).Inc()
	f.logger.Warn("Credential banned by admin",
		"credential", credentialName,
		"model", modelID,
		"reason", reason,
		"ban_duration", ttl,
		"permanent", ttl == 0)

	return f.banPairLocked(key, ban)
}

// adminReason prefixes reason with "admin:" so admin bans are recognisable in
// /health and logs even without looking at the origin field.
func adminReason(reason string) string {
	reason = strings.TrimSpace(reason)
	switch {
	case reason == "":
		return "admin"
	case strings.HasPrefix(reason, "admin:"):
		return reason
	default:
		return "admin: " + reason
	}
}

// AdminUnban lifts bans on credentialName and returns how many active bans
// were removed. An empty modelID removes every ban of the credential
// (wildcard and per-model); otherwise only the ban on that exact key is
// removed, so unbanning one model does not lift a credential-wide wildcard
// ban. Removing nothing is not an error.
func (f *Fail2Ban) AdminUnban(credentialName, modelID string) int {
	f.mu.Lock()
	defer f.mu.Unlock()

	if modelID == "" {
		return f.removeCredentialBansLocked(credentialName)
	}

	key := banKey(credentialName, modelID)
	ban, exists := f.banned[key]
	if !exists {
		return 0
	}
	active := ban.active()
	f.removeBanLocked(key, credentialName, modelID)
	if active {
		return 1
	}
	return 0
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
		pairs = append(pairs, f.banPairLocked(key, ban))
	}
	return pairs
}

// GetActiveBans returns the bans that have not expired yet, ordered by
// credential and model. Unlike GetBannedPairs it skips expired entries that
// have not been swept from the map.
func (f *Fail2Ban) GetActiveBans() []BanPair {
	f.mu.RLock()
	defer f.mu.RUnlock()

	pairs := make([]BanPair, 0, len(f.banned))
	for key, ban := range f.banned {
		if ban.active() {
			pairs = append(pairs, f.banPairLocked(key, ban))
		}
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].Credential != pairs[j].Credential {
			return pairs[i].Credential < pairs[j].Credential
		}
		return pairs[i].Model < pairs[j].Model
	})
	return pairs
}

// banPairLocked builds the exported view of one ban. Caller holds f.mu.
func (f *Fail2Ban) banPairLocked(key string, ban *banInfo) BanPair {
	credential, model := parseBanKey(key)
	counts := make(map[int]int)
	for code, count := range f.failures[key] {
		counts[code] = count
	}
	return BanPair{
		Credential:      credential,
		Model:           model,
		ErrorCode:       ban.errorCode,
		ErrorCodeCounts: counts,
		BanTime:         ban.banTime,
		BanDuration:     ban.banDuration,
		BanUntil:        banUntil(ban),
		Reason:          ban.reason,
		Origin:          ban.originOrDefault(),
	}
}

func banUntil(ban *banInfo) time.Time {
	if ban == nil || ban.banDuration == 0 {
		return time.Time{}
	}
	return ban.banTime.Add(ban.banDuration).UTC()
}

// RemainingBan returns the time remaining until credentialName becomes
// available again for modelID, if it is currently under a temporary ban.
// Used to build a Retry-After hint on a synthesized 429. Permanent bans
// (banDuration == 0) return (0, false) — there is no finite ETA to report.
func (f *Fail2Ban) RemainingBan(credentialName, modelID string) (time.Duration, bool) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	keys := []string{banKey(credentialName, modelID)}
	if modelID != WildcardModel {
		keys = append(keys, banKey(credentialName, WildcardModel))
	}

	// The credential is usable again only once every applicable ban has
	// lifted, so report the longest remaining one.
	var longest time.Duration
	found := false
	for _, key := range keys {
		ban, exists := f.banned[key]
		if !exists {
			continue
		}
		until := banUntil(ban)
		if until.IsZero() {
			return 0, false // permanent ban — no finite ETA
		}
		if remaining := until.Sub(utils.NowUTC()); remaining > longest {
			longest = remaining
			found = true
		}
	}
	return longest, found
}

// DefaultBanDuration returns the configured ban duration that would apply to
// credentialName if it were banned for statusCode right now, regardless of
// whether it is actually banned. Used to build a Retry-After hint on a 429
// even when no credential has crossed the failure threshold yet — a 429
// reaching the client must always carry a Retry-After. Returns (0, false)
// for a permanent-ban rule (BanDuration == 0), since there is no finite ETA
// to report.
func (f *Fail2Ban) DefaultBanDuration(credentialName string, statusCode int) (time.Duration, bool) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	rule := f.getRule(credentialName, statusCode)
	if rule.BanDuration <= 0 {
		return 0, false
	}
	return rule.BanDuration, true
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
