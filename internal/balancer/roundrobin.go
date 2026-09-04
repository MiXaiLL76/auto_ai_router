// Package balancer selects an upstream credential for each request across the configured providers.
package balancer

import (
	"errors"
	"io"
	"log/slog"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/scope"
)

// ModelChecker interface for checking model availability
type ModelChecker interface {
	HasModel(credentialName, modelID string) bool
	GetCredentialsForModel(modelID string) []string
	GetModelWeightForCredential(modelID, credentialName string) int
	// LearnedModelPriorityForCredential returns the per-model priority learned from an
	// upstream /health poll for a proxy/AIR credential, plus a flag that is false only
	// when nothing has been learned. A learned 0 (upstream's best group) is authoritative
	// and must not fall through to the credential's static priority: field.
	LearnedModelPriorityForCredential(modelID, credentialName string) (int, bool)
	// GetModelPriorityTiersForCredential returns the learned per-priority-tier breakdown
	// for a proxy/AIR credential serving modelID (sorted ascending by priority), or nil
	// when the credential is a single implicit tier. Design B (review_158 item 3): a
	// proxy credential fronting an upstream that spans several priority groups expands
	// into one balancer candidate per tier.
	GetModelPriorityTiersForCredential(modelID, credentialName string) []httputil.ModelPriorityTier
	IsEnabled() bool
}

var (
	ErrNoCredentialsAvailable = errors.New("no credentials available")
	ErrRateLimitExceeded      = errors.New("rate limit exceeded")
)

type candidateEntry struct {
	absIdx int
	cred   *config.CredentialConfig
	// tier is set only when this entry is one learned primary-priority tier of a
	// proxy/AIR credential for the request model (Design B). Picking a tier candidate
	// still routes the real request to cred — the tier only governs which primary
	// priority group the candidate sits in, its SWRR weight, and a local cumulative
	// rate-limit gate so this router cascades off its own tier-1 saturation without
	// waiting for the next /health poll.
	tier *tierCandidate
}

type tierCandidate struct {
	priority int
	weight   int
	// cumLimitRPM/TPM: this tier's own aggregate cap plus every lower-priority tier's,
	// so "current model usage >= cumLimitRPM" means tiers <= this one are full and the
	// candidate should be skipped in favour of the next group. <= 0 means uncapped.
	// Capacity of tiers the upstream reports really banned is left out — a banned
	// contributor's limit is not available right now and must not inflate a live tier's cap.
	cumLimitRPM int
	cumLimitTPM int
	// pricierCurrentRPM/TPM: summed upstream-reported usage of every tier with a strictly
	// higher priority number (a costlier fallback group). This router keeps a single
	// aggregate (cred, model) counter with no tier attribution; before comparing it with a
	// cheaper tier's cumulative cap we subtract the usage the upstream attributes to
	// pricier tiers, so fallback-tier load cannot spuriously close a cheaper tier that has
	// in fact recovered.
	pricierCurrentRPM int
	pricierCurrentTPM int
	// ownLimitRPM/TPM and ownCurrentRPM/TPM are this tier's own aggregate cap and the
	// upstream's last-polled usage against it. tierHasHeadroom gates on these (the
	// upstream's own view of this specific tier) in addition to the cumulative-cap check
	// against this router's local aggregate counter — the two can disagree when the
	// upstream does not consume capacity strictly in priority order. <= 0 limit means
	// uncapped.
	ownLimitRPM   int
	ownCurrentRPM int
	ownLimitTPM   int
	ownCurrentTPM int
	banned        bool // every upstream leaf credential in this tier is really banned (not merely saturated)
}

type RoundRobin struct {
	mu              sync.RWMutex
	credentials     []config.CredentialConfig
	staticCreds     []config.CredentialConfig // immutable snapshot of YAML-defined credentials
	credentialIndex map[string]int            // O(1) lookup by name instead of O(n) search
	swrr            map[schedKey]*swrrState   // smooth weighted round-robin state per selection cycle
	fail2ban        *fail2ban.Fail2Ban
	rateLimiter     *ratelimit.RPMLimiter
	modelChecker    ModelChecker
	logger          *slog.Logger
}

func New(credentials []config.CredentialConfig, f2b *fail2ban.Fail2Ban, rl *ratelimit.RPMLimiter) *RoundRobin {
	if f2b == nil {
		panic("balancer.New: fail2ban must not be nil")
	}
	if rl == nil {
		panic("balancer.New: rateLimiter must not be nil")
	}

	credentialIndex := make(map[string]int, len(credentials))
	for i, c := range credentials {
		// Normalize TPM: 0 means "not configured" → treat as unlimited (-1).
		// Convention: -1 = unlimited, positive = limit.
		tpm := c.TPM
		if tpm == 0 {
			tpm = -1
		}
		rl.AddCredentialWithTPM(c.Name, c.RPM, tpm)
		credentialIndex[c.Name] = i
	}

	rr := &RoundRobin{
		credentials:     credentials,
		staticCreds:     append([]config.CredentialConfig(nil), credentials...),
		credentialIndex: credentialIndex,
		swrr:            make(map[schedKey]*swrrState),
		fail2ban:        f2b,
		rateLimiter:     rl,
		modelChecker:    nil,
		logger:          slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{Level: slog.LevelInfo})),
	}

	rr.logCredentialTiers()

	return rr
}

// SetLogger sets the logger for the RoundRobin balancer
func (r *RoundRobin) SetLogger(logger *slog.Logger) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.logger = logger
}

// SetModelChecker sets the model checker for filtering credentials by model
func (r *RoundRobin) SetModelChecker(mc ModelChecker) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.modelChecker = mc
}

func (r *RoundRobin) UpdateProviderScopes(
	expected config.CredentialConfig,
	scopes, deniedScopes []string,
	expression *scope.Expression,
	known bool,
) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	cred := r.getCredentialByName(expected.Name)
	if cred == nil || !expected.SameProviderIdentity(*cred) {
		return false
	}

	cred.ProviderScopes = scope.NormalizeList(scopes)
	cred.ProviderDeniedScopes = scope.NormalizeList(deniedScopes)
	cred.ProviderScopeExpression = scope.NormalizeExpression(expression)
	cred.ProviderScopeKnown = known
	return true
}

// getCredentialByName finds a credential by name (must be called with lock held)
func (r *RoundRobin) getCredentialByName(name string) *config.CredentialConfig {
	idx, ok := r.credentialIndex[name]
	if !ok {
		return nil
	}
	return &r.credentials[idx]
}

// IsProxyCredential checks if a credential uses proxy/AIR remote-router transport.
func (r *RoundRobin) IsProxyCredential(credentialName string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	cred := r.getCredentialByName(credentialName)
	return cred != nil && cred.IsProxyLike()
}

// IsBanned checks if a specific credential+model pair is currently banned
func (r *RoundRobin) IsBanned(credentialName, modelID string) bool {
	return r.fail2ban.IsBanned(credentialName, modelID)
}

// HasAnyBan checks if a credential has any banned models
func (r *RoundRobin) HasAnyBan(credentialName string) bool {
	return r.fail2ban.HasAnyBan(credentialName)
}

// MinRemainingBanForModel returns the shortest time remaining until any
// fail2ban-banned credential for modelID becomes available again, scoped to
// the candidates actually relevant to this request: credentials visible
// under visibility that serve modelID, minus exclude (credentials already
// tried and rejected earlier in this same request). A ban on a credential
// outside that set can never affect when this caller will next succeed, so
// including it would report a Retry-After that's meaningless to them.
func (r *RoundRobin) MinRemainingBanForModel(modelID string, exclude map[string]bool, visibility scope.Context) (time.Duration, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var shortest time.Duration
	found := false
	for i := range r.credentials {
		cred := &r.credentials[i]
		if exclude[cred.Name] {
			continue
		}
		if !cred.VisibleTo(visibility) {
			continue
		}
		if modelID != "" && r.modelChecker != nil && r.modelChecker.IsEnabled() && !r.hasModel(cred.Name, modelID, visibility) {
			continue
		}
		remaining, ok := r.fail2ban.RemainingBan(cred.Name, modelID)
		if !ok {
			continue
		}
		if !found || remaining < shortest {
			shortest = remaining
			found = true
		}
	}
	return shortest, found
}

// defaultRetryAfterFallback is the last-resort Retry-After hint used when no
// active ban and no per-credential 429 ban-duration rule can be found for a
// model (e.g. the model has no eligible credentials left to inspect). A 429
// reaching the client must always carry a Retry-After, so this constant
// exists purely to guarantee that — it is never a precise ETA.
const defaultRetryAfterFallback = 30 * time.Second

// DefaultRetryAfterForModel returns a Retry-After duration to report on a 429
// for modelID, scoped to the same candidate set as MinRemainingBanForModel.
// It first tries an active ban (a precise ETA); if none is active, it falls
// back to the shortest *configured* 429 ban duration among eligible
// credentials, so the header is still meaningful; if that also yields
// nothing (no eligible credentials, or all have permanent-ban rules),
// it returns defaultRetryAfterFallback. This method never reports "no
// Retry-After" — callers use it precisely to guarantee the header is always
// present on a 429.
func (r *RoundRobin) DefaultRetryAfterForModel(modelID string, exclude map[string]bool, visibility scope.Context) time.Duration {
	if remaining, ok := r.MinRemainingBanForModel(modelID, exclude, visibility); ok {
		return remaining
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	var shortest time.Duration
	found := false
	for i := range r.credentials {
		cred := &r.credentials[i]
		if exclude[cred.Name] {
			continue
		}
		if !cred.VisibleTo(visibility) {
			continue
		}
		if modelID != "" && r.modelChecker != nil && r.modelChecker.IsEnabled() && !r.hasModel(cred.Name, modelID, visibility) {
			continue
		}
		duration, ok := r.fail2ban.DefaultBanDuration(cred.Name, http.StatusTooManyRequests)
		if !ok {
			continue
		}
		if !found || duration < shortest {
			shortest = duration
			found = true
		}
	}
	if !found {
		return defaultRetryAfterFallback
	}
	return shortest
}

// GetProxyCredentials returns all proxy/AIR remote-router credentials.
func (r *RoundRobin) GetProxyCredentials() []config.CredentialConfig {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var proxies []config.CredentialConfig
	for _, cred := range r.credentials {
		if cred.IsProxyLike() {
			proxies = append(proxies, cred)
		}
	}
	return proxies
}

// NextForModel returns the next available credential that supports the specified model
func (r *RoundRobin) NextForModel(modelID string) (*config.CredentialConfig, error) {
	return r.NextForModelScoped(modelID, scope.AdminContext())
}

func (r *RoundRobin) NextForModelScoped(modelID string, visibility scope.Context) (*config.CredentialConfig, error) {
	return r.nextExcludingScoped(modelID, "", nil, visibility)
}

// NextSpecific tries to return a specific credential by name without advancing the
// round-robin state. It still applies model availability, ban, and rate-limit checks.
func (r *RoundRobin) NextSpecific(credentialName, modelID string) (*config.CredentialConfig, error) {
	return r.NextSpecificScoped(credentialName, modelID, scope.AdminContext())
}

func (r *RoundRobin) NextSpecificScoped(credentialName, modelID string, visibility scope.Context) (*config.CredentialConfig, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	idx, ok := r.credentialIndex[credentialName]
	if !ok {
		return nil, ErrNoCredentialsAvailable
	}

	cred := &r.credentials[idx]
	if !cred.VisibleTo(visibility) {
		return nil, ErrNoCredentialsAvailable
	}
	if modelID != "" && r.modelChecker != nil && r.modelChecker.IsEnabled() {
		if !r.hasModel(credentialName, modelID, visibility) {
			return nil, ErrNoCredentialsAvailable
		}
	}

	if r.fail2ban.IsBanned(credentialName, modelID) {
		return nil, ErrNoCredentialsAvailable
	}

	if !r.rateLimiter.TryAllowAll(credentialName, modelID) {
		return nil, ErrRateLimitExceeded
	}

	return cred, nil
}

// nextExcludingScoped is the single credential selection path (initial pick and every
// retry). Excluded credentials are skipped entirely and don't count as candidates.
//
//  1. Build a candidate list via structural filters (exclude, requiredType, scope, model
//     availability) — time-stable properties that don't change between requests.
//  2. Expand proxy/AIR candidates whose upstream spans several priority groups into one
//     candidate per learned tier (Design B).
//  3. Cascade through priority groups ascending (credentials without an explicit
//     priority: share group 0, tried first; a last-resort credential sits at group 999).
//     SWRR runs within the lowest-numbered group that still has a live member; the
//     cascade drops to the next group only when the current one is fully banned or
//     rate-limited. Commit the first candidate that passes its rate limits.
//
// requiredType != "" restricts the pool to one provider type (proxy same-type retry).
func (r *RoundRobin) nextExcludingScoped(modelID string, requiredType config.ProviderType, exclude map[string]bool, visibility scope.Context) (*config.CredentialConfig, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	var candidates []candidateEntry

	for i := range r.credentials {
		cred := &r.credentials[i]

		if len(exclude) > 0 && exclude[cred.Name] {
			continue
		}

		if requiredType != "" && cred.Type != requiredType {
			monitoring.CredentialSelectionRejected.WithLabelValues("type_mismatch").Inc()
			continue
		}

		if !cred.VisibleTo(visibility) {
			monitoring.CredentialSelectionRejected.WithLabelValues("scope_not_allowed").Inc()
			continue
		}

		// Check model availability before ban/rate checks.
		// model_not_available is a structural property, not a temporary issue.
		if modelID != "" && r.modelChecker != nil && r.modelChecker.IsEnabled() {
			if !r.hasModel(cred.Name, modelID, visibility) {
				monitoring.CredentialSelectionRejected.WithLabelValues("model_not_available").Inc()
				continue
			}
		}

		candidates = append(candidates, candidateEntry{absIdx: i, cred: cred})
	}

	if len(candidates) == 0 {
		return nil, ErrNoCredentialsAvailable
	}

	candidates = r.expandTierCandidates(candidates, modelID)
	keyBase := r.schedKeyFor(modelID, requiredType, hasActiveExclusion(exclude), "")
	cred, found, rateLimitHit := r.selectPriorityGroupCandidate(modelID, candidates, keyBase, r.primaryPriority)
	if found {
		return cred, nil
	}
	if rateLimitHit {
		return nil, ErrRateLimitExceeded
	}
	return nil, ErrNoCredentialsAvailable
}

// expandTierCandidates replaces each proxy/AIR candidate that has a learned multi-tier
// breakdown for modelID with one candidate per tier (Design B). Everything else passes
// through unchanged, so the common single-tier / direct-provider case is a no-op.
func (r *RoundRobin) expandTierCandidates(candidates []candidateEntry, modelID string) []candidateEntry {
	if modelID == "" || r.modelChecker == nil || !r.modelChecker.IsEnabled() {
		return candidates
	}
	// Allocate the per-candidate breakdown slice lazily — the overwhelmingly common case
	// is that nothing expands (no proxy candidate, or none multi-tier), and this runs on
	// every primary selection under r.mu.
	var breakdowns [][]httputil.ModelPriorityTier
	for i, c := range candidates {
		if c.cred.IsProxyLike() {
			if tiers := r.modelChecker.GetModelPriorityTiersForCredential(modelID, c.cred.Name); tiersRoutable(tiers) {
				if breakdowns == nil {
					breakdowns = make([][]httputil.ModelPriorityTier, len(candidates))
				}
				breakdowns[i] = tiers
			}
		}
	}
	if breakdowns == nil {
		return candidates
	}

	out := make([]candidateEntry, 0, len(candidates)*2)
	for i, c := range candidates {
		tiers := breakdowns[i]
		if !tiersRoutable(tiers) {
			out = append(out, c)
			continue
		}
		// Grand total of the upstream's per-tier usage, so each tier can subtract the
		// share the upstream attributes to strictly-pricier tiers from this router's
		// tier-blind aggregate counter (see tierCandidate.pricierCurrentRPM).
		totalCurRPM, totalCurTPM := 0, 0
		for _, t := range tiers {
			totalCurRPM += t.CurrentRPM
			totalCurTPM += t.CurrentTPM
		}
		cumRPM, cumTPM := 0, 0
		uncappedRPM, uncappedTPM := false, false
		seenCurRPM, seenCurTPM := 0, 0
		for _, t := range tiers { // already sorted ascending by priority
			tc := &tierCandidate{
				priority:      t.Priority,
				weight:        t.Weight,
				banned:        t.Banned,
				ownLimitRPM:   t.LimitRPM,
				ownCurrentRPM: t.CurrentRPM,
				ownLimitTPM:   t.LimitTPM,
				ownCurrentTPM: t.CurrentTPM,
			}
			seenCurRPM += t.CurrentRPM
			seenCurTPM += t.CurrentTPM
			tc.pricierCurrentRPM = totalCurRPM - seenCurRPM
			tc.pricierCurrentTPM = totalCurTPM - seenCurTPM
			// A really-banned tier's capacity is not available right now — keep its limit
			// out of this and every cheaper live tier's cumulative cap (a banned cheap
			// tier must not let a live pricier tier over-admit). Its own candidate is
			// dropped in liveCandidates regardless.
			if !t.Banned {
				if t.LimitRPM <= 0 {
					uncappedRPM = true
				} else if !uncappedRPM {
					cumRPM += t.LimitRPM
				}
				if t.LimitTPM <= 0 {
					uncappedTPM = true
				} else if !uncappedTPM {
					cumTPM += t.LimitTPM
				}
			}
			if uncappedRPM {
				tc.cumLimitRPM = -1
			} else {
				tc.cumLimitRPM = cumRPM
			}
			if uncappedTPM {
				tc.cumLimitTPM = -1
			} else {
				tc.cumLimitTPM = cumTPM
			}
			out = append(out, candidateEntry{absIdx: c.absIdx, cred: c.cred, tier: tc})
		}
	}
	return out
}

// tiersRoutable reports whether a learned per-tier breakdown should drive candidate
// expansion. A 2+ tier breakdown always does. A single tier only does when it carries a
// real upstream ban — otherwise the credential stays on the scalar priority path
// unchanged, but a lone banned tier still has to reach the balancer so the credential is
// dropped for a model its sole upstream tier has banned (scalar priority holds no ban
// state, so the ban would otherwise be invisible to this router and the next one).
func tiersRoutable(tiers []httputil.ModelPriorityTier) bool {
	if len(tiers) >= 2 {
		return true
	}
	return len(tiers) == 1 && tiers[0].Banned
}

func (r *RoundRobin) liveCandidates(modelID string, candidates []candidateEntry) ([]candidateEntry, bool) {
	live := make([]candidateEntry, 0, len(candidates))
	rateLimitHit := false
	for _, c := range candidates {
		if c.tier != nil && c.tier.banned {
			// Upstream /health reports every leaf credential in this tier really banned
			// (not merely RPM/TPM-saturated — that is handled by tierHasHeadroom below,
			// which sets rateLimitHit so the caller surfaces 429, not 503).
			monitoring.CredentialSelectionRejected.WithLabelValues("banned").Inc()
			continue
		}
		if r.fail2ban.IsBanned(c.cred.Name, modelID) {
			monitoring.CredentialSelectionRejected.WithLabelValues("banned").Inc()
			continue
		}
		if c.tier != nil && !r.tierHasHeadroom(c.cred.Name, modelID, c.tier) {
			// Local cumulative gate: tiers up to and including this one are full, so the
			// upstream has cascaded to a pricier group — cascade locally too rather than
			// keep pouring tier-1-priced traffic at a proxy serving it at tier-N cost.
			monitoring.CredentialSelectionRejected.WithLabelValues("rate_limit").Inc()
			rateLimitHit = true
			continue
		}
		if !r.canPassRateLimits(c.cred.Name, modelID) {
			monitoring.CredentialSelectionRejected.WithLabelValues("rate_limit").Inc()
			rateLimitHit = true
			continue
		}
		live = append(live, c)
	}
	return live, rateLimitHit
}

// tierHasHeadroom reports whether this tier can still take traffic. Two independent gates,
// either of which closing the tier:
//
//  1. The upstream's own last-polled view of THIS tier: ownCurrentRPM/TPM >= ownLimitRPM/TPM.
//     The upstream may consume capacity out of priority order (a pricier tier filled while
//     a cheaper one recovered), so its per-tier counters are not reconstructable from this
//     router's single aggregate (cred, model) counter.
//  2. This router's own committed usage against the cumulative cap (this tier's cap plus
//     every cheaper tier's): reads the same aggregate counter that
//     selectWeightedLiveCandidate's TryAllowAll records into, so this router's own traffic
//     moves the cascade forward with no /health-poll lag. The counter has no tier
//     attribution, so usage the upstream reports against strictly-pricier tiers
//     (tc.pricierCurrent*) is subtracted first — otherwise fallback-tier load would count
//     against a cheaper tier's small cap and close it even after it recovered.
func (r *RoundRobin) tierHasHeadroom(credentialName, modelID string, tc *tierCandidate) bool {
	if tc.ownLimitRPM > 0 && tc.ownCurrentRPM >= tc.ownLimitRPM {
		return false
	}
	if tc.ownLimitTPM > 0 && tc.ownCurrentTPM >= tc.ownLimitTPM {
		return false
	}
	if tc.cumLimitRPM > 0 {
		localRPM := r.rateLimiter.GetCurrentModelRPM(credentialName, modelID) - tc.pricierCurrentRPM
		if localRPM >= tc.cumLimitRPM {
			return false
		}
	}
	if tc.cumLimitTPM > 0 {
		localTPM := r.rateLimiter.GetCurrentModelTPM(credentialName, modelID) - tc.pricierCurrentTPM
		if localTPM >= tc.cumLimitTPM {
			return false
		}
	}
	return true
}

func (r *RoundRobin) selectWeightedLiveCandidate(modelID string, live []candidateEntry, key schedKey, rateLimitHit bool) (*config.CredentialConfig, error) {
	if len(live) == 0 {
		if rateLimitHit {
			return nil, ErrRateLimitExceeded
		}
		return nil, ErrNoCredentialsAvailable
	}

	// Smooth weighted round-robin over candidates that are available now.
	// Banned/rate-limited credentials are dropped before this point so they don't
	// accumulate weight while down — otherwise a high-weight provider would burst on recovery.
	// With equal weights this degenerates to the historical round-robin sequence.
	state := r.swrrStateFor(key)
	liveWeights := make(map[string]int, len(live))
	for _, c := range live {
		liveWeights[c.cred.Name] = r.candidateWeight(c, modelID)
	}
	total := state.advance(liveWeights)

	// Order by running counter (desc); ties keep the structural candidate order so that
	// equal weights reproduce the historical ascending round-robin sequence.
	sort.SliceStable(live, func(i, j int) bool {
		return state.currentOf(live[i].cred.Name) > state.currentOf(live[j].cred.Name)
	})

	// Phase 3: Commit the highest-priority candidate that passes its rate limits.
	// TryAllowAll atomically checks credential + model RPM/TPM and records usage only on
	// success, preventing TOCTOU races after the non-recording precheck above.
	for _, c := range live {
		if !r.rateLimiter.TryAllowAll(c.cred.Name, modelID) {
			monitoring.CredentialSelectionRejected.WithLabelValues("rate_limit").Inc()
			rateLimitHit = true
			continue
		}
		state.commit(c.cred.Name, total)
		return c.cred, nil
	}

	// Prioritize rate limit error: if any candidate hit rate limit, surface it even if
	// others were banned. This gives callers accurate signal for backoff/retry logic.
	if rateLimitHit {
		return nil, ErrRateLimitExceeded
	}
	// All candidates are banned (or none remain after ban + rate-limit filtering).
	return nil, ErrNoCredentialsAvailable
}

func (r *RoundRobin) canPassRateLimits(credentialName, modelID string) bool {
	if !r.rateLimiter.CanAllow(credentialName) || !r.rateLimiter.AllowTokens(credentialName) {
		return false
	}
	if modelID == "" {
		return true
	}
	return r.rateLimiter.CanAllowModel(credentialName, modelID) &&
		r.rateLimiter.AllowModelTokens(credentialName, modelID)
}

func hasActiveExclusion(exclude map[string]bool) bool {
	for _, excluded := range exclude {
		if excluded {
			return true
		}
	}
	return false
}

func candidateCycleKey(candidates []candidateEntry) string {
	if len(candidates) == 0 {
		return ""
	}
	names := make([]string, 0, len(candidates))
	for _, c := range candidates {
		names = append(names, c.cred.Name)
	}
	sort.Strings(names)
	return strings.Join(names, "\x00")
}

// NextForModelExcluding returns the next available non-fallback credential that supports
// the specified model, excluding credentials in the exclude set.
func (r *RoundRobin) NextForModelExcluding(modelID string, exclude map[string]bool) (*config.CredentialConfig, error) {
	return r.NextForModelExcludingScoped(modelID, exclude, scope.AdminContext())
}

func (r *RoundRobin) NextForModelExcludingScoped(modelID string, exclude map[string]bool, visibility scope.Context) (*config.CredentialConfig, error) {
	return r.nextExcludingScoped(modelID, "", exclude, visibility)
}

// NextSameTypeForModelExcluding returns the next available non-fallback credential of the
// same type as credType, excluding credentials in the exclude set. Used for same-type
// credential retry on provider errors (429/5xx/auth errors) to prevent cross-type routing.
func (r *RoundRobin) NextSameTypeForModelExcluding(modelID string, credType config.ProviderType, exclude map[string]bool) (*config.CredentialConfig, error) {
	return r.NextSameTypeForModelExcludingScoped(modelID, credType, exclude, scope.AdminContext())
}

func (r *RoundRobin) NextSameTypeForModelExcludingScoped(modelID string, credType config.ProviderType, exclude map[string]bool, visibility scope.Context) (*config.CredentialConfig, error) {
	return r.nextExcludingScoped(modelID, credType, exclude, visibility)
}

// NextProxyForModelExcludingScoped returns the next untried proxy/AIR credential in the
// priority cascade for modelID. The extended retry phase forwards through the proxy path
// and cannot accept a direct-provider credential, so it walks past any non-proxy-like
// candidate (marking it tried) and keeps cascading.
func (r *RoundRobin) NextProxyForModelExcludingScoped(modelID string, exclude map[string]bool, visibility scope.Context) (*config.CredentialConfig, error) {
	local := make(map[string]bool, len(exclude))
	for k, v := range exclude {
		local[k] = v
	}
	for {
		cred, err := r.nextExcludingScoped(modelID, "", local, visibility)
		if err != nil {
			return nil, err
		}
		if cred.IsProxyLike() {
			return cred, nil
		}
		local[cred.Name] = true
	}
}

func (r *RoundRobin) NextRetryForModelExcluding(modelID string, current *config.CredentialConfig, exclude map[string]bool) (*config.CredentialConfig, error) {
	return r.NextRetryForModelExcludingScoped(modelID, current, exclude, scope.AdminContext())
}

// NextRetryForModelExcludingScoped picks the next credential after a retryable error.
// Post-unification this is just the normal selection cascade over the untried
// credentials: it continues through the ascending priority tiers (cross-provider-type),
// so a retry naturally walks from the current tier down to the last-resort group. The
// `current` credential is informational only.
func (r *RoundRobin) NextRetryForModelExcludingScoped(modelID string, current *config.CredentialConfig, exclude map[string]bool, visibility scope.Context) (*config.CredentialConfig, error) {
	if current == nil {
		return nil, ErrNoCredentialsAvailable
	}
	return r.nextExcludingScoped(modelID, "", exclude, visibility)
}

func (r *RoundRobin) hasModel(credentialName, modelID string, visibility scope.Context) bool {
	if scoped, ok := r.modelChecker.(interface {
		HasModelScoped(credentialName, modelID string, visibility scope.Context) bool
	}); ok {
		return scoped.HasModelScoped(credentialName, modelID, visibility)
	}
	return r.modelChecker.HasModel(credentialName, modelID)
}

// selectPriorityGroupCandidate picks a live candidate from candidates using priority-group
// cascading: candidates are bucketed by priorityOf (ascending), and SWRR runs within the
// lowest-priority group that has at least one live (not banned, not rate-limited) member.
// A group cascades to the next only when every member of the current group is currently
// down — so partial degradation (some group members down) keeps serving that same group
// instead of moving to the next one.
//
// priorityOf is always primaryPriority: the learned per-model proxy tier if any, else the
// static priority: field. Initial pick and retry share it.
//
// keyBase supplies the non-priority, non-scope fields of the SWRR schedKey (model,
// required type, excluding); this function fills in
// keyBase.priority and keyBase.scopeKey per selected group so each priority tier gets its
// own independent SWRR cycle, keyed by that tier's full (structural) membership so the
// cycle survives transient bans/unbans within the tier.
//
// Returns (cred, found, rateLimitHit). found is true only on a successful pick; when every
// group is fully down, found is false and the caller decides between ErrRateLimitExceeded
// and ErrNoCredentialsAvailable using rateLimitHit.
func (r *RoundRobin) selectPriorityGroupCandidate(modelID string, candidates []candidateEntry, keyBase schedKey, priorityOf func(*config.CredentialConfig, string) int) (*config.CredentialConfig, bool, bool) {
	if len(candidates) == 0 {
		return nil, false, false
	}

	groups := make(map[int][]candidateEntry, len(candidates))
	priorities := make([]int, 0, len(candidates))
	for _, c := range candidates {
		// A tier candidate (Design B) carries its own priority — the tier the upstream
		// serves this model from. Otherwise priorityOf resolves the credential's group:
		// a per-model priority learned from an upstream /health poll for proxy/AIR
		// credentials (learnedProxyPriority in weighted.go), else its static priority:.
		p := priorityOf(c.cred, modelID)
		if c.tier != nil {
			p = c.tier.priority
		}
		if _, ok := groups[p]; !ok {
			priorities = append(priorities, p)
		}
		groups[p] = append(groups[p], c)
	}
	sort.Ints(priorities)

	rateLimitHit := false
	for _, p := range priorities {
		group := groups[p]
		live, groupRateLimitHit := r.liveCandidates(modelID, group)
		if groupRateLimitHit {
			rateLimitHit = true
		}
		if len(live) == 0 {
			// Whole group is currently down (banned or rate-limited) — cascade to the
			// next priority group.
			continue
		}

		key := keyBase
		key.priority = p
		key.scopeKey = candidateCycleKey(group)
		cred, err := r.selectWeightedLiveCandidate(modelID, live, key, groupRateLimitHit)
		if err == nil {
			return cred, true, rateLimitHit
		}
		// live is non-empty here, so selectWeightedLiveCandidate's only failure mode
		// is every member losing the atomic TryAllowAll race in its Phase 3 loop (the
		// len(live)==0 short-circuit at its top can't fire) — always ErrRateLimitExceeded,
		// never ErrNoCredentialsAvailable. That's a real possibility near RPM/TPM limits
		// under concurrency: liveCandidates' non-recording precheck can pass for several
		// candidates that then all fail the atomic recheck. Previously this returned
		// immediately with that error, which surfaced as a 429 to the caller even when a
		// lower-priority group had full untouched capacity — the flat pool this cascade
		// replaced would have kept trying other candidates instead. Record the hit and
		// keep cascading; the caller falls back to ErrNoCredentialsAvailable only if
		// rateLimitHit never got set at all.
		rateLimitHit = true
	}
	return nil, false, rateLimitHit
}

func (r *RoundRobin) RecordResponse(credentialName, modelID string, statusCode int) {
	r.fail2ban.RecordResponse(credentialName, modelID, statusCode)
}

func (r *RoundRobin) BanUntil(credentialName, modelID string, statusCode int, until time.Time, reason string) {
	r.fail2ban.BanUntil(credentialName, modelID, statusCode, until, reason)
}

func (r *RoundRobin) GetCredentialsSnapshot() []config.CredentialConfig {
	r.mu.RLock()
	defer r.mu.RUnlock()
	creds := make([]config.CredentialConfig, len(r.credentials))
	copy(creds, r.credentials)
	return creds
}

func (r *RoundRobin) GetAvailableCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()

	count := 0
	for _, cred := range r.credentials {
		if cred.VisibleTo(scope.AdminContext()) && !r.fail2ban.HasAnyBan(cred.Name) {
			count++
		}
	}
	return count
}

func (r *RoundRobin) GetBannedCount() int {
	return r.fail2ban.GetBannedCount()
}

// GetBannedPairs returns all currently banned credential+model pairs with error details
func (r *RoundRobin) GetBannedPairs() []fail2ban.BanPair {
	return r.fail2ban.GetBannedPairs()
}

// UpdateDBCredentials atomically replaces the DB-sourced portion of the credential list.
// Static (YAML-defined) credentials are always preserved unchanged.
// New credentials are registered in the rate limiter; stale entries are left in the rate
// limiter but will never be selected since they are absent from the credential list.
func (r *RoundRobin) UpdateDBCredentials(dbCreds []config.CredentialConfig) {
	r.mu.Lock()
	defer r.mu.Unlock()

	existing := make(map[string]config.CredentialConfig, len(r.credentials))
	for _, credential := range r.credentials {
		existing[credential.Name] = credential
	}

	// Build name set of static creds so we can skip duplicates from DB.
	staticNames := make(map[string]bool, len(r.staticCreds))
	for _, c := range r.staticCreds {
		staticNames[c.Name] = true
	}

	// Filter out DB creds that clash with static names.
	filtered := make([]config.CredentialConfig, 0, len(dbCreds))
	for _, c := range dbCreds {
		if !staticNames[c.Name] {
			filtered = append(filtered, c)
		}
	}

	// Merge static + new DB creds.
	newCreds := append(append([]config.CredentialConfig(nil), r.staticCreds...), filtered...)
	if len(newCreds) == 0 {
		// Nothing to update — keep existing credentials to avoid empty-list panics.
		return
	}
	for i := range newCreds {
		preserveProviderScopeMetadata(&newCreds[i], existing[newCreds[i].Name])
	}

	// Upsert rate-limiter limits for all DB creds (not just new ones).
	// AddCredentialWithTPM overwrites the existing entry, so calling it every sync
	// guarantees that RPM/TPM changes in DB are picked up immediately.
	for _, c := range filtered {
		tpm := c.TPM
		if tpm == 0 {
			tpm = -1
		}
		r.rateLimiter.AddCredentialWithTPM(c.Name, c.RPM, tpm)
	}

	// Rebuild the O(1) index.
	newIndex := make(map[string]int, len(newCreds))
	for i, c := range newCreds {
		newIndex[c.Name] = i
	}

	r.credentials = newCreds
	r.credentialIndex = newIndex
}

func preserveProviderScopeMetadata(next *config.CredentialConfig, previous config.CredentialConfig) {
	if !next.IsProxyLike() {
		return
	}
	if next.SameProviderIdentity(previous) {
		next.ProviderScopes = append([]string(nil), previous.ProviderScopes...)
		next.ProviderDeniedScopes = append([]string(nil), previous.ProviderDeniedScopes...)
		next.ProviderScopeExpression = scope.NormalizeExpression(previous.ProviderScopeExpression)
		next.ProviderScopeKnown = previous.ProviderScopeKnown
		return
	}
	if next.ProviderScopeExpression == nil && len(next.ProviderScopes) == 0 && len(next.ProviderDeniedScopes) == 0 {
		next.ProviderScopeExpression = scope.FalseExpression()
	}
}

// logCredentialTiers logs how the credential pool is split across priority tiers.
func (r *RoundRobin) logCredentialTiers() {
	lastResort := 0
	tiered := 0
	for _, cred := range r.credentials {
		switch {
		case cred.IsLastResort():
			lastResort++
		case cred.Priority > 0:
			tiered++
		}
	}
	r.logger.Info("Credential priority tiers",
		"total_credentials", len(r.credentials),
		"explicit_tier_credentials", tiered,
		"last_resort_credentials", lastResort,
	)
}
