package ratelimit

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

// defaultContextTimeout is used when no context is provided
const defaultContextTimeout = 30 * time.Second

// RPMLimiter tracks and enforces RPM (Requests Per Minute) and TPM (Tokens Per Minute) limits.
// Use this for API rate limiting where you need to track usage against configurable limits.
//
// Different from TimeBasedRateLimiter:
// - RPMLimiter: tracks usage against RPM/TPM limits (e.g., allow max 100 requests per minute)
// - TimeBasedRateLimiter: enforces fixed minimum interval (e.g., wait 100ms between requests)
//
// Counter storage is delegated to a counterBackend (local in-process or distributed Redis).
// The RPMLimiter itself only stores the configured limits.
type RPMLimiter struct {
	mu          sync.RWMutex
	limits      map[string]*limiterConfig // credential name → limits
	modelLimits map[string]*limiterConfig // "credential:model" → limits
	backend     counterBackend
}

// limiterConfig holds the configured RPM and TPM limits for one key.
// -1 means unlimited.
type limiterConfig struct {
	rpm int
	tpm int
}

// New creates a new RPMLimiter backed by the in-process local backend.
func New() *RPMLimiter {
	return &RPMLimiter{
		limits:      make(map[string]*limiterConfig),
		modelLimits: make(map[string]*limiterConfig),
		backend:     newLocalBackend(),
	}
}

// NewWithRedis creates an RPMLimiter that stores counters in Redis/Valkey.
func NewWithRedis(b *RedisBackend) *RPMLimiter {
	return &RPMLimiter{
		limits:      make(map[string]*limiterConfig),
		modelLimits: make(map[string]*limiterConfig),
		backend:     b,
	}
}

// NewWithHybrid creates an RPMLimiter backed by a HybridBackend: all hot-path
// decisions are made locally (zero added latency) and Redis is updated
// asynchronously in batches. syncInterval controls how often remote stats are
// pulled (default 5s when zero).
func NewWithHybrid(b *HybridBackend) *RPMLimiter {
	return &RPMLimiter{
		limits:      make(map[string]*limiterConfig),
		modelLimits: make(map[string]*limiterConfig),
		backend:     b,
	}
}

// makeModelKey creates a composite key for a (credential, model) pair.
func makeModelKey(credentialName, modelName string) string {
	return fmt.Sprintf("%s:%s", credentialName, modelName)
}

// credKey returns the backend counter key for a credential.
func credKey(credName string) string { return "c:" + credName }

// modelCounterKey returns the backend counter key for a (credential, model) pair.
func modelCounterKey(credName, modelName string) string {
	return "m:" + makeModelKey(credName, modelName)
}

func (r *RPMLimiter) getCredentialConfig(name string) *limiterConfig {
	r.mu.RLock()
	cfg := r.limits[name]
	r.mu.RUnlock()
	return cfg
}

func (r *RPMLimiter) getModelConfig(credName, modelName string) *limiterConfig {
	r.mu.RLock()
	cfg := r.modelLimits[makeModelKey(credName, modelName)]
	r.mu.RUnlock()
	return cfg
}

func (r *RPMLimiter) AddCredential(name string, rpm int) {
	r.AddCredentialWithTPM(name, rpm, -1)
}

func (r *RPMLimiter) AddCredentialWithTPM(name string, rpm int, tpm int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.limits[name] = &limiterConfig{rpm: rpm, tpm: tpm}
}

func (r *RPMLimiter) AddModel(credentialName, modelName string, rpm int) {
	r.AddModelWithTPM(credentialName, modelName, rpm, -1)
}

func (r *RPMLimiter) AddModelWithTPM(credentialName, modelName string, rpm int, tpm int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	key := makeModelKey(credentialName, modelName)
	r.modelLimits[key] = &limiterConfig{rpm: rpm, tpm: tpm}
}

// RemoveModel removes model-level limits and usage counters for a credential/model pair.
func (r *RPMLimiter) RemoveModel(credentialName, modelName string) {
	r.RemoveModelCtx(context.Background(), credentialName, modelName)
}

// RemoveModelCtx removes model-level limits and usage counters for a credential/model pair.
func (r *RPMLimiter) RemoveModelCtx(ctx context.Context, credentialName, modelName string) {
	r.mu.Lock()
	delete(r.modelLimits, makeModelKey(credentialName, modelName))
	r.mu.Unlock()

	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultContextTimeout)
		defer cancel()
	}
	r.backend.deleteKey(ctx, modelCounterKey(credentialName, modelName))
}

// Allow checks if a request for credentialName is allowed (RPM) and records it.
func (r *RPMLimiter) Allow(credentialName string) bool {
	return r.AllowCtx(context.Background(), credentialName)
}

// AllowCtx checks if a request for credentialName is allowed (RPM) and records it.
// It uses the provided context for timeout/cancellation.
func (r *RPMLimiter) AllowCtx(ctx context.Context, credentialName string) bool {
	cfg := r.getCredentialConfig(credentialName)
	if cfg == nil {
		return false
	}
	// Use default timeout if context has no deadline
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultContextTimeout)
		defer cancel()
	}
	return r.backend.tryAllowRPM(ctx, credKey(credentialName), cfg.rpm)
}

// CanAllow checks whether a request would be allowed without recording it.
func (r *RPMLimiter) CanAllow(credentialName string) bool {
	return r.CanAllowCtx(context.Background(), credentialName)
}

// CanAllowCtx checks whether a request would be allowed without recording it.
func (r *RPMLimiter) CanAllowCtx(ctx context.Context, credentialName string) bool {
	cfg := r.getCredentialConfig(credentialName)
	if cfg == nil {
		return false
	}
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultContextTimeout)
		defer cancel()
	}
	return r.backend.canAllowRPM(ctx, credKey(credentialName), cfg.rpm)
}

// AllowModel checks model-level RPM and records if allowed.
// Returns true if the model is not tracked (no limit configured).
func (r *RPMLimiter) AllowModel(credentialName, modelName string) bool {
	return r.AllowModelCtx(context.Background(), credentialName, modelName)
}

// AllowModelCtx checks model-level RPM and records if allowed.
func (r *RPMLimiter) AllowModelCtx(ctx context.Context, credentialName, modelName string) bool {
	cfg := r.getModelConfig(credentialName, modelName)
	if cfg == nil {
		return true
	}
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultContextTimeout)
		defer cancel()
	}
	return r.backend.tryAllowRPM(ctx, modelCounterKey(credentialName, modelName), cfg.rpm)
}

// CanAllowModel checks model-level RPM without recording.
// Returns true if the model is not tracked.
func (r *RPMLimiter) CanAllowModel(credentialName, modelName string) bool {
	return r.CanAllowModelCtx(context.Background(), credentialName, modelName)
}

// CanAllowModelCtx checks model-level RPM without recording.
func (r *RPMLimiter) CanAllowModelCtx(ctx context.Context, credentialName, modelName string) bool {
	cfg := r.getModelConfig(credentialName, modelName)
	if cfg == nil {
		return true
	}
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultContextTimeout)
		defer cancel()
	}
	return r.backend.canAllowRPM(ctx, modelCounterKey(credentialName, modelName), cfg.rpm)
}

// AllowTokens checks whether the credential TPM limit permits further requests.
func (r *RPMLimiter) AllowTokens(credentialName string) bool {
	return r.AllowTokensCtx(context.Background(), credentialName)
}

// AllowTokensCtx checks whether the credential TPM limit permits further requests.
func (r *RPMLimiter) AllowTokensCtx(ctx context.Context, credentialName string) bool {
	cfg := r.getCredentialConfig(credentialName)
	if cfg == nil {
		return false
	}
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultContextTimeout)
		defer cancel()
	}
	return r.backend.canAllowTPM(ctx, credKey(credentialName), cfg.tpm)
}

// AllowModelTokens checks whether the model TPM limit permits further requests.
func (r *RPMLimiter) AllowModelTokens(credentialName, modelName string) bool {
	return r.AllowModelTokensCtx(context.Background(), credentialName, modelName)
}

// AllowModelTokensCtx checks whether the model TPM limit permits further requests.
func (r *RPMLimiter) AllowModelTokensCtx(ctx context.Context, credentialName, modelName string) bool {
	cfg := r.getModelConfig(credentialName, modelName)
	if cfg == nil {
		return true
	}
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultContextTimeout)
		defer cancel()
	}
	return r.backend.canAllowTPM(ctx, modelCounterKey(credentialName, modelName), cfg.tpm)
}

// ConsumeTokens records token usage for a credential.
func (r *RPMLimiter) ConsumeTokens(credentialName string, tokenCount int) {
	r.ConsumeTokensCtx(context.Background(), credentialName, tokenCount)
}

// ConsumeTokensCtx records token usage for a credential.
func (r *RPMLimiter) ConsumeTokensCtx(ctx context.Context, credentialName string, tokenCount int) {
	if r.getCredentialConfig(credentialName) == nil {
		return
	}
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultContextTimeout)
		defer cancel()
	}
	r.backend.consumeTokens(ctx, credKey(credentialName), tokenCount)
}

// ConsumeModelTokens records token usage for a model within a credential.
func (r *RPMLimiter) ConsumeModelTokens(credentialName, modelName string, tokenCount int) {
	r.ConsumeModelTokensCtx(context.Background(), credentialName, modelName, tokenCount)
}

// ConsumeModelTokensCtx records token usage for a model within a credential.
func (r *RPMLimiter) ConsumeModelTokensCtx(ctx context.Context, credentialName, modelName string, tokenCount int) {
	if r.getModelConfig(credentialName, modelName) == nil {
		return
	}
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultContextTimeout)
		defer cancel()
	}
	r.backend.consumeTokens(ctx, modelCounterKey(credentialName, modelName), tokenCount)
}

// GetCurrentRPM returns the number of requests in the last 60 seconds for a credential.
func (r *RPMLimiter) GetCurrentRPM(credentialName string) int {
	return r.GetCurrentRPMCtx(context.Background(), credentialName)
}

// GetCurrentRPMCtx returns the number of requests in the last 60 seconds for a credential.
func (r *RPMLimiter) GetCurrentRPMCtx(ctx context.Context, credentialName string) int {
	if r.getCredentialConfig(credentialName) == nil {
		return 0
	}
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultContextTimeout)
		defer cancel()
	}
	return r.backend.currentRPM(ctx, credKey(credentialName))
}

// GetCurrentTPM returns the sum of tokens in the last 60 seconds for a credential.
func (r *RPMLimiter) GetCurrentTPM(credentialName string) int {
	return r.GetCurrentTPMCtx(context.Background(), credentialName)
}

// GetCurrentTPMCtx returns the sum of tokens in the last 60 seconds for a credential.
func (r *RPMLimiter) GetCurrentTPMCtx(ctx context.Context, credentialName string) int {
	if r.getCredentialConfig(credentialName) == nil {
		return 0
	}
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultContextTimeout)
		defer cancel()
	}
	return r.backend.currentTPM(ctx, credKey(credentialName))
}

// GetCurrentModelRPM returns the RPM for a (credential, model) pair.
func (r *RPMLimiter) GetCurrentModelRPM(credentialName, modelName string) int {
	return r.GetCurrentModelRPMCtx(context.Background(), credentialName, modelName)
}

// GetCurrentModelRPMCtx returns the RPM for a (credential, model) pair.
func (r *RPMLimiter) GetCurrentModelRPMCtx(ctx context.Context, credentialName, modelName string) int {
	if r.getModelConfig(credentialName, modelName) == nil {
		return 0
	}
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultContextTimeout)
		defer cancel()
	}
	return r.backend.currentRPM(ctx, modelCounterKey(credentialName, modelName))
}

// GetCurrentModelTPM returns the TPM for a (credential, model) pair.
func (r *RPMLimiter) GetCurrentModelTPM(credentialName, modelName string) int {
	return r.GetCurrentModelTPMCtx(context.Background(), credentialName, modelName)
}

// GetCurrentModelTPMCtx returns the TPM for a (credential, model) pair.
func (r *RPMLimiter) GetCurrentModelTPMCtx(ctx context.Context, credentialName, modelName string) int {
	if r.getModelConfig(credentialName, modelName) == nil {
		return 0
	}
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultContextTimeout)
		defer cancel()
	}
	return r.backend.currentTPM(ctx, modelCounterKey(credentialName, modelName))
}

// TryAllowAll atomically checks credential RPM, credential TPM, model RPM, and model TPM.
// Records credential and model RPM if all checks pass. Returns true if allowed.
// modelName may be empty to skip model-level checks.
func (r *RPMLimiter) TryAllowAll(credentialName, modelName string) bool {
	return r.TryAllowAllCtx(context.Background(), credentialName, modelName)
}

// TryAllowAllCtx atomically checks credential RPM, credential TPM, model RPM, and model TPM.
func (r *RPMLimiter) TryAllowAllCtx(ctx context.Context, credentialName, modelName string) bool {
	credCfg := r.getCredentialConfig(credentialName)
	if credCfg == nil {
		return false
	}

	modelRPM := -1
	modelTPM := -1
	mKey := ""

	if modelName != "" {
		modCfg := r.getModelConfig(credentialName, modelName)
		if modCfg != nil {
			modelRPM = modCfg.rpm
			modelTPM = modCfg.tpm
			mKey = modelCounterKey(credentialName, modelName)
		}
	}

	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultContextTimeout)
		defer cancel()
	}

	return r.backend.tryAllowAll(
		ctx,
		credKey(credentialName), credCfg.rpm, credCfg.tpm,
		mKey, modelRPM, modelTPM,
	)
}

// SetCredentialCurrentUsage overwrites the sliding-window counters for a credential.
// Used to synchronize usage from remote proxies. No-op for Redis backend.
func (r *RPMLimiter) SetCredentialCurrentUsage(credentialName string, currentRPM, currentTPM int) {
	r.SetCredentialCurrentUsageCtx(context.Background(), credentialName, currentRPM, currentTPM)
}

// SetCredentialCurrentUsageCtx overwrites the sliding-window counters for a credential.
func (r *RPMLimiter) SetCredentialCurrentUsageCtx(ctx context.Context, credentialName string, currentRPM, currentTPM int) {
	if r.getCredentialConfig(credentialName) == nil {
		return
	}
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultContextTimeout)
		defer cancel()
	}
	r.backend.setCurrentUsage(ctx, credKey(credentialName), currentRPM, currentTPM)
}

// SetModelCurrentUsage overwrites the sliding-window counters for a (credential, model) pair.
// No-op for Redis backend.
func (r *RPMLimiter) SetModelCurrentUsage(credentialName, modelName string, currentRPM, currentTPM int) {
	r.SetModelCurrentUsageCtx(context.Background(), credentialName, modelName, currentRPM, currentTPM)
}

// SetModelCurrentUsageCtx overwrites the sliding-window counters for a (credential, model) pair.
func (r *RPMLimiter) SetModelCurrentUsageCtx(ctx context.Context, credentialName, modelName string, currentRPM, currentTPM int) {
	if r.getModelConfig(credentialName, modelName) == nil {
		return
	}
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultContextTimeout)
		defer cancel()
	}
	r.backend.setCurrentUsage(ctx, modelCounterKey(credentialName, modelName), currentRPM, currentTPM)
}

// GetLimitRPM returns the configured RPM limit for a credential (-1 = not tracked).
func (r *RPMLimiter) GetLimitRPM(credentialName string) int {
	cfg := r.getCredentialConfig(credentialName)
	if cfg == nil {
		return -1
	}
	return cfg.rpm
}

// GetLimitTPM returns the configured TPM limit for a credential (-1 = not tracked).
func (r *RPMLimiter) GetLimitTPM(credentialName string) int {
	cfg := r.getCredentialConfig(credentialName)
	if cfg == nil {
		return -1
	}
	return cfg.tpm
}

// GetModelLimitRPM returns the configured RPM limit for a model (-1 = not tracked).
func (r *RPMLimiter) GetModelLimitRPM(credentialName, modelName string) int {
	cfg := r.getModelConfig(credentialName, modelName)
	if cfg == nil {
		return -1
	}
	return cfg.rpm
}

// GetModelLimitTPM returns the configured TPM limit for a model (-1 = not tracked).
func (r *RPMLimiter) GetModelLimitTPM(credentialName, modelName string) int {
	cfg := r.getModelConfig(credentialName, modelName)
	if cfg == nil {
		return -1
	}
	return cfg.tpm
}

// KeyStats holds the current RPM and TPM for a single key.
type KeyStats struct {
	RPM int
	TPM int
}

// BatchCurrentStats fetches RPM and TPM for all given credentials and model pairs
// in a single backend round-trip (one Redis pipeline when using the Redis backend).
// Returns two maps: credName → stats, and "credName:modelName" → stats.
func (r *RPMLimiter) BatchCurrentStats(ctx context.Context, credNames []string, modelPairs []ModelPair) (map[string]KeyStats, map[string]KeyStats) {
	// Build the flat key list that the backend understands.
	allKeys := make([]string, 0, len(credNames)+len(modelPairs))
	for _, name := range credNames {
		allKeys = append(allKeys, credKey(name))
	}
	for _, p := range modelPairs {
		allKeys = append(allKeys, modelCounterKey(p.Credential, p.Model))
	}

	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultContextTimeout)
		defer cancel()
	}

	raw := r.backend.batchCurrentStats(ctx, allKeys)

	credStats := make(map[string]KeyStats, len(credNames))
	for _, name := range credNames {
		v := raw[credKey(name)]
		credStats[name] = KeyStats{RPM: v[0], TPM: v[1]}
	}

	modelStats := make(map[string]KeyStats, len(modelPairs))
	for _, p := range modelPairs {
		v := raw[modelCounterKey(p.Credential, p.Model)]
		modelStats[p.Credential+":"+p.Model] = KeyStats{RPM: v[0], TPM: v[1]}
	}

	return credStats, modelStats
}

// GetAllModels returns all tracked "credential:model" keys.
func (r *RPMLimiter) GetAllModels() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	keys := make([]string, 0, len(r.modelLimits))
	for k := range r.modelLimits {
		keys = append(keys, k)
	}
	return keys
}

// ModelPair represents a parsed credential:model pair.
type ModelPair struct {
	Credential string
	Model      string
}

// GetAllModelPairs returns all tracked (credential, model) pairs pre-parsed.
func (r *RPMLimiter) GetAllModelPairs() []ModelPair {
	r.mu.RLock()
	defer r.mu.RUnlock()
	pairs := make([]ModelPair, 0, len(r.modelLimits))
	for key := range r.modelLimits {
		parts := strings.SplitN(key, ":", 2)
		if len(parts) == 2 {
			pairs = append(pairs, ModelPair{Credential: parts[0], Model: parts[1]})
		}
	}
	return pairs
}
