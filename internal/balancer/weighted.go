package balancer

import (
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

// swrrNode holds the smooth weighted round-robin state for a single credential.
type swrrNode struct {
	current int
}

// schedKey identifies an independent SWRR cycle. Using a comparable struct (rather than a
// formatted string) keeps it allocation-free on the selection hot path.
type schedKey struct {
	model     string
	reqType   config.ProviderType
	excluding bool
	priority  int
	scopeKey  string
}

// swrrState is the SWRR scheduler for one schedKey. Nodes are keyed by credential name so
// the live set can be reconciled cheaply on every request.
type swrrState struct {
	nodes    map[string]*swrrNode
	lastUsed time.Time // updated on every swrrStateFor lookup; see PruneStaleSWRRState
}

func newSWRRState() *swrrState {
	return &swrrState{nodes: make(map[string]*swrrNode), lastUsed: time.Now()}
}

// advance reconciles the live node set, accumulates effective weight into each live node's
// running counter (nginx smooth weighted round-robin), and returns the total live weight.
// Banned/excluded credentials are absent from liveWeights, so they neither accumulate
// (which would cause a burst when they recover) nor get selected.
func (s *swrrState) advance(liveWeights map[string]int) int {
	for name := range s.nodes {
		if _, ok := liveWeights[name]; !ok {
			delete(s.nodes, name)
		}
	}
	total := 0
	for name, w := range liveWeights {
		n, ok := s.nodes[name]
		if !ok {
			n = &swrrNode{}
			s.nodes[name] = n
		}
		n.current += w
		total += w
	}
	return total
}

// commit charges the selected credential the total live weight, the SWRR step that keeps
// long-run selection proportional to configured weights.
func (s *swrrState) commit(name string, total int) {
	if n, ok := s.nodes[name]; ok {
		n.current -= total
	}
}

func (s *swrrState) currentOf(name string) int {
	if n, ok := s.nodes[name]; ok {
		return n.current
	}
	return 0
}

// schedKeyFor builds the SWRR cycle key. The model is only part of the key when model
// filtering is active; otherwise every model shares one candidate set, so they must share
// one cycle too — keeping the key out avoids unbounded map growth from arbitrary model
// names. Must be called with r.mu held.
func (r *RoundRobin) schedKeyFor(modelID string, requiredType config.ProviderType, excluding bool, scopeKey string) schedKey {
	model := modelID
	if model == "" || r.modelChecker == nil || !r.modelChecker.IsEnabled() {
		model = ""
	}
	return schedKey{model: model, reqType: requiredType, excluding: excluding, scopeKey: scopeKey}
}

// swrrStateFor returns (creating if needed) the SWRR scheduler for a selection cycle.
// Must be called with r.mu held.
//
// r.swrr is never pruned by removing unreachable keys as they go stale, only by
// PruneStaleSWRRState — schedKey.priority and .scopeKey (candidateCycleKey, the sorted
// membership of a priority group) are both derived from effectivePriority(), which for
// proxy/AIR credentials can fluctuate as an upstream's health-learned priority changes
// (sub-credentials banning/unbanning). Each distinct (priority, membership) combination
// ever observed leaves its own entry behind once priority moves on, so a caller that
// only touches lastUsed here and never removes entries would grow this map without
// bound over a long-running process's lifetime.
func (r *RoundRobin) swrrStateFor(key schedKey) *swrrState {
	st, ok := r.swrr[key]
	if !ok {
		st = newSWRRState()
		r.swrr[key] = st
	}
	st.lastUsed = time.Now()
	return st
}

// PruneStaleSWRRState removes SWRR scheduler entries not looked up (via swrrStateFor)
// within maxAge, returning the count removed. Intended to be called periodically from a
// background updater (see cmd/server/main.go's startProxyStatsUpdater) — selection
// itself never prunes, so without a periodic external caller r.swrr grows unbounded per
// swrrStateFor's doc comment.
func (r *RoundRobin) PruneStaleSWRRState(maxAge time.Duration) int {
	r.mu.Lock()
	defer r.mu.Unlock()

	cutoff := time.Now().Add(-maxAge)
	removed := 0
	for key, st := range r.swrr {
		// !After rather than Before so maxAge:0 honestly means "prune everything":
		// with a coarse monotonic clock (Windows, ~500µs steps) lastUsed and cutoff
		// can land in the same tick, and a strict Before would spare the entry.
		if !st.lastUsed.After(cutoff) {
			delete(r.swrr, key)
			removed++
		}
	}
	return removed
}

// EffectiveWeight resolves the weighted round-robin fallback chain: model-level override,
// then credential default, then 1.
func EffectiveWeight(modelWeight, credWeight int) int {
	if modelWeight > 0 {
		return modelWeight
	}
	if credWeight > 0 {
		return credWeight
	}
	return 1
}

// learnedProxyPriority returns the per-model priority learned from an upstream /health
// poll for a proxy/AIR credential (e.g. ru01's "grant-pol01"/"comet-ger01" proxy
// credentials picking up per-model priority from pol01/ger01). ok is false when there is
// no upstream to poll, the model checker is off, or nothing has been learned yet.
//
// A learned priority of 0 (the upstream serves the model in its best group) is a real,
// authoritative value — ok is true and callers must use it, not fall through to the
// credential's static priority: field. See internal/models/manager.go
// (LearnedModelPriorityForCredential) for the matching contract note.
func (r *RoundRobin) learnedProxyPriority(cred *config.CredentialConfig, modelID string) (int, bool) {
	if modelID != "" && cred.IsProxyLike() && r.modelChecker != nil && r.modelChecker.IsEnabled() {
		return r.modelChecker.LearnedModelPriorityForCredential(modelID, cred.Name)
	}
	return 0, false
}

// primaryPriority resolves the priority group for cred on the selection cascade: the
// learned per-model proxy priority if any (an upstream proxy/AIR credential inherits the
// tier its upstream actually serves the model from), else the static `priority:` field.
// This is the only priority resolver now — initial pick and retry both use it.
func (r *RoundRobin) primaryPriority(cred *config.CredentialConfig, modelID string) int {
	if p, ok := r.learnedProxyPriority(cred, modelID); ok {
		return p
	}
	return cred.Priority
}

// candidateWeight is the SWRR weight for one selection candidate: a tier candidate
// (Design B) uses its learned tier weight (the summed weight of the upstream leaf
// credentials in that tier), everything else falls through to effectiveWeight.
func (r *RoundRobin) candidateWeight(c candidateEntry, modelID string) int {
	if c.tier != nil && c.tier.weight > 0 {
		return c.tier.weight
	}
	return r.effectiveWeight(c.cred, modelID)
}

// effectiveWeight resolves the weight for a (credential, model) pair, mirroring how RPM
// is resolved: model-level override first, then the credential default, then 1.
func (r *RoundRobin) effectiveWeight(cred *config.CredentialConfig, modelID string) int {
	modelWeight := 0
	if modelID != "" && r.modelChecker != nil && r.modelChecker.IsEnabled() {
		modelWeight = r.modelChecker.GetModelWeightForCredential(modelID, cred.Name)
	}
	return EffectiveWeight(modelWeight, cred.Weight)
}
