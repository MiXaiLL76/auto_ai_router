package httputil

import "testing"

func TestEffectiveHealthPriority(t *testing.T) {
	tests := []struct {
		name       string
		modelStats ModelHealthStats
		credStats  CredentialHealthStats
		want       int
	}{
		{
			name:       "credential priority propagates",
			modelStats: ModelHealthStats{Priority: 200},
			credStats:  CredentialHealthStats{Priority: 200},
			want:       200,
		},
		{
			name:       "default group (0) is a valid result, not a sentinel",
			modelStats: ModelHealthStats{Priority: 0},
			credStats:  CredentialHealthStats{Priority: 0},
			want:       0,
		},
		{
			name: "modelStats.Priority is authoritative even if credStats disagrees " +
				"further downstream in a proxy-of-proxy chain, e.g. ru01 -> pol01 -> ... -> cheapgpt)",
			modelStats: ModelHealthStats{Priority: 999},
			credStats:  CredentialHealthStats{Priority: 300},
			want:       999,
		},
		{
			name:       "falls back to credential priority when the model entry has none",
			modelStats: ModelHealthStats{Priority: 0},
			credStats:  CredentialHealthStats{Priority: 150},
			want:       150,
		},
		{
			name:       "last-resort credential priority (999) propagates",
			modelStats: ModelHealthStats{Priority: 0},
			credStats:  CredentialHealthStats{Priority: 999, LastResort: true},
			want:       999,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := EffectiveHealthPriority(tt.modelStats, tt.credStats)
			if got != tt.want {
				t.Errorf("EffectiveHealthPriority() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestModelHealthEntryLive(t *testing.T) {
	tests := []struct {
		name string
		s    ModelHealthStats
		want bool
	}{
		{name: "no limits, not banned", s: ModelHealthStats{}, want: true},
		{name: "banned", s: ModelHealthStats{IsBanned: true}, want: false},
		{name: "rpm under limit", s: ModelHealthStats{LimitRPM: 100, CurrentRPM: 99}, want: true},
		{name: "rpm at limit", s: ModelHealthStats{LimitRPM: 100, CurrentRPM: 100}, want: false},
		{name: "rpm over limit", s: ModelHealthStats{LimitRPM: 100, CurrentRPM: 150}, want: false},
		{name: "tpm at limit", s: ModelHealthStats{LimitTPM: 1000, CurrentTPM: 1000}, want: false},
		{name: "unlimited rpm (0) never counts as exhausted", s: ModelHealthStats{LimitRPM: 0, CurrentRPM: 500}, want: true},
		{name: "untracked rpm (-1) never counts as exhausted", s: ModelHealthStats{LimitRPM: -1, CurrentRPM: 500}, want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ModelHealthEntryLive(tt.s); got != tt.want {
				t.Errorf("ModelHealthEntryLive(%+v) = %v, want %v", tt.s, got, tt.want)
			}
		})
	}
}

func TestModelPriorityTierLive(t *testing.T) {
	tests := []struct {
		name string
		t    ModelPriorityTier
		want bool
	}{
		{name: "uncapped, not banned", t: ModelPriorityTier{Priority: 1}, want: true},
		{name: "banned", t: ModelPriorityTier{Priority: 1, Banned: true}, want: false},
		{name: "rpm at cap", t: ModelPriorityTier{LimitRPM: 60, CurrentRPM: 60}, want: false},
		{name: "rpm under cap", t: ModelPriorityTier{LimitRPM: 60, CurrentRPM: 59}, want: true},
		{name: "tpm at cap", t: ModelPriorityTier{LimitTPM: 5000, CurrentTPM: 6000}, want: false},
		{name: "uncapped rpm ignores usage", t: ModelPriorityTier{LimitRPM: -1, CurrentRPM: 999}, want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ModelPriorityTierLive(tt.t); got != tt.want {
				t.Errorf("ModelPriorityTierLive(%+v) = %v, want %v", tt.t, got, tt.want)
			}
		})
	}
}

func TestEffectiveHealthWeight_StillWorks(t *testing.T) {
	// Sanity check that adding Priority alongside Weight in both structs didn't
	// disturb the existing weight resolution chain.
	if got := EffectiveHealthWeight(ModelHealthStats{Weight: 5}, CredentialHealthStats{Weight: 2}); got != 5 {
		t.Errorf("EffectiveHealthWeight() = %d, want 5", got)
	}
	if got := EffectiveHealthWeight(ModelHealthStats{}, CredentialHealthStats{Weight: 2}); got != 2 {
		t.Errorf("EffectiveHealthWeight() = %d, want 2", got)
	}
	if got := EffectiveHealthWeight(ModelHealthStats{}, CredentialHealthStats{}); got != 1 {
		t.Errorf("EffectiveHealthWeight() = %d, want 1", got)
	}
}
