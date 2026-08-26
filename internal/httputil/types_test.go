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
			name: "credStats.Priority is authoritative even if modelStats disagrees " +
				"(per-model override intentionally not modeled — see doc comment)",
			modelStats: ModelHealthStats{Priority: 999},
			credStats:  CredentialHealthStats{Priority: 300},
			want:       300,
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
