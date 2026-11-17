package config

import (
	"os"
	"path/filepath"
	"testing"
)

// writeConfig writes a config file and returns its path.
func writeConfig(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("writing the config: %v", err)
	}
	return path
}

// TestTheToolsAdaptersCanBeSwitchedOff checks the kill-switch: an explicit
// `enabled: false` in the adapter's section turns the adapter off, and an
// absent one leaves it on.
//
// The switch is what an operator reaches for during a provider outage or a
// runaway cost, so it has to work -- and the default (on) must not change
// the behaviour of a config that says nothing about it.
func TestTheToolsAdaptersCanBeSwitchedOff(t *testing.T) {
	cases := []struct {
		name         string
		body         string
		wantDeepSeek bool
		wantMoonshot bool
	}{
		{
			name: "no section: both on",
			body: `
server:
  port: 8080
  master_key: "test-key"
credentials:
  - name: "provider_1"
    type: "openai"
    api_key: "sk-xxxx"
    base_url: "https://api.openai.com"
    rpm: 60
`,
			wantDeepSeek: true,
			wantMoonshot: true,
		},
		{
			name: "an empty section: both on",
			body: `
server:
  port: 8080
  master_key: "test-key"
  deepseek_tools: {}
  moonshot_tools: {}

credentials:
  - name: "provider_1"
    type: "openai"
    api_key: "sk-xxxx"
    base_url: "https://api.openai.com"
    rpm: 60
`,
			wantDeepSeek: true,
			wantMoonshot: true,
		},
		{
			name: "explicitly off: both off",
			body: `
server:
  port: 8080
  master_key: "test-key"
  deepseek_tools:
    enabled: false
  moonshot_tools:
    enabled: false

credentials:
  - name: "provider_1"
    type: "openai"
    api_key: "sk-xxxx"
    base_url: "https://api.openai.com"
    rpm: 60
`,
			wantDeepSeek: false,
			wantMoonshot: false,
		},
		{
			name: "explicitly on: both on",
			body: `
server:
  port: 8080
  master_key: "test-key"
  deepseek_tools:
    enabled: true
  moonshot_tools:
    enabled: true

credentials:
  - name: "provider_1"
    type: "openai"
    api_key: "sk-xxxx"
    base_url: "https://api.openai.com"
    rpm: 60
`,
			wantDeepSeek: true,
			wantMoonshot: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := Load(writeConfig(t, tc.body))
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			deepSeekOn := cfg.Server.DeepSeekTools.Enabled == nil || *cfg.Server.DeepSeekTools.Enabled
			if deepSeekOn != tc.wantDeepSeek {
				t.Errorf("the DeepSeek adapter is %v, want %v", deepSeekOn, tc.wantDeepSeek)
			}
			moonshotOn := cfg.Server.MoonshotTools.Enabled == nil || *cfg.Server.MoonshotTools.Enabled
			if moonshotOn != tc.wantMoonshot {
				t.Errorf("the Moonshot adapter is %v, want %v", moonshotOn, tc.wantMoonshot)
			}
		})
	}
}

// TestTheToolsAdapterSettingsFallBackToTheDefaults checks that an absent
// value takes the provider's own default: the limits are the provider's, the
// step budget and the retries the router's.
func TestTheToolsAdapterSettingsFallBackToTheDefaults(t *testing.T) {
	cfg, err := Load(writeConfig(t, `
server:
  port: 8080
  master_key: "test-key"
  moonshot_tools: {}
credentials:
  - name: "provider_1"
    type: "openai"
    api_key: "sk-xxxx"
    base_url: "https://api.openai.com"
    rpm: 60
`))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Server.MoonshotTools.RequestTimeout != DefaultMoonshotToolsRequestTimeout {
		t.Errorf("the request timeout = %v, want %v", cfg.Server.MoonshotTools.RequestTimeout, DefaultMoonshotToolsRequestTimeout)
	}
	if cfg.Server.MoonshotTools.SearchQPS != DefaultMoonshotSearchQPS {
		t.Errorf("the search limit = %d, want %d", cfg.Server.MoonshotTools.SearchQPS, DefaultMoonshotSearchQPS)
	}
	if cfg.Server.MoonshotTools.SearchProQPS != DefaultMoonshotSearchProQPS {
		t.Errorf("the search_pro limit = %d, want %d", cfg.Server.MoonshotTools.SearchProQPS, DefaultMoonshotSearchProQPS)
	}
	if cfg.Server.MoonshotTools.FetchQPS != DefaultMoonshotFetchQPS {
		t.Errorf("the fetch limit = %d, want %d", cfg.Server.MoonshotTools.FetchQPS, DefaultMoonshotFetchQPS)
	}
	if cfg.Server.MoonshotTools.MaxSteps != DefaultMoonshotToolsMaxSteps {
		t.Errorf("the step budget = %d, want %d", cfg.Server.MoonshotTools.MaxSteps, DefaultMoonshotToolsMaxSteps)
	}
	if cfg.Server.MoonshotTools.MaxRetries != DefaultMoonshotToolsMaxRetries {
		t.Errorf("the retries = %d, want %d", cfg.Server.MoonshotTools.MaxRetries, DefaultMoonshotToolsMaxRetries)
	}
}

// TestTheToolsAdapterSettingsAreRead checks that the adapter's own settings
// are read, not just its switch.
func TestTheToolsAdapterSettingsAreRead(t *testing.T) {
	cfg, err := Load(writeConfig(t, `
server:
  port: 8080
  master_key: "test-key"
  deepseek_tools:
    request_timeout: 45s
    search_qps: 50
    fetch_qps: 10
    max_steps: 4
    max_retries: 1
  moonshot_tools:
    request_timeout: 20s
    search_qps: 30
    search_pro_qps: 20
    fetch_qps: 5
    max_steps: 3
    max_retries: 4

credentials:
  - name: "provider_1"
    type: "openai"
    api_key: "sk-xxxx"
    base_url: "https://api.openai.com"
    rpm: 60
`))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Server.DeepSeekTools.RequestTimeout.Seconds() != 45 {
		t.Errorf("the DeepSeek timeout = %v, want 45s", cfg.Server.DeepSeekTools.RequestTimeout)
	}
	if cfg.Server.DeepSeekTools.SearchQPS != 50 || cfg.Server.DeepSeekTools.FetchQPS != 10 {
		t.Errorf("the DeepSeek limits = %d/%d, want 50/10", cfg.Server.DeepSeekTools.SearchQPS, cfg.Server.DeepSeekTools.FetchQPS)
	}
	if cfg.Server.DeepSeekTools.MaxSteps != 4 || cfg.Server.DeepSeekTools.MaxRetries != 1 {
		t.Errorf("the DeepSeek budget = %d/%d, want 4/1", cfg.Server.DeepSeekTools.MaxSteps, cfg.Server.DeepSeekTools.MaxRetries)
	}
	if cfg.Server.MoonshotTools.RequestTimeout.Seconds() != 20 {
		t.Errorf("the Moonshot timeout = %v, want 20s", cfg.Server.MoonshotTools.RequestTimeout)
	}
	if cfg.Server.MoonshotTools.SearchQPS != 30 || cfg.Server.MoonshotTools.SearchProQPS != 20 || cfg.Server.MoonshotTools.FetchQPS != 5 {
		t.Errorf("the Moonshot limits = %d/%d/%d, want 30/20/5",
			cfg.Server.MoonshotTools.SearchQPS, cfg.Server.MoonshotTools.SearchProQPS, cfg.Server.MoonshotTools.FetchQPS)
	}
	if cfg.Server.MoonshotTools.MaxSteps != 3 || cfg.Server.MoonshotTools.MaxRetries != 4 {
		t.Errorf("the Moonshot budget = %d/%d, want 3/4", cfg.Server.MoonshotTools.MaxSteps, cfg.Server.MoonshotTools.MaxRetries)
	}
}
