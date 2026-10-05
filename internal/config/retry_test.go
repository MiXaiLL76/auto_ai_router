package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestRetryConfig_UnmarshalYAML(t *testing.T) {
	var cfg RetryConfig
	err := yaml.Unmarshal([]byte(`
status_codes: [429, "5xx", 408, 429]
non_retryable_markers: ["Prompt blocked"]
bad_request_markers: ["maximum context length"]
provider_overrides:
  hosted_vllm:
    status_codes: [429, "5xx"]
credential_overrides:
  flaky:
    status_codes: []
`), &cfg)
	require.NoError(t, err)

	assert.Len(t, cfg.StatusCodes, 102, "408, 429 and 500-599, deduplicated")
	assert.Equal(t, 408, cfg.StatusCodes[0])
	assert.Equal(t, []string{"Prompt blocked"}, cfg.NonRetryableMarkers)
	assert.Equal(t, []string{"maximum context length"}, cfg.BadRequestMarkers)
	require.Contains(t, cfg.ProviderOverrides, ProviderTypeVLLM, "provider type spellings are normalized")
	assert.Len(t, cfg.ProviderOverrides[ProviderTypeVLLM].StatusCodes, 101)
	require.Contains(t, cfg.CredentialOverrides, "flaky")
	assert.Empty(t, cfg.CredentialOverrides["flaky"].StatusCodes)
}

func TestRetryConfig_UnmarshalYAMLErrors(t *testing.T) {
	for name, doc := range map[string]string{
		"bad code":      `status_codes: [99]`,
		"bad class":     `status_codes: ["6xx"]`,
		"not a code":    `status_codes: ["abc"]`,
		"bad provider":  "provider_overrides:\n  nope:\n    status_codes: [500]",
		"bad override":  "credential_overrides:\n  c:\n    status_codes: [\"x\"]",
		"provider code": "provider_overrides:\n  vllm:\n    status_codes: [700]",
	} {
		t.Run(name, func(t *testing.T) {
			var cfg RetryConfig
			assert.Error(t, yaml.Unmarshal([]byte(doc), &cfg))
		})
	}
}

func TestRetryConfig_WithDefaults(t *testing.T) {
	def := DefaultRetryConfig()
	assert.Equal(t, DefaultRetryStatusCodes, def.StatusCodes)
	assert.Equal(t, DefaultNonRetryableMarkers, def.NonRetryableMarkers)
	assert.Equal(t, DefaultBadRequestMarkers, def.BadRequestMarkers)
	assert.NotContains(t, def.ProviderOverrides[ProviderTypeVLLM].StatusCodes, 400, "vLLM does not retry 400 by default")

	cfg := RetryConfig{
		StatusCodes:         []int{503},
		BadRequestMarkers:   []string{"  Too Long ", "penalty is not enabled"},
		ProviderOverrides:   map[ProviderType]RetryOverrideConfig{ProviderTypeOpenAI: {StatusCodes: []int{429}}},
		CredentialOverrides: map[string]RetryOverrideConfig{"c": {StatusCodes: []int{500}}},
	}.WithDefaults()
	assert.Equal(t, []int{503}, cfg.StatusCodes)
	assert.Equal(t, append(append([]string{}, DefaultBadRequestMarkers...), "too long"), cfg.BadRequestMarkers,
		"configured markers are lower-cased and appended after the built-in ones without duplicates")
	assert.Contains(t, cfg.ProviderOverrides, ProviderTypeVLLM, "built-in provider overrides are kept")
	assert.Equal(t, []int{429}, cfg.ProviderOverrides[ProviderTypeOpenAI].StatusCodes)
	assert.Equal(t, []int{500}, cfg.CredentialOverrides["c"].StatusCodes)
}

func TestLoad_RetrySection(t *testing.T) {
	cfg, err := Load(writeTempConfig(t, `
server:
  port: 8080
  master_key: sk-test
credentials:
  - name: vllm-a
    type: vllm
    base_url: http://vllm:8000/v1
    rpm: -1
    tpm: -1
retry:
  provider_overrides:
    vllm:
      status_codes: [400, "5xx"]
models:
  - name: qwen
    credential: vllm-a
    reasoning_effort_map: {minimal: low, high: medium, default: low}
  - name: glm
    credential: vllm-a
    reasoning_effort_map: '{"medium": "low", "default": "high"}'
`))
	require.NoError(t, err)
	assert.Contains(t, cfg.Retry.ProviderOverrides[ProviderTypeVLLM].StatusCodes, 400)

	require.Len(t, cfg.Models, 2)
	require.NotNil(t, cfg.Models[0].ReasoningEffortMap)
	assert.Equal(t, map[string]string{"minimal": "low", "high": "medium"}, cfg.Models[0].ReasoningEffortMap.Mapping)
	assert.Equal(t, "low", cfg.Models[0].ReasoningEffortMap.Default)
	require.NotNil(t, cfg.Models[1].ReasoningEffortMap, "a JSON string is accepted too")
	assert.Equal(t, "high", cfg.Models[1].ReasoningEffortMap.Default)
}

func writeTempConfig(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	require.NoError(t, os.WriteFile(path, []byte(content), 0o600))
	return path
}
