package config

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// providerMessagesTestConfigBase is a config that loads: the router refuses
// to start without a server, a master key and a credential.
const providerMessagesTestConfigBase = `
server:
  port: 8080
  master_key: test
  max_body_size_mb: 10
  response_body_multiplier: 1
credentials:
  - name: "provider_1"
    type: "openai"
    api_key: "sk-test"
    base_url: "https://api.openai.com"
    rpm: 60
`

// The provider_messages section is how a deployment adds the provider
// messages the router may relay to a client. It is a list of rules: the
// provider's own words, and the router's answer to them.
func TestProviderMessagesConfig_Parses(t *testing.T) {
	cfg, err := loadConfigFromString(t, providerMessagesTestConfigBase+`
provider_messages:
  messages:
    - provider: acme
      provider_code: InvalidParameter
      match: agent capabilities are not enabled
      message: This model does not support agent capabilities.
      code: invalid_request
      status: 400
    - match: unsupported model
      message: Invalid model
      code: invalid_model
      status: 404
`)
	require.NoError(t, err)

	require.Len(t, cfg.ProviderMessages.Messages, 2)

	first := cfg.ProviderMessages.Messages[0]
	assert.Equal(t, "acme", first.Provider)
	assert.Equal(t, "InvalidParameter", first.ProviderCode)
	assert.Equal(t, "agent capabilities are not enabled", first.Match)
	assert.Equal(t, "This model does not support agent capabilities.", first.Message)
	assert.Equal(t, "invalid_request", first.Code)
	assert.Equal(t, 400, first.Status)

	second := cfg.ProviderMessages.Messages[1]
	assert.Empty(t, second.Provider)
	assert.Empty(t, second.ProviderCode)
	assert.Equal(t, "unsupported model", second.Match)
	assert.Equal(t, "Invalid model", second.Message)
	assert.Equal(t, "invalid_model", second.Code)
	assert.Equal(t, 404, second.Status)
}

// TestProviderMessagesConfig_Rules checks that the section becomes the rules
// the router matches against.
func TestProviderMessagesConfig_Rules(t *testing.T) {
	cfg, err := loadConfigFromString(t, providerMessagesTestConfigBase+`
provider_messages:
  messages:
    - provider_code: InvalidParameter
      match: unsupported model
      message: Invalid model
      code: invalid_model
      status: 400
`)
	require.NoError(t, err)

	rules := cfg.ProviderMessages.Rules()
	require.Len(t, rules, 1)
	assert.Equal(t, "InvalidParameter", rules[0].ProviderCode)
	assert.Equal(t, "unsupported model", rules[0].Match)
	assert.Equal(t, "invalid_model", rules[0].Code)
	assert.Equal(t, "Invalid model", rules[0].Message)
	assert.Equal(t, 400, rules[0].Status)
}

// TestProviderMessagesConfig_AbsentSection checks that a config without the
// section is valid and yields no rules: the router's own built-in rules
// stand.
func TestProviderMessagesConfig_AbsentSection(t *testing.T) {
	cfg, err := loadConfigFromString(t, providerMessagesTestConfigBase)
	require.NoError(t, err)

	assert.Empty(t, cfg.ProviderMessages.Messages)
	assert.Empty(t, cfg.ProviderMessages.Rules())
	require.NoError(t, cfg.ProviderMessages.Validate())
}

// TestProviderMessagesConfig_RejectsBadRules checks what the section
// refuses.
//
// The checks are the section's whole safety argument: a rule that matched
// everything, or that named no wording, would put the provider's free text
// into a client's response body -- which is exactly what this mechanism
// exists to prevent. A bad rule must stop the router from starting, not
// surface later as a wrong answer, so the error comes out of the config load.
func TestProviderMessagesConfig_RejectsBadRules(t *testing.T) {
	for _, tt := range []struct {
		name    string
		yaml    string
		wantErr string
	}{
		{
			name: "a rule that matches everything",
			yaml: `
provider_messages:
  messages:
    - message: Invalid request
      code: invalid_request
      status: 400
`,
			wantErr: "match or provider_code is required",
		},
		{
			name: "a rule without a wording",
			yaml: `
provider_messages:
  messages:
    - provider_code: InvalidParameter
      code: invalid_request
      status: 400
`,
			wantErr: "message is required",
		},
		{
			name: "a rule without a code",
			yaml: `
provider_messages:
  messages:
    - match: unsupported model
      message: Invalid model
      status: 400
`,
			wantErr: "code is required",
		},
		{
			name: "a rule with a 5xx status",
			yaml: `
provider_messages:
  messages:
    - match: unsupported model
      message: Invalid model
      code: invalid_model
      status: 502
`,
			wantErr: "status must be a 4xx status",
		},
		{
			name: "a rule with a 2xx status",
			yaml: `
provider_messages:
  messages:
    - match: unsupported model
      message: Invalid model
      code: invalid_model
      status: 200
`,
			wantErr: "status must be a 4xx status",
		},
		{
			name: "a duplicated rule",
			yaml: `
provider_messages:
  messages:
    - match: unsupported model
      message: Invalid model
      code: invalid_model
      status: 400
    - match: unsupported model
      message: Another wording
      code: invalid_model
      status: 400
`,
			wantErr: "duplicate rule",
		},
		{
			name:    "a control character in the wording",
			yaml:    "provider_messages:\n  messages:\n    - match: \"unsupported\\u0007model\"\n      message: Invalid model\n      code: invalid_model\n      status: 400\n",
			wantErr: "control character",
		},
		{
			name: "an over-long wording",
			yaml: `
provider_messages:
  messages:
    - match: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
      message: Invalid model
      code: invalid_model
      status: 400
`,
			wantErr: "longer than",
		},
		{
			// A reserved code carries a meaning of its own, and the router's
			// own bodies are recognized by it -- so a rule reusing one
			// produces a body the router cannot recognize as its own, and
			// the operator's message is lost on the second classification
			// pass. The config must not load.
			name: "a reserved router code",
			yaml: `
provider_messages:
  messages:
    - match: insufficient permission to call this resource
      message: Caller lacks permission for this resource
      code: account_blocked
      status: 400
`,
			wantErr: "is reserved by the router",
		},
		{
			// Two rules that name no text match every failure carrying their
			// code -- the same rule twice. The second would never fire, and
			// the operator would see a rule that loads cleanly and does
			// nothing.
			name: "two rules with the same provider code and no text",
			yaml: `
provider_messages:
  messages:
    - provider_code: InvalidParameter
      message: First
      code: invalid_request
      status: 400
    - provider_code: InvalidParameter
      message: Second
      code: invalid_parameter
      status: 400
`,
			wantErr: "another rule already matches every failure with provider_code",
		},
		{
			// The same, the other way round: two rules that name no code
			// match every failure carrying their text.
			name: "two rules with the same text and no provider code",
			yaml: `
provider_messages:
  messages:
    - match: unsupported model
      message: First
      code: invalid_request
      status: 400
    - match: unsupported model
      message: Second
      code: invalid_model
      status: 400
`,
			wantErr: "another rule already matches every failure with match",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			// The section is validated with the rest of the config, so a
			// bad rule comes out of the load itself: the router must not
			// start with a rule that does not do what it says.
			_, err := loadConfigFromString(t, providerMessagesTestConfigBase+tt.yaml)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)
		})
	}
}

// TestProviderMessagesConfig_RejectsBadYAML checks that a section that is
// not a list of rules is refused rather than silently ignored: a config that
// does not do what it says is worse than one that fails.
func TestProviderMessagesConfig_RejectsBadYAML(t *testing.T) {
	_, err := loadConfigFromString(t, providerMessagesTestConfigBase+`
provider_messages:
  messages:
    - match: unsupported model
      message: Invalid model
      code: invalid_model
      status: not-a-number
`)
	require.Error(t, err)
}

// TestProviderMessagesConfig_EnvSubstitution checks that a value may come
// from the environment, like every other config value.
func TestProviderMessagesConfig_EnvSubstitution(t *testing.T) {
	t.Setenv("ROUTER_PROVIDER_MESSAGE", "unsupported model")

	cfg, err := loadConfigFromString(t, providerMessagesTestConfigBase+`
provider_messages:
  messages:
    - match: os.environ/ROUTER_PROVIDER_MESSAGE
      message: Invalid model
      code: invalid_model
      status: 400
`)
	require.NoError(t, err)

	require.Len(t, cfg.ProviderMessages.Messages, 1)
	assert.Equal(t, "unsupported model", cfg.ProviderMessages.Messages[0].Match)
}

// The status a rule asks for must be a 4xx one: a 5xx would tell the client
// the provider is broken, which is the opposite of what a rule describes.
func TestProviderMessagesConfig_StatusMustBeClientFault(t *testing.T) {
	cfg := &ProviderMessagesConfig{Messages: []ProviderMessageConfig{{
		Match:   "unsupported model",
		Message: "Invalid model",
		Code:    "invalid_model",
		Status:  http.StatusBadRequest,
	}}}
	require.NoError(t, cfg.Validate())
}

// TestProviderMessagesConfig_Retryable checks that the retryable field is
// read and carried into the rule.
//
// It is the answer to a question the status alone cannot answer: a 400 is
// retryable by default (a provider account may lack a model another account
// serves), but a rule about the request's own fault is not -- every
// credential refuses it identically.
func TestProviderMessagesConfig_Retryable(t *testing.T) {
	for _, tt := range []struct {
		name string
		yaml string
		want *bool
	}{
		{
			name: "not retryable",
			yaml: `
provider_messages:
  messages:
    - match: unsupported model
      message: Invalid model
      code: invalid_model
      status: 400
      retryable: false
`,
			want: boolPtr(false),
		},
		{
			name: "retryable",
			yaml: `
provider_messages:
  messages:
    - match: the model is not served by this account
      message: Invalid model
      code: invalid_model
      status: 400
      retryable: true
`,
			want: boolPtr(true),
		},
		{
			name: "unset",
			yaml: `
provider_messages:
  messages:
    - match: unsupported model
      message: Invalid model
      code: invalid_model
      status: 400
`,
			want: nil,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := loadConfigFromString(t, providerMessagesTestConfigBase+tt.yaml)
			require.NoError(t, err)
			require.Len(t, cfg.ProviderMessages.Messages, 1)

			rules := cfg.ProviderMessages.Rules()
			require.Len(t, rules, 1)

			if tt.want == nil {
				assert.Nil(t, rules[0].Retryable, "an unset retryable must stay unset")
				return
			}
			require.NotNil(t, rules[0].Retryable)
			assert.Equal(t, *tt.want, *rules[0].Retryable)
		})
	}
}

func boolPtr(v bool) *bool { return &v }
