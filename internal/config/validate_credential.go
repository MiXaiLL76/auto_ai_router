package config

import (
	"fmt"
	"os"
)

// validate checks one credential: its identity, its transport settings and
// the requirements its provider type imposes. index is only used in the
// "name is required" message, which must stay stable for operators.
func (cred *CredentialConfig) validate(index int) error {
	if err := cred.validateIdentity(index); err != nil {
		return err
	}
	if err := cred.validateProtocolFlags(); err != nil {
		return err
	}
	if err := validateCredentialRequestHeaders(cred.Name, cred.Type, cred.RequestHeaders); err != nil {
		return err
	}
	if err := cred.validateProviderRequirements(); err != nil {
		return err
	}
	return cred.validateLimits()
}

// validateIdentity checks the fields every credential needs to be addressable.
func (cred *CredentialConfig) validateIdentity(index int) error {
	if cred.Name == "" {
		return fmt.Errorf("credential %d: name is required", index)
	}
	if _, err := ParseProxyURL(cred.ProxyURL); err != nil {
		return fmt.Errorf("credential %s: %w", cred.Name, err)
	}
	return nil
}

// validateProtocolFlags checks the type/auth/protocol combinations that are
// independent of the provider's own requirements.
func (cred *CredentialConfig) validateProtocolFlags() error {
	// Validate provider type
	if !cred.Type.IsValid() {
		return fmt.Errorf("credential %s: invalid type: %s (must be 'openai', 'vertex-ai', 'gemini', 'anthropic', 'cometapi', 'proman', 'bedrock', 'proxy', 'air', or 'vllm')", cred.Name, cred.Type)
	}
	if cred.AuthType != "" && cred.AuthType != "bearer" && cred.AuthType != "x-api-key" {
		return fmt.Errorf("credential %s: invalid auth_type: %s (must be 'bearer' or 'x-api-key')", cred.Name, cred.AuthType)
	}
	if cred.OpenAIProtocol && cred.Type != ProviderTypeCometAPI {
		return fmt.Errorf("credential %s: openai_proto is only supported for 'cometapi' type, got: %s", cred.Name, cred.Type)
	}
	if cred.GoogleProtocol && cred.Type != ProviderTypeCometAPI {
		return fmt.Errorf("credential %s: google_proto is only supported for 'cometapi' type, got: %s", cred.Name, cred.Type)
	}
	if cred.OpenAIProtocol && cred.GoogleProtocol {
		return fmt.Errorf("credential %s: openai_proto and google_proto are mutually exclusive", cred.Name)
	}
	if cred.SystemOne && cred.EffectiveProviderType() != ProviderTypeOpenAI {
		return fmt.Errorf("credential %s: system_one needs an OpenAI-wire credential (openai, vllm), got: %s", cred.Name, cred.Type)
	}
	return nil
}

// validateProviderRequirements checks what each provider type requires:
// base_url, api_key, project_id/location for Vertex, credentials_file
// accessibility.
func (cred *CredentialConfig) validateProviderRequirements() error {
	switch cred.Type {
	case ProviderTypeProxy, ProviderTypeAIR:
		return cred.validateRemoteRouter()
	case ProviderTypeVLLM:
		return cred.validateVLLM()
	case ProviderTypeVertexAI:
		return cred.validateVertex()
	case ProviderTypeGemini:
		return cred.validateGemini()
	default:
		return cred.validateOpenAICompatible()
	}
}

// validateRemoteRouter checks proxy/AIR credentials, whose api_key is optional.
func (cred *CredentialConfig) validateRemoteRouter() error {
	// base_url is required for remote router/proxy credentials
	if cred.BaseURL == "" {
		return fmt.Errorf("credential %s: base_url is required for %s type", cred.Name, cred.Type)
	}
	// Validate base_url is a valid URL
	if err := validateBaseURL(cred.Name, cred.BaseURL); err != nil {
		return err
	}
	// api_key is optional for proxy/AIR
	return nil
}

// validateVLLM checks vLLM credentials. vLLM serves an OpenAI-compatible API
// and is commonly deployed without --api-key, so base_url is required but
// api_key is optional.
func (cred *CredentialConfig) validateVLLM() error {
	if cred.BaseURL == "" {
		return fmt.Errorf("credential %s: base_url is required for vllm type", cred.Name)
	}
	return validateBaseURL(cred.Name, cred.BaseURL)
}

// validateVertex checks Vertex AI credentials, whose base_url is optional
// (constructed dynamically).
func (cred *CredentialConfig) validateVertex() error {
	// For Vertex AI, project_id and location are required
	if cred.ProjectID == "" {
		return fmt.Errorf("credential %s: project_id is required for vertex-ai type", cred.Name)
	}
	if cred.Location == "" {
		return fmt.Errorf("credential %s: location is required for vertex-ai type", cred.Name)
	}
	// API Key is required for Vertex AI (Express Mode)
	if cred.APIKey == "" && cred.CredentialsFile == "" && cred.CredentialsJSON == "" {
		return fmt.Errorf("credential %s: api_key, credentials_file, or credentials_json is required for vertex-ai type", cred.Name)
	}
	// Validate credentials_file exists if provided
	if cred.CredentialsFile != "" {
		if _, err := os.Stat(cred.CredentialsFile); err != nil {
			return fmt.Errorf("credential %s: credentials_file does not exist or is not accessible: %w", cred.Name, err)
		}
	}
	return nil
}

// validateGemini checks Gemini (Google AI Studio) credentials, which need
// both api_key and base_url.
func (cred *CredentialConfig) validateGemini() error {
	if cred.APIKey == "" {
		return fmt.Errorf("credential %s: api_key is required for gemini type", cred.Name)
	}
	if cred.BaseURL == "" {
		return fmt.Errorf("credential %s: base_url is required for gemini type", cred.Name)
	}
	return validateBaseURL(cred.Name, cred.BaseURL)
}

// validateOpenAICompatible checks OpenAI- and Anthropic-wire credentials,
// which need both api_key and base_url.
func (cred *CredentialConfig) validateOpenAICompatible() error {
	if cred.APIKey == "" {
		return fmt.Errorf("credential %s: api_key is required", cred.Name)
	}
	if cred.BaseURL == "" {
		return fmt.Errorf("credential %s: base_url is required", cred.Name)
	}
	// Validate base_url is a valid URL
	return validateBaseURL(cred.Name, cred.BaseURL)
}

// validateLimits checks the rate-limit and ordering knobs shared by all
// credential types.
func (cred *CredentialConfig) validateLimits() error {
	// -1 means unlimited RPM
	if cred.RPM <= 0 && !isUnlimited(cred.RPM) {
		return fmt.Errorf("credential %s: invalid rpm: %d (must be -1 for unlimited or positive number)", cred.Name, cred.RPM)
	}
	// TPM: 0 or -1 means unlimited, positive means limited
	if cred.TPM < -1 {
		return fmt.Errorf("credential %s: invalid tpm: %d (must be -1 or 0 for unlimited, or positive number)", cred.Name, cred.TPM)
	}
	// Weight: 0 means default (1), positive means a higher share. Negative is invalid.
	if cred.Weight < 0 {
		return fmt.Errorf("credential %s: invalid weight: %d (must be 0 for default or positive number)", cred.Name, cred.Weight)
	}
	if cred.FallbackPriority < 0 {
		return fmt.Errorf("credential %s: invalid fallback_priority: %d (must be >= 0)", cred.Name, cred.FallbackPriority)
	}
	if cred.IsFallback && cred.FallbackPriority > 0 {
		return fmt.Errorf("credential %s: invalid fallback_priority: fallback credentials cannot set fallback_priority", cred.Name)
	}
	if cred.Priority < 0 {
		return fmt.Errorf("credential %s: invalid priority: %d (must be >= 0)", cred.Name, cred.Priority)
	}
	if cred.Priority > 0 && cred.FallbackPriority > 0 {
		return fmt.Errorf("credential %s: invalid priority: cannot set both priority and fallback_priority", cred.Name)
	}
	if cred.IsFallback && cred.Priority > 0 && cred.Priority != FallbackPriorityGroup {
		return fmt.Errorf("credential %s: invalid priority: fallback credentials cannot set priority (always tried last, pinned to group %d)", cred.Name, FallbackPriorityGroup)
	}
	return nil
}
