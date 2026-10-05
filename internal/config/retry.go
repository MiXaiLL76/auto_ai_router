package config

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// RetryConfig controls which upstream error responses make the router replay the
// request on the next credential of the same model (and, once those run out, on
// the fallback chain). Everything that is not retried is returned to the client
// as is.
//
// A response is retried when its status code is in the effective status code set
// (credential override > provider override > status_codes) and its body matches
// none of the non-retryable markers. A 400 is additionally kept on the first
// credential when its body matches a bad-request marker.
type RetryConfig struct {
	// StatusCodes are the retryable upstream status codes. An entry is either an
	// exact code (429) or a class ("5xx"). Empty = DefaultRetryStatusCodes.
	StatusCodes []int `yaml:"status_codes,omitempty"`

	// NonRetryableMarkers are case-insensitive substrings of a response body that
	// stop the retry for any status code. Added to DefaultNonRetryableMarkers.
	NonRetryableMarkers []string `yaml:"non_retryable_markers,omitempty"`

	// BadRequestMarkers are case-insensitive substrings of a 400 body that mark
	// the request itself as faulty (it fails identically on every credential).
	// Added to DefaultBadRequestMarkers.
	BadRequestMarkers []string `yaml:"bad_request_markers,omitempty"`

	// ProviderOverrides replace StatusCodes for credentials of one provider type,
	// keyed by the credential's configured type (vllm, openai, ...). Merged over
	// DefaultRetryProviderOverrides: a key given here replaces the built-in one.
	ProviderOverrides map[ProviderType]RetryOverrideConfig `yaml:"provider_overrides,omitempty"`

	// CredentialOverrides replace StatusCodes for one credential, keyed by
	// CredentialConfig.Name. Wins over ProviderOverrides.
	CredentialOverrides map[string]RetryOverrideConfig `yaml:"credential_overrides,omitempty"`
}

// RetryOverrideConfig is the scoped form of RetryConfig.StatusCodes.
type RetryOverrideConfig struct {
	StatusCodes []int `yaml:"status_codes"`
}

// DefaultRetryStatusCodes is the historical retry set: request errors that may
// depend on the credential (400 and 404 from a provider account), auth and
// payment problems, rate limits and every 5xx.
var DefaultRetryStatusCodes = append([]int{400, 401, 402, 403, 404, 429}, statusClass(5)...)

// DefaultNonRetryableMarkers are content-policy rejections: the provider refused
// the content, and every other credential would refuse it too.
var DefaultNonRetryableMarkers = []string{
	"content policy",
	"content management policy",
	"policy violation",
}

// DefaultBadRequestMarkers are upstream 400 texts that fault the request itself —
// a parameter the target model does not implement, a media part it cannot use —
// rather than anything about the credential that served it. Replaying such a
// request on the next credential returns the identical 400.
//
// Kept deliberately narrow: anything that could plausibly differ between
// credentials (a model missing from one account, a disabled API, a quota) must
// stay retryable, since moving to another credential is exactly what fixes those.
var DefaultBadRequestMarkers = []string{
	"penalty is not enabled",
	"thinking level is unsupported",
	"thinking level minimal is not supported",
	"unsupported mime type",
	"required oneof field",
	"but the supported range is from",
}

// DefaultRetryProviderOverrides drops 400 from the retry set of self-hosted vLLM:
// every replica of a vLLM model runs the same weights, chat template and limits,
// and there is no account or quota behind it, so a 400 (unsupported
// reasoning_effort, max_tokens over the context, prompt too long) comes back
// identical from every replica.
var DefaultRetryProviderOverrides = map[ProviderType]RetryOverrideConfig{
	ProviderTypeVLLM: {StatusCodes: append([]int{404, 429}, statusClass(5)...)},
}

// DefaultRetryConfig returns the built-in retry policy.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{}.WithDefaults()
}

// WithDefaults returns a copy with the built-in values filled in: the default
// status codes when none are set, the built-in markers ahead of the configured
// ones, and the built-in provider overrides under the configured ones.
func (c RetryConfig) WithDefaults() RetryConfig {
	out := RetryConfig{
		StatusCodes:         slices.Clone(c.StatusCodes),
		NonRetryableMarkers: mergeMarkers(DefaultNonRetryableMarkers, c.NonRetryableMarkers),
		BadRequestMarkers:   mergeMarkers(DefaultBadRequestMarkers, c.BadRequestMarkers),
		ProviderOverrides:   make(map[ProviderType]RetryOverrideConfig, len(DefaultRetryProviderOverrides)+len(c.ProviderOverrides)),
		CredentialOverrides: make(map[string]RetryOverrideConfig, len(c.CredentialOverrides)),
	}
	if len(out.StatusCodes) == 0 {
		out.StatusCodes = slices.Clone(DefaultRetryStatusCodes)
	}
	for k, v := range DefaultRetryProviderOverrides {
		out.ProviderOverrides[k] = v
	}
	for k, v := range c.ProviderOverrides {
		out.ProviderOverrides[k] = v
	}
	for k, v := range c.CredentialOverrides {
		out.CredentialOverrides[k] = v
	}
	return out
}

func mergeMarkers(builtin, extra []string) []string {
	out := make([]string, 0, len(builtin)+len(extra))
	seen := make(map[string]struct{}, len(builtin)+len(extra))
	for _, list := range [][]string{builtin, extra} {
		for _, m := range list {
			m = strings.ToLower(strings.TrimSpace(m))
			if m == "" {
				continue
			}
			if _, dup := seen[m]; dup {
				continue
			}
			seen[m] = struct{}{}
			out = append(out, m)
		}
	}
	return out
}

func statusClass(class int) []int {
	codes := make([]int, 0, 100)
	for code := class * 100; code < class*100+100; code++ {
		codes = append(codes, code)
	}
	return codes
}

// parseRetryStatusCodes expands status code entries: exact codes ("429") and
// classes ("4xx", "5xx"). Duplicates are dropped; the result is sorted.
func parseRetryStatusCodes(entries []string, field string) ([]int, error) {
	set := make(map[int]struct{}, len(entries))
	for _, raw := range entries {
		entry := strings.ToLower(strings.TrimSpace(resolveEnvString(raw)))
		if len(entry) == 3 && strings.HasSuffix(entry, "xx") {
			class := int(entry[0] - '0')
			if class < 1 || class > 5 {
				return nil, fmt.Errorf("invalid %s entry %q: status class must be 1xx..5xx", field, raw)
			}
			for _, code := range statusClass(class) {
				set[code] = struct{}{}
			}
			continue
		}
		code, err := strconv.Atoi(entry)
		if err != nil || code < 100 || code > 599 {
			return nil, fmt.Errorf("invalid %s entry %q: want a status code (100-599) or a class like \"5xx\"", field, raw)
		}
		set[code] = struct{}{}
	}
	codes := make([]int, 0, len(set))
	for code := range set {
		codes = append(codes, code)
	}
	slices.Sort(codes)
	return codes, nil
}

// UnmarshalYAML accepts status code classes ("5xx") next to exact codes.
func (c *RetryConfig) UnmarshalYAML(value *yaml.Node) error {
	type tempOverride struct {
		StatusCodes []string `yaml:"status_codes"`
	}
	type tempConfig struct {
		StatusCodes         []string                `yaml:"status_codes,omitempty"`
		NonRetryableMarkers []string                `yaml:"non_retryable_markers,omitempty"`
		BadRequestMarkers   []string                `yaml:"bad_request_markers,omitempty"`
		ProviderOverrides   map[string]tempOverride `yaml:"provider_overrides,omitempty"`
		CredentialOverrides map[string]tempOverride `yaml:"credential_overrides,omitempty"`
	}
	var temp tempConfig
	if err := value.Decode(&temp); err != nil {
		return err
	}

	var err error
	if c.StatusCodes, err = parseRetryStatusCodes(temp.StatusCodes, "retry.status_codes"); err != nil {
		return err
	}
	c.NonRetryableMarkers = temp.NonRetryableMarkers
	c.BadRequestMarkers = temp.BadRequestMarkers

	c.ProviderOverrides = nil
	if len(temp.ProviderOverrides) > 0 {
		c.ProviderOverrides = make(map[ProviderType]RetryOverrideConfig, len(temp.ProviderOverrides))
		for rawType, override := range temp.ProviderOverrides {
			field := "retry.provider_overrides." + rawType + ".status_codes"
			providerType := normalizeProviderType(rawType)
			if !providerType.IsValid() {
				return fmt.Errorf("invalid retry.provider_overrides key %q: unknown provider type", rawType)
			}
			codes, err := parseRetryStatusCodes(override.StatusCodes, field)
			if err != nil {
				return err
			}
			c.ProviderOverrides[providerType] = RetryOverrideConfig{StatusCodes: codes}
		}
	}

	c.CredentialOverrides = nil
	if len(temp.CredentialOverrides) > 0 {
		c.CredentialOverrides = make(map[string]RetryOverrideConfig, len(temp.CredentialOverrides))
		for credName, override := range temp.CredentialOverrides {
			codes, err := parseRetryStatusCodes(override.StatusCodes, "retry.credential_overrides."+credName+".status_codes")
			if err != nil {
				return err
			}
			c.CredentialOverrides[credName] = RetryOverrideConfig{StatusCodes: codes}
		}
	}
	return nil
}
