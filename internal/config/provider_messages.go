package config

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"gopkg.in/yaml.v3"

	"github.com/mixaill76/auto_ai_router/internal/upstreamerror"
)

const (
	// maxProviderMessageEntries bounds how many provider messages a config
	// may list. The list is read once at startup and matched per error, so a
	// bound is not about speed: it is about a config that is not a config
	// (a whole file pasted in by accident).
	maxProviderMessageEntries = 1024
	// maxProviderMessageTextBytes bounds one provider message's text. A
	// provider's message is a sentence about the request; anything longer is
	// not one, and a long one is the shape of a dump.
	maxProviderMessageTextBytes = 512
)

// ProviderMessageConfig is one provider message the router may relay to a
// client.
//
// The router does not hand a provider's free text to a client as it is: it
// hands out a small fixed set of messages instead. This config is how an
// operator adds to that set -- a provider message that was read, checked and
// approved, with the wording the client should see.
//
// The router's own code and message are what reach the client, so a change
// here changes what a client sees, not what the provider said.
type ProviderMessageConfig struct {
	// Provider is the provider whose message this is. It is a label only:
	// the router matches on the message and the code, not on the provider,
	// so a provider that writes the same message under a different name is
	// still recognized, and a provider that changes its name does not break
	// the rule.
	Provider string `yaml:"provider,omitempty"`
	// ProviderCode is the provider's own error code ("code" in the error
	// object). When set, the failure's code must be exactly this.
	//
	// It is what makes one rule cover a whole family of failures: every
	// "InvalidParameter" from any provider, whatever the model and the
	// request, is the same kind of refusal.
	ProviderCode string `yaml:"provider_code,omitempty"`
	// Match is text of the provider's message that the failure's message must
	// contain. It is matched with the case folded and runs of whitespace
	// collapsed, so how the config is written does not matter.
	//
	// It is what makes one rule cover the failures that share a wording
	// whatever the provider calls them ("Unsupported model: 'kimi-k2.7-code'"
	// and "Unsupported model: 'other-model'" are one failure).
	//
	// At least one of Match and ProviderCode is required: a rule with
	// neither would match every failure of every provider.
	Match string `yaml:"match,omitempty"`
	// Message is what the client is told. It is the router's own wording,
	// not the provider's: this is the text that goes on the wire.
	Message string `yaml:"message"`
	// Code is the router's error code for the failure (the "code" field of
	// the client's error object). It must be one of the router's own codes:
	// an unknown code would put a free-form string into a field clients
	// branch on.
	Code string `yaml:"code"`
	// Status is the HTTP status the client is answered with. It must be a
	// 4xx status: a 5xx would say the provider is broken, which is the
	// opposite of what a rule like this describes.
	Status int `yaml:"status"`
}

// ProviderMessagesConfig is the top-level provider_messages section: the
// provider messages the router may relay to a client.
type ProviderMessagesConfig struct {
	// Messages are the provider messages to relay.
	Messages []ProviderMessageConfig `yaml:"messages"`
}

// UnmarshalYAML reads the section, normalizing the texts so that the
// comparison the router does is not sensitive to how the config was written.
func (c *ProviderMessagesConfig) UnmarshalYAML(value *yaml.Node) error {
	if value == nil || value.Kind != yaml.MappingNode {
		return fmt.Errorf("provider_messages must be a mapping")
	}

	var raw struct {
		Messages []ProviderMessageConfig `yaml:"messages"`
	}
	if err := value.Decode(&raw); err != nil {
		return err
	}

	c.Messages = make([]ProviderMessageConfig, 0, len(raw.Messages))
	for _, entry := range raw.Messages {
		entry.Provider = strings.TrimSpace(resolveEnvString(entry.Provider))
		entry.ProviderCode = strings.TrimSpace(resolveEnvString(entry.ProviderCode))
		entry.Match = strings.TrimSpace(resolveEnvString(entry.Match))
		entry.Message = strings.TrimSpace(resolveEnvString(entry.Message))
		entry.Code = strings.TrimSpace(resolveEnvString(entry.Code))
		c.Messages = append(c.Messages, entry)
	}
	return nil
}

// Rules converts the section into the rules the router matches against.
//
// The router keeps its own built-in rules; the configured ones are added to
// them, so a config that names one provider's message does not have to
// repeat the rest.
func (c *ProviderMessagesConfig) Rules() []upstreamerror.ProviderMessageRule {
	if c == nil || len(c.Messages) == 0 {
		return nil
	}
	rules := make([]upstreamerror.ProviderMessageRule, 0, len(c.Messages))
	for _, entry := range c.Messages {
		rules = append(rules, upstreamerror.ProviderMessageRule{
			ProviderCode: entry.ProviderCode,
			Match:        entry.Match,
			Code:         entry.Code,
			Message:      entry.Message,
			Status:       entry.Status,
		})
	}
	return rules
}

// Validate checks the section.
//
// The checks are the section's whole safety argument, so they are strict: a
// rule that names no message, no wording, no code or no status would either
// match everything or produce an error body no client can act on.
func (c *ProviderMessagesConfig) Validate() error {
	if len(c.Messages) == 0 {
		return nil
	}
	if len(c.Messages) > maxProviderMessageEntries {
		return fmt.Errorf("provider_messages.messages exceeds %d entries", maxProviderMessageEntries)
	}

	seen := make(map[string]struct{}, len(c.Messages))
	for i, entry := range c.Messages {
		switch {
		case entry.Match == "" && entry.ProviderCode == "":
			return fmt.Errorf("provider_messages.messages[%d]: match or provider_code is required", i)
		case entry.Message == "":
			return fmt.Errorf("provider_messages.messages[%d]: message is required", i)
		case entry.Code == "":
			return fmt.Errorf("provider_messages.messages[%d]: code is required", i)
		case entry.Status < 400 || entry.Status > 499:
			return fmt.Errorf("provider_messages.messages[%d]: status must be a 4xx status, got %d", i, entry.Status)
		}
		if len(entry.Match) > maxProviderMessageTextBytes {
			return fmt.Errorf("provider_messages.messages[%d]: match is longer than %d bytes", i, maxProviderMessageTextBytes)
		}
		if len(entry.Message) > maxProviderMessageTextBytes {
			return fmt.Errorf("provider_messages.messages[%d]: message is longer than %d bytes", i, maxProviderMessageTextBytes)
		}
		if !isPrintableText(entry.Message) {
			return fmt.Errorf("provider_messages.messages[%d]: message contains a control character", i)
		}
		if !isPrintableText(entry.Match) {
			return fmt.Errorf("provider_messages.messages[%d]: match contains a control character", i)
		}

		key := strings.ToLower(entry.Match) + "\x00" + entry.ProviderCode + "\x00" + entry.Code
		if _, duplicate := seen[key]; duplicate {
			return fmt.Errorf("provider_messages.messages[%d]: duplicate rule for match %q / provider_code %q", i, entry.Match, entry.ProviderCode)
		}
		seen[key] = struct{}{}
	}
	return nil
}

// isPrintableText reports whether a text carries no control characters.
//
// A control character in a message would go into the client's response body
// as it is: a newline in the middle of a JSON string, a tab, an escape
// sequence the JSON encoder would have to guess at. The config is the
// operator's own text, but a body that does not render is worse than a
// rejected config.
func isPrintableText(text string) bool {
	if !utf8.ValidString(text) {
		return false
	}
	for _, r := range text {
		if r == '\t' {
			continue
		}
		if unicode.IsControl(r) {
			return false
		}
	}
	return true
}
