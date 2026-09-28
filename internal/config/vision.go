package config

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// VisionFallbackMode selects what happens to image inputs sent to a model
// configured with supports_vision: false.
type VisionFallbackMode string

const (
	// VisionFallbackReject fails the request with 400 before any upstream call.
	VisionFallbackReject VisionFallbackMode = "reject"
	// VisionFallbackStrip replaces every image with a text placeholder.
	VisionFallbackStrip VisionFallbackMode = "strip"
	// VisionFallbackDescribe replaces the images of the current turn with a text
	// description produced by describe_model, and older images with a placeholder.
	VisionFallbackDescribe VisionFallbackMode = "describe"
)

const (
	DefaultVisionMaxImages = 4
	DefaultVisionMaxTokens = 1024
	DefaultVisionTimeout   = 2 * time.Minute

	DefaultVisionDescribePrompt = "Describe this image as precisely and completely as possible: " +
		"objects, people, layout, colors, all visible text verbatim, numbers, tables, charts and code. " +
		"The description is passed to another model that cannot see the image, so do not omit details. " +
		"Answer in the language of the user's question."
)

// VisionFallbackConfig is the top-level vision_fallback section.
type VisionFallbackConfig struct {
	// Mode applies to models with supports_vision: false. Default: describe when
	// describe_model is set, reject otherwise.
	Mode VisionFallbackMode `yaml:"mode"`
	// DescribeModel is a vision-capable model served by this router. The describe
	// call goes through the normal pipeline with the caller's key, so it is
	// authenticated, rate-limited and billed like any other request.
	DescribeModel  string        `yaml:"describe_model"`
	DescribePrompt string        `yaml:"describe_prompt"`
	MaxImages      int           `yaml:"max_images"` // images described per request, the rest become placeholders; 0 = no limit
	MaxTokens      int           `yaml:"max_tokens"` // max_tokens of each describe call
	Timeout        time.Duration `yaml:"timeout"`    // per describe call
	// InjectIntoResponse prepends the descriptions to the assistant answer (describe
	// mode) so the client keeps them in its history and later turns restore them
	// without another describe call. nil = true.
	InjectIntoResponse *bool `yaml:"inject_into_response"`
}

// InjectEnabled reports whether descriptions are written into the answer.
func (c *VisionFallbackConfig) InjectEnabled() bool {
	return c.InjectIntoResponse == nil || *c.InjectIntoResponse
}

func (c *VisionFallbackConfig) UnmarshalYAML(value *yaml.Node) error {
	type rawVisionFallbackConfig struct {
		Mode           string `yaml:"mode"`
		DescribeModel  string `yaml:"describe_model"`
		DescribePrompt string `yaml:"describe_prompt"`
		MaxImages      string `yaml:"max_images"`
		MaxTokens      string `yaml:"max_tokens"`
		Timeout        string `yaml:"timeout"`
		Inject         string `yaml:"inject_into_response"`
	}
	var raw rawVisionFallbackConfig
	if err := value.Decode(&raw); err != nil {
		return err
	}

	c.Mode = VisionFallbackMode(strings.ToLower(strings.TrimSpace(resolveEnvString(raw.Mode))))
	c.DescribeModel = resolveEnvString(raw.DescribeModel)
	c.DescribePrompt = resolveEnvString(raw.DescribePrompt)
	var err error
	// Omitted max_images means the default; an explicit 0 means no limit.
	if c.MaxImages, err = parseField(raw.MaxImages, DefaultVisionMaxImages, strconv.Atoi, "vision_fallback.max_images"); err != nil {
		return err
	}
	if c.MaxTokens, err = parseField(raw.MaxTokens, 0, strconv.Atoi, "vision_fallback.max_tokens"); err != nil {
		return err
	}
	if c.Timeout, err = parseField(raw.Timeout, 0, time.ParseDuration, "vision_fallback.timeout"); err != nil {
		return err
	}
	if c.InjectIntoResponse, err = parseOptionalBool(raw.Inject, "vision_fallback.inject_into_response"); err != nil {
		return err
	}
	return nil
}

// defaultVisionFallbackConfig is used when the vision_fallback section is absent.
func defaultVisionFallbackConfig() VisionFallbackConfig {
	return VisionFallbackConfig{MaxImages: DefaultVisionMaxImages}
}

// ApplyDefaults fills omitted values. max_images is not touched: its default is
// applied while parsing, because an explicit 0 means "no limit".
func (c *VisionFallbackConfig) ApplyDefaults() {
	if c.Mode == "" {
		if c.DescribeModel != "" {
			c.Mode = VisionFallbackDescribe
		} else {
			c.Mode = VisionFallbackReject
		}
	}
	if c.DescribePrompt == "" {
		c.DescribePrompt = DefaultVisionDescribePrompt
	}
	if c.MaxTokens == 0 {
		c.MaxTokens = DefaultVisionMaxTokens
	}
	if c.Timeout == 0 {
		c.Timeout = DefaultVisionTimeout
	}
}

func (c *VisionFallbackConfig) Validate() error {
	switch c.Mode {
	case "", VisionFallbackReject, VisionFallbackStrip: // "" = defaults not applied (config built in code)
	case VisionFallbackDescribe:
		if c.DescribeModel == "" {
			return fmt.Errorf("vision_fallback.describe_model is required for mode %q", c.Mode)
		}
	default:
		return fmt.Errorf("invalid vision_fallback.mode %q (want reject, strip or describe)", c.Mode)
	}
	if c.MaxImages < 0 || c.MaxTokens < 0 || c.Timeout < 0 {
		return fmt.Errorf("vision_fallback.max_images, max_tokens and timeout must not be negative")
	}
	return nil
}
