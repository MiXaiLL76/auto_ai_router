package config

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// LoopGateMode selects what the loop gate does with a streamed generation it
// sees looping (the same long word sequence repeated until max_tokens).
type LoopGateMode string

const (
	// LoopGateOff disables the detector.
	LoopGateOff LoopGateMode = "off"
	// LoopGateObserve runs the detector and logs/counts firings, never cutting
	// anything. Used to calibrate thresholds on real traffic.
	LoopGateObserve LoopGateMode = "observe"
)

// Loop gate defaults, from the loop-gate reference calibration
// (https://github.com/vakovalskii/loop-gate-skill): honest answers never
// reached a repeat ratio of 0.35, loops climb to 0.8-0.99.
const (
	DefaultLoopGateNGram     = 32
	DefaultLoopGateArmWords  = 3000
	DefaultLoopGateMinRepeat = 5
	DefaultLoopGateMinRatio  = 0.35
)

// DefaultLoopGateProviders are the provider types the global mode applies to:
// self-hosted engines only, since a paid provider bills the abandoned part of
// a cut or retried generation too.
var DefaultLoopGateProviders = []ProviderType{ProviderTypeVLLM}

// LoopGateSettings are the effective detector settings of one request.
type LoopGateSettings struct {
	Mode      LoopGateMode `yaml:"mode"`
	NGram     int          `yaml:"ngram"`
	ArmWords  int          `yaml:"arm_words"`
	MinRepeat int          `yaml:"min_repeat"`
	MinRatio  float64      `yaml:"min_ratio"`
}

// Enabled reports whether the detector runs.
func (s LoopGateSettings) Enabled() bool {
	return s.Mode != "" && s.Mode != LoopGateOff
}

// LoopGateConfig is the top-level loop_gate section.
type LoopGateConfig struct {
	LoopGateSettings `yaml:",inline"`
	// Providers are the credential types the global mode applies to. A mode set
	// explicitly on a credential or model applies whatever the provider type.
	// Empty = DefaultLoopGateProviders.
	Providers []ProviderType `yaml:"providers"`
}

// LoopGateOverride is the loop_gate block of a credential or a model: every
// field it sets replaces the inherited one.
type LoopGateOverride struct {
	Mode      *LoopGateMode `yaml:"mode,omitempty"`
	NGram     *int          `yaml:"ngram,omitempty"`
	ArmWords  *int          `yaml:"arm_words,omitempty"`
	MinRepeat *int          `yaml:"min_repeat,omitempty"`
	MinRatio  *float64      `yaml:"min_ratio,omitempty"`
}

type rawLoopGateFields struct {
	Mode      string `yaml:"mode"`
	NGram     string `yaml:"ngram"`
	ArmWords  string `yaml:"arm_words"`
	MinRepeat string `yaml:"min_repeat"`
	MinRatio  string `yaml:"min_ratio"`
}

func parseLoopGateMode(raw string) LoopGateMode {
	return LoopGateMode(strings.ToLower(strings.TrimSpace(resolveEnvString(raw))))
}

func parseOptionalLoopGateField[T any](raw string, parser parseFunc[T], fieldPath string) (*T, error) {
	if resolveEnvString(raw) == "" {
		return nil, nil
	}
	v, err := parseField(raw, *new(T), parser, fieldPath)
	if err != nil {
		return nil, err
	}
	return &v, nil
}

func parseFloat64(s string) (float64, error) { return strconv.ParseFloat(s, 64) }

func (o *LoopGateOverride) UnmarshalYAML(value *yaml.Node) error {
	var raw rawLoopGateFields
	if err := value.Decode(&raw); err != nil {
		return err
	}
	*o = LoopGateOverride{}
	if mode := parseLoopGateMode(raw.Mode); mode != "" {
		o.Mode = &mode
	}
	var err error
	if o.NGram, err = parseOptionalLoopGateField(raw.NGram, strconv.Atoi, "loop_gate.ngram"); err != nil {
		return err
	}
	if o.ArmWords, err = parseOptionalLoopGateField(raw.ArmWords, strconv.Atoi, "loop_gate.arm_words"); err != nil {
		return err
	}
	if o.MinRepeat, err = parseOptionalLoopGateField(raw.MinRepeat, strconv.Atoi, "loop_gate.min_repeat"); err != nil {
		return err
	}
	if o.MinRatio, err = parseOptionalLoopGateField(raw.MinRatio, parseFloat64, "loop_gate.min_ratio"); err != nil {
		return err
	}
	return nil
}

func (c *LoopGateConfig) UnmarshalYAML(value *yaml.Node) error {
	type rawLoopGateConfig struct {
		rawLoopGateFields `yaml:",inline"`
		Providers         []string `yaml:"providers"`
	}
	var raw rawLoopGateConfig
	if err := value.Decode(&raw); err != nil {
		return err
	}
	*c = LoopGateConfig{}
	c.Mode = parseLoopGateMode(raw.Mode)
	var err error
	if c.NGram, err = parseField(raw.NGram, 0, strconv.Atoi, "loop_gate.ngram"); err != nil {
		return err
	}
	if c.ArmWords, err = parseField(raw.ArmWords, DefaultLoopGateArmWords, strconv.Atoi, "loop_gate.arm_words"); err != nil {
		return err
	}
	if c.MinRepeat, err = parseField(raw.MinRepeat, 0, strconv.Atoi, "loop_gate.min_repeat"); err != nil {
		return err
	}
	if c.MinRatio, err = parseField(raw.MinRatio, 0, parseFloat64, "loop_gate.min_ratio"); err != nil {
		return err
	}
	for _, p := range raw.Providers {
		if p = strings.TrimSpace(resolveEnvString(p)); p != "" {
			c.Providers = append(c.Providers, normalizeProviderType(p))
		}
	}
	return nil
}

// DefaultLoopGateConfig is used when the loop_gate section is absent.
func DefaultLoopGateConfig() LoopGateConfig {
	return LoopGateConfig{LoopGateSettings: LoopGateSettings{ArmWords: DefaultLoopGateArmWords}}
}

// ApplyDefaults fills omitted values. arm_words is not touched: its default is
// applied while parsing, because an explicit 0 means "armed from the first word".
func (c *LoopGateConfig) ApplyDefaults() {
	if c.Mode == "" {
		c.Mode = LoopGateOff
	}
	if c.NGram == 0 {
		c.NGram = DefaultLoopGateNGram
	}
	if c.MinRepeat == 0 {
		c.MinRepeat = DefaultLoopGateMinRepeat
	}
	if c.MinRatio == 0 {
		c.MinRatio = DefaultLoopGateMinRatio
	}
	if len(c.Providers) == 0 {
		c.Providers = slices.Clone(DefaultLoopGateProviders)
	}
}

func validateLoopGateMode(mode LoopGateMode, path string) error {
	switch mode {
	case "", LoopGateOff, LoopGateObserve:
		return nil
	default:
		return fmt.Errorf("invalid %s.mode %q (want off or observe)", path, mode)
	}
}

func (s LoopGateSettings) validate(path string) error {
	if err := validateLoopGateMode(s.Mode, path); err != nil {
		return err
	}
	if s.NGram < 2 {
		return fmt.Errorf("%s.ngram must be at least 2, got %d", path, s.NGram)
	}
	if s.ArmWords < 0 {
		return fmt.Errorf("%s.arm_words must not be negative, got %d", path, s.ArmWords)
	}
	if s.MinRepeat < 2 {
		return fmt.Errorf("%s.min_repeat must be at least 2, got %d", path, s.MinRepeat)
	}
	if s.MinRatio <= 0 || s.MinRatio > 1 {
		return fmt.Errorf("%s.min_ratio must be in (0, 1], got %v", path, s.MinRatio)
	}
	return nil
}

// Validate checks the section after ApplyDefaults. A config built in code
// without defaults (Mode "") is accepted as disabled.
func (c *LoopGateConfig) Validate() error {
	if c.Mode == "" {
		return nil
	}
	return c.validate("loop_gate")
}

// Resolve returns the settings of a request served by a credential of type
// providerType, applying the credential override and then the model override
// (either may be nil). The global mode applies only to the configured provider
// types; a mode set explicitly by an override applies to any provider. The
// result is validated, so a bad override disables the gate instead of
// running the detector with nonsense thresholds.
func (c *LoopGateConfig) Resolve(providerType ProviderType, overrides ...*LoopGateOverride) (LoopGateSettings, error) {
	s := c.LoopGateSettings
	if !slices.Contains(c.Providers, providerType) {
		s.Mode = LoopGateOff
	}
	for _, o := range overrides {
		if o == nil {
			continue
		}
		if o.Mode != nil {
			s.Mode = *o.Mode
		}
		if o.NGram != nil {
			s.NGram = *o.NGram
		}
		if o.ArmWords != nil {
			s.ArmWords = *o.ArmWords
		}
		if o.MinRepeat != nil {
			s.MinRepeat = *o.MinRepeat
		}
		if o.MinRatio != nil {
			s.MinRatio = *o.MinRatio
		}
	}
	if !s.Enabled() {
		return s, nil
	}
	if err := s.validate("loop_gate"); err != nil {
		return LoopGateSettings{Mode: LoopGateOff}, err
	}
	return s, nil
}

// Validate checks the fields the override sets. Thresholds are checked in
// combination with the inherited ones by LoopGateConfig.Resolve.
func (o *LoopGateOverride) Validate(path string) error {
	if o == nil {
		return nil
	}
	if o.Mode != nil {
		if err := validateLoopGateMode(*o.Mode, path); err != nil {
			return err
		}
	}
	probe := LoopGateSettings{
		NGram: DefaultLoopGateNGram, ArmWords: DefaultLoopGateArmWords,
		MinRepeat: DefaultLoopGateMinRepeat, MinRatio: DefaultLoopGateMinRatio,
	}
	if o.NGram != nil {
		probe.NGram = *o.NGram
	}
	if o.ArmWords != nil {
		probe.ArmWords = *o.ArmWords
	}
	if o.MinRepeat != nil {
		probe.MinRepeat = *o.MinRepeat
	}
	if o.MinRatio != nil {
		probe.MinRatio = *o.MinRatio
	}
	return probe.validate(path)
}
