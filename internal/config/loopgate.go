package config

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"

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
	// LoopGateEnforce also acts on a loop: the upstream generation is aborted
	// and, while only reasoning has been streamed, restarted with a new seed
	// and a higher presence_penalty in the same stream; otherwise the streamed
	// answer is cut with finish_reason "length".
	LoopGateEnforce LoopGateMode = "enforce"
)

// Loop gate defaults, from the loop-gate reference calibration
// (https://github.com/vakovalskii/loop-gate-skill): honest answers never
// reached a repeat ratio of 0.35, loops climb to 0.8-0.99.
const (
	DefaultLoopGateNGram     = 32
	DefaultLoopGateArmWords  = 3000
	DefaultLoopGateMinRepeat = 5
	DefaultLoopGateMinRatio  = 0.35

	// DefaultLoopGateMaxAttempts counts the first attempt: up to two restarts.
	DefaultLoopGateMaxAttempts   = 3
	DefaultLoopGateRetryDeadline = 10 * time.Minute
)

// DefaultLoopGateRetryPresencePenalty is the presence_penalty of the first and
// of every later restart. repetition_penalty is never touched: in the
// reference production it made generation itself crawl.
var DefaultLoopGateRetryPresencePenalty = []float64{1.0, 1.5}

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
	// CutMinRatio is the repeat ratio at which enforce mode cuts a stream the
	// client already reads. 0 = MinRatio. A higher value keeps the start of a
	// borderline answer (long reviews rereading the same material) longer.
	CutMinRatio float64 `yaml:"cut_min_ratio,omitempty"`
	// MaxAttempts bounds the attempts of one request, the first included.
	// 1 = never restart, only cut.
	MaxAttempts int `yaml:"max_attempts"`
	// RetryDeadline is how long after the start of the first stream a restart
	// may still begin.
	RetryDeadline time.Duration `yaml:"retry_deadline"`
	// RetryPresencePenalty is the presence_penalty sent with restart N
	// (element N-1); the last element applies to every later restart.
	RetryPresencePenalty []float64 `yaml:"retry_presence_penalty"`
	// ForceStream makes enforce mode send a non-streamed chat completion
	// upstream as a stream, so a loop can be stopped and restarted before the
	// client gets anything; the client still receives one chat.completion.
	// nil = true. false leaves non-streamed requests to post-hoc observation.
	ForceStream *bool `yaml:"force_stream,omitempty"`
}

// ForceStreamEnabled reports whether enforce mode streams non-streamed requests.
func (s LoopGateSettings) ForceStreamEnabled() bool {
	return s.ForceStream == nil || *s.ForceStream
}

// PresencePenaltyForRestart returns the presence_penalty of restart n (1-based).
func (s LoopGateSettings) PresencePenaltyForRestart(n int) float64 {
	if len(s.RetryPresencePenalty) == 0 {
		return 0
	}
	return s.RetryPresencePenalty[min(n, len(s.RetryPresencePenalty))-1]
}

// Enabled reports whether the detector runs.
func (s LoopGateSettings) Enabled() bool {
	return s.Mode != "" && s.Mode != LoopGateOff
}

// CutRatio returns the effective cut_min_ratio.
func (s LoopGateSettings) CutRatio() float64 {
	if s.CutMinRatio > 0 {
		return s.CutMinRatio
	}
	return s.MinRatio
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
	Mode                 *LoopGateMode  `yaml:"mode,omitempty"`
	NGram                *int           `yaml:"ngram,omitempty"`
	ArmWords             *int           `yaml:"arm_words,omitempty"`
	MinRepeat            *int           `yaml:"min_repeat,omitempty"`
	MinRatio             *float64       `yaml:"min_ratio,omitempty"`
	CutMinRatio          *float64       `yaml:"cut_min_ratio,omitempty"`
	MaxAttempts          *int           `yaml:"max_attempts,omitempty"`
	RetryDeadline        *time.Duration `yaml:"retry_deadline,omitempty"`
	RetryPresencePenalty []float64      `yaml:"retry_presence_penalty,omitempty"`
	ForceStream          *bool          `yaml:"force_stream,omitempty"`
}

type rawLoopGateFields struct {
	Mode                 string   `yaml:"mode"`
	NGram                string   `yaml:"ngram"`
	ArmWords             string   `yaml:"arm_words"`
	MinRepeat            string   `yaml:"min_repeat"`
	MinRatio             string   `yaml:"min_ratio"`
	CutMinRatio          string   `yaml:"cut_min_ratio"`
	MaxAttempts          string   `yaml:"max_attempts"`
	RetryDeadline        string   `yaml:"retry_deadline"`
	RetryPresencePenalty []string `yaml:"retry_presence_penalty"`
	ForceStream          string   `yaml:"force_stream"`
}

func parsePresencePenalties(raw []string, fieldPath string) ([]float64, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	out := make([]float64, 0, len(raw))
	for _, v := range raw {
		f, err := parseField(v, 0.0, parseFloat64, fieldPath)
		if err != nil {
			return nil, err
		}
		out = append(out, f)
	}
	return out, nil
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
	if o.CutMinRatio, err = parseOptionalLoopGateField(raw.CutMinRatio, parseFloat64, "loop_gate.cut_min_ratio"); err != nil {
		return err
	}
	if o.MaxAttempts, err = parseOptionalLoopGateField(raw.MaxAttempts, strconv.Atoi, "loop_gate.max_attempts"); err != nil {
		return err
	}
	if o.RetryDeadline, err = parseOptionalLoopGateField(raw.RetryDeadline, time.ParseDuration, "loop_gate.retry_deadline"); err != nil {
		return err
	}
	if o.RetryPresencePenalty, err = parsePresencePenalties(raw.RetryPresencePenalty, "loop_gate.retry_presence_penalty"); err != nil {
		return err
	}
	if o.ForceStream, err = parseOptionalBool(raw.ForceStream, "loop_gate.force_stream"); err != nil {
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
	if c.CutMinRatio, err = parseField(raw.CutMinRatio, 0, parseFloat64, "loop_gate.cut_min_ratio"); err != nil {
		return err
	}
	if c.MaxAttempts, err = parseField(raw.MaxAttempts, 0, strconv.Atoi, "loop_gate.max_attempts"); err != nil {
		return err
	}
	if c.RetryDeadline, err = parseField(raw.RetryDeadline, 0, time.ParseDuration, "loop_gate.retry_deadline"); err != nil {
		return err
	}
	if c.RetryPresencePenalty, err = parsePresencePenalties(raw.RetryPresencePenalty, "loop_gate.retry_presence_penalty"); err != nil {
		return err
	}
	if c.ForceStream, err = parseOptionalBool(raw.ForceStream, "loop_gate.force_stream"); err != nil {
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
// cut_min_ratio stays 0 (= min_ratio), so a model override of min_ratio moves
// the cut threshold with it.
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
	if c.MaxAttempts == 0 {
		c.MaxAttempts = DefaultLoopGateMaxAttempts
	}
	if c.RetryDeadline == 0 {
		c.RetryDeadline = DefaultLoopGateRetryDeadline
	}
	if len(c.RetryPresencePenalty) == 0 {
		c.RetryPresencePenalty = slices.Clone(DefaultLoopGateRetryPresencePenalty)
	}
	if len(c.Providers) == 0 {
		c.Providers = slices.Clone(DefaultLoopGateProviders)
	}
}

func validateLoopGateMode(mode LoopGateMode, path string) error {
	switch mode {
	case "", LoopGateOff, LoopGateObserve, LoopGateEnforce:
		return nil
	default:
		return fmt.Errorf("invalid %s.mode %q (want off, observe or enforce)", path, mode)
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
	if s.CutMinRatio != 0 && (s.CutMinRatio < s.MinRatio || s.CutMinRatio > 1) {
		return fmt.Errorf("%s.cut_min_ratio must be 0 (= min_ratio) or in [min_ratio %v, 1], got %v", path, s.MinRatio, s.CutMinRatio)
	}
	return s.validateRetry(path)
}

func (s LoopGateSettings) validateRetry(path string) error {
	if s.MaxAttempts < 1 {
		return fmt.Errorf("%s.max_attempts must be at least 1, got %d", path, s.MaxAttempts)
	}
	if s.RetryDeadline < 0 {
		return fmt.Errorf("%s.retry_deadline must not be negative, got %v", path, s.RetryDeadline)
	}
	for _, v := range s.RetryPresencePenalty {
		if v < -2 || v > 2 {
			return fmt.Errorf("%s.retry_presence_penalty values must be in [-2, 2], got %v", path, v)
		}
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
// result is validated, so a bad combination disables the gate instead of
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
		if o.CutMinRatio != nil {
			s.CutMinRatio = *o.CutMinRatio
		}
		if o.MaxAttempts != nil {
			s.MaxAttempts = *o.MaxAttempts
		}
		if o.RetryDeadline != nil {
			s.RetryDeadline = *o.RetryDeadline
		}
		if len(o.RetryPresencePenalty) > 0 {
			s.RetryPresencePenalty = o.RetryPresencePenalty
		}
		if o.ForceStream != nil {
			s.ForceStream = o.ForceStream
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
		MaxAttempts: DefaultLoopGateMaxAttempts, RetryDeadline: DefaultLoopGateRetryDeadline,
		RetryPresencePenalty: o.RetryPresencePenalty,
	}
	if o.MaxAttempts != nil {
		probe.MaxAttempts = *o.MaxAttempts
	}
	if o.RetryDeadline != nil {
		probe.RetryDeadline = *o.RetryDeadline
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
	if o.CutMinRatio != nil {
		// Checked against min_ratio by Resolve, where the inherited value is known.
		if *o.CutMinRatio < 0 || *o.CutMinRatio > 1 {
			return fmt.Errorf("%s.cut_min_ratio must be in [0, 1], got %v", path, *o.CutMinRatio)
		}
	}
	return probe.validate(path)
}
