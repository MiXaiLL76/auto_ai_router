package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func loopGatePtr[T any](v T) *T { return &v }

func defaultLoopGateSettings(mode LoopGateMode) LoopGateSettings {
	return LoopGateSettings{
		Mode: mode, NGram: DefaultLoopGateNGram, ArmWords: DefaultLoopGateArmWords,
		MinRepeat: DefaultLoopGateMinRepeat, MinRatio: DefaultLoopGateMinRatio,
		MaxAttempts: DefaultLoopGateMaxAttempts, RetryDeadline: DefaultLoopGateRetryDeadline,
		RetryPresencePenalty: DefaultLoopGateRetryPresencePenalty,
	}
}

func TestLoad_LoopGate(t *testing.T) {
	t.Setenv("TEST_LOOP_GATE_MODE", "Observe")
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	require.NoError(t, os.WriteFile(configPath, []byte(`
server:
  master_key: sk-test
credentials:
  - name: vllm-node
    type: vllm
    base_url: http://vllm:8000
    rpm: -1
    loop_gate:
      min_ratio: "0.5"
    models:
      - name: qwen
        loop_gate:
          arm_words: 0
          mode: off
  - name: oai
    type: openai
    api_key: sk-x
    base_url: https://api.openai.com
    rpm: -1
models:
  - name: glm
    credential: vllm-node
loop_gate:
  mode: os.environ/TEST_LOOP_GATE_MODE
  ngram: 16
`), 0600))

	cfg, err := Load(configPath)
	require.NoError(t, err)

	want := defaultLoopGateSettings(LoopGateObserve)
	want.NGram = 16
	assert.Equal(t, want, cfg.LoopGate.LoopGateSettings)
	assert.Equal(t, []ProviderType{ProviderTypeVLLM}, cfg.LoopGate.Providers)

	var vllm *CredentialConfig
	for i := range cfg.Credentials {
		if cfg.Credentials[i].Name == "vllm-node" {
			vllm = &cfg.Credentials[i]
		}
	}
	require.NotNil(t, vllm)
	require.NotNil(t, vllm.LoopGate)
	assert.Equal(t, loopGatePtr(0.5), vllm.LoopGate.MinRatio)
	assert.Nil(t, vllm.LoopGate.Mode)

	var qwen *ModelRPMConfig
	for i := range cfg.Models {
		if cfg.Models[i].Name == "qwen" {
			qwen = &cfg.Models[i]
		}
	}
	require.NotNil(t, qwen)
	require.NotNil(t, qwen.LoopGate)
	assert.Equal(t, loopGatePtr(LoopGateOff), qwen.LoopGate.Mode)
	assert.Equal(t, loopGatePtr(0), qwen.LoopGate.ArmWords, "explicit 0 is kept")
}

func TestLoad_LoopGateDefaultsWhenAbsent(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	require.NoError(t, os.WriteFile(configPath, []byte(`
server:
  master_key: sk-test
credentials:
  - name: vllm-node
    type: vllm
    base_url: http://vllm:8000
    rpm: -1
loop_gate:
  # mode: observe
`), 0600))
	cfg, err := Load(configPath)
	require.NoError(t, err)
	assert.Equal(t, defaultLoopGateSettings(LoopGateOff), cfg.LoopGate.LoopGateSettings)
	assert.Equal(t, DefaultLoopGateProviders, cfg.LoopGate.Providers)
}

func TestLoad_LoopGateInvalid(t *testing.T) {
	for name, section := range map[string]string{
		"bad mode":             "loop_gate:\n  mode: enforce-everything\n",
		"ngram too small":      "loop_gate:\n  mode: observe\n  ngram: 1\n",
		"ratio above one":      "loop_gate:\n  mode: observe\n  min_ratio: 1.5\n",
		"min_repeat too small": "loop_gate:\n  mode: observe\n  min_repeat: 1\n",
		"bad number":           "loop_gate:\n  ngram: many\n",
		"bad model override":   "models:\n  - name: glm\n    credential: vllm-node\n    loop_gate:\n      min_ratio: 0\n",
	} {
		t.Run(name, func(t *testing.T) {
			configPath := filepath.Join(t.TempDir(), "config.yaml")
			require.NoError(t, os.WriteFile(configPath, []byte(`
server:
  master_key: sk-test
credentials:
  - name: vllm-node
    type: vllm
    base_url: http://vllm:8000
    rpm: -1
`+section), 0600))
			_, err := Load(configPath)
			assert.ErrorContains(t, err, "loop_gate")
		})
	}
}

func TestLoopGateOverride_UnmarshalYAML(t *testing.T) {
	t.Setenv("TEST_LOOP_GATE_RATIO", "0.42")
	var o LoopGateOverride
	require.NoError(t, yaml.Unmarshal([]byte("mode: OBSERVE\nmin_ratio: os.environ/TEST_LOOP_GATE_RATIO\nmin_repeat: 7\n"), &o))
	assert.Equal(t, LoopGateOverride{
		Mode: loopGatePtr(LoopGateObserve), MinRatio: loopGatePtr(0.42), MinRepeat: loopGatePtr(7),
	}, o)
	assert.ErrorContains(t, yaml.Unmarshal([]byte("arm_words: lots\n"), &o), "loop_gate.arm_words")
}

func TestLoopGateConfig_Resolve(t *testing.T) {
	global := LoopGateConfig{LoopGateSettings: defaultLoopGateSettings(LoopGateObserve)}
	global.ApplyDefaults()

	s, err := global.Resolve(ProviderTypeVLLM)
	require.NoError(t, err)
	assert.Equal(t, defaultLoopGateSettings(LoopGateObserve), s)

	s, err = global.Resolve(ProviderTypeOpenAI)
	require.NoError(t, err)
	assert.False(t, s.Enabled(), "global mode applies only to the listed providers")

	s, err = global.Resolve(ProviderTypeOpenAI, &LoopGateOverride{MinRatio: loopGatePtr(0.5)})
	require.NoError(t, err)
	assert.False(t, s.Enabled(), "a threshold-only override does not opt a provider in")

	s, err = global.Resolve(ProviderTypeOpenAI, &LoopGateOverride{Mode: loopGatePtr(LoopGateObserve)})
	require.NoError(t, err)
	assert.True(t, s.Enabled(), "an explicit mode opts any provider in")

	s, err = global.Resolve(ProviderTypeVLLM,
		&LoopGateOverride{MinRatio: loopGatePtr(0.5), NGram: loopGatePtr(20)},
		nil,
		&LoopGateOverride{MinRatio: loopGatePtr(0.6), ArmWords: loopGatePtr(100)},
	)
	require.NoError(t, err)
	want := defaultLoopGateSettings(LoopGateObserve)
	want.MinRatio, want.NGram, want.ArmWords = 0.6, 20, 100
	assert.Equal(t, want, s, "model override wins over credential override")

	s, err = global.Resolve(ProviderTypeVLLM, nil, &LoopGateOverride{Mode: loopGatePtr(LoopGateOff)})
	require.NoError(t, err)
	assert.False(t, s.Enabled(), "a model can switch the gate off")

	s, err = global.Resolve(ProviderTypeVLLM, &LoopGateOverride{NGram: loopGatePtr(1)})
	assert.Error(t, err)
	assert.False(t, s.Enabled(), "invalid effective settings disable the gate")

	off := LoopGateConfig{}
	off.ApplyDefaults()
	s, err = off.Resolve(ProviderTypeVLLM)
	require.NoError(t, err)
	assert.False(t, s.Enabled())
}

func TestLoopGate_CutMinRatio(t *testing.T) {
	s := defaultLoopGateSettings(LoopGateEnforce)
	assert.Equal(t, DefaultLoopGateMinRatio, s.CutRatio(), "0 = min_ratio")
	s.CutMinRatio = 0.8
	assert.Equal(t, 0.8, s.CutRatio())

	global := LoopGateConfig{LoopGateSettings: defaultLoopGateSettings(LoopGateEnforce)}
	global.ApplyDefaults()
	_, err := global.Resolve(ProviderTypeVLLM, &LoopGateOverride{CutMinRatio: loopGatePtr(0.2)})
	assert.ErrorContains(t, err, "cut_min_ratio", "below the effective min_ratio")
	got, err := global.Resolve(ProviderTypeVLLM, &LoopGateOverride{MinRatio: loopGatePtr(0.5)})
	require.NoError(t, err)
	assert.Equal(t, 0.5, got.CutRatio(), "an unset cut_min_ratio follows an overridden min_ratio")
}

func TestLoad_LoopGateCutMinRatioCombinationIsValidated(t *testing.T) {
	for name, tc := range map[string]struct {
		section string
		wantErr string
	}{
		"global below min_ratio": {section: "loop_gate:\n  mode: enforce\n  cut_min_ratio: 0.2\n", wantErr: "cut_min_ratio"},
		"model below inherited min_ratio": {
			section: "loop_gate:\n  mode: enforce\n  min_ratio: 0.5\nmodels:\n  - name: glm\n    credential: vllm-node\n    loop_gate:\n      cut_min_ratio: 0.4\n",
			wantErr: "model glm on credential vllm-node",
		},
		"enforce with cut above min_ratio": {section: "loop_gate:\n  mode: enforce\n  cut_min_ratio: 0.8\n"},
	} {
		t.Run(name, func(t *testing.T) {
			configPath := filepath.Join(t.TempDir(), "config.yaml")
			require.NoError(t, os.WriteFile(configPath, []byte(`
server:
  master_key: sk-test
credentials:
  - name: vllm-node
    type: vllm
    base_url: http://vllm:8000
    rpm: -1
`+tc.section), 0600))
			cfg, err := Load(configPath)
			if tc.wantErr != "" {
				assert.ErrorContains(t, err, tc.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, LoopGateEnforce, cfg.LoopGate.Mode)
			assert.Equal(t, 0.8, cfg.LoopGate.CutRatio())
		})
	}
}

func TestLoopGate_RetrySettings(t *testing.T) {
	var cfg LoopGateConfig
	require.NoError(t, yaml.Unmarshal([]byte("mode: enforce\nmax_attempts: 2\nretry_deadline: 90s\nretry_presence_penalty: [0.5, \"0.8\", 1.2]\n"), &cfg))
	cfg.ApplyDefaults()
	require.NoError(t, cfg.Validate())
	assert.Equal(t, 2, cfg.MaxAttempts)
	assert.Equal(t, 90*time.Second, cfg.RetryDeadline)
	assert.Equal(t, 0.5, cfg.PresencePenaltyForRestart(1))
	assert.Equal(t, 1.2, cfg.PresencePenaltyForRestart(3))
	assert.Equal(t, 1.2, cfg.PresencePenaltyForRestart(9), "the last value repeats")

	got, err := cfg.Resolve(ProviderTypeVLLM, &LoopGateOverride{MaxAttempts: loopGatePtr(1), RetryPresencePenalty: []float64{2}})
	require.NoError(t, err)
	assert.Equal(t, 1, got.MaxAttempts)
	assert.Equal(t, 2.0, got.PresencePenaltyForRestart(1))

	for name, doc := range map[string]string{
		"zero attempts":   "mode: enforce\nmax_attempts: 0\n",
		"penalty too big": "mode: enforce\nretry_presence_penalty: [3]\n",
	} {
		t.Run(name, func(t *testing.T) {
			var o LoopGateOverride
			require.NoError(t, yaml.Unmarshal([]byte(doc), &o))
			assert.ErrorContains(t, o.Validate("loop_gate"), "loop_gate")
		})
	}
	var bad LoopGateConfig
	assert.ErrorContains(t, yaml.Unmarshal([]byte("retry_deadline: soon\n"), &bad), "retry_deadline")
}

func TestLoopGate_ForceStream(t *testing.T) {
	var cfg LoopGateConfig
	require.NoError(t, yaml.Unmarshal([]byte("mode: enforce\n"), &cfg))
	cfg.ApplyDefaults()
	assert.False(t, cfg.ForceStreamEnabled(), "off by default")

	got, err := cfg.Resolve(ProviderTypeVLLM, nil, &LoopGateOverride{ForceStream: loopGatePtr(true)})
	require.NoError(t, err)
	assert.True(t, got.ForceStreamEnabled(), "a model override opts in")

	require.NoError(t, yaml.Unmarshal([]byte("mode: enforce\nforce_stream: true\n"), &cfg))
	assert.True(t, cfg.ForceStreamEnabled())

	assert.ErrorContains(t, yaml.Unmarshal([]byte("force_stream: maybe\n"), &cfg), "force_stream")
}
