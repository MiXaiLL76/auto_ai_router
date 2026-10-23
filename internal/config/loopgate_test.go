package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func loopGatePtr[T any](v T) *T { return &v }

func defaultLoopGateSettings(mode LoopGateMode) LoopGateSettings {
	return LoopGateSettings{
		Mode: mode, NGram: DefaultLoopGateNGram, ArmWords: DefaultLoopGateArmWords,
		MinRepeat: DefaultLoopGateMinRepeat, MinRatio: DefaultLoopGateMinRatio,
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
