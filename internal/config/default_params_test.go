package config

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestModelRPMConfig_DefaultParamsFromYAML(t *testing.T) {
	var model ModelRPMConfig
	require.NoError(t, yaml.Unmarshal([]byte(`
name: unlimited-ocr
credential: k8s_unlimited_ocr_0_1xH200
rpm: -1
default_params:
  skip_special_tokens: false
  temperature: 0.0
  vllm_xargs:
    ngram_size: 35
    window_size: 128
`), &model))

	assert.Equal(t, map[string]any{
		"skip_special_tokens": false,
		"temperature":         0.0,
		"vllm_xargs":          map[string]any{"ngram_size": 35, "window_size": 128},
	}, model.DefaultParams)
}

func TestModelRPMConfig_WithoutDefaultParams(t *testing.T) {
	var model ModelRPMConfig
	require.NoError(t, yaml.Unmarshal([]byte("name: m\nrpm: 1\n"), &model))
	assert.Nil(t, model.DefaultParams)
}

func TestModelRPMConfig_DefaultParamsRejectReservedKeys(t *testing.T) {
	for _, key := range defaultParamsReservedKeys {
		var model ModelRPMConfig
		err := yaml.Unmarshal([]byte("name: m\ndefault_params:\n  "+key+": x\n"), &model)
		require.Error(t, err, key)
		assert.Contains(t, err.Error(), key)
	}
}
