package vertex

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

func TestConvertVertexFunctionCallToStreamingOpenAI(t *testing.T) {
	t.Run("valid function call with name and args", func(t *testing.T) {
		fc := &genai.FunctionCall{
			Name: "get_weather",
			Args: map[string]interface{}{
				"city": "Tokyo",
			},
		}
		thoughtSig := []byte("test-signature")

		result := convertVertexFunctionCallToStreamingOpenAI(fc, thoughtSig, 0)

		assert.Equal(t, 0, result.Index)
		assert.Equal(t, "function", result.Type)
		assert.NotEmpty(t, result.ID)
		require.NotNil(t, result.Function)
		assert.Equal(t, "get_weather", result.Function.Name)
		assert.Contains(t, result.Function.Arguments, "Tokyo")

		// Check thoughtSignature is preserved in provider_specific_fields
		require.NotNil(t, result.ProviderSpecificFields)
		assert.NotEmpty(t, result.ProviderSpecificFields["thought_signature"])
	})

	t.Run("function call with nil args produces empty JSON object", func(t *testing.T) {
		fc := &genai.FunctionCall{
			Name: "no_args_fn",
			Args: nil,
		}

		result := convertVertexFunctionCallToStreamingOpenAI(fc, nil, 1)

		assert.Equal(t, 1, result.Index)
		require.NotNil(t, result.Function)
		assert.Equal(t, "no_args_fn", result.Function.Name)
		assert.Equal(t, "{}", result.Function.Arguments)
	})

	t.Run("nil thought signature sets skip flag", func(t *testing.T) {
		fc := &genai.FunctionCall{
			Name: "test_fn",
			Args: map[string]interface{}{},
		}

		result := convertVertexFunctionCallToStreamingOpenAI(fc, nil, 0)

		require.NotNil(t, result.ProviderSpecificFields)
		assert.Equal(t, true, result.ProviderSpecificFields["skip_thought_signature_validator"])
	})

	t.Run("empty thought signature sets skip flag", func(t *testing.T) {
		fc := &genai.FunctionCall{
			Name: "test_fn",
			Args: map[string]interface{}{},
		}

		result := convertVertexFunctionCallToStreamingOpenAI(fc, []byte{}, 0)

		require.NotNil(t, result.ProviderSpecificFields)
		assert.Equal(t, true, result.ProviderSpecificFields["skip_thought_signature_validator"])
	})

	t.Run("index is preserved", func(t *testing.T) {
		fc := &genai.FunctionCall{
			Name: "fn",
			Args: map[string]interface{}{},
		}

		result := convertVertexFunctionCallToStreamingOpenAI(fc, nil, 5)
		assert.Equal(t, 5, result.Index)
	})
}
