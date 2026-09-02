package vertex

import (
	"encoding/base64"
	"net/http"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConvertContentToParts_InputAudioOversizedReturns413(t *testing.T) {
	withMaxBase64Size(t, 16)
	oversized := strings.Repeat("A", 17)
	content := []interface{}{
		map[string]interface{}{
			"type": "input_audio",
			"input_audio": map[string]interface{}{
				"data":   oversized,
				"format": "wav",
			},
		},
	}

	parts, err := convertContentToParts(content)

	require.Nil(t, parts)
	require.Error(t, err)
	var validationErr *converterutil.RequestValidationError
	require.ErrorAs(t, err, &validationErr)
	assert.Equal(t, http.StatusRequestEntityTooLarge, validationErr.StatusCode)
}

func TestConvertContentToParts_InputAudioValidDecodes(t *testing.T) {
	raw := []byte("fake wav bytes")
	encoded := base64.StdEncoding.EncodeToString(raw)
	content := []interface{}{
		map[string]interface{}{
			"type": "input_audio",
			"input_audio": map[string]interface{}{
				"data":   encoded,
				"format": "wav",
			},
		},
	}

	parts, err := convertContentToParts(content)

	require.NoError(t, err)
	require.Len(t, parts, 1)
	require.NotNil(t, parts[0].InlineData)
	assert.Equal(t, raw, parts[0].InlineData.Data)
	assert.Equal(t, "audio/wav", parts[0].InlineData.MIMEType)
}
