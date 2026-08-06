package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExtractBetaHeader(t *testing.T) {
	tests := []struct {
		name      string
		body      string
		wantBetas []string
		wantField bool
	}{
		{
			name:      "array",
			body:      `{"model":"claude","anthropic_beta":["effort-2025-11-24"," effort-2025-11-24 ","interleaved-thinking-2025-05-14"]}`,
			wantBetas: []string{"effort-2025-11-24", "interleaved-thinking-2025-05-14"},
		},
		{
			name:      "string",
			body:      `{"model":"claude","anthropic_beta":"effort-2025-11-24"}`,
			wantBetas: []string{"effort-2025-11-24"},
		},
		{
			name:      "absent",
			body:      `{"model":"claude"}`,
			wantField: false,
		},
		{
			name:      "invalid",
			body:      `{"model":"claude","anthropic_beta":42}`,
			wantField: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, betas := ExtractBetaHeader([]byte(tt.body))
			assert.Equal(t, tt.wantBetas, betas)
			var request map[string]any
			require.NoError(t, json.Unmarshal(body, &request))
			_, hasField := request["anthropic_beta"]
			assert.Equal(t, tt.wantField, hasField)
		})
	}
}
