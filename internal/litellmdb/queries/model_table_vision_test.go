package queries

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestModelTableSupportsVision(t *testing.T) {
	assert.Nil(t, ModelTable{}.SupportsVision())
	assert.Nil(t, ModelTable{ModelInfo: map[string]interface{}{"supports_vision": "yes"}}.SupportsVision(), "non-boolean is ignored")

	v := ModelTable{ModelInfo: map[string]interface{}{"supports_vision": false}}.SupportsVision()
	require.NotNil(t, v)
	assert.False(t, *v)
}
