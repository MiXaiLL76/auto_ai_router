package queries

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestModelTable_Pools(t *testing.T) {
	assert.Equal(t, []string{"pool-01"}, ModelTable{ModelInfo: map[string]interface{}{"pools": []string{"pool-01"}}}.Pools())
	assert.Equal(t, []string{"pool-01", "pool-02"}, ModelTable{ModelInfo: map[string]interface{}{"pools": []interface{}{"pool-01", "pool-02"}}}.Pools())
	assert.Nil(t, ModelTable{}.Pools(), "a row without model_info belongs to no pool")
	assert.Nil(t, ModelTable{ModelInfo: map[string]interface{}{"pools": "pool-01"}}.Pools(), "a bare string is not a tag list")
	assert.Nil(t, ModelTable{ModelInfo: map[string]interface{}{"pools": []interface{}{"pool-01", 1}}}.Pools(), "a non-string drops the whole tag list")
}
