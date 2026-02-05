package openai

import (
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestGenerateID(t *testing.T) {
	id := GenerateID()

	assert.NotEmpty(t, id)
	assert.True(t, strings.HasPrefix(id, "chatcmpl-"))
	assert.Equal(t, 29, len(id)) // "chatcmpl-" (9) + 20 hex chars
}

func TestGenerateID_Uniqueness(t *testing.T) {
	id1 := GenerateID()
	id2 := GenerateID()
	id3 := GenerateID()

	assert.NotEqual(t, id1, id2)
	assert.NotEqual(t, id2, id3)
	assert.NotEqual(t, id1, id3)
}

func TestGetCurrentTimestamp(t *testing.T) {
	before := time.Now().UTC().Unix()
	timestamp := GetCurrentTimestamp()
	after := time.Now().UTC().Unix()

	assert.GreaterOrEqual(t, timestamp, before)
	assert.LessOrEqual(t, timestamp, after)
}

func TestGetCurrentTimestamp_IsReasonable(t *testing.T) {
	timestamp := GetCurrentTimestamp()

	// Timestamp should be after 2020-01-01
	afterYear2020 := int64(1577836800)
	assert.Greater(t, timestamp, afterYear2020)

	// Timestamp should be before 2100-01-01
	beforeYear2100 := int64(4102444800)
	assert.Less(t, timestamp, beforeYear2100)
}

func TestGetString_ExistingKey(t *testing.T) {
	m := map[string]interface{}{
		"name":  "test",
		"value": "data",
	}

	result := GetString(m, "name")
	assert.Equal(t, "test", result)
}

func TestGetString_NonExistentKey(t *testing.T) {
	m := map[string]interface{}{
		"name": "test",
	}

	result := GetString(m, "missing")
	assert.Equal(t, "", result)
}

func TestGetString_WrongType(t *testing.T) {
	m := map[string]interface{}{
		"number": 123,
		"bool":   true,
		"slice":  []string{"a", "b"},
	}

	assert.Equal(t, "", GetString(m, "number"))
	assert.Equal(t, "", GetString(m, "bool"))
	assert.Equal(t, "", GetString(m, "slice"))
}

func TestGetString_EmptyMap(t *testing.T) {
	m := map[string]interface{}{}

	result := GetString(m, "key")
	assert.Equal(t, "", result)
}

func TestGetString_NilMap(t *testing.T) {
	var m map[string]interface{}

	// Should not panic
	result := GetString(m, "key")
	assert.Equal(t, "", result)
}

func TestGetString_EmptyStringValue(t *testing.T) {
	m := map[string]interface{}{
		"empty": "",
	}

	result := GetString(m, "empty")
	assert.Equal(t, "", result)
}

func TestGetString_NestedStructure(t *testing.T) {
	m := map[string]interface{}{
		"nested": map[string]interface{}{
			"value": "test",
		},
	}

	// Should return empty string since value is not a string
	result := GetString(m, "nested")
	assert.Equal(t, "", result)
}

func TestGenerateID_Format(t *testing.T) {
	for i := 0; i < 10; i++ {
		id := GenerateID()

		// Check prefix
		assert.True(t, strings.HasPrefix(id, "chatcmpl-"))

		// Check suffix is hex
		suffix := id[9:]
		_ = strings.ToLower(suffix)
		for _, ch := range suffix {
			assert.True(t, (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F'),
				"Invalid hex character: %c", ch)
		}
	}
}

func TestGetString_MultipleStringValues(t *testing.T) {
	m := map[string]interface{}{
		"first":  "value1",
		"second": "value2",
		"third":  "value3",
	}

	assert.Equal(t, "value1", GetString(m, "first"))
	assert.Equal(t, "value2", GetString(m, "second"))
	assert.Equal(t, "value3", GetString(m, "third"))
}

func TestGetString_LongString(t *testing.T) {
	longString := strings.Repeat("x", 10000)
	m := map[string]interface{}{
		"long": longString,
	}

	result := GetString(m, "long")
	assert.Equal(t, longString, result)
}
