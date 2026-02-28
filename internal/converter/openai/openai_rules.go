package openai

import (
	"bytes"
	"encoding/json"
	"strings"
)

type ModelParamsMapping struct {
	KeysToReplace map[string]string
	KeysToRemove  []string
}

func UpdateJSONField(body []byte, mapping ModelParamsMapping) []byte {
	// Используем any вместо interface{}, это современный стандарт Go
	var data map[string]any

	if err := json.Unmarshal(body, &data); err != nil {
		return body
	}

	// 1. Замена ключей
	for oldKey, newKey := range mapping.KeysToReplace {
		if val, ok := data[oldKey]; ok {
			// Сначала удаляем старый, потом пишем новый,
			// чтобы не было дублей, если старый и новый ключи совпали
			delete(data, oldKey)
			data[newKey] = val
		}
	}

	// 2. Удаление ключей
	for _, key := range mapping.KeysToRemove {
		delete(data, key)
	}

	// 3. Собираем обратно
	updatedBody, err := json.Marshal(data)
	if err != nil {
		return body
	}

	return updatedBody
}

// replaceModelInBody replaces the "model" field value in a JSON body.
// Uses simple byte-level replacement of `"model":"oldValue"` to avoid full re-serialization.
func ReplaceModelInBody(body []byte, oldModel, newModel string) []byte {
	oldToken, _ := json.Marshal(oldModel)
	newToken, _ := json.Marshal(newModel)

	// Replace "model":"oldModel" → "model":"newModel"
	// Handles both with and without spaces after colon
	patterns := [][]byte{
		append([]byte(`"model":`), oldToken...),
		append([]byte(`"model": `), oldToken...),
	}
	replacements := [][]byte{
		append([]byte(`"model":`), newToken...),
		append([]byte(`"model": `), newToken...),
	}

	for i, pattern := range patterns {
		if bytes.Contains(body, pattern) {
			return bytes.Replace(body, pattern, replacements[i], 1)
		}
	}

	return body
}

var gpt5mapping = ModelParamsMapping{
	KeysToReplace: map[string]string{
		"max_tokens": "max_completion_tokens",
	},
	KeysToRemove: []string{"top_p", "temperature"},
}

func ReplaceBodyParam(modelID string, body []byte) []byte {
	// Проверка на gpt-5
	if strings.Contains(modelID, "gpt-5") {
		return UpdateJSONField(body, gpt5mapping)
	}
	return body
}
