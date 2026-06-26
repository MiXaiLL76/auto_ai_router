package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

type sosanaTaskResponse struct {
	Status          string          `json:"status"`
	CreatedAt       string          `json:"created_at"`
	ResultFileURL   string          `json:"result_file_url"`
	OptimizedPrompt string          `json:"optimized_prompt"`
	Error           json.RawMessage `json:"error"`
}

func normalizeOpenAIImageResponseBody(body []byte) ([]byte, error) {
	var envelope struct {
		Error json.RawMessage `json:"error"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return nil, fmt.Errorf("parse image response: %w", err)
	}

	var resp openai.OpenAIImageResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("parse OpenAI image response: %w", err)
	}
	if len(resp.Data) == 0 {
		if sosanaResp, ok, err := openAIImageResponseFromSosanaTask(body); err != nil {
			return nil, err
		} else if ok {
			resp = sosanaResp
		} else {
			if hasJSONValue(envelope.Error) {
				return nil, fmt.Errorf("image response contains upstream error envelope")
			}
			return nil, fmt.Errorf("image response contains no image data")
		}
	}
	for i, item := range resp.Data {
		if item.B64JSON == "" && item.URL == "" {
			return nil, fmt.Errorf("image response item %d contains neither b64_json nor url", i)
		}
	}
	if resp.Created == 0 {
		resp.Created = converterutil.GetCurrentTimestamp()
	}

	normalized, err := json.Marshal(resp)
	if err != nil {
		return nil, fmt.Errorf("marshal OpenAI image response: %w", err)
	}
	return append(normalized, '\n'), nil
}

func openAIImageResponseFromSosanaTask(body []byte) (openai.OpenAIImageResponse, bool, error) {
	var task sosanaTaskResponse
	if err := json.Unmarshal(body, &task); err != nil {
		return openai.OpenAIImageResponse{}, false, nil
	}

	switch task.Status {
	case "":
		return openai.OpenAIImageResponse{}, false, nil
	case "COMPLETED":
		if task.ResultFileURL == "" {
			return openai.OpenAIImageResponse{}, true, fmt.Errorf("image task completed response contains no result URL")
		}
		resp := openai.OpenAIImageResponse{
			Created: parseSosanaCreatedAt(task.CreatedAt),
			Data: []openai.OpenAIImageData{{
				URL:           task.ResultFileURL,
				RevisedPrompt: task.OptimizedPrompt,
			}},
		}
		return resp, true, nil
	case "PROCESSING", "FAILED", "MODERATED":
		return openai.OpenAIImageResponse{}, true, fmt.Errorf("image task response is not a completed image result")
	default:
		return openai.OpenAIImageResponse{}, true, fmt.Errorf("image task response contains an unknown status")
	}
}

func parseSosanaCreatedAt(value string) int64 {
	if value == "" {
		return converterutil.GetCurrentTimestamp()
	}
	if ts, err := time.Parse(time.RFC3339Nano, value); err == nil {
		return ts.Unix()
	}
	return converterutil.GetCurrentTimestamp()
}

func hasJSONValue(raw json.RawMessage) bool {
	raw = bytes.TrimSpace(raw)
	return len(raw) > 0 && !bytes.Equal(raw, []byte("null"))
}
