package sosana

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/stretchr/testify/require"
)

func TestSosanaLiveAcceptance(t *testing.T) {
	if os.Getenv("SOSANA_ACCEPTANCE") != "1" {
		t.Skip("SOSANA_ACCEPTANCE=1 not set, skipping paid Sosana live acceptance test")
	}

	apiKey := os.Getenv("SOSANA_API_KEY")
	require.NotEmpty(t, apiKey, "SOSANA_API_KEY is required for Sosana live acceptance test")

	baseURL := os.Getenv("SOSANA_BASE_URL")
	if baseURL == "" {
		baseURL = "https://sosana.art"
	}
	model := os.Getenv("SOSANA_MODEL")
	if model == "" {
		model = "banana-2-1k-compliant"
	}
	prompt := os.Getenv("SOSANA_PROMPT")
	if prompt == "" {
		prompt = "a small blue cube on a white background"
	}

	openAIBody, err := json.Marshal(map[string]any{
		"model":  model,
		"prompt": prompt,
		"size":   "1024x1024",
		"n":      1,
	})
	require.NoError(t, err)

	createBody, _, err := ImageGenerationRequest(openAIBody, model)
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	taskResult, err := DoTaskRequest(ctx, http.DefaultClient, http.MethodPost, CreateURL(baseURL), apiKey, createBody)
	require.NoError(t, err)
	require.Less(t, taskResult.StatusCode, http.StatusBadRequest, "response body: %s", string(taskResult.RawBody))

	task := taskResult.Task
	for task.Status == StatusProcessing {
		select {
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		case <-time.After(time.Duration(PollInterval)):
		}
		taskResult, err = DoTaskRequest(ctx, http.DefaultClient, http.MethodGet, PollURL(baseURL, task.UID), apiKey, nil)
		require.NoError(t, err)
		require.Less(t, taskResult.StatusCode, http.StatusBadRequest, "response body: %s", string(taskResult.RawBody))
		task = taskResult.Task
	}
	require.Equal(t, StatusCompleted, task.Status, "response body: %s", string(taskResult.RawBody))

	image, err := DownloadResultImage(ctx, http.DefaultClient, task)
	require.NoError(t, err)
	require.NotEmpty(t, image.Bytes)

	body, err := OpenAIImageResponse(task, image.Bytes)
	require.NoError(t, err)
	var resp openai.OpenAIImageResponse
	require.NoError(t, json.Unmarshal(body, &resp))
	require.Len(t, resp.Data, 1)
	require.Empty(t, resp.Data[0].URL)
	require.NotEmpty(t, resp.Data[0].B64JSON)
	_, err = base64.StdEncoding.DecodeString(resp.Data[0].B64JSON)
	require.NoError(t, err)
}
