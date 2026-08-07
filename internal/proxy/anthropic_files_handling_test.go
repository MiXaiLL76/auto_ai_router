package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProxyRequest_AnthropicResponsesInputFileValidation(t *testing.T) {
	const providerType config.ProviderType = "test-anthropic-responses"
	responses.RegisterProviderResponses(providerType, func(mode responses.ResponsesRequestMode) responses.ProviderResponses {
		return testAnthropicProviderResponses{mode: mode}
	})

	tests := []struct {
		name         string
		filePart     string
		wantStatus   int
		wantSource   map[string]string
		wantUpstream bool
		wantError    string
	}{
		{
			name:       "file_data becomes document base64",
			filePart:   `{"type":"input_file","filename":"test.pdf","file_data":"data:application/pdf;base64,JVBERi0="}`,
			wantStatus: http.StatusOK,
			wantSource: map[string]string{
				"type":       "base64",
				"media_type": "application/pdf",
				"data":       "JVBERi0=",
			},
			wantUpstream: true,
		},
		{
			name:       "file_url stays document url",
			filePart:   `{"type":"input_file","file_url":"https://example.com/document.pdf"}`,
			wantStatus: http.StatusOK,
			wantSource: map[string]string{
				"type": "url",
				"url":  "https://example.com/document.pdf",
			},
			wantUpstream: true,
		},
		{
			name:         "file_id unsupported returns 400",
			filePart:     `{"type":"input_file","file_id":"file-abc"}`,
			wantStatus:   http.StatusBadRequest,
			wantUpstream: false,
			wantError:    "input_file.file_id",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var upstreamCalls int32
			var upstreamBody map[string]interface{}
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				atomic.AddInt32(&upstreamCalls, 1)
				require.NoError(t, json.NewDecoder(r.Body).Decode(&upstreamBody))
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{
					"id":"msg_1",
					"type":"message",
					"role":"assistant",
					"model":"claude-sonnet-4-5",
					"content":[{"type":"text","text":"ok"}],
					"stop_reason":"end_turn",
					"usage":{"input_tokens":1,"output_tokens":1}
				}`))
			}))
			defer upstream.Close()

			prx := NewTestProxyBuilder().
				WithSingleCredential("anthropic", providerType, upstream.URL, "upstream-key").
				Build()
			body := fmt.Sprintf(`{
				"model":"claude-sonnet-4-5",
				"input":[{"role":"user","content":[%s,{"type":"input_text","text":"read it"}]}],
				"max_output_tokens":128
			}`, tt.filePart)
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			prx.ProxyRequest(w, req)

			require.Equal(t, tt.wantStatus, w.Code, w.Body.String())
			assert.Equal(t, tt.wantUpstream, atomic.LoadInt32(&upstreamCalls) > 0)
			if tt.wantError != "" {
				assert.Contains(t, w.Body.String(), tt.wantError)
			}
			if tt.wantSource != nil {
				block := firstUpstreamContentBlock(t, upstreamBody)
				assert.Equal(t, "document", block["type"])
				source := block["source"].(map[string]interface{})
				for key, want := range tt.wantSource {
					assert.Equal(t, want, source[key])
				}
			}
		})
	}
}

func TestProxyRequest_NativeResponsesInternalConversionErrorRemains500(t *testing.T) {
	const providerType config.ProviderType = "test-native-responses-internal-error"
	responses.RegisterProviderResponses(providerType, func(mode responses.ResponsesRequestMode) responses.ProviderResponses {
		return failingProviderResponses{}
	})

	var upstreamCalls int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&upstreamCalls, 1)
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("failing", providerType, upstream.URL, "upstream-key").
		Build()
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"any","input":"hello"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusInternalServerError, w.Code, w.Body.String())
	assert.Equal(t, int32(0), atomic.LoadInt32(&upstreamCalls))
}

func TestSanitizeRequestBodyForLogRedactsDocumentPayloads(t *testing.T) {
	body := []byte(`{
		"messages":[{
			"content":[
				{"type":"file","file":{"file_data":"data:application/pdf;base64,JVBERi0="}},
				{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}
			]
		}]
	}`)

	got := sanitizeRequestBodyForLog(body)

	assert.NotContains(t, got, "data:application/pdf;base64,JVBERi0=")
	assert.NotContains(t, got, `"data":"JVBERi0="`)
	assert.Contains(t, got, `"file_data":"<redacted bytes=36>"`)
	assert.Contains(t, got, `"data":"<redacted bytes=8>"`)
	assert.Contains(t, got, `"media_type":"application/pdf"`)
}

type failingProviderResponses struct{}

type testAnthropicProviderResponses struct {
	mode responses.ResponsesRequestMode
}

func (t testAnthropicProviderResponses) RequestFrom(body []byte) ([]byte, string, error) {
	var request map[string]interface{}
	if err := json.Unmarshal(body, &request); err != nil {
		return nil, "", err
	}
	source, err := testInputFileSource(request)
	if err != nil {
		return nil, "", err
	}
	model := t.mode.ModelID
	if model == "" {
		model, _ = request["model"].(string)
	}
	converted := map[string]interface{}{
		"model":      model,
		"max_tokens": 128,
		"messages": []interface{}{
			map[string]interface{}{
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{"type": "document", "source": source},
				},
			},
		},
	}
	encoded, err := json.Marshal(converted)
	if err != nil {
		return nil, "", err
	}
	return encoded, "application/json", nil
}

func (t testAnthropicProviderResponses) ResponseTo(body []byte, displayModelID string) (*responses.Response, error) {
	model := displayModelID
	if model == "" {
		model = t.mode.DisplayModel()
	}
	return &responses.Response{
		ID:         "resp_test",
		Object:     "response",
		Model:      model,
		Status:     "completed",
		Output:     []responses.OutputItem{},
		Tools:      []responses.Tool{},
		Metadata:   map[string]string{},
		Usage:      &responses.Usage{},
		Truncation: "disabled",
	}, nil
}

func (t testAnthropicProviderResponses) StreamTo(
	reader io.Reader,
	writer io.Writer,
	displayModelID string,
	meta *responses.ResponsesMetadata,
	onComplete func(*responses.Response),
) error {
	return fmt.Errorf("not used")
}

func (t testAnthropicProviderResponses) BuildURL(cred *config.CredentialConfig) string {
	return cred.BaseURL + "/v1/messages"
}

func testInputFileSource(request map[string]interface{}) (map[string]interface{}, error) {
	input, _ := request["input"].([]interface{})
	if len(input) == 0 {
		return nil, fmt.Errorf("missing input")
	}
	message, _ := input[0].(map[string]interface{})
	content, _ := message["content"].([]interface{})
	if len(content) == 0 {
		return nil, fmt.Errorf("missing content")
	}
	filePart, _ := content[0].(map[string]interface{})
	fileData, _ := filePart["file_data"].(string)
	if fileData != "" {
		const prefix = "data:application/pdf;base64,"
		if !strings.HasPrefix(fileData, prefix) {
			return nil, converterutil.NewRequestValidationError("input_file.file_data", "malformed file_data")
		}
		return map[string]interface{}{
			"type":       "base64",
			"media_type": "application/pdf",
			"data":       strings.TrimPrefix(fileData, prefix),
		}, nil
	}
	if fileURL, _ := filePart["file_url"].(string); fileURL != "" {
		return map[string]interface{}{
			"type": "url",
			"url":  fileURL,
		}, nil
	}
	if fileID, _ := filePart["file_id"].(string); fileID != "" {
		return nil, converterutil.NewRequestValidationError("input_file.file_id", "input_file.file_id is not supported for Anthropic-backed models")
	}
	return nil, converterutil.NewRequestValidationError("input_file", "missing supported file source")
}

func (f failingProviderResponses) RequestFrom([]byte) ([]byte, string, error) {
	return nil, "", fmt.Errorf("synthetic internal conversion failure")
}

func (f failingProviderResponses) ResponseTo([]byte, string) (*responses.Response, error) {
	return nil, fmt.Errorf("not used")
}

func (f failingProviderResponses) StreamTo(
	reader io.Reader,
	writer io.Writer,
	displayModelID string,
	meta *responses.ResponsesMetadata,
	onComplete func(*responses.Response),
) error {
	return fmt.Errorf("not used")
}

func (f failingProviderResponses) BuildURL(*config.CredentialConfig) string {
	return "http://unused.local/v1/messages"
}

func firstUpstreamContentBlock(t *testing.T, body map[string]interface{}) map[string]interface{} {
	t.Helper()
	messages := body["messages"].([]interface{})
	require.NotEmpty(t, messages)
	content := messages[0].(map[string]interface{})["content"].([]interface{})
	require.NotEmpty(t, content)
	return content[0].(map[string]interface{})
}
