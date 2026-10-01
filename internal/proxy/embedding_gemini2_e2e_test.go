package proxy

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	pricing "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// geminiEmbedding2TestPrice is the VseLLM gemini-embedding-2 price entry.
func geminiEmbedding2TestPrice() *pricing.ModelPrice {
	return &pricing.ModelPrice{
		InputCostPerToken:      0.00000026,
		InputCostPerImageToken: 0.000000585,
		InputCostPerAudioToken: 0.00000845,
		InputCostPerVideoToken: 0.0000156,
	}
}

func newGeminiEmbeddingProxy(t *testing.T, upstreamURL string) (*Proxy, *stubLiteLLMManager) {
	t.Helper()
	dbStub := &stubLiteLLMManager{}
	prx := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name:    "gemini-api",
			Type:    config.ProviderTypeGemini,
			BaseURL: upstreamURL,
			APIKey:  "gemini-key",
			RPM:     100,
			TPM:     1000000,
		}).
		WithMasterKey("master-key").
		Build()
	prx.LiteLLMDB = dbStub
	setTestModelPrice(prx, "gemini-embedding-2", geminiEmbedding2TestPrice())
	return prx, dbStub
}

func postEmbeddings(prx *Proxy, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest("POST", "/v1/embeddings", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	return w
}

func TestProxyRequest_GeminiEmbedding2BatchBillsEachModality(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		require.Equal(t, "/v1beta/models/gemini-embedding-2:batchEmbedContents", r.URL.Path)
		require.Equal(t, "gemini-key", r.Header.Get("x-goog-api-key"))
		var batch struct {
			Requests []struct {
				Model   string `json:"model"`
				Content struct {
					Parts []map[string]any `json:"parts"`
				} `json:"content"`
				OutputDimensionality int `json:"outputDimensionality"`
			} `json:"requests"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&batch))
		require.Len(t, batch.Requests, 2)
		assert.Equal(t, "models/gemini-embedding-2", batch.Requests[0].Model)
		assert.Equal(t, 768, batch.Requests[0].OutputDimensionality)
		assert.Len(t, batch.Requests[0].Content.Parts, 1)
		assert.Len(t, batch.Requests[1].Content.Parts, 4, "the mixed item stays one content")

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"embeddings": [{"values": [0.1, 0.2]}, {"values": [0.3, 0.4]}],
			"usageMetadata": {
				"promptTokenCount": 1300,
				"promptTokenDetails": [
					{"modality": "TEXT", "tokenCount": 132},
					{"modality": "IMAGE", "tokenCount": 258},
					{"modality": "AUDIO", "tokenCount": 250},
					{"modality": "VIDEO", "tokenCount": 660}
				]
			}
		}`))
	}))
	defer upstream.Close()
	prx, dbStub := newGeminiEmbeddingProxy(t, upstream.URL)

	w := postEmbeddings(prx, `{
		"model": "gemini-embedding-2",
		"dimensions": 768,
		"input": [
			"task: search result | query: dog on a beach",
			[
				"title: none | text: a dog",
				{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
				{"type": "input_audio", "input_audio": {"data": "SUQz", "format": "mp3"}},
				{"type": "video_url", "video_url": {"url": "gs://bucket/dog.mp4"}}
			]
		]
	}`)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(1), calls.Load())
	var resp struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
		Usage struct {
			PromptTokens        int            `json:"prompt_tokens"`
			TotalTokens         int            `json:"total_tokens"`
			PromptTokensDetails map[string]int `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	require.Len(t, resp.Data, 2)
	assert.Equal(t, 1, resp.Data[1].Index)
	assert.Equal(t, []float64{0.3, 0.4}, resp.Data[1].Embedding)
	assert.Equal(t, 1300, resp.Usage.PromptTokens)
	assert.Equal(t, map[string]int{"text_tokens": 132, "image_tokens": 258, "audio_tokens": 250, "video_tokens": 660}, resp.Usage.PromptTokensDetails)

	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	assert.Equal(t, 1300, entry.PromptTokens)
	assert.Equal(t, 0, entry.CompletionTokens)
	wantSpend := 132*0.00000026 + 258*0.000000585 + 250*0.00000845 + 660*0.0000156
	assert.InDelta(t, wantSpend, entry.Spend, 1e-12)

	metadata := decodeMetadata(t, entry.Metadata)
	details := metadata["usage_object"].(map[string]any)["prompt_tokens_details"].(map[string]any)
	assert.Equal(t, float64(258), details["image_tokens"])
	assert.Equal(t, float64(250), details["audio_tokens"])
	assert.Equal(t, float64(660), details["video_tokens"])
	costBreakdown := metadata["cost_breakdown"].(map[string]any)
	assert.InDelta(t, 132*0.00000026, costBreakdown["input_cost"], 1e-15)
	assert.InDelta(t, 258*0.000000585, costBreakdown["image_cost"], 1e-15)
	assert.InDelta(t, 250*0.00000845, costBreakdown["audio_input_cost"], 1e-15)
	assert.InDelta(t, 660*0.0000156, costBreakdown["video_input_cost"], 1e-15)
	assert.Equal(t, float64(0), costBreakdown["output_cost"])
}

func TestProxyRequest_GeminiEmbedding2SingleInputUsesEmbedContent(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1beta/models/gemini-embedding-2:embedContent", r.URL.Path)
		body, _ := io.ReadAll(r.Body)
		assert.JSONEq(t, `{"model":"models/gemini-embedding-2","content":{"parts":[{"text":"What is the meaning of life?"}]}}`, string(body))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embedding":{"values":[0.5,0.25]},"usageMetadata":{"promptTokenCount":8}}`))
	}))
	defer upstream.Close()
	prx, dbStub := newGeminiEmbeddingProxy(t, upstream.URL)

	w := postEmbeddings(prx, `{"model":"gemini-embedding-2","input":"What is the meaning of life?"}`)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	var resp struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	require.Len(t, resp.Data, 1)
	assert.Equal(t, []float64{0.5, 0.25}, resp.Data[0].Embedding)
	assert.Equal(t, 8, resp.Usage.PromptTokens, "provider count, not a length/4 estimate")
	require.Len(t, dbStub.loggedEntries, 1)
	assert.InDelta(t, 8*0.00000026, dbStub.loggedEntries[0].Spend, 1e-15)
}

func TestProxyRequest_GeminiEmbedding2UnusableInputIs400WithoutUpstreamCall(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()
	prx, _ := newGeminiEmbeddingProxy(t, upstream.URL)

	w := postEmbeddings(prx, `{"model":"gemini-embedding-2","input":[[{"type":"image_url","image_url":{"url":"https://example.com/no-extension"}}]]}`)

	assert.Equal(t, http.StatusBadRequest, w.Code, w.Body.String())
	assert.Contains(t, w.Body.String(), "mime_type")
	assert.Equal(t, int32(0), calls.Load())
}

func TestDoEmbedContentFanOut_MergesRepliesInInputOrder(t *testing.T) {
	var mu sync.Mutex
	var seen []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1beta1/projects/p/locations/global/publishers/google/models/gemini-embedding-2:embedContent", r.URL.Path)
		require.Equal(t, "Bearer vertex-token", r.Header.Get("Authorization"))
		var req struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		text := req.Content.Parts[0].Text
		mu.Lock()
		seen = append(seen, text)
		mu.Unlock()
		// Answer the first input last so completion order differs from input order.
		if text == "a" {
			time.Sleep(50 * time.Millisecond)
		}
		values := map[string]string{"a": "[1]", "b": "[2]", "c": "[3]"}[text]
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embedding":{"values":` + values + `},"usageMetadata":{"promptTokenCount":1,"totalTokenCount":1,"promptTokensDetails":[{"modality":"TEXT","tokenCount":1}]}}`))
	}))
	defer upstream.Close()
	prx := NewTestProxyBuilder().Build()

	template, err := http.NewRequest(http.MethodPost, upstream.URL+"/v1beta1/projects/p/locations/global/publishers/google/models/gemini-embedding-2:embedContent", nil)
	require.NoError(t, err)
	template.Header.Set("Authorization", "Bearer vertex-token")
	template.Header.Set("Content-Type", "application/json")

	resp, inputFault, err := prx.doEmbedContentFanOut(template, "gemini-embedding-2", [][]byte{
		[]byte(`{"content":{"parts":[{"text":"a"}]}}`),
		[]byte(`{"content":{"parts":[{"text":"b"}]}}`),
		[]byte(`{"content":{"parts":[{"text":"c"}]}}`),
	}, &embedContentFanOutReplies{})

	require.NoError(t, err)
	assert.False(t, inputFault)
	require.Equal(t, http.StatusOK, resp.StatusCode)
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	assert.JSONEq(t, `{
		"embeddings": [{"values":[1]},{"values":[2]},{"values":[3]}],
		"usageMetadata": {"promptTokenCount":3,"totalTokenCount":3,"promptTokensDetails":[{"modality":"TEXT","tokenCount":3}]}
	}`, string(body))
	assert.ElementsMatch(t, []string{"a", "b", "c"}, seen)
}

func TestDoEmbedContentFanOut_ReturnsProviderErrorOfLowestInput(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		if strings.Contains(string(body), `"bad"`) {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"code":429,"status":"RESOURCE_EXHAUSTED"}}`))
			return
		}
		_, _ = w.Write([]byte(`{"embedding":{"values":[1]}}`))
	}))
	defer upstream.Close()
	prx := NewTestProxyBuilder().Build()
	template, err := http.NewRequest(http.MethodPost, upstream.URL+"/x:embedContent", nil)
	require.NoError(t, err)

	replies := &embedContentFanOutReplies{}
	resp, inputFault, err := prx.doEmbedContentFanOut(template, "gemini-embedding-2", [][]byte{
		[]byte(`{"content":{"parts":[{"text":"ok"}]}}`),
		[]byte(`{"content":{"parts":[{"text":"bad"}]}}`),
	}, replies)

	require.NoError(t, err)
	assert.False(t, inputFault, "a quota error is about the credential, not the input")
	assert.Len(t, replies.byInput, 1, "the reply that did arrive is kept for the retry")
	assert.Equal(t, http.StatusTooManyRequests, resp.StatusCode)
	body, _ := io.ReadAll(resp.Body)
	assert.JSONEq(t, `{"error":{"code":429,"status":"RESOURCE_EXHAUSTED"}}`, string(body))
}

func TestDoEmbedContentFanOut_TransportErrorIsReturned(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	upstreamURL := upstream.URL
	upstream.Close()
	prx := NewTestProxyBuilder().Build()
	template, err := http.NewRequest(http.MethodPost, upstreamURL+"/x:embedContent", nil)
	require.NoError(t, err)

	resp, inputFault, err := prx.doEmbedContentFanOut(template, "gemini-embedding-2", [][]byte{[]byte(`{}`), []byte(`{}`)}, &embedContentFanOutReplies{})

	require.Error(t, err)
	assert.False(t, inputFault)
	assert.Nil(t, resp)
}

func TestBuildKafkaSpendEvent_VideoInputFields(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	logCtx := testLogCtx(t)
	logCtx.TokenUsage = &converter.TokenUsage{PromptTokens: 700, ImageTokens: 40, VideoInputTokens: 660}

	event := prx.buildKafkaSpendEvent(logCtx, "cred", "cred:gemini-embedding-2", "hash",
		"", "", "", "", "aiplatform.googleapis.com", "success", 0.0103,
		&converter.TokenCosts{ImageCost: 0.0000234, VideoInputCost: 0.010296, TotalCost: 0.0103}, 0, logCtx.StartTime)

	assert.Equal(t, 40, event.ImageTokens)
	assert.Equal(t, 660, event.VideoInputTokens)
	assert.Equal(t, 0.0000234, event.ImageCost)
	assert.Equal(t, 0.010296, event.VideoInputCost)
	encoded, err := json.Marshal(event)
	require.NoError(t, err)
	assert.Contains(t, string(encoded), `"video_input_tokens":660`)
	assert.Contains(t, string(encoded), `"video_input_cost":0.010296`)
}

// rewriteHostTransport sends every request to target, keeping path and query:
// the Vertex AI URL builder always targets *.googleapis.com.
type rewriteHostTransport struct {
	target *url.URL
}

func (t rewriteHostTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	clone := req.Clone(req.Context())
	clone.URL.Scheme = t.target.Scheme
	clone.URL.Host = t.target.Host
	clone.Host = t.target.Host
	return http.DefaultTransport.RoundTrip(clone)
}

// fakeServiceAccountJSON returns service account credentials whose OAuth token
// endpoint is tokenURL, so the real token manager obtains a token offline.
func fakeServiceAccountJSON(t *testing.T, tokenURL string) string {
	t.Helper()
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	require.NoError(t, err)
	der, err := x509.MarshalPKCS8PrivateKey(key)
	require.NoError(t, err)
	encoded, err := json.Marshal(map[string]string{
		"type":           "service_account",
		"project_id":     "grant",
		"private_key_id": "test-key",
		"private_key":    string(pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der})),
		"client_email":   "embedder@grant.iam.gserviceaccount.com",
		"client_id":      "1",
		"token_uri":      tokenURL,
	})
	require.NoError(t, err)
	return string(encoded)
}

func TestProxyRequest_VertexGeminiEmbedding2FansOutAndBills(t *testing.T) {
	var embedCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/token" {
			_, _ = w.Write([]byte(`{"access_token":"vertex-token","token_type":"Bearer","expires_in":3600}`))
			return
		}
		embedCalls.Add(1)
		require.Equal(t, "/v1beta1/projects/grant/locations/global/publishers/google/models/gemini-embedding-2:embedContent", r.URL.Path)
		require.Equal(t, "Bearer vertex-token", r.Header.Get("Authorization"))
		var req struct {
			Content struct {
				Parts []map[string]any `json:"parts"`
			} `json:"content"`
			EmbedContentConfig struct {
				OutputDimensionality int `json:"outputDimensionality"`
			} `json:"embedContentConfig"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		assert.Equal(t, 1536, req.EmbedContentConfig.OutputDimensionality)
		if len(req.Content.Parts) == 1 {
			_, _ = w.Write([]byte(`{"embedding":{"values":[1,0]},"usageMetadata":{"promptTokenCount":6,"totalTokenCount":6,"promptTokensDetails":[{"modality":"TEXT","tokenCount":6}]}}`))
			return
		}
		_, _ = w.Write([]byte(`{"embedding":{"values":[0,1]},"usageMetadata":{"promptTokenCount":264,"totalTokenCount":264,"promptTokensDetails":[{"modality":"TEXT","tokenCount":6},{"modality":"DOCUMENT","tokenCount":258}]}}`))
	}))
	defer upstream.Close()
	target, err := url.Parse(upstream.URL)
	require.NoError(t, err)

	dbStub := &stubLiteLLMManager{}
	prx := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name:            "grant-global",
			Type:            config.ProviderTypeVertexAI,
			ProjectID:       "grant",
			Location:        "global",
			CredentialsJSON: fakeServiceAccountJSON(t, upstream.URL+"/token"),
			RPM:             -1,
			TPM:             -1,
		}).
		WithMasterKey("master-key").
		Build()
	prx.client = &http.Client{Transport: rewriteHostTransport{target: target}}
	prx.LiteLLMDB = dbStub
	setTestModelPrice(prx, "gemini-embedding-2", geminiEmbedding2TestPrice())

	w := postEmbeddings(prx, `{
		"model": "gemini-embedding-2",
		"dimensions": 1536,
		"input": [
			"task: search result | query: invoice",
			[{"type": "text", "text": "title: none | text:"}, {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBERi0xLjc="}}]
		]
	}`)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(2), embedCalls.Load(), "one embedContent call per input")
	var resp struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
		Usage struct {
			PromptTokens        int            `json:"prompt_tokens"`
			PromptTokensDetails map[string]int `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	require.Len(t, resp.Data, 2)
	assert.Equal(t, []float64{1, 0}, resp.Data[0].Embedding)
	assert.Equal(t, []float64{0, 1}, resp.Data[1].Embedding)
	assert.Equal(t, 270, resp.Usage.PromptTokens)
	assert.Equal(t, map[string]int{"text_tokens": 12, "image_tokens": 258}, resp.Usage.PromptTokensDetails, "PDF pages bill as images")

	require.Len(t, dbStub.loggedEntries, 1)
	assert.Equal(t, 270, dbStub.loggedEntries[0].PromptTokens)
	assert.InDelta(t, 12*0.00000026+258*0.000000585, dbStub.loggedEntries[0].Spend, 1e-15)
}

// embedTextUpstream serves Vertex AI embedContent calls from answer, keyed by
// the first part's text, and records each call as "project/text".
func embedTextUpstream(t *testing.T, answer func(project, text string, w http.ResponseWriter)) (*httptest.Server, func() []string) {
	t.Helper()
	var mu sync.Mutex
	var calls []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/token" {
			_, _ = w.Write([]byte(`{"access_token":"vertex-token","token_type":"Bearer","expires_in":3600}`))
			return
		}
		project := strings.Split(strings.TrimPrefix(r.URL.Path, "/v1beta1/projects/"), "/")[0]
		var req struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		text := req.Content.Parts[0].Text
		mu.Lock()
		calls = append(calls, project+"/"+text)
		mu.Unlock()
		answer(project, text, w)
	}))
	return upstream, func() []string {
		mu.Lock()
		defer mu.Unlock()
		return append([]string(nil), calls...)
	}
}

func writeTextEmbedding(w http.ResponseWriter, value string) {
	_, _ = w.Write([]byte(`{"embedding":{"values":[` + value + `]},"usageMetadata":{"promptTokenCount":1,"totalTokenCount":1,"promptTokensDetails":[{"modality":"TEXT","tokenCount":1}]}}`))
}

func writeInvalidArgument(w http.ResponseWriter) {
	w.WriteHeader(http.StatusBadRequest)
	_, _ = w.Write([]byte(`{"error":{"code":400,"message":"Request contains an invalid argument.","status":"INVALID_ARGUMENT"}}`))
}

func textEmbedBodies(texts ...string) [][]byte {
	bodies := make([][]byte, len(texts))
	for i, text := range texts {
		bodies[i] = []byte(`{"content":{"parts":[{"text":"` + text + `"}]}}`)
	}
	return bodies
}

func fanOutTemplate(t *testing.T, upstreamURL string) *http.Request {
	t.Helper()
	template, err := http.NewRequest(http.MethodPost, upstreamURL+"/v1beta1/projects/p/locations/global/publishers/google/models/gemini-embedding-2:embedContent", nil)
	require.NoError(t, err)
	return template
}

func TestDoEmbedContentFanOut_InputRefusalBesideSuccessIsInputFault(t *testing.T) {
	upstream, _ := embedTextUpstream(t, func(_, text string, w http.ResponseWriter) {
		if text == "bad" {
			time.Sleep(50 * time.Millisecond) // let the good input succeed first
			writeInvalidArgument(w)
			return
		}
		writeTextEmbedding(w, "1")
	})
	defer upstream.Close()
	prx := NewTestProxyBuilder().Build()

	resp, inputFault, err := prx.doEmbedContentFanOut(fanOutTemplate(t, upstream.URL), "gemini-embedding-2", textEmbedBodies("ok", "bad"), &embedContentFanOutReplies{})

	require.NoError(t, err)
	assert.Equal(t, http.StatusBadRequest, resp.StatusCode)
	assert.True(t, inputFault, "the credential embedded the other input, so the refused one is at fault")
}

func TestDoEmbedContentFanOut_RefusalWithoutSuccessIsNotInputFault(t *testing.T) {
	upstream, _ := embedTextUpstream(t, func(_, _ string, w http.ResponseWriter) {
		writeInvalidArgument(w)
	})
	defer upstream.Close()
	prx := NewTestProxyBuilder().Build()

	resp, inputFault, err := prx.doEmbedContentFanOut(fanOutTemplate(t, upstream.URL), "gemini-embedding-2", textEmbedBodies("x", "y"), &embedContentFanOutReplies{})

	require.NoError(t, err)
	assert.Equal(t, http.StatusBadRequest, resp.StatusCode)
	assert.False(t, inputFault, "nothing succeeded here, so the credential itself may be the problem")
}

func TestDoEmbedContentFanOut_RetrySendsOnlyMissingInputs(t *testing.T) {
	var cFailed atomic.Bool
	upstream, calls := embedTextUpstream(t, func(_, text string, w http.ResponseWriter) {
		if text == "c" && cFailed.CompareAndSwap(false, true) {
			time.Sleep(50 * time.Millisecond) // a and b are in flight and must finish
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"code":429,"status":"RESOURCE_EXHAUSTED"}}`))
			return
		}
		writeTextEmbedding(w, map[string]string{"a": "1", "b": "2", "c": "3"}[text])
	})
	defer upstream.Close()
	prx := NewTestProxyBuilder().Build()
	replies := &embedContentFanOutReplies{}
	bodies := textEmbedBodies("a", "b", "c")

	first, _, err := prx.doEmbedContentFanOut(fanOutTemplate(t, upstream.URL), "gemini-embedding-2", bodies, replies)
	require.NoError(t, err)
	require.Equal(t, http.StatusTooManyRequests, first.StatusCode)

	second, inputFault, err := prx.doEmbedContentFanOut(fanOutTemplate(t, upstream.URL), "gemini-embedding-2", bodies, replies)
	require.NoError(t, err)
	assert.False(t, inputFault)
	require.Equal(t, http.StatusOK, second.StatusCode)
	body, err := io.ReadAll(second.Body)
	require.NoError(t, err)
	assert.JSONEq(t, `{
		"embeddings": [{"values":[1]},{"values":[2]},{"values":[3]}],
		"usageMetadata": {"promptTokenCount":3,"totalTokenCount":3,"promptTokensDetails":[{"modality":"TEXT","tokenCount":3}]}
	}`, string(body), "replies from both attempts, each billed once")
	assert.ElementsMatch(t, []string{"p/a", "p/b", "p/c", "p/c"}, calls(), "the retry re-sent only the missing input")
}

func TestDoEmbedContentFanOut_RepliesAreNotReusedAcrossModels(t *testing.T) {
	upstream, calls := embedTextUpstream(t, func(_, _ string, w http.ResponseWriter) {
		writeTextEmbedding(w, "1")
	})
	defer upstream.Close()
	prx := NewTestProxyBuilder().Build()
	replies := &embedContentFanOutReplies{}
	bodies := textEmbedBodies("a", "b")

	_, _, err := prx.doEmbedContentFanOut(fanOutTemplate(t, upstream.URL), "gemini-embedding-2-preview", bodies, replies)
	require.NoError(t, err)
	_, _, err = prx.doEmbedContentFanOut(fanOutTemplate(t, upstream.URL), "gemini-embedding-2", bodies, replies)
	require.NoError(t, err)

	assert.Len(t, calls(), 4, "vectors of another model are never mixed in")
}

func newVertexEmbeddingProxy(t *testing.T, upstreamURL string, projects ...string) (*Proxy, *stubLiteLLMManager) {
	t.Helper()
	target, err := url.Parse(upstreamURL)
	require.NoError(t, err)
	creds := make([]config.CredentialConfig, len(projects))
	for i, project := range projects {
		creds[i] = config.CredentialConfig{
			Name:            project,
			Type:            config.ProviderTypeVertexAI,
			ProjectID:       project,
			Location:        "global",
			CredentialsJSON: fakeServiceAccountJSON(t, upstreamURL+"/token"),
			RPM:             -1,
			TPM:             -1,
		}
	}
	dbStub := &stubLiteLLMManager{}
	prx := NewTestProxyBuilder().WithCredentials(creds...).WithMasterKey("master-key").Build()
	prx.client = &http.Client{Transport: rewriteHostTransport{target: target}}
	prx.maxProviderRetries = 4
	prx.LiteLLMDB = dbStub
	setTestModelPrice(prx, "gemini-embedding-2", geminiEmbedding2TestPrice())
	return prx, dbStub
}

func TestProxyRequest_VertexGeminiEmbedding2InputRefusalIsNotRetried(t *testing.T) {
	var refused, embedded atomic.Int32
	upstream, _ := embedTextUpstream(t, func(_, text string, w http.ResponseWriter) {
		if text == "bad" {
			refused.Add(1)
			time.Sleep(50 * time.Millisecond)
			writeInvalidArgument(w)
			return
		}
		embedded.Add(1)
		writeTextEmbedding(w, "1")
	})
	defer upstream.Close()
	prx, dbStub := newVertexEmbeddingProxy(t, upstream.URL, "grant-a", "grant-b", "grant-c")

	inputs := make([]string, 20)
	for i := range inputs {
		inputs[i] = fmt.Sprintf("%q", fmt.Sprintf("text %d", i))
	}
	inputs[15] = `"bad"`
	w := postEmbeddings(prx, `{"model":"gemini-embedding-2","input":[`+strings.Join(inputs, ",")+`]}`)

	assert.Equal(t, http.StatusBadRequest, w.Code, w.Body.String())
	assert.Equal(t, int32(1), refused.Load(), "a refused input is not replayed on the other credentials")
	assert.LessOrEqual(t, embedded.Load(), int32(19), "each other input is sent at most once")
	require.Len(t, dbStub.loggedEntries, 1)
	assert.Zero(t, dbStub.loggedEntries[0].Spend)
}

func TestProxyRequest_VertexGeminiEmbedding2RetrySendsOnlyMissingInputs(t *testing.T) {
	var cFailed atomic.Bool
	upstream, calls := embedTextUpstream(t, func(_, text string, w http.ResponseWriter) {
		if text == "c" && cFailed.CompareAndSwap(false, true) {
			time.Sleep(50 * time.Millisecond)
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"code":429,"status":"RESOURCE_EXHAUSTED"}}`))
			return
		}
		writeTextEmbedding(w, map[string]string{"a": "1", "b": "2", "c": "3"}[text])
	})
	defer upstream.Close()
	prx, dbStub := newVertexEmbeddingProxy(t, upstream.URL, "grant-a", "grant-b")

	w := postEmbeddings(prx, `{"model":"gemini-embedding-2","input":["a","b","c"]}`)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	var resp struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	require.Len(t, resp.Data, 3)
	assert.Equal(t, []float64{3}, resp.Data[2].Embedding)
	assert.Equal(t, 3, resp.Usage.PromptTokens)

	seen := calls()
	require.Len(t, seen, 4, "a and b once, c on both credentials")
	var cCalls []string
	for _, call := range seen {
		if strings.HasSuffix(call, "/c") {
			cCalls = append(cCalls, call)
		}
	}
	require.Len(t, cCalls, 2)
	assert.NotEqual(t, cCalls[0], cCalls[1], "c was retried on the other credential")
	require.Len(t, dbStub.loggedEntries, 1)
	assert.InDelta(t, 3*0.00000026, dbStub.loggedEntries[0].Spend, 1e-15)
}
