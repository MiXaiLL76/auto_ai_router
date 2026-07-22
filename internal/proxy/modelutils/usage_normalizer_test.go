package modelutils

import (
	"bytes"
	"encoding/json"
	"io"
	"strconv"
	"strings"
	"testing"
)

func TestNormalizeCompletionUsage_IncidentResponses(t *testing.T) {
	tests := []struct {
		name             string
		modelID          string
		completionTokens int64
		reasoningTokens  int64
		textTokens       int64
		wantTextTokens   int64
	}{
		{
			name:             "first incident request with provider prefix",
			modelID:          "openai/qwen3.6-35b-a3b",
			completionTokens: 4774,
			reasoningTokens:  4417,
			textTokens:       4774,
			wantTextTokens:   357,
		},
		{
			name:             "second incident request with version suffix",
			modelID:          "QWEN3.6-35B-A3B-20260415",
			completionTokens: 4726,
			reasoningTokens:  4357,
			textTokens:       4726,
			wantTextTokens:   369,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := []byte(`{
				"id":"chatcmpl-test",
				"choices":[{"index":0,"message":{"content":"ok"}}],
				"vendor_extension":{"trace":"preserve-me"},
				"usage":{
					"prompt_tokens":1370,
					"completion_tokens":` + strconv.FormatInt(tt.completionTokens, 10) + `,
					"total_tokens":6144,
					"provider_usage":{"cached":17},
					"completion_tokens_details":{
						"text_tokens":` + strconv.FormatInt(tt.textTokens, 10) + `,
						"reasoning_tokens":` + strconv.FormatInt(tt.reasoningTokens, 10) + `,
						"vendor_detail":"preserve-me"
					}
				}
			}`)

			normalized, changed := NormalizeCompletionUsage(body, tt.modelID)
			if !changed {
				t.Fatal("expected usage to be normalized")
			}

			var response struct {
				ID              string `json:"id"`
				Choices         []any  `json:"choices"`
				VendorExtension struct {
					Trace string `json:"trace"`
				} `json:"vendor_extension"`
				Usage struct {
					PromptTokens  int64 `json:"prompt_tokens"`
					ProviderUsage struct {
						Cached int `json:"cached"`
					} `json:"provider_usage"`
					CompletionTokensDetails struct {
						TextTokens      int64  `json:"text_tokens"`
						ReasoningTokens int64  `json:"reasoning_tokens"`
						VendorDetail    string `json:"vendor_detail"`
					} `json:"completion_tokens_details"`
				} `json:"usage"`
			}
			if err := json.Unmarshal(normalized, &response); err != nil {
				t.Fatalf("normalized response is invalid JSON: %v", err)
			}
			if response.Usage.CompletionTokensDetails.TextTokens != tt.wantTextTokens {
				t.Fatalf("text_tokens = %d, want %d", response.Usage.CompletionTokensDetails.TextTokens, tt.wantTextTokens)
			}
			if response.Usage.CompletionTokensDetails.ReasoningTokens != tt.reasoningTokens {
				t.Fatalf("reasoning_tokens changed: got %d, want %d", response.Usage.CompletionTokensDetails.ReasoningTokens, tt.reasoningTokens)
			}
			if response.ID != "chatcmpl-test" || len(response.Choices) != 1 || response.VendorExtension.Trace != "preserve-me" {
				t.Fatal("top-level unknown fields were not preserved")
			}
			if response.Usage.PromptTokens != 1370 || response.Usage.ProviderUsage.Cached != 17 || response.Usage.CompletionTokensDetails.VendorDetail != "preserve-me" {
				t.Fatal("unknown usage fields were not preserved")
			}
		})
	}
}

func TestNormalizeCompletionUsage_AmbiguousOrUnrelatedPayloadsStayUnchanged(t *testing.T) {
	validOverlap := `{"usage":{"completion_tokens":100,"completion_tokens_details":{"text_tokens":100,"reasoning_tokens":75}}}`
	tests := []struct {
		name    string
		modelID string
		body    string
	}{
		{name: "other model", modelID: "qwen3.6-32b", body: validOverlap},
		{name: "similar model prefix", modelID: "qwen3.6-35b-a3bother", body: validOverlap},
		{name: "missing text", modelID: qwenReasoningUsageModel, body: `{"usage":{"completion_tokens":100,"completion_tokens_details":{"reasoning_tokens":75}}}`},
		{name: "zero text", modelID: qwenReasoningUsageModel, body: `{"usage":{"completion_tokens":100,"completion_tokens_details":{"text_tokens":0,"reasoning_tokens":75}}}`},
		{name: "zero completion", modelID: qwenReasoningUsageModel, body: `{"usage":{"completion_tokens":0,"completion_tokens_details":{"text_tokens":10,"reasoning_tokens":5}}}`},
		{name: "zero reasoning", modelID: qwenReasoningUsageModel, body: `{"usage":{"completion_tokens":100,"completion_tokens_details":{"text_tokens":100,"reasoning_tokens":0}}}`},
		{name: "negative completion", modelID: qwenReasoningUsageModel, body: `{"usage":{"completion_tokens":-100,"completion_tokens_details":{"text_tokens":100,"reasoning_tokens":75}}}`},
		{name: "negative text", modelID: qwenReasoningUsageModel, body: `{"usage":{"completion_tokens":100,"completion_tokens_details":{"text_tokens":-1,"reasoning_tokens":75}}}`},
		{name: "negative reasoning", modelID: qwenReasoningUsageModel, body: `{"usage":{"completion_tokens":100,"completion_tokens_details":{"text_tokens":100,"reasoning_tokens":-1}}}`},
		{name: "reasoning exceeds completion", modelID: qwenReasoningUsageModel, body: `{"usage":{"completion_tokens":100,"completion_tokens_details":{"text_tokens":100,"reasoning_tokens":101}}}`},
		{name: "valid partition", modelID: qwenReasoningUsageModel, body: `{"usage":{"completion_tokens":100,"completion_tokens_details":{"text_tokens":25,"reasoning_tokens":75}}}`},
		{name: "smaller text partition", modelID: qwenReasoningUsageModel, body: `{"usage":{"completion_tokens":100,"completion_tokens_details":{"text_tokens":20,"reasoning_tokens":75}}}`},
		{name: "fractional token value", modelID: qwenReasoningUsageModel, body: `{"usage":{"completion_tokens":100,"completion_tokens_details":{"text_tokens":100.0,"reasoning_tokens":75}}}`},
		{name: "missing usage", modelID: qwenReasoningUsageModel, body: `{"choices":[]}`},
		{name: "invalid JSON", modelID: qwenReasoningUsageModel, body: `{"usage":`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			original := []byte(tt.body)
			normalized, changed := NormalizeCompletionUsage(original, tt.modelID)
			if changed {
				t.Fatal("ambiguous or unrelated response must not be normalized")
			}
			if !bytes.Equal(normalized, original) {
				t.Fatalf("response changed:\n got %s\nwant %s", normalized, original)
			}
		})
	}
}

func TestQwenUsageNormalizingReadCloser_PreservesSSEFramingAcrossFragmentedReads(t *testing.T) {
	payload := []byte(`{"id":"chunk","usage":{"completion_tokens":4774,"completion_tokens_details":{"text_tokens":4774,"reasoning_tokens":4417}},"extension":{"keep":true}}`)
	normalizedPayload, changed := normalizeQwenCompletionUsage(payload, qwenReasoningUsageModel)
	if !changed {
		t.Fatal("test payload must require normalization")
	}

	input := strings.Join([]string{
		": keep-alive\r\n",
		"event: message\n",
		"data: [DONE]\r\n",
		"\n",
		"data: " + string(payload) + "\n",
		"data:" + string(payload) + "   \r\n",
		"data: not-json\n",
		"data: " + string(payload),
	}, "")
	want := strings.Join([]string{
		": keep-alive\r\n",
		"event: message\n",
		"data: [DONE]\r\n",
		"\n",
		"data: " + string(normalizedPayload) + "\n",
		"data:" + string(normalizedPayload) + "   \r\n",
		"data: not-json\n",
		"data: " + string(normalizedPayload),
	}, "")

	source := &oneByteReadCloser{reader: strings.NewReader(input)}
	wrapped, ok := NewUsageNormalizingReadCloser(source, "proxy/qwen3.6-35b-a3b-20260415")
	if !ok {
		t.Fatal("expected Qwen stream to be wrapped")
	}
	got, err := readWithSmallBuffer(wrapped, 7)
	if err != nil {
		t.Fatalf("read normalized SSE: %v", err)
	}
	if string(got) != want {
		t.Fatalf("normalized SSE mismatch:\n got %q\nwant %q", got, want)
	}
	if err := wrapped.Close(); err != nil {
		t.Fatalf("close wrapper: %v", err)
	}
	if !source.closed {
		t.Fatal("closing wrapper did not close source")
	}
}

func TestQwenUsageNormalizingReadCloser_OtherModelStaysByteForByteUnchanged(t *testing.T) {
	input := "data: {\"usage\":{\"completion_tokens\":100,\"completion_tokens_details\":{\"text_tokens\":100,\"reasoning_tokens\":75}}}\r\n\r\n"
	source := &oneByteReadCloser{reader: strings.NewReader(input)}
	wrapped, ok := NewUsageNormalizingReadCloser(source, "gpt-5-mini")
	if ok {
		t.Fatal("unrelated model must not be wrapped")
	}
	if wrapped != source {
		t.Fatal("unrelated model must not be wrapped or line-buffered")
	}
	got, err := io.ReadAll(wrapped)
	if err != nil {
		t.Fatalf("read SSE: %v", err)
	}
	if string(got) != input {
		t.Fatalf("unrelated model response changed:\n got %q\nwant %q", got, input)
	}
}

func TestQwenUsageNormalizingReadCloser_OversizedLinePassesThroughAndResynchronizes(t *testing.T) {
	payload := []byte(`{"usage":{"completion_tokens":4774,"completion_tokens_details":{"text_tokens":4774,"reasoning_tokens":4417}}}`)
	normalizedPayload, changed := normalizeQwenCompletionUsage(payload, qwenReasoningUsageModel)
	if !changed {
		t.Fatal("test payload must require normalization")
	}

	oversizedLine := "data: " + strings.Repeat("x", maxQwenSSELineBytes+8192) + "\n"
	input := oversizedLine + "data: " + string(payload) + "\n"
	wrapped := newQwenUsageNormalizingReadCloser(
		io.NopCloser(strings.NewReader(input)),
		qwenReasoningUsageModel,
	).(*qwenUsageNormalizingReadCloser)

	first := make([]byte, 17)
	n, err := wrapped.Read(first)
	if err != nil {
		t.Fatalf("first read: %v", err)
	}
	if len(wrapped.line) > maxQwenSSELineBytes {
		t.Fatalf("line buffer grew to %d bytes", len(wrapped.line))
	}
	if len(wrapped.pending) > maxQwenSSELineBytes+wrapped.reader.Size() {
		t.Fatalf("pending buffer grew to %d bytes", len(wrapped.pending))
	}

	rest, err := io.ReadAll(wrapped)
	if err != nil {
		t.Fatalf("read remainder: %v", err)
	}
	got := append(append([]byte(nil), first[:n]...), rest...)
	want := oversizedLine + "data: " + string(normalizedPayload) + "\n"
	if string(got) != want {
		t.Fatal("oversized line was not preserved or following line was not normalized")
	}
}

func TestQwenUsageNormalizingReadCloser_OversizedUnterminatedLinePassesThrough(t *testing.T) {
	input := "data: " + strings.Repeat("x", maxQwenSSELineBytes+8192)
	wrapped := newQwenUsageNormalizingReadCloser(
		io.NopCloser(strings.NewReader(input)),
		qwenReasoningUsageModel,
	)

	got, err := io.ReadAll(wrapped)
	if err != nil {
		t.Fatalf("read oversized unterminated line: %v", err)
	}
	if string(got) != input {
		t.Fatal("oversized unterminated line changed")
	}
}

func TestNewUsageNormalizingReadCloser_NilSource(t *testing.T) {
	got, ok := NewUsageNormalizingReadCloser(nil, qwenReasoningUsageModel)
	if ok {
		t.Fatal("nil source must not report wrapper")
	}
	if got != nil {
		t.Fatalf("nil source returned %T, want nil", got)
	}
}

type oneByteReadCloser struct {
	reader *strings.Reader
	closed bool
}

func (r *oneByteReadCloser) Read(p []byte) (int, error) {
	if len(p) > 1 {
		p = p[:1]
	}
	return r.reader.Read(p)
}

func (r *oneByteReadCloser) Close() error {
	r.closed = true
	return nil
}

func readWithSmallBuffer(r io.Reader, size int) ([]byte, error) {
	var output bytes.Buffer
	buffer := make([]byte, size)
	for {
		n, err := r.Read(buffer)
		if n > 0 {
			_, _ = output.Write(buffer[:n])
		}
		if err == io.EOF {
			return output.Bytes(), nil
		}
		if err != nil {
			return nil, err
		}
	}
}
