package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	anthropicconv "github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	promanutils "github.com/mixaill76/auto_ai_router/internal/converter/proman/utils"
	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/mixaill76/auto_ai_router/internal/proxy/modelutils"
)

// streamChunkWriteTimeout is the per-chunk write deadline for streaming responses.
// If no data flows for this duration, the connection is terminated.
const streamChunkWriteTimeout = 60 * time.Second

// streamDrainTimeout caps how long we wait for the upstream to finish after a
// client disconnects. The provider charges for the full generation regardless,
// so we keep reading to capture the real usage chunk.
const streamDrainTimeout = 60 * time.Second

// streamTTFTDetectionLimit caps how many bytes streamToClient accumulates while
// looking for the first real content delta (for CompletionStartTime/ttft_ms).
// Streams that never produce a detectable content delta within this many bytes
// (error-only or keep-alive-only streams) stop being scanned; CompletionStartTime
// stays zero, matching existing "never reached" semantics.
const streamTTFTDetectionLimit = 64 * 1024

// streamInitialCommitBufferLimit bounds how much of a successful upstream SSE
// response we hold before committing downstream headers. This small preflight
// window lets an immediate terminal SSE error (for example response.failed with
// rate_limit_exceeded) become a real HTTP error status instead of an HTTP 200
// stream whose first body event is failure.
const streamInitialCommitBufferLimit = 64 * 1024

// ttftScanState incrementally scans SSE lines for the first detectable content
// delta, byte-for-byte equivalent to calling extractCompletionDeltaText on the
// whole buffer accumulated so far after every Read — but each byte is scanned
// exactly once (as part of the line it completes) instead of once per Read,
// which made the old approach O(n²) in the bytes accumulated before a match.
// A line can only ever contain a complete JSON payload once its trailing
// newline has arrived, so deferring extraction until then loses no matches
// extractCompletionDeltaText could have found on a still-partial line — for
// real SSE framing (every current caller feeds streamToClient real "data:
// ...\n\n" events). This is NOT a universal guarantee: a hypothetical stream
// whose final content line never terminates with '\n' would leave that line
// stuck in s.pending forever, and TTFT would never be stamped for it. Narrow
// in practice (only the CompletionStartTime metric is affected — no
// correctness/billing impact — and no current caller produces such a
// stream), but worth knowing before reusing this pattern somewhere newline
// termination isn't guaranteed.
type ttftScanState struct {
	pending []byte
	total   int
}

// observe feeds a newly read chunk into the scan and reports whether a
// detectable content delta was found. total tracks cumulative bytes observed
// (not just the still-unprocessed tail in pending) so streamTTFTDetectionLimit
// keeps its original "give up after this many bytes with no match" meaning.
func (s *ttftScanState) observe(chunk []byte) bool {
	s.total += len(chunk)
	s.pending = append(s.pending, chunk...)
	for {
		idx := bytes.IndexByte(s.pending, '\n')
		if idx < 0 {
			break
		}
		line := s.pending[:idx]
		s.pending = s.pending[idx+1:]
		if extractCompletionDeltaText(line) != "" {
			return true
		}
	}
	return false
}

// streamFlushCoalesceWindow throttles how often streamToClient calls Flush().
// Profiled under concurrent streaming load: a Flush() (and the write syscall
// behind it) on every single upstream Read is a dominant CPU cost at high
// concurrency. 10ms is well under human-perceptible inter-token latency, so
// coalescing flushes within that window costs nothing on the "feels live"
// front while cutting syscall volume whenever reads arrive in a burst (e.g.
// providers/mocks that batch several SSE frames per write). The very first
// flush always fires immediately (TTFT accuracy), and every write is
// guaranteed to be flushed by the time streamToClient returns (see
// flushPending in that function — no buffered bytes are ever left behind at
// stream end). That guarantee does NOT extend to a live mid-stream pause: a
// chunk written just before the reader blocks on the next upstream Read can
// still sit unflushed for the whole pause, since nothing outside the read
// loop can force a flush — see TestStreamToClient_FlushesTailOnMidStreamPause.
const streamFlushCoalesceWindow = 10 * time.Millisecond

var streamBufPool = sync.Pool{
	New: func() any {
		buf := make([]byte, 8192)
		return &buf
	},
}

// StreamUsageInfo holds extracted usage information from streaming responses.
// It provides a unified structure for token counts across all providers.
// Not all fields will be populated; some providers don't report certain metrics.
type StreamUsageInfo struct {
	PromptTokens             int // May be 0 if not provided in streaming response
	CompletionTokens         int
	CachedTokens             int // Tokens from cached prompt content (prompt_caching feature)
	CachedAudioTokens        int // Cached prompt tokens whose modality is audio
	AudioInputTokens         int // Audio tokens in the request
	AudioOutputTokens        int // Audio tokens in the response
	ImageTokens              int // Input image/video tokens (if reported)
	OutputImageTokens        int // Output image/video tokens (if reported)
	ReasoningTokens          int // Reasoning/thoughts tokens (output)
	AcceptedPredictionTokens int
	RejectedPredictionTokens int
	CachedOutputTokens       int
	CacheCreationTokens      int // Tokens created for cache (billed at different rate)
	CacheCreation5mTokens    int
	CacheCreation1hTokens    int
	CacheReadTokens          int // Tokens read from cache (billed at cheaper rate)
	WebSearchRequests        int // Confirmed built-in web search executions
}

// StreamUsageExtractor provides a provider-agnostic interface for extracting
// usage information from streaming response chunks.
// Each provider may use different JSON structures and field names,
// so implementations handle provider-specific parsing.
type StreamUsageExtractor interface {
	// ExtractUsage attempts to extract usage information from the given chunk.
	// Returns nil if the chunk doesn't contain usage information.
	// Errors are logged internally; the function never returns error.
	ExtractUsage(chunk []byte) *StreamUsageInfo
}

// openAIStreamUsageExtractor implements StreamUsageExtractor for OpenAI format
type openAIStreamUsageExtractor struct {
	audioInputAlreadyExcludesCachedAudio bool
}

func (o *openAIStreamUsageExtractor) ExtractUsage(chunk []byte) *StreamUsageInfo {
	// Supports two OpenAI streaming formats:
	//
	// 1. Chat Completions API:
	//    {"choices":[...],"usage":{"prompt_tokens":100,"completion_tokens":50,...}}
	//
	// 2. Responses API (GPT-5, /v1/responses):
	//    {"type":"response.completed","response":{"usage":{"input_tokens":100,"output_tokens":50,...}}}
	//    Usage fields use input_tokens/output_tokens and output_tokens_details instead of
	//    prompt_tokens/completion_tokens and completion_tokens_details.

	payloads := splitSSEPayloads(chunk, nil)
	for i := len(payloads) - 1; i >= 0; i-- {
		if info := o.extractChatCompletionUsage(payloads[i]); info != nil {
			return info
		}
		if info := o.extractResponsesAPIUsage(payloads[i]); info != nil {
			return info
		}
	}

	return nil
}

// extractChatCompletionUsage parses usage from Chat Completions streaming format.
// Format: {"usage":{"prompt_tokens":N,"completion_tokens":N,...}}
func (o *openAIStreamUsageExtractor) extractChatCompletionUsage(payload []byte) *StreamUsageInfo {
	var data struct {
		Usage *struct {
			PromptTokens        *int `json:"prompt_tokens"`
			CompletionTokens    *int `json:"completion_tokens"`
			PromptTokensDetails struct {
				CachedTokens              int `json:"cached_tokens,omitempty"`
				CachedAudioTokens         int `json:"cached_audio_tokens,omitempty"`
				CacheCreationTokens       int `json:"cache_creation_tokens,omitempty"`
				CacheWriteTokens          int `json:"cache_write_tokens,omitempty"`
				CacheCreationTokenDetails struct {
					Ephemeral5mInputTokens int `json:"ephemeral_5m_input_tokens,omitempty"`
					Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens,omitempty"`
				} `json:"cache_creation_token_details,omitempty"`
				AudioTokens int `json:"audio_tokens,omitempty"`
				ImageTokens int `json:"image_tokens,omitempty"`
			} `json:"prompt_tokens_details,omitempty"`
			CompletionTokensDetails struct {
				AcceptedPredictionTokens int `json:"accepted_prediction_tokens,omitempty"`
				AudioTokens              int `json:"audio_tokens,omitempty"`
				CachedTokens             int `json:"cached_tokens,omitempty"`
				ImageTokens              int `json:"image_tokens,omitempty"`
				ReasoningTokens          int `json:"reasoning_tokens,omitempty"`
				RejectedPredictionTokens int `json:"rejected_prediction_tokens,omitempty"`
			} `json:"completion_tokens_details,omitempty"`
			ServerToolUse struct {
				WebSearchRequests int `json:"web_search_requests,omitempty"`
			} `json:"server_tool_use,omitempty"`
			WebSearchRequests int `json:"web_search_requests,omitempty"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(payload, &data); err != nil {
		return nil
	}

	if data.Usage == nil || (data.Usage.PromptTokens == nil && data.Usage.CompletionTokens == nil) {
		return nil
	}

	cacheCreationTokens := data.Usage.PromptTokensDetails.CacheCreationTokens
	if cacheCreationTokens == 0 {
		cacheCreationTokens = data.Usage.PromptTokensDetails.CacheWriteTokens
	}
	if cacheCreationTokens == 0 {
		cacheCreationTokens = data.Usage.PromptTokensDetails.CacheCreationTokenDetails.Ephemeral5mInputTokens +
			data.Usage.PromptTokensDetails.CacheCreationTokenDetails.Ephemeral1hInputTokens
	}
	cachedTokens, cachedAudioTokens := converterutil.NormalizeCachedAudioBreakdown(
		data.Usage.PromptTokensDetails.CachedTokens,
		data.Usage.PromptTokensDetails.CachedAudioTokens,
	)

	return &StreamUsageInfo{
		PromptTokens:          intValue(data.Usage.PromptTokens),
		CompletionTokens:      intValue(data.Usage.CompletionTokens),
		CachedTokens:          cachedTokens,
		CachedAudioTokens:     cachedAudioTokens,
		CacheCreationTokens:   cacheCreationTokens,
		CacheCreation5mTokens: data.Usage.PromptTokensDetails.CacheCreationTokenDetails.Ephemeral5mInputTokens,
		CacheCreation1hTokens: data.Usage.PromptTokensDetails.CacheCreationTokenDetails.Ephemeral1hInputTokens,
		AudioInputTokens: normalizeStreamAudioInput(
			data.Usage.PromptTokensDetails.AudioTokens,
			cachedTokens,
			cachedAudioTokens,
			o.audioInputAlreadyExcludesCachedAudio,
		),
		AudioOutputTokens:        data.Usage.CompletionTokensDetails.AudioTokens,
		ImageTokens:              data.Usage.PromptTokensDetails.ImageTokens,
		OutputImageTokens:        data.Usage.CompletionTokensDetails.ImageTokens,
		ReasoningTokens:          data.Usage.CompletionTokensDetails.ReasoningTokens,
		AcceptedPredictionTokens: data.Usage.CompletionTokensDetails.AcceptedPredictionTokens,
		RejectedPredictionTokens: data.Usage.CompletionTokensDetails.RejectedPredictionTokens,
		CachedOutputTokens:       data.Usage.CompletionTokensDetails.CachedTokens,
		WebSearchRequests: webSearchRequestsFromUsage(
			data.Usage.ServerToolUse.WebSearchRequests,
			data.Usage.WebSearchRequests,
		),
	}
}

// extractResponsesAPIUsage parses usage from Responses API streaming format.
// The usage can appear at two levels:
//   - Top-level: {"usage":{"input_tokens":N,"output_tokens":N,...}}
//   - Nested in response.completed: {"type":"response.completed","response":{"usage":{...}}}
func (o *openAIStreamUsageExtractor) extractResponsesAPIUsage(payload []byte) *StreamUsageInfo {
	var data struct {
		// Top-level usage (some Responses API events)
		Usage  *responsesAPIUsage        `json:"usage,omitempty"`
		Output []streamingResponseOutput `json:"output,omitempty"`
		// Nested usage in response.completed event
		Response struct {
			Usage  *responsesAPIUsage        `json:"usage,omitempty"`
			Output []streamingResponseOutput `json:"output,omitempty"`
		} `json:"response,omitempty"`
	}

	if err := json.Unmarshal(payload, &data); err != nil {
		return nil
	}

	// Prefer nested response.usage (response.completed event), fall back to top-level
	usage := data.Response.Usage
	if usage == nil {
		usage = data.Usage
	}
	if usage == nil || (usage.InputTokens == nil && usage.OutputTokens == nil) {
		return nil
	}

	cacheCreationTokens := usage.InputTokensDetails.CacheCreationTokens
	if cacheCreationTokens == 0 {
		cacheCreationTokens = usage.InputTokensDetails.CacheWriteTokens
	}
	if cacheCreationTokens == 0 {
		cacheCreationTokens = usage.InputTokensDetails.CacheCreationTokenDetails.Ephemeral5mInputTokens +
			usage.InputTokensDetails.CacheCreationTokenDetails.Ephemeral1hInputTokens
	}
	cachedTokens, cachedAudioTokens := converterutil.NormalizeCachedAudioBreakdown(
		usage.InputTokensDetails.CachedTokens,
		usage.InputTokensDetails.CachedAudioTokens,
	)
	webSearchRequests := webSearchRequestsFromUsage(
		usage.ServerToolUse.WebSearchRequests,
		usage.WebSearchRequests,
	)
	if webSearchRequests == 0 {
		webSearchRequests = countCompletedStreamingWebSearchItems(data.Response.Output)
	}
	if webSearchRequests == 0 {
		webSearchRequests = countCompletedStreamingWebSearchItems(data.Output)
	}

	return &StreamUsageInfo{
		PromptTokens:          intValue(usage.InputTokens),
		CompletionTokens:      intValue(usage.OutputTokens),
		CachedTokens:          cachedTokens,
		CachedAudioTokens:     cachedAudioTokens,
		CacheCreationTokens:   cacheCreationTokens,
		CacheCreation5mTokens: usage.InputTokensDetails.CacheCreationTokenDetails.Ephemeral5mInputTokens,
		CacheCreation1hTokens: usage.InputTokensDetails.CacheCreationTokenDetails.Ephemeral1hInputTokens,
		AudioInputTokens: normalizeStreamAudioInput(
			usage.InputTokensDetails.AudioTokens,
			cachedTokens,
			cachedAudioTokens,
			o.audioInputAlreadyExcludesCachedAudio,
		),
		AudioOutputTokens:        usage.OutputTokensDetails.AudioTokens,
		ImageTokens:              usage.InputTokensDetails.ImageTokens,
		OutputImageTokens:        usage.OutputTokensDetails.ImageTokens,
		ReasoningTokens:          usage.OutputTokensDetails.ReasoningTokens,
		AcceptedPredictionTokens: usage.OutputTokensDetails.AcceptedPredictionTokens,
		RejectedPredictionTokens: usage.OutputTokensDetails.RejectedPredictionTokens,
		CachedOutputTokens:       usage.OutputTokensDetails.CachedTokens,
		WebSearchRequests:        webSearchRequests,
	}
}

type streamingResponseOutput struct {
	Type   string `json:"type"`
	Status string `json:"status,omitempty"`
}

func countCompletedStreamingWebSearchItems(output []streamingResponseOutput) int {
	count := 0
	for _, item := range output {
		if item.Type == "web_search_call" && (item.Status == "" || item.Status == "completed") {
			count++
		}
	}
	return count
}

// responsesAPIUsage represents the usage object in OpenAI Responses API format.
type responsesAPIUsage struct {
	InputTokens        *int `json:"input_tokens"`
	OutputTokens       *int `json:"output_tokens"`
	InputTokensDetails struct {
		CachedTokens              int `json:"cached_tokens,omitempty"`
		CachedAudioTokens         int `json:"cached_audio_tokens,omitempty"`
		CacheCreationTokens       int `json:"cache_creation_tokens,omitempty"`
		CacheWriteTokens          int `json:"cache_write_tokens,omitempty"`
		CacheCreationTokenDetails struct {
			Ephemeral5mInputTokens int `json:"ephemeral_5m_input_tokens,omitempty"`
			Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens,omitempty"`
		} `json:"cache_creation_token_details,omitempty"`
		AudioTokens int `json:"audio_tokens,omitempty"`
		ImageTokens int `json:"image_tokens,omitempty"`
	} `json:"input_tokens_details,omitempty"`
	OutputTokensDetails struct {
		AcceptedPredictionTokens int `json:"accepted_prediction_tokens,omitempty"`
		AudioTokens              int `json:"audio_tokens,omitempty"`
		CachedTokens             int `json:"cached_tokens,omitempty"`
		ImageTokens              int `json:"image_tokens,omitempty"`
		ReasoningTokens          int `json:"reasoning_tokens,omitempty"`
		RejectedPredictionTokens int `json:"rejected_prediction_tokens,omitempty"`
	} `json:"output_tokens_details,omitempty"`
	ServerToolUse struct {
		WebSearchRequests int `json:"web_search_requests,omitempty"`
	} `json:"server_tool_use,omitempty"`
	WebSearchRequests int `json:"web_search_requests,omitempty"`
}

// anthropicStreamUsageExtractor implements StreamUsageExtractor for Anthropic format
type anthropicStreamUsageExtractor struct{}

func (a *anthropicStreamUsageExtractor) ExtractUsage(chunk []byte) *StreamUsageInfo {
	// Anthropic streaming format (message_delta event):
	// {"type":"message_delta","delta":{...},"usage":{"input_tokens":100,"output_tokens":50}}
	// Usage appears in the message_delta event at the end of streaming

	var data struct {
		Usage *struct {
			InputTokens              *int `json:"input_tokens"`
			OutputTokens             *int `json:"output_tokens"`
			CacheCreationInputTokens int  `json:"cache_creation_input_tokens,omitempty"`
			CacheReadInputTokens     int  `json:"cache_read_input_tokens,omitempty"`
			CacheCreation            struct {
				Ephemeral5mInputTokens int `json:"ephemeral_5m_input_tokens,omitempty"`
				Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens,omitempty"`
			} `json:"cache_creation,omitempty"`
			ServerToolUse struct {
				WebSearchRequests int `json:"web_search_requests,omitempty"`
			} `json:"server_tool_use,omitempty"`
		} `json:"usage"`
	}

	payloads := splitSSEPayloads(chunk, nil)
	for i := len(payloads) - 1; i >= 0; i-- {
		if err := json.Unmarshal(payloads[i], &data); err != nil {
			continue
		}

		if data.Usage == nil || (data.Usage.InputTokens == nil && data.Usage.OutputTokens == nil) {
			continue
		}

		cacheCreationTokens := data.Usage.CacheCreationInputTokens
		if cacheCreationTokens == 0 {
			cacheCreationTokens = data.Usage.CacheCreation.Ephemeral5mInputTokens + data.Usage.CacheCreation.Ephemeral1hInputTokens
		}
		return &StreamUsageInfo{
			PromptTokens:          intValue(data.Usage.InputTokens),
			CompletionTokens:      intValue(data.Usage.OutputTokens),
			CacheCreationTokens:   cacheCreationTokens,
			CacheCreation5mTokens: data.Usage.CacheCreation.Ephemeral5mInputTokens,
			CacheCreation1hTokens: data.Usage.CacheCreation.Ephemeral1hInputTokens,
			CacheReadTokens:       data.Usage.CacheReadInputTokens,
			WebSearchRequests:     data.Usage.ServerToolUse.WebSearchRequests,
			// Anthropic separates cache_creation (cached prompt tokens)
			// For logging purposes, we combine under CachedTokens
			CachedTokens: data.Usage.CacheReadInputTokens,
		}
	}

	return nil
}

func intValue(value *int) int {
	if value == nil {
		return 0
	}
	return *value
}

func webSearchRequestsFromUsage(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

// sseDataPrefix and sseDoneSentinel are shared byte-literal needles for
// splitSSEPayloads — declared once to avoid re-allocating a []byte from a
// string literal on every call.
var (
	sseDataPrefix   = []byte("data:")
	sseDoneSentinel = []byte("[DONE]")
	// sseUsageNeedle is the byte-level prefilter needle (Reviewer #2 / plan
	// item D): stream_options.include_usage forces usage to arrive only in
	// the final chunk, so a chunk that doesn't even contain this substring
	// cannot possibly carry usage/total_tokens — skip the unmarshal attempt
	// entirely rather than paying for it on every content-only chunk.
	sseUsageNeedle = []byte(`"usage"`)
	// sseWebSearchCallNeedle and sseAnnotationsNeedle catch the two other
	// shapes converter.ExtractTokenUsageWithOptions can derive a non-nil,
	// billable TokenUsage from *without* any "usage" key present at all:
	// a completed output[]/response.output[] item of type "web_search_call",
	// or a choices[].message.annotations[] entry with type "url_citation".
	// Found by review: a chunk carrying only one of these (e.g. a
	// Chat-Completions-shaped full "message" frame relayed by an upstream
	// AIR/proxy-type credential, separate from the frame that carries usage)
	// was being silently skipped by the "usage"-only prefilter below,
	// dropping billed WebSearchRequests. Checking chunk-wide (not per-payload)
	// keeps this a single cheap scan like the usage check.
	sseWebSearchCallNeedle = []byte(`"web_search_call"`)
	sseAnnotationsNeedle   = []byte(`"annotations"`)
	// sseErrorNeedle and sseResponseFailedNeedle prefilter
	// extractStreamErrorEvent's json.Unmarshal (called from
	// proxyStreamErrorCapture.Observe/Finalize on every assembled SSE frame):
	// it only ever matches a frame containing an "error" field key, or an
	// eventType of "error"/"response.error" (all covered by the unquoted
	// substring "error"), or "response.failed" (which doesn't contain
	// "error", hence the second needle). Checked against the
	// fully-assembled frame (post nextSSEFrameEnd), not a possibly-split raw
	// read, so there's no risk of a false negative from a match straddling
	// two reads.
	sseErrorNeedle          = []byte("error")
	sseResponseFailedNeedle = []byte("response.failed")
)

// frameMayCarryStreamError reports whether frame could possibly make
// extractStreamErrorEvent return non-empty. See sseErrorNeedle/
// sseResponseFailedNeedle above.
func frameMayCarryStreamError(frame []byte) bool {
	return bytes.Contains(frame, sseErrorNeedle) || bytes.Contains(frame, sseResponseFailedNeedle)
}

// chunkMayCarryTokenUsage reports whether chunk could possibly yield a
// non-nil result from extractTokenUsageFromPayloads/
// converter.ExtractTokenUsageWithOptions — i.e. it contains "usage", or
// either of the web-search-only signal shapes those functions also read
// (see sseWebSearchCallNeedle/sseAnnotationsNeedle above). Used to gate the
// per-chunk usage-extraction attempt (plan item D) without dropping the
// web-search billing signal for chunks that carry it without any "usage" key.
func chunkMayCarryTokenUsage(chunk []byte) bool {
	return bytes.Contains(chunk, sseUsageNeedle) ||
		bytes.Contains(chunk, sseWebSearchCallNeedle) ||
		bytes.Contains(chunk, sseAnnotationsNeedle)
}

// splitSSEPayloads splits an SSE-formatted chunk into its "data:" JSON payload
// sub-slices using a single bytes.IndexByte('\n') scan — no strings.Split, no
// per-payload []byte(...) copy. If the chunk contains no "data:" marker at
// all, the whole (trimmed) chunk is treated as one plain-JSON payload (fast
// path for non-SSE callers, e.g. a bare response.completed JSON event).
//
// dst is reused via the dst[:0] pattern: pass the same backing slice back in
// on the next call (typically a field alongside a stream's other per-request
// state, e.g. completionTokenAccumulator.payloadBuf or a local var captured
// by an onChunk closure) to avoid allocating a new [][]byte per chunk. Pass
// nil for one-off, non-hot-path calls.
//
// ⚠️ Every returned payload is a sub-slice of chunk itself — zero copies.
// The result is only valid until chunk's backing array is next overwritten
// (e.g. the next Read() into a buffer pulled from streamBufPool). Every
// current caller is synchronous within the Write/onChunk call that produced
// chunk and finishes before returning, so this is safe today — but don't
// stash a returned payload past that call. rememberLastStreamDataChunk is the
// one exception that must survive past the current chunk's lifetime, and it
// copies explicitly (see its own doc comment).
func splitSSEPayloads(chunk []byte, dst [][]byte) [][]byte {
	dst = dst[:0]
	trimmed := bytes.TrimSpace(chunk)
	if len(trimmed) == 0 {
		return dst
	}

	// Fast path: non-SSE plain JSON.
	if !bytes.Contains(trimmed, sseDataPrefix) {
		return append(dst, trimmed)
	}

	rest := trimmed
	for {
		var line []byte
		if idx := bytes.IndexByte(rest, '\n'); idx >= 0 {
			line = rest[:idx]
			rest = rest[idx+1:]
		} else {
			line = rest
			rest = nil
		}
		line = bytes.TrimSpace(line)
		if bytes.HasPrefix(line, sseDataPrefix) {
			payload := bytes.TrimSpace(line[len(sseDataPrefix):])
			if len(payload) > 0 && !bytes.Equal(payload, sseDoneSentinel) {
				dst = append(dst, payload)
			}
		}
		if rest == nil {
			break
		}
	}

	return dst
}

// getStreamUsageExtractor returns the appropriate usage extractor for a provider.
// This factory method ensures all providers use the correct parsing logic.
// If the provider is unknown, defaults to OpenAI extractor (most compatible fallback).
func getStreamUsageExtractor(providerName string) StreamUsageExtractor {
	switch strings.ToLower(strings.TrimSpace(providerName)) {
	case "openai":
		return &openAIStreamUsageExtractor{}
	case "anthropic":
		// Anthropic streaming goes through handleTransformedStreaming which converts
		// chunks to OpenAI format, so we use OpenAI extractor for the transformed response
		return &openAIStreamUsageExtractor{audioInputAlreadyExcludesCachedAudio: true}
	case "vertex ai":
		// Vertex AI transforms to OpenAI format during streaming,
		// so we use OpenAI extractor for the transformed response
		return &openAIStreamUsageExtractor{audioInputAlreadyExcludesCachedAudio: true}
	case "bedrock":
		// Bedrock transforms to OpenAI format during streaming (via Anthropic converter),
		// so we use OpenAI extractor for the transformed response
		return &openAIStreamUsageExtractor{audioInputAlreadyExcludesCachedAudio: true}
	case "native_responses":
		// Native Responses converters emit billing-normalized usage.
		return &openAIStreamUsageExtractor{audioInputAlreadyExcludesCachedAudio: true}
	default:
		// Fallback: try OpenAI format first (most common)
		return &openAIStreamUsageExtractor{}
	}
}

func normalizeStreamAudioInput(audioTokens, cachedTokens, cachedAudioTokens int, alreadyExcludesCachedAudio bool) int {
	return converterutil.NormalizeAudioInputTokens(audioTokens, cachedTokens, cachedAudioTokens, !alreadyExcludesCachedAudio)
}

func IsStreamingResponse(resp *http.Response) bool {
	contentType := resp.Header.Get("Content-Type")
	return strings.Contains(contentType, "text/event-stream") ||
		strings.Contains(contentType, "application/stream+json") ||
		strings.Contains(contentType, "application/vnd.amazon.eventstream")
}

type streamTransformer func(io.Reader, string, io.Writer) error

func (p *Proxy) handleProviderStreaming(
	w http.ResponseWriter,
	resp *http.Response,
	cred *config.CredentialConfig,
	realModelID, displayModelID string,
	logCtx *RequestLogContext,
) error {
	publicModel := clientVisibleResponseModel(logCtx, displayModelID)
	switch cred.Type {
	case config.ProviderTypeVertexAI, config.ProviderTypeGemini:
		return p.handleVertexStreaming(w, resp, cred.Name, realModelID, publicModel, logCtx)
	case config.ProviderTypeAnthropic:
		return p.handleAnthropicCompatibleStreaming(w, resp, cred, realModelID, publicModel, cred.Type, "Anthropic", logCtx)
	case config.ProviderTypeCometAPI:
		return p.handleAnthropicCompatibleStreaming(w, resp, cred, realModelID, publicModel, cred.Type, "Comet API", logCtx)
	case config.ProviderTypeProMan:
		return p.handleAnthropicCompatibleStreaming(w, resp, cred, realModelID, publicModel, cred.Type, "ProMan", logCtx)
	case config.ProviderTypeBedrock:
		return p.handleBedrockStreaming(w, resp, cred.Name, realModelID, publicModel, logCtx)
	default:
		if resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
			resp.Body, _ = modelutils.NewUsageNormalizingReadCloser(resp.Body, publicModel)
		}
		return p.handleStreamingWithTokens(w, resp, cred.Name, displayModelID, logCtx)
	}
}

func (p *Proxy) handleVertexStreaming(w http.ResponseWriter, resp *http.Response, credName, modelID, displayModelID string, logCtx *RequestLogContext) error {
	conv := converter.New(config.ProviderTypeVertexAI, converter.RequestMode{ModelID: modelID, DisplayModelID: displayModelID, IsStreaming: true})
	transformer := func(r io.Reader, id string, w io.Writer) error {
		return conv.StreamTo(r, w)
	}
	return p.handleTransformedStreaming(w, resp, credName, modelID, "Vertex AI", transformer, logCtx)
}

func (p *Proxy) handleAnthropicCompatibleStreaming(w http.ResponseWriter, resp *http.Response, cred *config.CredentialConfig, modelID, displayModelID string, providerType config.ProviderType, providerLabel string, logCtx *RequestLogContext) error {
	conv := converter.New(providerType, converter.RequestMode{ModelID: modelID, DisplayModelID: displayModelID, IsStreaming: true})
	transformer := func(r io.Reader, id string, w io.Writer) error {
		// Sanitize unconditionally, matching handleTransformedStreaming's output-side pass:
		// the converter can fold a raw upstream error event's message into a plain
		// delta.content string (see anthropic.TransformAnthropicStreamToOpenAI's "error"
		// case), which the output-side sanitizer never inspects since it only masks
		// structural error.message keys. Sanitizing here, before conversion, is the only
		// place that still sees the message under its original "error.message" key.
		r = promanutils.NewSanitizingSSEReader(r, displayModelID)
		return conv.StreamTo(r, w)
	}
	return p.handleTransformedStreaming(w, resp, cred.Name, modelID, providerLabel, transformer, logCtx)
}

func (p *Proxy) handleBedrockStreaming(w http.ResponseWriter, resp *http.Response, credName, modelID, displayModelID string, logCtx *RequestLogContext) error {
	conv := converter.New(config.ProviderTypeBedrock, converter.RequestMode{ModelID: modelID, DisplayModelID: displayModelID, IsStreaming: true})
	transformer := func(r io.Reader, id string, w io.Writer) error {
		return conv.StreamTo(r, w)
	}
	return p.handleTransformedStreaming(w, resp, credName, modelID, "Bedrock", transformer, logCtx)
}

type tokenCapturingWriter struct {
	writer     io.Writer
	tokens     *int
	completion *completionTokenAccumulator
	logger     *slog.Logger
	// payloadBuf is reused across Write calls (dst[:0] pattern) — one split
	// per chunk, shared between the token count below, the completion
	// accumulator, and onChunk, instead of each doing its own copy+split.
	// Safe because a tokenCapturingWriter is created fresh per stream, never
	// shared across concurrent streams.
	payloadBuf [][]byte
	// onChunk is invoked for each chunk with the payloads already split out
	// of it (see splitSSEPayloads — valid only for the duration of this call)
	// and whether the chunk contains the literal `"usage"` substring, so
	// callers can skip their own usage-unmarshal attempt without re-scanning
	// the chunk. Optional; used to capture the last chunk for usage
	// extraction.
	onChunk func(chunk []byte, payloads [][]byte, hasUsage bool)
}

func (tcw *tokenCapturingWriter) Write(p []byte) (n int, err error) {
	// One parse per chunk: split once, reuse the sub-slices for the token
	// count below, the completion accumulator, and onChunk.
	tcw.payloadBuf = splitSSEPayloads(p, tcw.payloadBuf)

	// Byte-level prefilter (plan item D): stream_options.include_usage means
	// total_tokens can only ever appear in a chunk containing "usage" — but
	// converter.ExtractTokenUsageWithOptions (via the onChunk callback below)
	// can also derive a billable WebSearchRequests from "web_search_call"/
	// "annotations" shapes with no "usage" key at all, so the gate has to
	// cover those too (see chunkMayCarryTokenUsage's doc comment).
	hasUsage := chunkMayCarryTokenUsage(p)
	if hasUsage {
		// Extract tokens from the data being written.
		// Use assignment (not +=) because Vertex/Gemini include cumulative total_tokens in every
		// streaming chunk. Accumulating across chunks would multiply the real count by the number
		// of chunks (e.g. 50 chunks × 1000 tokens = 50 000 instead of 1 000).
		// OpenAI only emits total_tokens in the final usage chunk, so assignment is equivalent there.
		if tokens := extractTokensFromPayloads(tcw.payloadBuf); tokens > 0 {
			*tcw.tokens = tokens
		}
	}

	if tcw.completion != nil {
		tcw.completion.AddPayloads(tcw.payloadBuf)
	}

	// Invoke callback if provided (used to capture last chunk for usage extraction)
	if tcw.onChunk != nil {
		tcw.onChunk(p, tcw.payloadBuf, hasUsage)
	}

	return tcw.writer.Write(p)
}

// rememberLastStreamDataChunk stores each chunk, keeping only the last one
// that contains actual data and isn't [DONE]. *dst's backing array is reused
// across calls (append((*dst)[:0], ...)) instead of allocating fresh on every
// chunk — this is the one place that must copy rather than sub-slice chunk,
// since it's the only state here that survives past the current chunk's
// lifetime (see splitSSEPayloads' doc comment on that sharp edge).
func rememberLastStreamDataChunk(dst *[]byte, chunk []byte) {
	trimmed := bytes.TrimSpace(chunk)
	if len(trimmed) == 0 || bytes.Equal(trimmed, sseDoneSentinel) || bytes.Equal(trimmed, []byte("data: [DONE]")) {
		return
	}
	*dst = append((*dst)[:0], chunk...)
}

func (p *Proxy) handleTransformedStreaming(
	w http.ResponseWriter,
	resp *http.Response,
	credName string,
	modelID string,
	providerName string,
	transformFunc streamTransformer,
	logCtx *RequestLogContext,
) error {
	p.logger.DebugContext(respCtx(resp), "Starting streaming response", "provider", providerName, "credential", credName)

	pr, pw := io.Pipe()
	defer func() {
		_ = pr.Close()
	}()
	var totalTokens int
	completion := p.newCompletionTokenAccumulator(modelID)

	// Capture last chunk for usage extraction (Solution 3: Hybrid approach)
	var lastChunk []byte
	detectProviderStreamError := resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices
	rawProviderStreamError := &proxyStreamErrorCapture{}
	outputStreamError := &proxyStreamErrorCapture{}
	providerReader := io.Reader(resp.Body)
	if detectProviderStreamError {
		// Some converters normalize a provider error event into an apparently
		// successful output chunk. Observe both sides so the original terminal
		// event still controls accounting and session semantics.
		providerReader = io.TeeReader(providerReader, proxyStreamErrorObserver{capture: rawProviderStreamError})
	}

	// WaitGroup ensures the transform goroutine completes before we read
	// lastChunk and totalTokens, preventing a data race.
	var wg sync.WaitGroup
	clientReader := normalizeSuccessfulResponseModelStream(
		promanutils.NewSanitizingSSEReader(pr, clientVisibleResponseModel(logCtx, modelID)),
		resp.StatusCode,
		logCtx,
		modelID,
	)
	wg.Add(1)
	chunkCount := 0
	go func() {
		defer wg.Done()
		err := transformFunc(providerReader, modelID, &tokenCapturingWriter{
			writer:     pw,
			tokens:     &totalTokens,
			completion: completion,
			logger:     p.logger,
			onChunk: func(chunk []byte, payloads [][]byte, hasUsage bool) {
				chunkCount++

				if detectProviderStreamError {
					outputStreamError.Observe(chunk)
				}
				if logCtx != nil && hasUsage {
					if usage := extractTokenUsageFromPayloads(payloads, converter.TokenUsageExtractionOptions{}); usage != nil {
						if logCtx.TokenUsage == nil {
							logCtx.TokenUsage = &converter.TokenUsage{}
						}
						// Merge rather than replace — see MergeNonZero's doc
						// comment: a web-search-only chunk's WebSearchRequests
						// must not be clobbered back to zero by a later
						// chunk's usage read that doesn't carry it.
						logCtx.TokenUsage.MergeNonZero(usage)
					}
				}

				// Store each chunk, keeping only the last one that contains actual data and isn't [DONE]
				rememberLastStreamDataChunk(&lastChunk, chunk)
			},
		})
		if err != nil {
			p.logger.ErrorContext(respCtx(resp), "Transform goroutine error",
				"provider", providerName, "error", err, "chunks_written", chunkCount)
			_ = pw.CloseWithError(fmt.Errorf("%s transform: %w", providerName, err))
		} else {
			p.logger.DebugContext(respCtx(resp), "Transform goroutine completed OK",
				"provider", providerName, "chunks_written", chunkCount, "total_tokens", totalTokens)
			_ = pw.Close()
		}
	}()

	if err := p.streamToClient(respCtx(resp), w, clientReader, credName, metricModelID(modelID, logCtx), endpointFromLogContext(logCtx), resp.StatusCode, nil, func() { _ = pr.Close() }, logCtx); err != nil {
		p.logStreamHandlerError(respCtx(resp), "streamToClient error in handleTransformedStreaming", err,
			"credential", credName, "provider", providerName, "model", modelID)
		wg.Wait()
		// Stream aborted before the final usage chunk — log with whatever tokens we have.
		// Fall back to local token counting when totalTokens is still 0.
		estimated := totalTokens
		if estimated == 0 {
			estimated = completion.TokenCount()
		}
		if detectProviderStreamError {
			err = resolveCapturedProviderStreamError(logCtx, resp.StatusCode, err, rawProviderStreamError, outputStreamError)
		}
		markStreamFailure(logCtx, err)
		p.finalizeStreamingLog(logCtx, estimated, lastChunk, providerName, resp.StatusCode, true)
		return err
	}
	wg.Wait()

	var streamErr error
	if detectProviderStreamError {
		streamErr = resolveCapturedProviderStreamError(logCtx, resp.StatusCode, nil, rawProviderStreamError, outputStreamError)
		if streamErr != nil {
			markStreamFailure(logCtx, streamErr)
		}
	}

	p.logger.DebugContext(respCtx(resp), "handleTransformedStreaming completed",
		"provider", providerName, "total_tokens", totalTokens,
		"chunks_written", chunkCount, "last_chunk_len", len(lastChunk))

	// When no usage chunk arrived (provider disconnected without sending one),
	// fall back to local token counting from accumulated delta text.
	logTokens := totalTokens
	if logTokens == 0 {
		logTokens = completion.TokenCount()
		p.logger.Debug("No usage chunk received; counted completion tokens from delta text",
			"tokens", logTokens, "provider", providerName, "model", modelID)
	}

	if logTokens > 0 {
		p.rateLimiter.ConsumeTokens(credName, logTokens)
		if modelID != "" {
			p.rateLimiter.ConsumeModelTokens(credName, modelID, logTokens)
		}
		p.logger.DebugContext(respCtx(resp), "Streaming token usage recorded", "credential", credName, "model", modelID, "tokens", totalTokens)
	}

	p.finalizeStreamingLog(logCtx, logTokens, lastChunk, providerName, resp.StatusCode, true)

	if streamErr == nil {
		p.logger.DebugContext(respCtx(resp), "Streaming response completed", "provider", providerName, "credential", credName)
	}
	return streamErr
}

func (p *Proxy) handleStreamingWithTokens(w http.ResponseWriter, resp *http.Response, credName, modelID string, logCtx *RequestLogContext) error {
	p.logger.DebugContext(respCtx(resp), "Starting streaming response with token tracking (passthrough)",
		"credential", credName, "model", modelID,
		"content_type", resp.Header.Get("Content-Type"))

	var totalTokens int
	completion := p.newCompletionTokenAccumulator(modelID)
	chunkCount := 0

	// Capture last chunk for usage extraction (Solution 3: Hybrid approach)
	var lastChunk []byte
	detectProviderStreamError := resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices
	providerStreamError := &proxyStreamErrorCapture{}
	providerReader := io.Reader(resp.Body)
	if detectProviderStreamError {
		providerReader = io.TeeReader(providerReader, proxyStreamErrorObserver{capture: providerStreamError})
	}

	var payloadBuf [][]byte
	onChunk := func(chunk []byte) {
		chunkCount++

		// One split per chunk (plan item C), reused below for both usage and
		// total-tokens extraction; the completion accumulator keeps its own
		// reused buffer (see completionTokenAccumulator.payloadBuf) since it
		// also has to run on hasUsage==false chunks.
		payloadBuf = splitSSEPayloads(chunk, payloadBuf)
		if hasUsage := chunkMayCarryTokenUsage(chunk); hasUsage {
			if logCtx != nil {
				if usage := extractTokenUsageFromPayloads(payloadBuf, converter.TokenUsageExtractionOptions{AudioInputIncludesCachedAudio: true}); usage != nil {
					if logCtx.TokenUsage == nil {
						logCtx.TokenUsage = &converter.TokenUsage{}
					}
					// Merge rather than replace — see MergeNonZero's doc
					// comment: a web-search-only chunk's WebSearchRequests
					// must not be clobbered back to zero by a later chunk's
					// usage read that doesn't carry it.
					logCtx.TokenUsage.MergeNonZero(usage)
				}
			}

			if tokens := extractTokensFromPayloads(payloadBuf); tokens > 0 {
				totalTokens += tokens
			}
		}
		completion.AddPayloads(payloadBuf)

		// Don't let a bare [DONE] sentinel or empty chunks overwrite a lastChunk that carries usage data.
		rememberLastStreamDataChunk(&lastChunk, chunk)
	}

	clientReader := normalizeSuccessfulResponseModelStream(
		promanutils.NewSanitizingSSEReader(providerReader, clientVisibleResponseModel(logCtx, modelID)),
		resp.StatusCode,
		logCtx,
		modelID,
	)
	if err := p.streamToClient(respCtx(resp), w, clientReader, credName, metricModelID(modelID, logCtx), endpointFromLogContext(logCtx), resp.StatusCode, onChunk, nil, logCtx); err != nil {
		p.logStreamHandlerError(respCtx(resp), "streamToClient error in handleStreamingWithTokens", err,
			"credential", credName, "model", modelID, "chunks_received", chunkCount)
		if p.drainUpstreamOnAbort {
			// Keep reading upstream to capture the real usage chunk.
			// The provider charges for the full generation regardless of client disconnect.
			drainCtx, cancel := context.WithTimeout(context.Background(), streamDrainTimeout)
			defer cancel()
			p.drainUpstream(drainCtx, clientReader, onChunk, credName)
		}
		estimated := totalTokens
		if estimated == 0 {
			estimated = completion.TokenCount()
		}
		if detectProviderStreamError {
			err = resolveCapturedProviderStreamError(logCtx, resp.StatusCode, err, providerStreamError)
		}
		markStreamFailure(logCtx, err)
		p.finalizeStreamingLog(logCtx, estimated, lastChunk, "openai", resp.StatusCode, false)
		return err
	}

	var streamErr error
	if detectProviderStreamError {
		streamErr = resolveCapturedProviderStreamError(logCtx, resp.StatusCode, nil, providerStreamError)
		if streamErr != nil {
			markStreamFailure(logCtx, streamErr)
		}
	}

	p.logger.DebugContext(respCtx(resp), "handleStreamingWithTokens completed",
		"credential", credName, "model", modelID,
		"chunks_received", chunkCount, "total_tokens", totalTokens,
		"last_chunk_len", len(lastChunk))

	// When no usage chunk arrived (provider disconnected without sending one),
	// fall back to local token counting from accumulated delta text.
	logTokens := totalTokens
	if logTokens == 0 {
		logTokens = completion.TokenCount()
		p.logger.Debug("No usage chunk received; counted completion tokens from delta text",
			"tokens", logTokens, "credential", credName, "model", modelID)
	}

	if logTokens > 0 {
		p.rateLimiter.ConsumeTokens(credName, logTokens)
		if modelID != "" {
			p.rateLimiter.ConsumeModelTokens(credName, modelID, logTokens)
		}
		p.logger.DebugContext(respCtx(resp), "Streaming token usage recorded", "credential", credName, "model", modelID, "tokens", totalTokens)
	}

	p.finalizeStreamingLog(logCtx, logTokens, lastChunk, "openai", resp.StatusCode, false)

	if streamErr == nil {
		p.logger.DebugContext(respCtx(resp), "Streaming response completed", "credential", credName)
	}
	return streamErr
}

// finalizeStreamingLog extracts usage info from the last streaming chunk and logs spend to LiteLLM DB.
func (p *Proxy) finalizeStreamingLog(logCtx *RequestLogContext, totalTokens int, lastChunk []byte, providerName string, statusCode int, audioInputAlreadyExcludesCachedAudio bool) {
	if logCtx == nil || logCtx.Logged {
		return
	}
	if logCtx.StreamOutcome == "" {
		logCtx.StreamOutcome = "completed"
	}

	if logCtx.TokenUsage == nil {
		logCtx.TokenUsage = &converter.TokenUsage{}
	}
	fallbackCompletion := totalTokens

	providerUsage := false
	// Tracked separately from providerUsage: some providers (e.g. CometAPI's Anthropic-
	// compatible streaming) send a syntactically valid usage object whose message_start
	// carries a placeholder usage:{input_tokens:0,output_tokens:0} while output tokens
	// land correctly later via message_delta. Gating the local tiktoken fallback on the
	// coarse providerUsage flag silently accepts that placeholder zero as truth and never
	// estimates prompt tokens for the request. Gate per-field instead, so a provider that
	// reports some fields but not others still gets a fallback for the missing one.
	providerReportedPromptTokens := false
	providerReportedCompletionTokens := false
	if len(lastChunk) > 0 {
		extractor := getStreamUsageExtractor(providerName)
		if audioInputAlreadyExcludesCachedAudio {
			extractor = &openAIStreamUsageExtractor{audioInputAlreadyExcludesCachedAudio: true}
		}
		if usageInfo := extractor.ExtractUsage(lastChunk); usageInfo != nil {
			providerUsage = true
			if usageInfo.PromptTokens > 0 {
				logCtx.TokenUsage.PromptTokens = usageInfo.PromptTokens
				providerReportedPromptTokens = true
			}
			if usageInfo.CompletionTokens > 0 {
				logCtx.TokenUsage.CompletionTokens = usageInfo.CompletionTokens
				providerReportedCompletionTokens = true
			}

			if usageInfo.CachedTokens > 0 {
				logCtx.TokenUsage.CachedInputTokens = usageInfo.CachedTokens
			}
			if usageInfo.CachedAudioTokens > 0 {
				logCtx.TokenUsage.CachedAudioInputTokens = usageInfo.CachedAudioTokens
			}
			if usageInfo.AudioInputTokens > 0 {
				logCtx.TokenUsage.AudioInputTokens = usageInfo.AudioInputTokens
			}
			if usageInfo.AudioOutputTokens > 0 {
				logCtx.TokenUsage.AudioOutputTokens = usageInfo.AudioOutputTokens
			}
			if usageInfo.ImageTokens > 0 {
				logCtx.TokenUsage.ImageTokens = usageInfo.ImageTokens
			}
			if usageInfo.OutputImageTokens > 0 {
				logCtx.TokenUsage.OutputImageTokens = usageInfo.OutputImageTokens
			}
			if usageInfo.ReasoningTokens > 0 {
				logCtx.TokenUsage.ReasoningTokens = usageInfo.ReasoningTokens
			}
			if usageInfo.AcceptedPredictionTokens > 0 {
				logCtx.TokenUsage.AcceptedPredictionTokens = usageInfo.AcceptedPredictionTokens
			}
			if usageInfo.RejectedPredictionTokens > 0 {
				logCtx.TokenUsage.RejectedPredictionTokens = usageInfo.RejectedPredictionTokens
			}
			if usageInfo.CachedOutputTokens > 0 {
				logCtx.TokenUsage.CachedOutputTokens = usageInfo.CachedOutputTokens
			}

			if usageInfo.CacheCreationTokens > 0 {
				logCtx.TokenUsage.CacheCreationTokens = usageInfo.CacheCreationTokens
			}
			if usageInfo.CacheCreation5mTokens > 0 {
				logCtx.TokenUsage.CacheCreation5mTokens = usageInfo.CacheCreation5mTokens
			}
			if usageInfo.CacheCreation1hTokens > 0 {
				logCtx.TokenUsage.CacheCreation1hTokens = usageInfo.CacheCreation1hTokens
			}
			if usageInfo.WebSearchRequests > 0 {
				logCtx.TokenUsage.WebSearchRequests = usageInfo.WebSearchRequests
			}

			p.logger.DebugContext(logCtx.Context(), "Extracted usage from streaming response",
				"provider", providerName,
				"prompt_tokens", usageInfo.PromptTokens,
				"completion_tokens", usageInfo.CompletionTokens,
				"cached_tokens", usageInfo.CachedTokens,
				"audio_input_tokens", usageInfo.AudioInputTokens,
				"audio_output_tokens", usageInfo.AudioOutputTokens,
				"image_tokens", usageInfo.ImageTokens,
				"output_image_tokens", usageInfo.OutputImageTokens,
				"reasoning_tokens", usageInfo.ReasoningTokens,
				"web_search_requests", usageInfo.WebSearchRequests,
			)
		}
	}

	// explicitZeroUsage: the provider sent a usage object and reported BOTH fields as zero
	// together (e.g. a filtered/cancelled request that legitimately produced nothing) — a
	// coherent signal, trusted as-is, same as before this fix.
	//
	// A zero PromptTokens alongside a genuine non-zero CompletionTokens is distrusted and
	// falls back to the local estimate for prompt tokens specifically: this is internally
	// inconsistent (a real completion cannot be produced from zero input) and matches the
	// CometAPI bug, where message_start's placeholder usage:{input_tokens:0,output_tokens:0}
	// never gets a real input count even though output lands correctly later via
	// message_delta.
	//
	// The reverse — a genuine non-zero PromptTokens alongside a zero CompletionTokens — is
	// NOT distrusted: it's a normal, legitimate outcome (e.g. output filtered/stopped
	// immediately), and no known provider bug produces a stuck-zero completion count while
	// input is reported correctly. Completion only falls back to the local estimate when no
	// provider usage object arrived at all.
	explicitZeroUsage := providerUsage && !providerReportedPromptTokens && !providerReportedCompletionTokens
	if !providerReportedPromptTokens && !explicitZeroUsage && logCtx.TokenUsage.PromptTokens == 0 {
		logCtx.TokenUsage.PromptTokens = logCtx.promptTokensEstimate()
	}
	if !providerUsage && logCtx.TokenUsage.CompletionTokens == 0 {
		logCtx.TokenUsage.CompletionTokens = fallbackCompletion
	}
	logCtx.TokenUsage.Normalize()
	if providerUsage {
		logCtx.UsageSource = "provider"
	} else if logCtx.UsageSource == "" {
		logCtx.UsageSource = "estimated"
	}

	if logCtx.StreamOutcome != "stream_error" || logCtx.HTTPStatus < http.StatusBadRequest {
		logCtx.HTTPStatus = statusCode
	}
	if logCtx.StreamOutcome == "client_aborted" || logCtx.StreamOutcome == "stream_error" {
		logCtx.Status = "failure"
		if logCtx.StreamOutcome == "client_aborted" {
			logCtx.HTTPStatus = 499
		}
		if logCtx.ErrorMsg == "" {
			logCtx.ErrorMsg = logCtx.StreamOutcome
		}
	} else if statusCode >= 400 {
		logCtx.Status = "failure"
		if logCtx.ErrorMsg == "" {
			logCtx.ErrorMsg = extractErrorMessage(lastChunk)
		}
	} else if streamErr := extractStreamErrorEvent(lastChunk); streamErr != "" {
		// Provider returned HTTP 2xx but sent an error event inside the stream
		// (e.g. `data: {"error":...}`, `event: error`, response.failed). Without
		// this check such requests are logged as success and never hit ERROR.
		logCtx.Status = "failure"
		logCtx.ErrorMsg = streamErr
		p.logUpstreamError(logCtx.Context(), "Provider sent error event in stream", statusCode,
			logCtx.Credential, logCtx.ModelID, []byte(streamErr),
			"request_id", logCtx.RequestID)
	} else {
		logCtx.Status = "success"
		if logCtx.Credential != nil {
			p.metrics.RecordTokenUsage(logCtx.Credential.Name, logCtx.ModelID,
				logCtx.TokenUsage.PromptTokens, logCtx.TokenUsage.CompletionTokens,
				logCtx.TokenUsage.ReasoningTokens, logCtx.TokenUsage.CachedInputTokens)
		}
	}
	// TTFT: only meaningful if a real content delta actually arrived (streamToClient
	// stamps CompletionStartTime the moment it does) — a stream that errored before
	// any content, or wasn't streaming at all, has nothing to report here.
	if logCtx.Credential != nil && !logCtx.CompletionStartTime.IsZero() {
		p.metrics.RecordTTFT(logCtx.Credential.Name, endpointFromLogContext(logCtx),
			logCtx.CompletionStartTime.Sub(logCtx.StartTime))
	}
	if err := p.logSpendToLiteLLMDB(logCtx); err != nil {
		p.logger.WarnContext(logCtx.Context(), "Failed to queue streaming spend log",
			"error", err,
			"request_id", logCtx.RequestID,
		)
	}
	logCtx.Logged = true
}

func markStreamFailure(logCtx *RequestLogContext, err error) {
	if logCtx == nil || err == nil {
		return
	}
	logCtx.Status = "failure"
	if isClientDisconnectError(err) {
		logCtx.StreamOutcome = "client_aborted"
		logCtx.HTTPStatus = 499
		if logCtx.ErrorMsg == "" {
			logCtx.ErrorMsg = logCtx.StreamOutcome
		}
		return
	}
	logCtx.StreamOutcome = "stream_error"
	if logCtx.ErrorMsg == "" {
		logCtx.ErrorMsg = logCtx.StreamOutcome
	}
}

// extractStreamErrorEvent scans an SSE chunk (or plain JSON payload) for an
// error event sent by the provider inside an HTTP 2xx stream. It returns the
// error payload as a string, or "" when the chunk carries no error.
// Recognized shapes:
//   - data: {"error": {...}}              (OpenAI-style mid-stream error)
//   - data: {"type":"error", ...}         (Anthropic `event: error`)
//   - data: {"type":"response.failed"...} (Responses API failed event)
//   - bare JSON object with the same shapes (non-SSE final chunk)
func extractStreamErrorEvent(chunk []byte) string {
	if len(chunk) == 0 {
		return ""
	}

	checkPayload := func(payload string) string {
		var evt struct {
			Type  string          `json:"type"`
			Error json.RawMessage `json:"error"`
		}
		if err := json.Unmarshal([]byte(payload), &evt); err != nil {
			return ""
		}
		hasError := len(evt.Error) > 0 && string(evt.Error) != "null"
		if isStreamErrorEvent(evt.Type, hasError) {
			return payload
		}
		return ""
	}

	found := ""
	sawDataLine := false
	for _, line := range strings.Split(string(chunk), "\n") {
		line = strings.TrimRight(line, "\r")
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		sawDataLine = true
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		if errPayload := checkPayload(payload); errPayload != "" {
			found = errPayload
		}
	}
	if found != "" {
		return found
	}

	// Non-SSE chunk: try the whole payload as a single JSON object.
	if !sawDataLine {
		return checkPayload(strings.TrimSpace(string(chunk)))
	}
	return ""
}

func isStreamErrorEvent(eventType string, hasError bool) bool {
	return hasError || eventType == "error" || eventType == "response.error" || eventType == "response.failed"
}

// drainUpstream reads from body until EOF, ctx expiry, or an error, calling
// onChunk for every read. Invoked after a client disconnects so that the usage
// chunk emitted by the provider at the end of the stream is captured for
// accurate spend logging — the provider charges for the full generation
// regardless of whether the client waited for it.
// Must be called from the same goroutine that already owns body (no concurrent reads).
func (p *Proxy) drainUpstream(ctx context.Context, body io.Reader, onChunk func([]byte), credName string) {
	buf := make([]byte, 4096)
	drained := 0
	for {
		select {
		case <-ctx.Done():
			p.logger.Debug("Upstream drain timeout", "credential", credName, "bytes_drained", drained)
			return
		default:
		}
		n, err := body.Read(buf)
		if n > 0 {
			drained += n
			if onChunk != nil {
				onChunk(buf[:n])
			}
		}
		if err != nil {
			p.logger.Debug("Upstream drain complete", "credential", credName, "bytes_drained", drained)
			return
		}
	}
}

type streamInitialCommitGate struct {
	pending []byte
	capture proxyStreamErrorCapture
}

func (g *streamInitialCommitGate) Observe(chunk []byte) (payload string, ready bool) {
	if len(chunk) == 0 {
		return "", false
	}
	g.pending = append(g.pending, chunk...)
	if payload := g.capture.Observe(chunk); payload != "" {
		return payload, false
	}
	if len(g.pending) > streamInitialCommitBufferLimit {
		return "", true
	}

	trimmed := bytes.TrimSpace(g.pending)
	if len(trimmed) == 0 {
		return "", false
	}
	if looksLikeEventStream(trimmed) {
		return "", nextSSEFrameEnd(g.pending) >= 0
	}
	if frameMayCarryStreamError(trimmed) {
		if payload := extractStreamErrorEvent(trimmed); payload != "" {
			return payload, false
		}
	}
	return "", true
}

func (g *streamInitialCommitGate) FinalizeTerminalError() string {
	if payload := g.capture.Finalize(); payload != "" {
		return payload
	}
	trimmed := bytes.TrimSpace(g.pending)
	if frameMayCarryStreamError(trimmed) {
		return extractStreamErrorEvent(trimmed)
	}
	return ""
}

func (g *streamInitialCommitGate) Release() []byte {
	pending := g.pending
	g.pending = nil
	return pending
}

func (p *Proxy) streamToClient(
	ctx context.Context,
	w http.ResponseWriter,
	reader io.Reader,
	credName string,
	modelID string,
	endpoint string,
	statusCode int,
	onChunk func([]byte),
	onWriteErr func(),
	logCtx *RequestLogContext,
) error {
	_, ok := w.(http.Flusher)
	if !ok {
		p.logger.ErrorContext(ctx, "Streaming not supported", "credential", credName)
		WriteErrorInternal(w, "Streaming Not Supported")
		return fmt.Errorf("streaming not supported")
	}
	controller := http.NewResponseController(w)

	buf := streamBufPool.Get().(*[]byte)
	defer streamBufPool.Put(buf)

	// TTFT detection: separate from the bytes forwarded to the client via
	// onChunk (forwarding must stay byte-for-byte immediate for latency). A real
	// content/tool/reasoning delta can straddle multiple Read calls, or be
	// preceded by content-free SSE events (ping, role-only delta) that arrive in
	// their own Read — so lines accumulate across reads via ttftScan until a
	// complete line yields actual content, rather than stamping
	// CompletionStartTime on the first non-empty Read regardless of content.
	var ttftScan ttftScanState
	ttftPending := logCtx != nil && logCtx.CompletionStartTime.IsZero()
	var lastFlush time.Time
	// flushPending tracks whether the most recent write was buffered rather
	// than flushed (coalescing skip below). It is guaranteed to be flushed
	// before this function returns — see the unconditional check on every
	// loop-exit path — fixing the bug where a terminal Read() returning (0,
	// io.EOF) never reached the "if n > 0" block at all, leaving any
	// buffered tail stuck for the rest of the connection's lifetime.
	// KNOWN LIMITATION (not fixed by this flag): a chunk written just before
	// a live upstream pause can still sit unflushed for the pause's whole
	// duration, since this loop is blocked synchronously in reader.Read()
	// and nothing forces a flush until the next Read() returns. Closing that
	// gap needs a timer-driven background flush with its own
	// synchronization against concurrent Write() — deliberately out of scope
	// here; see TestStreamToClient_FlushesTailOnMidStreamPause.
	flushPending := false
	var responseIDScanner clientResponseIDScanner
	if statusCode == 0 {
		statusCode = http.StatusOK
	}
	preflightEnabled := statusCode >= http.StatusOK && statusCode < http.StatusMultipleChoices
	committed := !preflightEnabled
	if !preflightEnabled {
		w.WriteHeader(statusCode)
	}
	var initialCommit streamInitialCommitGate

	writeStreamChunk := func(chunk []byte) error {
		if len(chunk) == 0 {
			return nil
		}
		// Set write deadline before each write — keeps active streams alive,
		// terminates if client stops reading for streamChunkWriteTimeout.
		_ = controller.SetWriteDeadline(time.Now().Add(streamChunkWriteTimeout))
		if _, writeErr := w.Write(chunk); writeErr != nil {
			if isClientDisconnectError(writeErr) {
				p.logger.DebugContext(ctx, "Client disconnected during streaming", "error", writeErr, "credential", credName)
				p.recordAbortedRequest(credName, endpoint, modelID)
			} else {
				p.logger.ErrorContext(ctx, "Failed to write streaming chunk", "error", writeErr, "credential", credName)
			}
			if onWriteErr != nil {
				onWriteErr()
			}
			return writeErr
		}
		// Coalesce flushes within streamFlushCoalesceWindow: always flush the
		// first chunk (TTFT); in between, skip a flush if the previous one was
		// very recent — bursty reads (e.g. an upstream that batches several SSE
		// frames per write) then cost one syscall, not N. Whatever gets skipped
		// here is guaranteed to be flushed before this function returns (see
		// the unconditional flushPending check below) — do NOT rely on "err !=
		// nil" to force a flush from inside this block: a terminal Read()
		// commonly returns (0, io.EOF), which never enters "if n > 0" at all.
		now := time.Now()
		if lastFlush.IsZero() || now.Sub(lastFlush) >= streamFlushCoalesceWindow {
			if flushErr := p.flushStreaming(ctx, controller, credName); flushErr != nil {
				if isClientDisconnectError(flushErr) {
					p.logger.DebugContext(ctx, "Client disconnected during streaming flush", "error", flushErr, "credential", credName)
					p.recordAbortedRequest(credName, endpoint, modelID)
				}
				if onWriteErr != nil {
					onWriteErr()
				}
				return flushErr
			}
			lastFlush = now
			flushPending = false
		} else {
			flushPending = true
		}
		return nil
	}

	writeEarlyStreamError := func(payload string) error {
		statusCode := statusCodeFromProviderStreamError(payload)
		markProxyProviderStreamError(logCtx, statusCode, payload)
		if logCtx != nil {
			logCtx.StreamOutcome = "stream_error"
		}
		writeProviderStreamErrorBeforeCommit(w, statusCode)
		if onWriteErr != nil {
			onWriteErr()
		}
		return proxyProviderStreamError{payload: payload, statusCode: statusCode, beforeCommit: true}
	}

	for {
		n, err := reader.Read(*buf)
		if n > 0 {
			chunk := (*buf)[:n]
			responseIDScanner.observe(logCtx, chunk)
			if ttftPending {
				if ttftScan.observe(chunk) {
					logCtx.CompletionStartTime = time.Now()
					ttftPending = false
				} else if ttftScan.total > streamTTFTDetectionLimit {
					ttftPending = false
				}
				if !ttftPending {
					ttftScan = ttftScanState{}
				}
			}
			if onChunk != nil {
				onChunk(chunk)
			}

			if !committed {
				payload, ready := initialCommit.Observe(chunk)
				if payload != "" {
					return writeEarlyStreamError(payload)
				}
				if !ready {
					if err == nil {
						continue
					}
				} else {
					chunk = initialCommit.Release()
					w.WriteHeader(statusCode)
					committed = true
				}
			}
			if committed {
				if writeErr := writeStreamChunk(chunk); writeErr != nil {
					return writeErr
				}
			}
		}
		if err != nil {
			if !committed {
				if payload := initialCommit.FinalizeTerminalError(); payload != "" {
					return writeEarlyStreamError(payload)
				}
				w.WriteHeader(statusCode)
				if writeErr := writeStreamChunk(initialCommit.Release()); writeErr != nil {
					return writeErr
				}
			}
			if flushPending {
				// No need to clear flushPending here — every path below
				// returns immediately, so nothing would ever read it again.
				if flushErr := p.flushStreaming(ctx, controller, credName); flushErr != nil {
					if isClientDisconnectError(flushErr) {
						p.logger.DebugContext(ctx, "Client disconnected during streaming flush", "error", flushErr, "credential", credName)
						p.recordAbortedRequest(credName, endpoint, modelID)
					}
					if onWriteErr != nil {
						onWriteErr()
					}
					return flushErr
				}
			}
			if err != io.EOF {
				if sink, ok := w.(responseStreamErrorSink); ok {
					sink.setStreamError(err)
				}
				p.logStreamHandlerError(ctx, "Streaming read error", err, "credential", credName)
				return err
			}
			return nil
		}
	}
}

func (p *Proxy) flushStreaming(ctx context.Context, controller *http.ResponseController, credName string) (retErr error) {
	defer func() {
		if r := recover(); r != nil {
			p.logger.ErrorContext(ctx, "Flusher panic", "panic", r, "credential", credName)
			retErr = fmt.Errorf("flusher panic: %v", r)
		}
	}()
	if err := controller.Flush(); err != nil {
		switch {
		case isClientDisconnectError(err):
			// Disconnects are logged once by the caller; don't double-log here.
		case errors.Is(err, http.ErrNotSupported):
			p.logger.ErrorContext(ctx, "Streaming not supported", "credential", credName)
		default:
			p.logger.ErrorContext(ctx, "Flusher error", "error", err, "credential", credName)
		}
		return err
	}
	return nil
}

// handleResponsesAPIStreaming handles streaming for Responses API requests.
// It first converts the provider stream to OpenAI Chat Completions SSE format,
// then converts that to Responses API SSE format.
// The optional onComplete callback is invoked with the fully-built Response
// once the stream finishes (used for store persistence).
// meta (may be nil) is used to echo store/previous_response_id/metadata fields
// back in every emitted SSE response object.
func (p *Proxy) handleResponsesAPIStreaming(
	w http.ResponseWriter,
	resp *http.Response,
	cred *config.CredentialConfig,
	modelID string,
	logCtx *RequestLogContext,
	onComplete func(*responses.Response),
	meta ...*responses.ResponsesMetadata,
) error {
	p.logger.DebugContext(respCtx(resp), "Starting Responses API streaming", "credential", cred.Name, "provider", cred.Type)
	publicModel := clientVisibleResponseModel(logCtx, modelID)

	// For providers that need transformation (Vertex, Anthropic, Bedrock),
	// first transform to OpenAI Chat Completions SSE, then to Responses API SSE.
	// For OpenAI (passthrough), the stream is already in Chat Completions SSE format.

	// Use realModelID for provider dispatch (e.g. isAnthropicBedrockModel check),
	// keep modelID (alias) as DisplayModelID so response chunks show the alias.
	converterModelID := modelID
	if logCtx != nil && logCtx.RealModelID != "" {
		converterModelID = logCtx.RealModelID
	}
	conv := converter.New(cred.Type, converter.RequestMode{
		ModelID:        converterModelID,
		DisplayModelID: publicModel,
		IsStreaming:    true,
	})

	// Extract optional metadata for field echoing (nil is fine — echoing is skipped)
	var reqMeta *responses.ResponsesMetadata
	if len(meta) > 0 {
		reqMeta = meta[0]
	}

	// Create a wrapper transformer that chains:
	// Provider SSE -> Chat Completions SSE -> Responses API SSE
	transformer := func(r io.Reader, id string, w io.Writer) error {
		if conv.IsPassthrough() {
			p.logger.DebugContext(respCtx(resp), "Responses API streaming: passthrough mode (Chat Completions SSE → Responses SSE)",
				"model", modelID, "provider", cred.Type)
			usageOptions := tokenUsageExtractionOptionsForResponse(cred, resp.Header)
			return responses.TransformChatStreamToResponsesWithMetaAndUsage(
				r, w, publicModel, reqMeta, usageOptions.AudioInputIncludesCachedAudio, onComplete,
			)
		}

		p.logger.DebugContext(respCtx(resp), "Responses API streaming: converted mode (Provider SSE → Chat Completions SSE → Responses SSE)",
			"model", modelID, "provider", cred.Type)

		// Non-passthrough providers: first convert to Chat Completions SSE via pipe
		pr, pw := io.Pipe()
		var transformErr error
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			transformErr = conv.StreamTo(r, pw)
			if transformErr != nil {
				p.logger.ErrorContext(respCtx(resp), "Responses API streaming: provider→ChatCompletions transform failed",
					"error", transformErr, "provider", cred.Type)
				_ = pw.CloseWithError(transformErr)
			} else {
				p.logger.DebugContext(respCtx(resp), "Responses API streaming: provider→ChatCompletions transform completed OK",
					"provider", cred.Type)
				_ = pw.Close()
			}
		}()

		// Then convert Chat Completions SSE to Responses API SSE
		err := responses.TransformChatStreamToResponsesWithMetaAndUsage(
			pr, w, publicModel, reqMeta, false, onComplete,
		)
		_ = pr.Close()
		wg.Wait() // ensure goroutine completes before reading transformErr
		if err != nil {
			p.logger.ErrorContext(respCtx(resp), "Responses API streaming: ChatCompletions→Responses transform failed",
				"error", err, "provider", cred.Type)
			return err
		}
		if transformErr != nil {
			p.logger.ErrorContext(respCtx(resp), "Responses API streaming: provider transform error after Responses transform",
				"error", transformErr, "provider", cred.Type)
		}
		return transformErr
	}

	return p.handleTransformedStreaming(w, resp, cred.Name, modelID, string(cred.Type), transformer, logCtx)
}

func (p *Proxy) handleMessagesAPIStreaming(
	w http.ResponseWriter,
	resp *http.Response,
	cred *config.CredentialConfig,
	modelID string,
	logCtx *RequestLogContext,
	metadata anthropicconv.MessagesAdapterMetadata,
) error {
	publicModel := clientVisibleResponseModel(logCtx, modelID)
	converterModelID := modelID
	if logCtx != nil && logCtx.RealModelID != "" {
		converterModelID = logCtx.RealModelID
	}
	conv := converter.New(cred.Type, converter.RequestMode{
		ModelID:        converterModelID,
		DisplayModelID: publicModel,
		IsStreaming:    true,
	})
	transformer := func(reader io.Reader, _ string, writer io.Writer) error {
		if conv.IsPassthrough() {
			return anthropicconv.TransformChatStreamToMessages(reader, writer, publicModel, metadata)
		}
		chatReader, chatWriter := io.Pipe()
		var providerErr error
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			providerErr = conv.StreamTo(reader, chatWriter)
			if providerErr != nil {
				_ = chatWriter.CloseWithError(providerErr)
				return
			}
			_ = chatWriter.Close()
		}()
		err := anthropicconv.TransformChatStreamToMessages(chatReader, writer, publicModel, metadata)
		_ = chatReader.Close()
		wg.Wait()
		if err != nil {
			return err
		}
		return providerErr
	}
	return p.handleTransformedStreaming(w, resp, cred.Name, modelID, "messages", transformer, logCtx)
}

// handleNativeResponsesStreaming handles Responses API streaming via the Phase 4
// ProviderResponses converter (Vertex AI, Anthropic). The provider SSE is converted
// directly to Responses API SSE format by the provider-specific StreamTo implementation.
func (p *Proxy) handleNativeResponsesStreaming(
	w http.ResponseWriter,
	resp *http.Response,
	provResponses responses.ProviderResponses,
	modelID string,
	logCtx *RequestLogContext,
	onComplete func(*responses.Response),
	meta *responses.ResponsesMetadata,
) error {
	credName := ""
	if logCtx.Credential != nil {
		credName = logCtx.Credential.Name
	}
	transformer := func(r io.Reader, _ string, ww io.Writer) error {
		return provResponses.StreamTo(r, ww, clientVisibleResponseModel(logCtx, modelID), meta, onComplete)
	}
	return p.handleTransformedStreaming(w, resp, credName, modelID, "native_responses", transformer, logCtx)
}

// handlePassthroughResponsesStreaming handles Responses API streaming for codex models
// that natively support the /v1/responses endpoint. The provider SSE stream is forwarded
// to the client as-is.
//
// Token counts and the optional save callback are driven by the response.completed SSE
// event. Because the event payload can be very large (full response JSON with reasoning),
// it often spans multiple 8 KB buffer reads. This function maintains a line-level
// accumulator (partialSSELine) so that a data: line that arrives in pieces is reassembled
// before JSON parsing, avoiding the silent json.Unmarshal failures that would otherwise
// leave totalTokens = 0 and the store callback never invoked.
func (p *Proxy) handlePassthroughResponsesStreaming(
	w http.ResponseWriter,
	resp *http.Response,
	credName, modelID string,
	logCtx *RequestLogContext,
	onComplete func(*responses.Response),
	usageOptions ...converter.TokenUsageExtractionOptions,
) error {
	p.logger.DebugContext(respCtx(resp), "Starting passthrough Responses API streaming",
		"credential", credName, "model", modelID)
	tokenUsageOptions := converter.TokenUsageExtractionOptions{AudioInputIncludesCachedAudio: true}
	if len(usageOptions) > 0 {
		tokenUsageOptions = usageOptions[0]
	}
	audioInputAlreadyExcludesCachedAudio := !tokenUsageOptions.AudioInputIncludesCachedAudio

	var (
		totalTokens           int
		chunkCount            int
		lastRawChunk          []byte // last raw buffer for fallback in finalizeStreamingLog
		completedEventPayload []byte // JSON payload of response.completed (used instead of lastRawChunk)
		partialSSELine        string // partial SSE line accumulator across buffer reads
		completion            = p.newCompletionTokenAccumulator(modelID)
		detectStreamError     = resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices
		providerStreamError   = &proxyStreamErrorCapture{}
	)
	providerReader := io.Reader(resp.Body)
	if detectStreamError {
		providerReader = io.TeeReader(providerReader, proxyStreamErrorObserver{capture: providerStreamError})
	}

	onChunk := func(chunk []byte) {
		chunkCount++
		completion.AddChunk(chunk)
		// Same [DONE]-skip as handleStreamingWithTokens: don't overwrite a useful
		// lastRawChunk with the bare sentinel — keeps the usage event accessible.
		rememberLastStreamDataChunk(&lastRawChunk, chunk)

		// Combine the partial line buffered from the previous read with the new chunk.
		// SSE data: lines can be arbitrarily long (e.g. response.completed with reasoning)
		// and will be split across multiple 8 KB buffer reads.
		combined := partialSSELine + string(chunk)
		partialSSELine = ""

		lastNL := strings.LastIndex(combined, "\n")
		if lastNL < 0 {
			// No newline yet — entire content is an incomplete line.
			partialSSELine = combined
			return
		}
		if lastNL < len(combined)-1 {
			// Characters after the last newline are an incomplete line.
			partialSSELine = combined[lastNL+1:]
		}

		// Walk every complete line in this chunk.
		for _, line := range strings.Split(combined[:lastNL+1], "\n") {
			line = strings.TrimRight(line, "\r")
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			jsonData := strings.TrimPrefix(line, "data: ")
			if jsonData == "" || jsonData == "[DONE]" {
				continue
			}

			var event struct {
				Type     string             `json:"type"`
				Response responses.Response `json:"response"`
			}
			if json.Unmarshal([]byte(jsonData), &event) == nil && event.Type == "response.completed" {
				if event.Response.Usage != nil {
					totalTokens = event.Response.Usage.TotalTokens
					if logCtx != nil {
						if logCtx.TokenUsage == nil {
							logCtx.TokenUsage = &converter.TokenUsage{}
						}
						cachedTokens, cachedAudioTokens := converterutil.NormalizeCachedAudioBreakdown(
							event.Response.Usage.InputTokensDetails.CachedTokens,
							event.Response.Usage.InputTokensDetails.CachedAudioTokens,
						)
						logCtx.TokenUsage.PromptTokens = event.Response.Usage.InputTokens
						logCtx.TokenUsage.CompletionTokens = event.Response.Usage.OutputTokens
						logCtx.TokenUsage.CachedInputTokens = cachedTokens
						logCtx.TokenUsage.CachedAudioInputTokens = cachedAudioTokens
						logCtx.TokenUsage.CacheCreationTokens = event.Response.Usage.InputTokensDetails.CacheCreationTokens
						if details := event.Response.Usage.InputTokensDetails.CacheCreationTokenDetails; details != nil {
							logCtx.TokenUsage.CacheCreation5mTokens = details.Ephemeral5mInputTokens
							logCtx.TokenUsage.CacheCreation1hTokens = details.Ephemeral1hInputTokens
						}
						logCtx.TokenUsage.AudioInputTokens = normalizeStreamAudioInput(
							event.Response.Usage.InputTokensDetails.AudioTokens,
							cachedTokens,
							cachedAudioTokens,
							audioInputAlreadyExcludesCachedAudio,
						)
						logCtx.TokenUsage.AudioOutputTokens = event.Response.Usage.OutputTokensDetails.AudioTokens
						logCtx.TokenUsage.ReasoningTokens = event.Response.Usage.OutputTokensDetails.ReasoningTokens
						if event.Response.Usage.ServerToolUse != nil {
							logCtx.TokenUsage.WebSearchRequests = event.Response.Usage.ServerToolUse.WebSearchRequests
						}
					}
				}
				completedEventPayload = []byte(jsonData) // plain JSON; extractResponsesAPIUsage handles it
				if onComplete != nil {
					onComplete(&event.Response)
				}
			}
		}
	}

	clientReader := normalizeSuccessfulResponseModelStream(
		promanutils.NewSanitizingSSEReader(providerReader, clientVisibleResponseModel(logCtx, modelID)),
		resp.StatusCode,
		logCtx,
		modelID,
	)
	if err := p.streamToClient(respCtx(resp), w, clientReader, credName, metricModelID(modelID, logCtx), endpointFromLogContext(logCtx), resp.StatusCode, onChunk, nil, logCtx); err != nil {
		p.logStreamHandlerError(respCtx(resp), "streamToClient error in handlePassthroughResponsesStreaming", err,
			"credential", credName, "model", modelID, "chunks_received", chunkCount)
		if p.drainUpstreamOnAbort {
			drainCtx, cancel := context.WithTimeout(context.Background(), streamDrainTimeout)
			defer cancel()
			p.drainUpstream(drainCtx, clientReader, onChunk, credName)
		}
		finalChunk := lastRawChunk
		if len(completedEventPayload) > 0 {
			finalChunk = completedEventPayload
		}
		logTokens := totalTokens
		if logTokens == 0 {
			logTokens = completion.TokenCount()
		}
		if detectStreamError {
			err = resolveCapturedProviderStreamError(logCtx, resp.StatusCode, err, providerStreamError)
		}
		markStreamFailure(logCtx, err)
		p.finalizeStreamingLog(logCtx, logTokens, finalChunk, "openai", resp.StatusCode, audioInputAlreadyExcludesCachedAudio)
		return err
	}

	var streamErr error
	if detectStreamError {
		streamErr = resolveCapturedProviderStreamError(logCtx, resp.StatusCode, nil, providerStreamError)
		if streamErr != nil {
			markStreamFailure(logCtx, streamErr)
		}
	}

	p.logger.DebugContext(respCtx(resp), "handlePassthroughResponsesStreaming completed",
		"credential", credName, "model", modelID,
		"chunks_received", chunkCount, "total_tokens", totalTokens)

	logTokens := totalTokens
	if logTokens == 0 {
		logTokens = completion.TokenCount()
	}

	if logTokens > 0 {
		p.rateLimiter.ConsumeTokens(credName, logTokens)
		if modelID != "" {
			p.rateLimiter.ConsumeModelTokens(credName, modelID, logTokens)
		}
		p.logger.DebugContext(respCtx(resp), "Streaming token usage recorded",
			"credential", credName, "model", modelID, "tokens", totalTokens)
	}

	// Prefer the parsed response.completed payload for detailed token extraction.
	// The raw lastRawChunk may contain only `data: [DONE]` with no usage info.
	// completedEventPayload is plain JSON; extractJSONPayloadsFromStreamChunk handles it
	// via the non-SSE fast path, so extractResponsesAPIUsage works correctly.
	finalChunk := lastRawChunk
	if len(completedEventPayload) > 0 {
		finalChunk = completedEventPayload
	}

	p.finalizeStreamingLog(logCtx, logTokens, finalChunk, "openai", resp.StatusCode, audioInputAlreadyExcludesCachedAudio)
	return streamErr
}
