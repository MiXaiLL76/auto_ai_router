package vertex

import (
	"cmp"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"google.golang.org/genai"
)

// This file holds Gemini usage accounting shared by the chat, Responses API and
// streaming routes and by the proxy: the OpenAI usage conversion, the billed
// Google Search queries and the tool-use prompt tokens a model does not charge.

// CountWebSearchRequests returns the number of Google Search queries confirmed by
// the grounding metadata of one response, web and image search together.
func CountWebSearchRequests(candidates []*genai.Candidate) int {
	var queries WebSearchQueries
	queries.Add(candidates)
	return queries.Count()
}

// imageSearchQueryKey keeps image queries apart from web ones: the same text searched
// both ways is two billed queries.
const imageSearchQueryKey = "\x00image_search\x00"

// WebSearchQueries counts the Google Search queries a generation ran, web and
// image search together, as Google bills them: one per query run, even when two
// runs share their text. A stream repeats grounding metadata it already sent, so
// within one upstream response a query text counts as many times as a single
// chunk lists it at most; the responses of a continued generation (see
// continuation.go) ran their own queries, so theirs add up.
type WebSearchQueries struct {
	response map[string]int // per query, the most times one chunk of the current response listed it
	closed   int            // queries of the earlier responses of a continued generation
}

// Add records the queries listed by one response or stream chunk.
func (q *WebSearchQueries) Add(candidates []*genai.Candidate) {
	listed := make(map[string]int)
	for _, candidate := range candidates {
		if candidate == nil || candidate.GroundingMetadata == nil {
			continue
		}
		for _, query := range candidate.GroundingMetadata.WebSearchQueries {
			if query = strings.TrimSpace(query); query != "" {
				listed[query]++
			}
		}
		for _, query := range candidate.GroundingMetadata.ImageSearchQueries {
			if query = strings.TrimSpace(query); query != "" {
				listed[imageSearchQueryKey+query]++
			}
		}
	}
	for query, times := range listed {
		if q.response == nil {
			q.response = make(map[string]int)
		}
		q.response[query] = max(q.response[query], times)
	}
}

// CloseResponse ends the current upstream response of a continued generation:
// queries listed after it are new ones.
func (q *WebSearchQueries) CloseResponse() {
	q.closed = q.Count()
	q.response = nil
}

// Count returns the number of queries recorded so far.
func (q *WebSearchQueries) Count() int {
	count := q.closed
	for _, times := range q.response {
		count += times
	}
	return count
}

// ToolUseSources records, over a response or a whole stream, what fed
// toolUsePromptTokenCount and which model answered.
type ToolUseSources struct {
	search       bool   // Google Search (web or image) queries ran
	other        bool   // url_context fetches, code execution, Maps or retrieval grounding
	modelVersion string // Gemini's modelVersion of the response
}

// Add records the tool use in candidates and the modelVersion, which names the model
// even when the router knows it by an alias.
func (s *ToolUseSources) Add(candidates []*genai.Candidate, modelVersion string) {
	if modelVersion != "" {
		s.modelVersion = modelVersion
	}
	for _, candidate := range candidates {
		if candidate == nil {
			continue
		}
		if candidate.URLContextMetadata != nil {
			s.other = true
		}
		if candidate.Content != nil {
			for _, part := range candidate.Content.Parts {
				if part != nil && (part.ExecutableCode != nil || part.CodeExecutionResult != nil) {
					s.other = true
				}
			}
		}
		gm := candidate.GroundingMetadata
		if gm == nil {
			continue
		}
		if len(gm.WebSearchQueries) > 0 || len(gm.ImageSearchQueries) > 0 {
			s.search = true
		}
		if len(gm.RetrievalQueries) > 0 {
			s.other = true
		}
		for _, chunk := range gm.GroundingChunks {
			if chunk != nil && (chunk.Maps != nil || chunk.RetrievedContext != nil) {
				s.other = true
			}
		}
	}
}

// unbilledSearchContext reports whether the answering model does not charge Google
// Search context as input. Gemini's modelVersion names that model; the router's model
// is only the fallback when the response carries none, so an alias answered by a
// model that charges the context stays billed.
func (s ToolUseSources) unbilledSearchContext(model string) bool {
	profile := lookupGeminiModelProfile(cmp.Or(s.modelVersion, model))
	return profile != nil && profile.unbilledSearchContext
}

// BillableUsageMetadata drops the tool-use prompt tokens from meta when the model does
// not charge Google Search context and only Google Search produced it. With any other
// tool (url_context is billed as input) the split is unknown, so all of it is billed.
func BillableUsageMetadata(meta *genai.GenerateContentResponseUsageMetadata, sources ToolUseSources, model string) *genai.GenerateContentResponseUsageMetadata {
	if meta == nil || meta.ToolUsePromptTokenCount <= 0 || !sources.search || sources.other ||
		!sources.unbilledSearchContext(model) {
		return meta
	}
	billable := *meta
	if billable.TotalTokenCount >= billable.ToolUsePromptTokenCount {
		billable.TotalTokenCount -= billable.ToolUsePromptTokenCount
	}
	billable.ToolUsePromptTokenCount = 0
	billable.ToolUsePromptTokensDetails = nil
	return &billable
}

// streamUsageMetadata is BillableUsageMetadata for one stream chunk. Stream usage is
// merged with later non-zero values winning, so a zero can't undo an earlier value;
// until the last chunk (which carries the grounding) the tool-use modality breakdown
// is left out, or search image tokens would stick even when they end up unbilled.
func streamUsageMetadata(meta *genai.GenerateContentResponseUsageMetadata, sources ToolUseSources, model string, finished bool) *genai.GenerateContentResponseUsageMetadata {
	meta = BillableUsageMetadata(meta, sources, model)
	if finished || meta == nil || len(meta.ToolUsePromptTokensDetails) == 0 || !sources.unbilledSearchContext(model) {
		return meta
	}
	partial := *meta
	partial.ToolUsePromptTokensDetails = nil
	return &partial
}

// convertVertexUsageMetadata converts Vertex AI usage metadata to OpenAI format.
func convertVertexUsageMetadata(meta *genai.GenerateContentResponseUsageMetadata) *openai.OpenAIUsage {
	// Include thinking/reasoning tokens in completion tokens for accurate conversion
	// Vertex AI reasoning models include thoughts_token_count which are part of the response
	completionTokens := int(meta.CandidatesTokenCount)
	if meta.ThoughtsTokenCount > 0 {
		completionTokens += int(meta.ThoughtsTokenCount)
	}

	usage := &openai.OpenAIUsage{
		PromptTokens:     int(meta.PromptTokenCount + meta.ToolUsePromptTokenCount),
		CompletionTokens: completionTokens,
		TotalTokens:      int(meta.PromptTokenCount+meta.ToolUsePromptTokenCount) + completionTokens,
	}

	// Map Vertex thinking tokens to OpenAI reasoning_tokens
	if meta.ThoughtsTokenCount > 0 {
		if usage.CompletionTokensDetails == nil {
			usage.CompletionTokensDetails = &openai.CompletionTokenDetails{}
		}
		usage.CompletionTokensDetails.ReasoningTokens = int(meta.ThoughtsTokenCount)
	}

	if meta.CachedContentTokenCount > 0 {
		if usage.PromptTokensDetails == nil {
			usage.PromptTokensDetails = &openai.TokenDetails{}
		}
		usage.PromptTokensDetails.CachedTokens = int(meta.CachedContentTokenCount)
	}

	if len(meta.CandidatesTokensDetails) > 0 {
		if usage.CompletionTokensDetails == nil {
			usage.CompletionTokensDetails = &openai.CompletionTokenDetails{}
		}
		for _, detail := range meta.CandidatesTokensDetails {
			if detail == nil {
				continue
			}
			switch genai.MediaModality(detail.Modality) {
			case genai.MediaModalityAudio:
				usage.CompletionTokensDetails.AudioTokens += int(detail.TokenCount)
			case genai.MediaModalityImage, genai.MediaModalityVideo:
				// LiteLLM supports image_tokens as an extension to the OpenAI
				// completion token details. Preserve the modality so generated
				// images are billed with output_cost_per_image_token instead of
				// the much lower text output rate.
				usage.CompletionTokensDetails.ImageTokens += int(detail.TokenCount)
			}
		}
	}

	if len(meta.PromptTokensDetails) > 0 {
		if usage.PromptTokensDetails == nil {
			usage.PromptTokensDetails = &openai.TokenDetails{}
		}
		for _, detail := range meta.PromptTokensDetails {
			if detail == nil {
				continue
			}
			switch genai.MediaModality(detail.Modality) {
			case genai.MediaModalityAudio:
				usage.PromptTokensDetails.AudioTokens += int(detail.TokenCount)
			case genai.MediaModalityImage:
				usage.PromptTokensDetails.ImageTokens += int(detail.TokenCount)
			case genai.MediaModalityVideo:
				// Reported apart from images: the two can carry different prices
				// (input_cost_per_video_token falls back to the image rate).
				usage.PromptTokensDetails.VideoTokens += int(detail.TokenCount)
			}
		}
	}

	if len(meta.ToolUsePromptTokensDetails) > 0 {
		if usage.PromptTokensDetails == nil {
			usage.PromptTokensDetails = &openai.TokenDetails{}
		}
		for _, detail := range meta.ToolUsePromptTokensDetails {
			if detail == nil {
				continue
			}
			switch genai.MediaModality(detail.Modality) {
			case genai.MediaModalityAudio:
				usage.PromptTokensDetails.AudioTokens += int(detail.TokenCount)
			case genai.MediaModalityImage:
				usage.PromptTokensDetails.ImageTokens += int(detail.TokenCount)
			case genai.MediaModalityVideo:
				// Reported apart from images: the two can carry different prices
				// (input_cost_per_video_token falls back to the image rate).
				usage.PromptTokensDetails.VideoTokens += int(detail.TokenCount)
			}
		}
	}

	// Avoid double-charging cached modality tokens as regular audio/image input.
	// Cached tokens are billed separately via CachedTokens.
	if len(meta.CacheTokensDetails) > 0 && usage.PromptTokensDetails != nil {
		for _, detail := range meta.CacheTokensDetails {
			if detail == nil {
				continue
			}
			switch genai.MediaModality(detail.Modality) {
			case genai.MediaModalityAudio:
				usage.PromptTokensDetails.CachedAudioTokens += int(detail.TokenCount)
				usage.PromptTokensDetails.AudioTokens -= int(detail.TokenCount)
				if usage.PromptTokensDetails.AudioTokens < 0 {
					usage.PromptTokensDetails.AudioTokens = 0
				}
			case genai.MediaModalityImage:
				usage.PromptTokensDetails.ImageTokens -= int(detail.TokenCount)
				if usage.PromptTokensDetails.ImageTokens < 0 {
					usage.PromptTokensDetails.ImageTokens = 0
				}
			case genai.MediaModalityVideo:
				usage.PromptTokensDetails.VideoTokens -= int(detail.TokenCount)
				if usage.PromptTokensDetails.VideoTokens < 0 {
					usage.PromptTokensDetails.VideoTokens = 0
				}
			}
		}
	}

	return usage
}
