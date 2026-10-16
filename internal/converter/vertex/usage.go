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

// CountWebSearchRequests returns the number of distinct Google Search queries
// confirmed by Vertex grounding metadata, web and image search together.
func CountWebSearchRequests(candidates []*genai.Candidate) int {
	queries := make(map[string]struct{})
	AddWebSearchQueries(queries, candidates)
	return len(queries)
}

// imageSearchQueryKey keeps image queries apart from web ones: the same text searched
// both ways is two billed queries.
const imageSearchQueryKey = "\x00image_search\x00"

// AddWebSearchQueries adds distinct, non-empty Google Search queries from the
// supplied candidates (web and image search alike) to queries. Callers can reuse the
// same set across streaming chunks so each provider query is billed exactly once.
func AddWebSearchQueries(queries map[string]struct{}, candidates []*genai.Candidate) {
	for _, candidate := range candidates {
		if candidate == nil || candidate.GroundingMetadata == nil {
			continue
		}
		for _, query := range candidate.GroundingMetadata.WebSearchQueries {
			query = strings.TrimSpace(query)
			if query != "" {
				queries[query] = struct{}{}
			}
		}
		for _, query := range candidate.GroundingMetadata.ImageSearchQueries {
			query = strings.TrimSpace(query)
			if query != "" {
				queries[imageSearchQueryKey+query] = struct{}{}
			}
		}
	}
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
