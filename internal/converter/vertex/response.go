package vertex

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	converterutil "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"google.golang.org/genai"
)

// VertexToOpenAI converts Vertex AI response to OpenAI format
func VertexToOpenAI(vertexBody []byte, model string) ([]byte, error) {
	var vertexResp genai.GenerateContentResponse
	if err := json.Unmarshal(vertexBody, &vertexResp); err != nil {
		return nil, fmt.Errorf("failed to parse Vertex response: %w", err)
	}

	openAIResp := openai.OpenAIResponse{
		ID:      converterutil.GenerateID(),
		Object:  "chat.completion",
		Created: converterutil.GetCurrentTimestamp(),
		Model:   model,
		Choices: make([]openai.OpenAIChoice, 0),
	}

	// Convert candidates to choices
	for _, candidate := range vertexResp.Candidates {
		var content string
		var reasoningContent string
		var images []openai.ImageData
		var toolCalls []openai.OpenAIToolCall

		if candidate.Content != nil && candidate.Content.Parts != nil {
			for _, part := range candidate.Content.Parts {
				// Handle thinking/reasoning parts (Thought == true means this is a reasoning token)
				if part.Thought {
					reasoningContent += part.Text
					continue
				}
				if part.Text != "" {
					content += part.Text
				}
				// Handle inline data (images) from Vertex response
				if part.InlineData != nil {
					if imageData, ok := inlineDataToChatImage(len(images), part.InlineData); ok {
						images = append(images, imageData)
					}
				}
				// Handle function calls from Vertex response
				if part.FunctionCall != nil {
					toolCall := convertGenaiToOpenAIFunctionCall(part.FunctionCall, part.ThoughtSignature)
					toolCalls = append(toolCalls, toolCall)
				}
				// Handle code execution results (model executed code and returned output)
				if part.CodeExecutionResult != nil {
					if part.CodeExecutionResult.Output != "" {
						content += "\n```\n" + part.CodeExecutionResult.Output + "\n```"
					}
				}
				// Handle executable code (model-generated code to be executed)
				if part.ExecutableCode != nil {
					if part.ExecutableCode.Code != "" {
						lang := strings.ToLower(string(part.ExecutableCode.Language))
						if lang == "" || lang == "language_unspecified" {
							lang = "python" // Vertex default language
						}
						content += "\n```" + lang + "\n" + part.ExecutableCode.Code + "\n```"
					}
				}
			}
		}

		// A candidate with no parts back at all (MAX_TOKENS entirely consumed
		// by hidden reasoning, an image-generation candidate that produced no
		// image -- NO_IMAGE, IMAGE_SAFETY, IMAGE_PROHIBITED_CONTENT, etc. --
		// or any other empty finish) has no partial model output to show.
		// Leave content empty and let finish_reason carry the signal, same as
		// every other provider route: a synthetic English placeholder here
		// isn't real model output, is silently indistinguishable from one,
		// and (unlike an empty string) can't be detected by a client without
		// hardcoding an exact string.

		message := openai.OpenAIResponseMessage{
			Role:    "assistant",
			Content: content,
			Images:  images,
		}

		// Set reasoning content if present (thinking/reasoning tokens)
		if reasoningContent != "" {
			message.ReasoningContent = reasoningContent
		}

		// Only include tool_calls if there are any
		if len(toolCalls) > 0 {
			message.ToolCalls = toolCalls
		}

		// Set refusal message when content is filtered for safety
		if candidate.FinishReason == genai.FinishReasonSafety && content == "" && len(toolCalls) == 0 {
			message.Refusal = "Content was filtered for safety reasons"
			message.Content = "" // ensure empty
		}

		finishReason := mapFinishReason(string(candidate.FinishReason))
		// Vertex API returns "STOP" even when there are function calls (Gemini 3+).
		// Override to "tool_calls" for OpenAI compatibility — clients rely on this
		// to detect that tool results need to be sent back.
		if len(toolCalls) > 0 && finishReason != "tool_calls" {
			finishReason = "tool_calls"
		}

		choice := openai.OpenAIChoice{
			Index:        int(candidate.Index),
			Message:      message,
			FinishReason: finishReason,
		}
		openAIResp.Choices = append(openAIResp.Choices, choice)
	}

	// Convert usage metadata
	if vertexResp.UsageMetadata != nil {
		var toolUse ToolUseSources
		toolUse.Add(vertexResp.Candidates, vertexResp.ModelVersion)
		openAIResp.Usage = convertVertexUsageMetadata(BillableUsageMetadata(vertexResp.UsageMetadata, toolUse, model))
	}
	if webSearchRequests := CountWebSearchRequests(vertexResp.Candidates); webSearchRequests > 0 {
		if openAIResp.Usage == nil {
			openAIResp.Usage = &openai.OpenAIUsage{}
		}
		openAIResp.Usage.ServerToolUse = &openai.ServerToolUseDetails{
			WebSearchRequests: webSearchRequests,
		}
	}
	return json.Marshal(openAIResp)
}

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

// unbilledSearchContext reports whether the answering model (by modelVersion, else
// model) does not charge Google Search context as input.
func (s ToolUseSources) unbilledSearchContext(model string) bool {
	for _, name := range []string{s.modelVersion, model} {
		if profile := lookupGeminiModelProfile(name); profile != nil {
			return profile.unbilledSearchContext
		}
	}
	return false
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

// HasFinishReason reports whether a candidate carries Gemini's finish reason; it
// is absent on every stream chunk but the last.
func HasFinishReason(candidate *genai.Candidate) bool {
	return candidate != nil && candidate.FinishReason != "" && candidate.FinishReason != genai.FinishReasonUnspecified
}

func inlineDataToChatImage(index int, blob *genai.Blob) (openai.ImageData, bool) {
	mimeType := blob.MIMEType
	if mimeType == "" {
		mimeType = http.DetectContentType(blob.Data)
	}
	if !strings.HasPrefix(strings.ToLower(mimeType), "image/") {
		return openai.ImageData{}, false
	}

	b64Data := base64.StdEncoding.EncodeToString(blob.Data)
	return openai.ImageData{
		Type:  "image_url",
		Index: &index,
		ImageURL: &openai.ImageURL{
			URL: "data:" + mimeType + ";base64," + b64Data,
		},
	}, true
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

// mapFinishReason maps Vertex AI finish reason to OpenAI finish reason
func mapFinishReason(vertexReason string) string {
	switch vertexReason {
	case "STOP":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "SAFETY", "RECITATION":
		return "content_filter"
	case "BLOCKLIST", "PROHIBITED_CONTENT", "SPII":
		return "content_filter"
	case "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION":
		return "content_filter"
	case "TOOL_CALL":
		return "tool_calls"
	default:
		return "stop"
	}
}

// convertGenaiToOpenAIFunctionCall converts genai.FunctionCall to OpenAI tool call format.
// Preserves thoughtSignature in provider_specific_fields for Gemini 3.x multi-turn conversations.
// Per litellm >= 1.80.5 and Google Gemini 3 requirements, thoughtSignature must be preserved
// when sending tool results back to maintain context and avoid 400 errors.
func convertGenaiToOpenAIFunctionCall(genaiCall *genai.FunctionCall, thoughtSignature []byte) openai.OpenAIToolCall {
	// Convert args to JSON string
	argsJSON := "{}"
	if genaiCall.Args != nil {
		if data, err := json.Marshal(genaiCall.Args); err == nil {
			argsJSON = string(data)
		}
	}

	toolCall := openai.OpenAIToolCall{
		ID:   converterutil.GenerateID(),
		Type: "function",
		Function: openai.OpenAIToolFunction{
			Name:      genaiCall.Name,
			Arguments: argsJSON,
		},
	}

	// Preserve thoughtSignature in provider_specific_fields for Gemini 3 function calling.
	// This is required by Gemini API when sending tool results in subsequent requests.
	providerFields := make(map[string]interface{})

	if len(thoughtSignature) > 0 {
		// Store thoughtSignature as base64 string for JSON compatibility
		providerFields["thought_signature"] = converterutil.EncodeBase64(thoughtSignature)
	} else {
		// Per litellm and Google docs: use dummy validator when thought_signature is missing.
		// This allows seamless model switching (e.g., from gemini-2.5-flash to gemini-3-pro).
		providerFields["skip_thought_signature_validator"] = true
	}

	toolCall.ProviderSpecificFields = providerFields
	return toolCall
}
