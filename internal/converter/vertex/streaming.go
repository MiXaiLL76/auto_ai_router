package vertex

import (
	"bufio"
	"fmt"
	"io"
	"log/slog"
	"strings"

	// goccy/go-json instead of encoding/json: every json.* call in this file is
	// on the per-SSE-chunk hot path (VertexStreamingChunk unmarshal, outgoing
	// OpenAIStreamingChunk marshal, function-call-args marshal). Benchmarked
	// ~3.5x faster decoding VertexStreamingChunk (which wraps genai.Candidate
	// et al.) than encoding/json; a round-trip test
	// (TestVertexStreamingChunkUnmarshal_GoccyMatchesStdlib) confirms identical
	// decode results, including genai's []byte/base64 fields like
	// ThoughtSignature.
	json "github.com/goccy/go-json"

	converterutil "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"google.golang.org/genai"
)

// TransformVertexStreamToOpenAI converts Vertex AI SSE stream to OpenAI SSE format.
//
// The stream may be a generation continued over several upstream requests (see
// continuation.go), one after the other: a request's CONTINUATION finish is not
// passed on, its usage and search queries are added to the next request's, and a
// stream left at such a finish -- not continued -- ends as "length".
func TransformVertexStreamToOpenAI(vertexStream io.Reader, model string, output io.Writer) error {
	scanner := bufio.NewScanner(vertexStream)
	scanner.Buffer(make([]byte, 1024*1024), converterutil.MaxSSELineBytes)
	chatID := converterutil.GenerateID()
	timestamp := converterutil.GetCurrentTimestamp()
	isFirstChunk := true
	sawDone := false // the upstream sent [DONE] itself
	var webSearch WebSearchQueries
	var usage ContinuationUsage
	var toolUse ToolUseSources
	finished := false // a candidate has reported its final finish reason
	// Candidates (by index) that have streamed a function call. Gemini may send the
	// call and its finish reason in different chunks, and that finish reason is STOP.
	calledTools := make(map[int32]bool)
	// Function calls streamed so far per candidate: a call's index in the OpenAI
	// stream, which must not restart when the calls come in separate chunks.
	toolCallCounts := make(map[int32]int)
	// Choices whose upstream request stopped with CONTINUATION, until the next
	// request's chunks come.
	continuing := make(map[int]bool)

	vertexLineCount := 0
	vertexChunkCount := 0
	for scanner.Scan() {
		line := scanner.Text()
		vertexLineCount++

		// Skip empty lines and non-data lines
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		// Extract JSON data
		jsonData := strings.TrimPrefix(line, "data: ")
		if jsonData == "[DONE]" {
			slog.Debug("[vertex/streaming] received [DONE]", "lines_read", vertexLineCount, "chunks_processed", vertexChunkCount)
			sawDone = true
			break
		}

		// Parse Vertex AI chunk
		var vertexChunk VertexStreamingChunk
		if err := json.Unmarshal([]byte(jsonData), &vertexChunk); err != nil {
			slog.Debug("[vertex/streaming] failed to parse Vertex chunk",
				"error", err, "json_prefix", jsonData[:min(len(jsonData), 200)])
			continue // Skip malformed chunks
		}
		toolUse.Add(vertexChunk.Candidates, vertexChunk.ModelVersion)
		answeringModel := toolUse.AnsweringModel(model)
		keepCallIDs := UsesFunctionCallIDs(answeringModel)
		webSearch.Add(vertexChunk.Candidates)
		webSearchRequests := webSearch.Count(answeringModel)
		requestContinues := false // this chunk ends an upstream request that goes on
		for _, candidate := range vertexChunk.Candidates {
			switch {
			case candidate != nil && IsContinuation(string(candidate.FinishReason)):
				requestContinues = true
			case HasFinishReason(candidate):
				finished = true
			}
		}
		var chunkUsage *openai.OpenAIUsage
		if vertexChunk.UsageMetadata != nil {
			chunkUsage = convertVertexUsageMetadata(usage.Observe(streamUsageMetadata(vertexChunk.UsageMetadata, toolUse, model, finished)))
			setVertexWebSearchUsage(chunkUsage, webSearchRequests)
			setInitialPromptTokens(chunkUsage, usage)
		}
		if requestContinues {
			usage.CloseRequest()
			webSearch.CloseResponse()
		}

		// Skip chunks with no candidates
		if len(vertexChunk.Candidates) == 0 {
			slog.Debug("[vertex/streaming] chunk with no candidates",
				"has_usage", vertexChunk.UsageMetadata != nil)
			// Still emit usage-only chunks (they have no candidates but have usage metadata)
			if chunkUsage != nil {
				openAIChunk := openai.OpenAIStreamingChunk{
					ID:      chatID,
					Object:  "chat.completion.chunk",
					Created: timestamp,
					Model:   model,
					Choices: []openai.OpenAIStreamingChoice{},
					Usage:   chunkUsage,
				}
				chunkJSON, err := json.Marshal(openAIChunk)
				if err == nil {
					slog.Debug("[vertex/streaming] emitting usage-only chunk",
						"prompt_tokens", vertexChunk.UsageMetadata.PromptTokenCount,
						"candidates_tokens", vertexChunk.UsageMetadata.CandidatesTokenCount,
						"total_tokens", vertexChunk.UsageMetadata.TotalTokenCount)
					if _, werr := fmt.Fprintf(output, "data: %s\n\n", chunkJSON); werr != nil {
						return fmt.Errorf("write usage chunk: %w", werr)
					}
				}
			}
			continue
		}

		vertexChunkCount++

		// Convert to OpenAI format
		openAIChunk := openai.OpenAIStreamingChunk{
			ID:      chatID,
			Object:  "chat.completion.chunk",
			Created: timestamp,
			Model:   model,
			Choices: make([]openai.OpenAIStreamingChoice, 0),
		}

		// Process candidates
		for i, candidate := range vertexChunk.Candidates {
			choice := openai.OpenAIStreamingChoice{
				Index: i,
				Delta: openai.OpenAIStreamingDelta{},
			}

			// Set role only for first chunk (OpenAI convention)
			if isFirstChunk {
				choice.Delta.Role = "assistant"
			}

			// Extract content and function calls from parts
			var content string
			var reasoningContent string
			var toolCalls []openai.OpenAIStreamingToolCall
			var images []openai.ImageData

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
					// Handle function calls
					if part.FunctionCall != nil {
						toolCall := convertVertexFunctionCallToStreamingOpenAI(part.FunctionCall, part.ThoughtSignature, toolCallCounts[candidate.Index], keepCallIDs)
						toolCalls = append(toolCalls, toolCall)
						toolCallCounts[candidate.Index]++
					}
					if part.InlineData != nil {
						if imageData, ok := inlineDataToChatImage(len(images), part.InlineData); ok {
							images = append(images, imageData)
						}
					}
				}
			}

			choice.Delta.Content = content
			if reasoningContent != "" {
				choice.Delta.ReasoningContent = reasoningContent
			}
			if len(toolCalls) > 0 {
				choice.Delta.ToolCalls = toolCalls
				calledTools[candidate.Index] = true
			}
			if len(images) > 0 {
				choice.Delta.Images = images
			}

			// Handle finish reason
			// An absent finish reason is "", not FINISH_REASON_UNSPECIFIED.
			continuing[i] = false
			switch {
			case IsContinuation(string(candidate.FinishReason)):
				// The generation goes on in the next upstream request.
				continuing[i] = true
			case HasFinishReason(candidate):
				finishReason := mapFinishReason(string(candidate.FinishReason))
				// Vertex returns "STOP" even with function calls (Gemini 3+).
				// Override for OpenAI compatibility, also when the calls came in
				// earlier chunks than the finish reason.
				if calledTools[candidate.Index] {
					finishReason = "tool_calls"
				}
				choice.FinishReason = &finishReason
			}

			openAIChunk.Choices = append(openAIChunk.Choices, choice)
		}

		// Usage of the generation so far
		openAIChunk.Usage = chunkUsage

		// Write OpenAI formatted chunk
		chunkJSON, err := json.Marshal(openAIChunk)
		if err != nil {
			continue
		}

		if _, werr := fmt.Fprintf(output, "data: %s\n\n", chunkJSON); werr != nil {
			return fmt.Errorf("write chunk: %w", werr)
		}
		isFirstChunk = false
	}

	slog.Debug("[vertex/streaming] scan finished",
		"lines_read", vertexLineCount, "chunks_processed", vertexChunkCount,
		"scanner_err", scanner.Err())

	// A request stopped at the per-request output cap and no continuation came: the
	// answer is cut short.
	if err := writeUncontinuedFinish(output, continuing, chatID, timestamp, model); err != nil {
		return err
	}

	// always send [DONE] even if stream ended without it
	if _, err := fmt.Fprintf(output, "data: [DONE]\n\n"); err != nil && sawDone {
		return fmt.Errorf("write [DONE]: %w", err)
	}

	return scanner.Err()
}

// writeUncontinuedFinish closes, with finish reason "length", the choices whose
// upstream request stopped with CONTINUATION when no further request followed.
func writeUncontinuedFinish(output io.Writer, continuing map[int]bool, chatID string, timestamp int64, model string) error {
	var choices []openai.OpenAIStreamingChoice
	for index := range len(continuing) + 1 {
		if !continuing[index] {
			continue
		}
		finishReason := mapFinishReason(string(genai.FinishReasonContinuation))
		choices = append(choices, openai.OpenAIStreamingChoice{Index: index, FinishReason: &finishReason})
	}
	if len(choices) == 0 {
		return nil
	}
	chunkJSON, err := json.Marshal(openai.OpenAIStreamingChunk{
		ID:      chatID,
		Object:  "chat.completion.chunk",
		Created: timestamp,
		Model:   model,
		Choices: choices,
	})
	if err != nil {
		return nil
	}
	if _, err := fmt.Fprintf(output, "data: %s\n\n", chunkJSON); err != nil {
		return fmt.Errorf("write finish chunk: %w", err)
	}
	return nil
}

// setInitialPromptTokens adds the prompt the client sent to the usage of a
// generation continued over several requests (see ContinuationUsage).
func setInitialPromptTokens(usage *openai.OpenAIUsage, generation ContinuationUsage) {
	if usage == nil || !generation.Continued() || generation.InitialPromptTokens <= 0 {
		return
	}
	if usage.PromptTokensDetails == nil {
		usage.PromptTokensDetails = &openai.TokenDetails{}
	}
	usage.PromptTokensDetails.InitialPromptTokens = int(generation.InitialPromptTokens)
}

// HasFinishReason reports whether a candidate carries Gemini's finish reason; it
// is absent on every stream chunk but the last.
func HasFinishReason(candidate *genai.Candidate) bool {
	return candidate != nil && candidate.FinishReason != "" && candidate.FinishReason != genai.FinishReasonUnspecified
}

func setVertexWebSearchUsage(usage *openai.OpenAIUsage, requests int) {
	if usage == nil || requests <= 0 {
		return
	}
	usage.ServerToolUse = &openai.ServerToolUseDetails{WebSearchRequests: requests}
}

// convertVertexFunctionCallToStreamingOpenAI converts Vertex function call to OpenAI streaming tool call format.
// Preserves thoughtSignature in provider_specific_fields for Gemini 3.x multi-turn streaming.
func convertVertexFunctionCallToStreamingOpenAI(genaiCall *genai.FunctionCall, thoughtSignature []byte, index int, keepID bool) openai.OpenAIStreamingToolCall {
	// Convert args to JSON string
	argsJSON := "{}"
	if genaiCall.Args != nil {
		if data, err := json.Marshal(genaiCall.Args); err == nil {
			argsJSON = string(data)
		}
	}

	toolCall := openai.OpenAIStreamingToolCall{
		Index: index,
		ID:    functionCallID(genaiCall, keepID),
		Type:  "function",
		Function: &openai.OpenAIStreamingToolFunction{
			Name:      genaiCall.Name,
			Arguments: argsJSON,
		},
	}

	// Preserve thoughtSignature for streaming mode
	providerFields := make(map[string]interface{})
	if len(thoughtSignature) > 0 {
		providerFields["thought_signature"] = converterutil.EncodeBase64(thoughtSignature)
	} else {
		providerFields["skip_thought_signature_validator"] = true
	}
	toolCall.ProviderSpecificFields = providerFields

	return toolCall
}
