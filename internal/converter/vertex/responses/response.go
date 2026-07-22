package vertexresponses

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"google.golang.org/genai"
)

// VertexToResponsesResponse converts a Vertex AI GenerateContent response to a
// responses.Response, using displayModelID as the echoed model name.
func VertexToResponsesResponse(body []byte, displayModelID, responseID string, createdAt int64) (*responses.Response, error) {
	var vertexResp genai.GenerateContentResponse
	if err := json.Unmarshal(body, &vertexResp); err != nil {
		return nil, fmt.Errorf("VertexToResponsesResponse: parse: %w", err)
	}
	if responseID == "" {
		responseID = responses.GenerateResponseID()
	}
	if createdAt == 0 {
		createdAt = time.Now().Unix()
	}
	return buildResponsesResponse(&vertexResp, displayModelID, responseID, createdAt), nil
}

func buildResponsesResponse(
	vertexResp *genai.GenerateContentResponse,
	model, responseID string,
	createdAt int64,
) *responses.Response {
	status, incompleteDetails := finishReasonToStatus(vertexResp)
	output := candidatesToOutputItems(vertexResp)
	usage := usageMetadataToUsage(vertexResp.UsageMetadata)
	completedAt := createdAt
	return responses.BuildCompletedResponse(responses.CompletedResponseParams{
		ID:                responseID,
		Model:             model,
		CreatedAt:         createdAt,
		CompletedAt:       &completedAt,
		Status:            status,
		IncompleteDetails: incompleteDetails,
		Output:            output,
		Usage:             usage,
	})
}

// candidatesToOutputItems converts Vertex candidates to Responses API OutputItems.
func candidatesToOutputItems(vertexResp *genai.GenerateContentResponse) []responses.OutputItem {
	var output []responses.OutputItem

	for _, candidate := range vertexResp.Candidates {
		if candidate.Content == nil {
			continue
		}

		// Accumulate text/refusal parts into a message item; other parts become separate items.
		var msgContent []responses.OutputContent
		var codeCallID string // active code_interpreter_call item ID
		partRefs := map[int]textPartRef{}
		var currentMessagePartIndices []int
		flushMessage := func() {
			if len(msgContent) == 0 {
				return
			}
			output = append(output, buildMessageItem(msgContent))
			outputIdx := len(output) - 1
			for contentIdx, partIdx := range currentMessagePartIndices {
				partRefs[partIdx] = textPartRef{
					outputIdx:  outputIdx,
					contentIdx: contentIdx,
				}
			}
			msgContent = nil
			currentMessagePartIndices = nil
		}

		for partIdx, part := range candidate.Content.Parts {
			switch {
			case part.Thought:
				// Reasoning / thinking part → reasoning OutputItem.
				if part.Text != "" {
					output = append(output, responses.OutputItem{
						Type:   "reasoning",
						ID:     responses.GenerateItemID("rs_"),
						Status: "completed",
						Summary: []responses.OutputContent{
							{Type: "summary_text", Text: part.Text},
						},
					})
				}

			case part.FunctionCall != nil:
				// Flush accumulated message content first.
				flushMessage()
				// Function call → function_call OutputItem.
				argsJSON := ""
				if part.FunctionCall.Args != nil {
					if b, err := json.Marshal(part.FunctionCall.Args); err == nil {
						argsJSON = string(b)
					}
				}
				callID := part.FunctionCall.ID
				if callID == "" {
					callID = responses.GenerateItemID("call_")
				}
				output = append(output, responses.OutputItem{
					Type:      "function_call",
					ID:        responses.GenerateItemID("fc_"),
					Status:    "completed",
					CallID:    callID,
					Name:      part.FunctionCall.Name,
					Arguments: argsJSON,
				})

			case part.ExecutableCode != nil:
				// Code to be executed → start or continue code_interpreter_call item.
				flushMessage()
				codeCallID = responses.GenerateItemID("ci_")
				output = append(output, responses.OutputItem{
					Type:   "code_interpreter_call",
					ID:     codeCallID,
					Status: "completed",
					Code:   part.ExecutableCode.Code,
				})

			case part.CodeExecutionResult != nil:
				// Code execution result → append outputs to the most recent code_interpreter_call.
				if codeCallID != "" {
					// Find and update the last code_interpreter_call.
					for i := len(output) - 1; i >= 0; i-- {
						if output[i].Type == "code_interpreter_call" && output[i].ID == codeCallID {
							output[i].Outputs = []map[string]interface{}{
								{"type": "text", "text": part.CodeExecutionResult.Output},
							}
							break
						}
					}
				}

			case part.InlineData != nil:
				// Gemini image models return generated media as inlineData. The
				// Responses API represents a generated image as a standalone
				// image_generation_call whose result is raw base64 (without the
				// data-URL prefix).
				if strings.HasPrefix(strings.ToLower(part.InlineData.MIMEType), "image/") {
					flushMessage()
					output = append(output, responses.OutputItem{
						Type:         "image_generation_call",
						ID:           responses.GenerateItemID("ig_"),
						Status:       "completed",
						Result:       base64.StdEncoding.EncodeToString(part.InlineData.Data),
						OutputFormat: imageOutputFormat(part.InlineData.MIMEType),
					})
				}

			case part.Text != "":
				// Regular text → accumulate as output_text content part.
				msgContent = append(msgContent, responses.OutputContent{
					Type:        "output_text",
					Text:        part.Text,
					Annotations: []responses.Annotation{},
				})
				currentMessagePartIndices = append(currentMessagePartIndices, partIdx)
			}
		}

		// Flush any remaining message content before grounding annotations are applied.
		flushMessage()

		if candidate.GroundingMetadata != nil {
			applyGroundingAnnotations(output, partRefs, candidate.GroundingMetadata)
		}

		// Grounding metadata → web_search_call item.
		if candidate.GroundingMetadata != nil {
			wsItem := groundingMetadataToWebSearchCall(candidate.GroundingMetadata)
			if wsItem != nil {
				output = append(output, *wsItem)
			}
		}
	}

	if len(output) == 0 {
		output = []responses.OutputItem{
			{
				Type:   "message",
				ID:     responses.GenerateItemID("msg_"),
				Status: "completed",
				Role:   "assistant",
				Content: []responses.OutputContent{
					{Type: "output_text", Text: "", Annotations: []responses.Annotation{}},
				},
			},
		}
	}

	return output
}

func imageOutputFormat(mimeType string) string {
	format := strings.TrimPrefix(strings.ToLower(strings.TrimSpace(mimeType)), "image/")
	if format == "jpg" {
		return "jpeg"
	}
	return format
}

type textPartRef struct {
	outputIdx  int
	contentIdx int
}

func buildMessageItem(content []responses.OutputContent) responses.OutputItem {
	return responses.OutputItem{
		Type:    "message",
		ID:      responses.GenerateItemID("msg_"),
		Status:  "completed",
		Role:    "assistant",
		Content: content,
	}
}

func applyGroundingAnnotations(output []responses.OutputItem, partRefs map[int]textPartRef, gm *genai.GroundingMetadata) {
	if gm == nil {
		return
	}

	for _, support := range gm.GroundingSupports {
		if support == nil || support.Segment == nil {
			continue
		}
		ref, ok := partRefs[int(support.Segment.PartIndex)]
		if !ok || ref.outputIdx >= len(output) || ref.contentIdx >= len(output[ref.outputIdx].Content) {
			continue
		}

		annotations := buildGroundingAnnotations(gm.GroundingChunks, support)
		if len(annotations) == 0 {
			continue
		}

		output[ref.outputIdx].Content[ref.contentIdx].Annotations = append(
			output[ref.outputIdx].Content[ref.contentIdx].Annotations,
			annotations...,
		)
	}
}

func buildGroundingAnnotations(chunks []*genai.GroundingChunk, support *genai.GroundingSupport) []responses.Annotation {
	if support == nil || support.Segment == nil {
		return nil
	}

	var annotations []responses.Annotation
	for _, chunkIdx := range support.GroundingChunkIndices {
		if chunkIdx < 0 || int(chunkIdx) >= len(chunks) {
			continue
		}
		chunk := chunks[chunkIdx]
		if chunk == nil || chunk.Web == nil || chunk.Web.URI == "" {
			continue
		}
		annotations = append(annotations, responses.Annotation{
			Type:       "url_citation",
			URL:        chunk.Web.URI,
			Title:      chunk.Web.Title,
			StartIndex: int(support.Segment.StartIndex),
			EndIndex:   int(support.Segment.EndIndex),
		})
	}
	return annotations
}

// groundingMetadataToWebSearchCall converts Vertex grounding metadata to a web_search_call item.
func groundingMetadataToWebSearchCall(gm *genai.GroundingMetadata) *responses.OutputItem {
	if gm == nil {
		return nil
	}
	var queries []string
	for _, q := range gm.WebSearchQueries {
		if q != "" {
			queries = append(queries, q)
		}
	}
	if len(queries) == 0 && len(gm.GroundingChunks) == 0 {
		return nil
	}

	item := &responses.OutputItem{
		Type:    "web_search_call",
		ID:      responses.GenerateItemID("ws_"),
		Status:  "completed",
		Queries: queries,
	}
	return item
}

// finishReasonToStatus maps Vertex finish reason to Responses API status.
func finishReasonToStatus(vertexResp *genai.GenerateContentResponse) (string, *responses.IncompleteDetails) {
	if len(vertexResp.Candidates) == 0 {
		return "incomplete", &responses.IncompleteDetails{Reason: "other"}
	}
	switch vertexResp.Candidates[0].FinishReason {
	case genai.FinishReasonStop:
		return "completed", nil
	case genai.FinishReasonMaxTokens:
		return "incomplete", &responses.IncompleteDetails{Reason: "max_output_tokens"}
	case genai.FinishReasonSafety, genai.FinishReasonRecitation:
		return "incomplete", &responses.IncompleteDetails{Reason: "content_filter"}
	default:
		return "completed", nil
	}
}

// usageMetadataToUsage converts Vertex UsageMetadata to responses.Usage.
func usageMetadataToUsage(meta *genai.GenerateContentResponseUsageMetadata) *responses.Usage {
	if meta == nil {
		return nil
	}

	// Extract specialized token counts from modality details.
	thoughtTokens := int(meta.ThoughtsTokenCount)
	cachedTokens := int(meta.CachedContentTokenCount)
	imageTokens := 0
	for _, detail := range meta.CandidatesTokensDetails {
		if detail == nil {
			continue
		}
		switch genai.MediaModality(detail.Modality) {
		case genai.MediaModalityImage, genai.MediaModalityVideo:
			imageTokens += int(detail.TokenCount)
		}
	}

	return &responses.Usage{
		InputTokens:  int(meta.PromptTokenCount),
		OutputTokens: int(meta.CandidatesTokenCount),
		TotalTokens:  int(meta.TotalTokenCount),
		InputTokensDetails: responses.InputDetails{
			CachedTokens: cachedTokens,
		},
		OutputTokensDetails: responses.OutputDetails{
			ReasoningTokens: thoughtTokens,
			ImageTokens:     imageTokens,
		},
	}
}
