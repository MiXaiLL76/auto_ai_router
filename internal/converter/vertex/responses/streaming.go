package vertexresponses

import (
	"bufio"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"sort"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/converter/responses"
	"github.com/mixaill76/auto_ai_router/internal/converter/vertex"
	"google.golang.org/genai"
)

// vertexStreamAccumulator tracks state across Vertex SSE streaming chunks.
type vertexStreamAccumulator struct {
	responseID string
	model      string
	createdAt  int64
	meta       *responses.ResponsesMetadata

	// Accumulated content
	fullText       string
	fullReasoning  string
	toolCalls      []accumulatedCall
	images         []accumulatedImage
	codeCallID     string
	codeCallCode   string
	codeCallOutput string

	// Completed output items (for building the final Response)
	// outputItems []responses.OutputItem

	// Usage from the final chunk
	usage *genai.GenerateContentResponseUsageMetadata

	// Finish reason (string)
	finishReason string

	// SSE state
	headerEmitted        bool
	messageStarted       bool
	messageItemID        string
	messageOutputIndex   int // output_index where the message item was placed
	reasoningStarted     bool
	reasoningItemID      string
	reasoningOutputIndex int // output_index where the reasoning item was placed
	codeOutputIndex      int // output_index where the code interpreter item was placed
	sequenceNumber       int
}

type accumulatedCall struct {
	callID      string
	name        string
	arguments   string
	itemID      string
	outputIndex int // output_index at which the item was announced
}

type accumulatedImage struct {
	itemID      string
	result      string
	format      string
	outputIndex int
}

// TransformVertexStreamToResponses reads Vertex AI SSE and writes Responses API SSE events.
// onComplete is called with the fully-built Response once the stream ends.
func TransformVertexStreamToResponses(
	reader io.Reader,
	writer io.Writer,
	model, responseID string,
	meta *responses.ResponsesMetadata,
	onComplete func(*responses.Response),
) error {
	if responseID == "" {
		responseID = generateResponseID()
	}
	acc := &vertexStreamAccumulator{
		responseID: responseID,
		model:      model,
		createdAt:  time.Now().Unix(),
		meta:       meta,
	}

	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	lineCount := 0
	for scanner.Scan() {
		line := scanner.Text()
		lineCount++

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var chunk vertex.VertexStreamingChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			slog.Debug("[vertexresponses/streaming] failed to parse chunk", "error", err)
			continue
		}

		if chunk.UsageMetadata != nil {
			acc.usage = chunk.UsageMetadata
		}

		if len(chunk.Candidates) == 0 {
			continue
		}

		candidate := chunk.Candidates[0]
		if candidate.FinishReason != genai.FinishReasonUnspecified {
			acc.finishReason = string(candidate.FinishReason)
		}

		if candidate.Content == nil {
			continue
		}

		for _, part := range candidate.Content.Parts {
			if err := processPart(writer, acc, part); err != nil {
				return err
			}
		}
	}

	if err := scanner.Err(); err != nil && err != io.EOF {
		return fmt.Errorf("stream read error: %w", err)
	}

	// Emit completion events.
	if err := emitVertexCompletionEvents(writer, acc); err != nil {
		return err
	}

	if onComplete != nil {
		onComplete(buildVertexCompletedResponse(acc))
	}

	return nil
}

// processPart handles a single genai.Part from a streaming chunk.
func processPart(w io.Writer, acc *vertexStreamAccumulator, part *genai.Part) error {
	switch {
	case part.Thought && part.Text != "":
		return processThoughtDelta(w, acc, part.Text)

	case part.FunctionCall != nil:
		return processFunctionCallPart(w, acc, part.FunctionCall)

	case part.ExecutableCode != nil:
		return processCodePart(w, acc, part.ExecutableCode)

	case part.CodeExecutionResult != nil:
		acc.codeCallOutput += part.CodeExecutionResult.Output
		return nil

	case part.InlineData != nil:
		return processImagePart(w, acc, part.InlineData)

	case part.Text != "":
		return processTextDelta(w, acc, part.Text)
	}
	return nil
}

func processImagePart(w io.Writer, acc *vertexStreamAccumulator, blob *genai.Blob) error {
	if blob == nil || !strings.HasPrefix(strings.ToLower(blob.MIMEType), "image/") {
		return nil
	}
	if !acc.headerEmitted {
		if err := emitVertexHeaderEvents(w, acc); err != nil {
			return err
		}
	}

	image := accumulatedImage{
		itemID:      generateItemID("ig_"),
		result:      base64.StdEncoding.EncodeToString(blob.Data),
		format:      imageOutputFormat(blob.MIMEType),
		outputIndex: currentOutputIndex(acc),
	}
	acc.images = append(acc.images, image)

	return writeVertexSSE(w, "response.output_item.added", map[string]interface{}{
		"type":         "response.output_item.added",
		"output_index": image.outputIndex,
		"item": map[string]interface{}{
			"type":          "image_generation_call",
			"id":            image.itemID,
			"status":        "in_progress",
			"output_format": image.format,
		},
	}, acc)
}

func processTextDelta(w io.Writer, acc *vertexStreamAccumulator, delta string) error {
	if !acc.headerEmitted {
		if err := emitVertexHeaderEvents(w, acc); err != nil {
			return err
		}
	}
	if !acc.messageStarted {
		if err := emitVertexMessageStart(w, acc); err != nil {
			return err
		}
	}
	acc.fullText += delta
	return writeVertexSSE(w, "response.output_text.delta",
		responses.BuildOutputTextDeltaEvent(acc.messageItemID, acc.messageOutputIndex, 0, delta), acc)
}

func processThoughtDelta(w io.Writer, acc *vertexStreamAccumulator, delta string) error {
	if !acc.headerEmitted {
		if err := emitVertexHeaderEvents(w, acc); err != nil {
			return err
		}
	}
	if !acc.reasoningStarted {
		// Capture index BEFORE setting reasoningStarted — currentOutputIndex counts it.
		outputIdx := currentOutputIndex(acc)
		acc.reasoningStarted = true
		acc.reasoningItemID = generateItemID("rs_")
		acc.reasoningOutputIndex = outputIdx
		if err := writeVertexSSE(w, "response.output_item.added", map[string]interface{}{
			"type":         "response.output_item.added",
			"output_index": outputIdx,
			"item": map[string]interface{}{
				"type":    "reasoning",
				"id":      acc.reasoningItemID,
				"status":  "in_progress",
				"summary": []interface{}{},
			},
		}, acc); err != nil {
			return err
		}
	}
	acc.fullReasoning += delta
	// Reasoning deltas have no standardized SSE type yet — accumulate only.
	return nil
}

func processFunctionCallPart(w io.Writer, acc *vertexStreamAccumulator, fc *genai.FunctionCall) error {
	if !acc.headerEmitted {
		if err := emitVertexHeaderEvents(w, acc); err != nil {
			return err
		}
	}
	argsJSON := "{}"
	if fc.Args != nil {
		if b, err := json.Marshal(fc.Args); err == nil {
			argsJSON = string(b)
		}
	}
	callID := fc.ID
	if callID == "" {
		callID = generateItemID("call_")
	}
	itemID := generateItemID("fc_")
	outputIdx := currentOutputIndex(acc)

	// Emit function_call output item
	if err := writeVertexSSE(w, "response.output_item.added", map[string]interface{}{
		"type":         "response.output_item.added",
		"output_index": outputIdx,
		"item": map[string]interface{}{
			"type":      "function_call",
			"id":        itemID,
			"status":    "in_progress",
			"call_id":   callID,
			"name":      fc.Name,
			"arguments": "",
		},
	}, acc); err != nil {
		return err
	}

	// Emit full arguments as a single delta, then immediately close.
	if err := writeVertexSSE(w, "response.function_call_arguments.delta", map[string]interface{}{
		"type":         "response.function_call_arguments.delta",
		"item_id":      itemID,
		"output_index": outputIdx,
		"delta":        argsJSON,
	}, acc); err != nil {
		return err
	}
	if err := writeVertexSSE(w, "response.function_call_arguments.done", map[string]interface{}{
		"type":         "response.function_call_arguments.done",
		"item_id":      itemID,
		"output_index": outputIdx,
		"arguments":    argsJSON,
	}, acc); err != nil {
		return err
	}

	acc.toolCalls = append(acc.toolCalls, accumulatedCall{
		callID:      callID,
		name:        fc.Name,
		arguments:   argsJSON,
		itemID:      itemID,
		outputIndex: outputIdx,
	})
	return nil
}

func processCodePart(w io.Writer, acc *vertexStreamAccumulator, ec *genai.ExecutableCode) error {
	if !acc.headerEmitted {
		if err := emitVertexHeaderEvents(w, acc); err != nil {
			return err
		}
	}
	// Capture index BEFORE setting codeCallID — currentOutputIndex counts it.
	outputIdx := currentOutputIndex(acc)
	acc.codeCallID = generateItemID("ci_")
	acc.codeCallCode = ec.Code
	acc.codeOutputIndex = outputIdx
	return writeVertexSSE(w, "response.output_item.added", map[string]interface{}{
		"type":         "response.output_item.added",
		"output_index": outputIdx,
		"item": map[string]interface{}{
			"type":   "code_interpreter_call",
			"id":     acc.codeCallID,
			"status": "in_progress",
			"code":   ec.Code,
		},
	}, acc)
}

// currentOutputIndex returns the next output item index based on what's been started.
func currentOutputIndex(acc *vertexStreamAccumulator) int {
	idx := 0
	if acc.reasoningStarted {
		idx++
	}
	if acc.messageStarted {
		idx++
	}
	idx += len(acc.toolCalls)
	idx += len(acc.images)
	if acc.codeCallID != "" {
		idx++
	}
	return idx
}

func emitVertexHeaderEvents(w io.Writer, acc *vertexStreamAccumulator) error {
	acc.headerEmitted = true
	respObj := buildVertexInProgressResponse(acc)
	if err := writeVertexSSE(w, "response.created",
		responses.BuildResponseEvent("response.created", respObj), acc); err != nil {
		return err
	}
	return writeVertexSSE(w, "response.in_progress",
		responses.BuildResponseEvent("response.in_progress", respObj), acc)
}

func emitVertexMessageStart(w io.Writer, acc *vertexStreamAccumulator) error {
	acc.messageStarted = true
	acc.messageItemID = generateItemID("msg_")
	outputIdx := currentOutputIndex(acc) - 1 // -1 because messageStarted was just set
	acc.messageOutputIndex = outputIdx       // save for text delta events

	if err := writeVertexSSE(w, "response.output_item.added",
		responses.BuildMessageItemAddedEvent(outputIdx, acc.messageItemID), acc); err != nil {
		return err
	}

	return writeVertexSSE(w, "response.content_part.added",
		responses.BuildContentPartAddedEvent(acc.messageItemID, outputIdx, 0), acc)
}

func emitVertexCompletionEvents(w io.Writer, acc *vertexStreamAccumulator) error {
	if !acc.headerEmitted {
		if err := emitVertexHeaderEvents(w, acc); err != nil {
			return err
		}
	}

	// Build close operations keyed by the output_index at which each item was
	// announced during streaming, then sort so done-events mirror added-events.
	type closureItem struct {
		outputIndex int
		fn          func() error
	}
	var closures []closureItem

	if acc.reasoningStarted && acc.fullReasoning != "" {
		idx := acc.reasoningOutputIndex
		closures = append(closures, closureItem{
			outputIndex: idx,
			fn: func() error {
				return writeVertexSSE(w, "response.output_item.done", map[string]interface{}{
					"type":         "response.output_item.done",
					"output_index": idx,
					"item": map[string]interface{}{
						"type":   "reasoning",
						"id":     acc.reasoningItemID,
						"status": "completed",
						"summary": []interface{}{
							map[string]interface{}{"type": "summary_text", "text": acc.fullReasoning},
						},
					},
				}, acc)
			},
		})
	}

	if acc.messageStarted && acc.fullText != "" {
		idx := acc.messageOutputIndex
		closures = append(closures, closureItem{
			outputIndex: idx,
			fn: func() error {
				if err := writeVertexSSE(w, "response.output_text.done",
					responses.BuildOutputTextDoneEvent(acc.messageItemID, idx, 0, acc.fullText), acc); err != nil {
					return err
				}
				if err := writeVertexSSE(w, "response.content_part.done",
					responses.BuildContentPartDoneEvent(acc.messageItemID, idx, 0, acc.fullText), acc); err != nil {
					return err
				}
				return writeVertexSSE(w, "response.output_item.done", map[string]interface{}{
					"type":         "response.output_item.done",
					"output_index": idx,
					"item": map[string]interface{}{
						"type":    "message",
						"id":      acc.messageItemID,
						"status":  "completed",
						"role":    "assistant",
						"content": []interface{}{map[string]interface{}{"type": "output_text", "text": acc.fullText, "annotations": []interface{}{}}},
					},
				}, acc)
			},
		})
	}

	for _, tc := range acc.toolCalls {
		tc := tc // capture for closure
		closures = append(closures, closureItem{
			outputIndex: tc.outputIndex,
			fn: func() error {
				return writeVertexSSE(w, "response.output_item.done", map[string]interface{}{
					"type":         "response.output_item.done",
					"output_index": tc.outputIndex,
					"item": map[string]interface{}{
						"type":      "function_call",
						"id":        tc.itemID,
						"status":    "completed",
						"call_id":   tc.callID,
						"name":      tc.name,
						"arguments": tc.arguments,
					},
				}, acc)
			},
		})
	}

	for _, image := range acc.images {
		image := image
		closures = append(closures, closureItem{
			outputIndex: image.outputIndex,
			fn: func() error {
				return writeVertexSSE(w, "response.output_item.done", map[string]interface{}{
					"type":         "response.output_item.done",
					"output_index": image.outputIndex,
					"item": map[string]interface{}{
						"type":          "image_generation_call",
						"id":            image.itemID,
						"status":        "completed",
						"result":        image.result,
						"output_format": image.format,
					},
				}, acc)
			},
		})
	}

	if acc.codeCallID != "" {
		idx := acc.codeOutputIndex
		closures = append(closures, closureItem{
			outputIndex: idx,
			fn: func() error {
				return writeVertexSSE(w, "response.output_item.done", map[string]interface{}{
					"type":         "response.output_item.done",
					"output_index": idx,
					"item": map[string]interface{}{
						"type":    "code_interpreter_call",
						"id":      acc.codeCallID,
						"status":  "completed",
						"code":    acc.codeCallCode,
						"outputs": []interface{}{map[string]interface{}{"type": "text", "text": acc.codeCallOutput}},
					},
				}, acc)
			},
		})
	}

	sort.Slice(closures, func(i, j int) bool {
		return closures[i].outputIndex < closures[j].outputIndex
	})

	for _, c := range closures {
		if err := c.fn(); err != nil {
			return err
		}
	}

	return writeVertexSSE(w, "response.completed",
		responses.BuildResponseEvent("response.completed", buildVertexCompletedResponse(acc)), acc)
}

func buildVertexInProgressResponse(acc *vertexStreamAccumulator) map[string]interface{} {
	resp := responses.BuildInProgressResponse(acc.responseID, acc.model, acc.createdAt)
	if acc.meta != nil {
		resp["store"] = acc.meta.Store
		if acc.meta.PreviousResponseID != "" {
			resp["previous_response_id"] = acc.meta.PreviousResponseID
		}
	}
	return resp
}

func buildVertexCompletedResponse(acc *vertexStreamAccumulator) *responses.Response {
	status, incompleteDetails := vertexFinishReasonToStatus(acc.finishReason)

	type indexedOutputItem struct {
		index int
		item  responses.OutputItem
	}
	var indexedOutput []indexedOutputItem
	if acc.reasoningStarted && acc.fullReasoning != "" {
		indexedOutput = append(indexedOutput, indexedOutputItem{
			index: acc.reasoningOutputIndex,
			item: responses.OutputItem{
				Type:   "reasoning",
				ID:     acc.reasoningItemID,
				Status: "completed",
				Summary: []responses.OutputContent{
					{Type: "summary_text", Text: acc.fullReasoning},
				},
			},
		})
	}
	if acc.messageStarted {
		indexedOutput = append(indexedOutput, indexedOutputItem{
			index: acc.messageOutputIndex,
			item: responses.OutputItem{
				Type:    "message",
				ID:      acc.messageItemID,
				Status:  "completed",
				Role:    "assistant",
				Content: []responses.OutputContent{{Type: "output_text", Text: acc.fullText, Annotations: []responses.Annotation{}}},
			},
		})
	}
	for _, tc := range acc.toolCalls {
		indexedOutput = append(indexedOutput, indexedOutputItem{
			index: tc.outputIndex,
			item: responses.OutputItem{
				Type:      "function_call",
				ID:        tc.itemID,
				Status:    "completed",
				CallID:    tc.callID,
				Name:      tc.name,
				Arguments: tc.arguments,
			},
		})
	}
	for _, image := range acc.images {
		indexedOutput = append(indexedOutput, indexedOutputItem{
			index: image.outputIndex,
			item: responses.OutputItem{
				Type:         "image_generation_call",
				ID:           image.itemID,
				Status:       "completed",
				Result:       image.result,
				OutputFormat: image.format,
			},
		})
	}
	if acc.codeCallID != "" {
		indexedOutput = append(indexedOutput, indexedOutputItem{
			index: acc.codeOutputIndex,
			item: responses.OutputItem{
				Type:    "code_interpreter_call",
				ID:      acc.codeCallID,
				Status:  "completed",
				Code:    acc.codeCallCode,
				Outputs: []map[string]interface{}{{"type": "text", "text": acc.codeCallOutput}},
			},
		})
	}
	sort.SliceStable(indexedOutput, func(i, j int) bool {
		return indexedOutput[i].index < indexedOutput[j].index
	})
	output := make([]responses.OutputItem, 0, len(indexedOutput))
	for _, indexed := range indexedOutput {
		output = append(output, indexed.item)
	}
	if len(output) == 0 {
		output = []responses.OutputItem{{
			Type:    "message",
			ID:      generateItemID("msg_"),
			Status:  "completed",
			Role:    "assistant",
			Content: []responses.OutputContent{{Type: "output_text", Text: "", Annotations: []responses.Annotation{}}},
		}}
	}

	var usage *responses.Usage
	if acc.usage != nil {
		usage = usageMetadataToUsage(acc.usage)
	}

	completedAt := acc.createdAt
	metadata := map[string]string{}
	var prevRespID interface{}
	if acc.meta != nil {
		for k, v := range acc.meta.Metadata {
			metadata[k] = v
		}
		if acc.meta.PreviousResponseID != "" {
			prevRespID = acc.meta.PreviousResponseID
		}
	}

	return responses.BuildCompletedResponse(responses.CompletedResponseParams{
		ID:                 acc.responseID,
		Model:              acc.model,
		CreatedAt:          acc.createdAt,
		CompletedAt:        &completedAt,
		Status:             status,
		IncompleteDetails:  incompleteDetails,
		Output:             output,
		Usage:              usage,
		Metadata:           metadata,
		PreviousResponseID: prevRespID,
	})
}

func vertexFinishReasonToStatus(reason string) (string, *responses.IncompleteDetails) {
	switch reason {
	case string(genai.FinishReasonMaxTokens):
		return "incomplete", &responses.IncompleteDetails{Reason: "max_output_tokens"}
	case string(genai.FinishReasonSafety), string(genai.FinishReasonRecitation):
		return "incomplete", &responses.IncompleteDetails{Reason: "content_filter"}
	default:
		return "completed", nil
	}
}

func writeVertexSSE(w io.Writer, eventType string, data interface{}, acc *vertexStreamAccumulator) error {
	return responses.WriteSSEEvent(w, eventType, data, &acc.sequenceNumber)
}
