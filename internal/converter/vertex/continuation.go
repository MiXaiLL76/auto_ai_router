package vertex

import (
	"bytes"
	"encoding/json"
	"fmt"

	"google.golang.org/genai"
)

// Long decode (Gemini 4). A generation longer than one request's output cap --
// 65,536 tokens streamed, 32,768 for a plain generateContent call -- stops with
// finish reason CONTINUATION and a continuation token. The same request sent again
// with that token as its top-level continuationToken, to the same account (v1beta1
// on Vertex AI), resumes it: the partial answer is not added to contents, and a
// request may have spent all of its tokens thinking and returned no text. The
// requests are billed one by one; each reports its usage cumulatively within itself,
// re-reads the prompt and the output so far as input (mostly as implicit cache hits)
// and searches on its own, so the generation's usage and search queries are the sums
// over its requests. The proxy sends the requests; this file holds the formats.

// MaxContinuations bounds the follow-up requests of one generation: its whole
// 1,048,576-token budget at 32,768 tokens a request takes 32 of them.
const MaxContinuations = 40

// SupportsLongDecode reports whether model continues a long generation with
// continuation tokens.
func SupportsLongDecode(model string) bool {
	profile := lookupGeminiModelProfile(model)
	return profile != nil && profile.longDecode
}

// continuationProbe is the part of a response or stream chunk that decides
// whether the generation goes on.
type continuationProbe struct {
	Candidates []struct {
		FinishReason      string `json:"finishReason"`
		ContinuationToken string `json:"continuationToken"`
	} `json:"candidates"`
}

// ReadContinuation returns the first candidate's finish reason and continuation
// token from a generateContent response or one streamed chunk; ok is false when
// payload is not such a JSON object. The token is kept as sent (base64 text), to
// go back unchanged.
func ReadContinuation(payload []byte) (finishReason, token string, ok bool) {
	var probe continuationProbe
	if err := json.Unmarshal(payload, &probe); err != nil {
		return "", "", false
	}
	if len(probe.Candidates) == 0 {
		return "", "", true
	}
	return probe.Candidates[0].FinishReason, probe.Candidates[0].ContinuationToken, true
}

// IsContinuation reports whether a finish reason asks for a continuation request.
// A token without a finish reason only marks a checkpoint of a running stream.
func IsContinuation(finishReason string) bool {
	return finishReason == string(genai.FinishReasonContinuation)
}

// ContinuationRequest returns the request body with token set as its top-level
// continuationToken; everything else -- contents, systemInstruction, tools,
// generationConfig with maxOutputTokens -- goes again unchanged.
func ContinuationRequest(body []byte, token string) ([]byte, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, fmt.Errorf("continuation request: %w", err)
	}
	encoded, err := json.Marshal(token)
	if err != nil {
		return nil, fmt.Errorf("continuation request: %w", err)
	}
	fields["continuationToken"] = encoded
	return json.Marshal(fields)
}

// MergeContinuationResponses joins the generateContent responses of one continued
// generation into the response the client gets: each candidate's parts in order,
// the last request's finish reason, the grounding and citations of all requests,
// and usage summed over the requests.
func MergeContinuationResponses(bodies [][]byte) ([]byte, error) {
	if len(bodies) == 1 {
		return bodies[0], nil
	}
	var merged *genai.GenerateContentResponse
	var initialPromptTokens int32
	for i, body := range bodies {
		var resp genai.GenerateContentResponse
		if err := json.Unmarshal(body, &resp); err != nil {
			return nil, fmt.Errorf("continuation response %d: %w", i, err)
		}
		if merged == nil {
			merged = &resp
			if resp.UsageMetadata != nil {
				initialPromptTokens = resp.UsageMetadata.PromptTokenCount
			}
			continue
		}
		mergeContinuationResponse(merged, &resp)
	}
	if merged == nil {
		return nil, fmt.Errorf("no continuation responses")
	}
	body, err := json.Marshal(merged)
	if err != nil || initialPromptTokens <= 0 {
		return body, err
	}
	// The summed usage counts the prompt once per request; the size of the
	// context the client sent rides along under a field of our own.
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, err
	}
	fields[initialPromptTokensField] = json.RawMessage(fmt.Sprint(initialPromptTokens))
	return json.Marshal(fields)
}

// initialPromptTokensField is the router's own top-level field of a merged
// response: the first request's promptTokenCount (see InitialPromptTokens).
const initialPromptTokensField = "initialPromptTokenCount"

// InitialPromptTokens returns the prompt of the first request of a generation
// that MergeContinuationResponses joined from several, or 0 for any other body.
func InitialPromptTokens(body []byte) int32 {
	if !bytes.Contains(body, []byte(`"`+initialPromptTokensField+`"`)) {
		return 0
	}
	var probe struct {
		InitialPromptTokenCount int32 `json:"initialPromptTokenCount"`
	}
	if json.Unmarshal(body, &probe) != nil {
		return 0
	}
	return probe.InitialPromptTokenCount
}

func mergeContinuationResponse(merged, next *genai.GenerateContentResponse) {
	for i, candidate := range next.Candidates {
		if candidate == nil {
			continue
		}
		switch {
		case i >= len(merged.Candidates):
			merged.Candidates = append(merged.Candidates, candidate)
		case merged.Candidates[i] == nil:
			merged.Candidates[i] = candidate
		default:
			mergeContinuationCandidate(merged.Candidates[i], candidate)
		}
	}
	merged.UsageMetadata = AddUsageMetadata(merged.UsageMetadata, next.UsageMetadata)
	if next.ModelVersion != "" {
		merged.ModelVersion = next.ModelVersion
	}
	if next.ResponseID != "" {
		merged.ResponseID = next.ResponseID
	}
	if merged.PromptFeedback == nil {
		merged.PromptFeedback = next.PromptFeedback
	}
}

func mergeContinuationCandidate(merged, next *genai.Candidate) {
	partOffset := 0
	if merged.Content == nil {
		merged.Content = next.Content
	} else if next.Content != nil {
		partOffset = len(merged.Content.Parts)
		merged.Content.Parts = append(merged.Content.Parts, next.Content.Parts...)
	}
	merged.FinishReason = next.FinishReason
	merged.FinishMessage = next.FinishMessage
	merged.ContinuationToken = next.ContinuationToken
	merged.TokenCount += next.TokenCount
	if next.AvgLogprobs != 0 {
		merged.AvgLogprobs = next.AvgLogprobs
	}
	if len(next.SafetyRatings) > 0 {
		merged.SafetyRatings = next.SafetyRatings
	}
	merged.GroundingMetadata = mergeGroundingMetadata(merged.GroundingMetadata, next.GroundingMetadata, partOffset)
	if next.CitationMetadata != nil {
		if merged.CitationMetadata == nil {
			merged.CitationMetadata = &genai.CitationMetadata{}
		}
		merged.CitationMetadata.Citations = append(merged.CitationMetadata.Citations, next.CitationMetadata.Citations...)
	}
	if next.URLContextMetadata != nil {
		if merged.URLContextMetadata == nil {
			merged.URLContextMetadata = &genai.URLContextMetadata{}
		}
		merged.URLContextMetadata.URLMetadata = append(merged.URLContextMetadata.URLMetadata, next.URLContextMetadata.URLMetadata...)
	}
	if next.LogprobsResult != nil {
		if merged.LogprobsResult == nil {
			merged.LogprobsResult = &genai.LogprobsResult{}
		}
		merged.LogprobsResult.ChosenCandidates = append(merged.LogprobsResult.ChosenCandidates, next.LogprobsResult.ChosenCandidates...)
		merged.LogprobsResult.TopCandidates = append(merged.LogprobsResult.TopCandidates, next.LogprobsResult.TopCandidates...)
	}
}

// mergeGroundingMetadata appends next's grounding to merged. next's supports point
// at its own parts and chunks, which follow merged's after the merge: partOffset
// parts and len(merged.GroundingChunks) chunks further on.
func mergeGroundingMetadata(merged, next *genai.GroundingMetadata, partOffset int) *genai.GroundingMetadata {
	if next == nil {
		return merged
	}
	if merged == nil {
		merged = &genai.GroundingMetadata{}
	}
	chunkOffset := ClampInt32(len(merged.GroundingChunks))
	merged.WebSearchQueries = append(merged.WebSearchQueries, next.WebSearchQueries...)
	merged.ImageSearchQueries = append(merged.ImageSearchQueries, next.ImageSearchQueries...)
	merged.RetrievalQueries = append(merged.RetrievalQueries, next.RetrievalQueries...)
	merged.GroundingChunks = append(merged.GroundingChunks, next.GroundingChunks...)
	merged.SourceFlaggingUris = append(merged.SourceFlaggingUris, next.SourceFlaggingUris...)
	for _, support := range next.GroundingSupports {
		if support == nil {
			continue
		}
		shifted := *support
		shifted.GroundingChunkIndices = make([]int32, len(support.GroundingChunkIndices))
		for i, index := range support.GroundingChunkIndices {
			shifted.GroundingChunkIndices[i] = index + chunkOffset
		}
		if support.Segment != nil {
			segment := *support.Segment
			segment.PartIndex += ClampInt32(partOffset)
			shifted.Segment = &segment
		}
		merged.GroundingSupports = append(merged.GroundingSupports, &shifted)
	}
	if next.SearchEntryPoint != nil {
		merged.SearchEntryPoint = next.SearchEntryPoint
	}
	if next.RetrievalMetadata != nil {
		merged.RetrievalMetadata = next.RetrievalMetadata
	}
	if next.GoogleMapsWidgetContextToken != "" {
		merged.GoogleMapsWidgetContextToken = next.GoogleMapsWidgetContextToken
	}
	return merged
}

// AddUsageMetadata returns the usage of two upstream requests together: every
// count summed, the modality breakdowns summed per modality. Either may be nil.
// Each request's usage must be its final one -- a stream's usage is cumulative
// within the stream, so summing its chunks would count tokens many times over.
func AddUsageMetadata(a, b *genai.GenerateContentResponseUsageMetadata) *genai.GenerateContentResponseUsageMetadata {
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}
	sum := &genai.GenerateContentResponseUsageMetadata{
		CacheTokensDetails:         addModalityTokenCounts(a.CacheTokensDetails, b.CacheTokensDetails),
		CachedContentTokenCount:    a.CachedContentTokenCount + b.CachedContentTokenCount,
		CandidatesTokenCount:       a.CandidatesTokenCount + b.CandidatesTokenCount,
		CandidatesTokensDetails:    addModalityTokenCounts(a.CandidatesTokensDetails, b.CandidatesTokensDetails),
		PromptTokenCount:           a.PromptTokenCount + b.PromptTokenCount,
		PromptTokensDetails:        addModalityTokenCounts(a.PromptTokensDetails, b.PromptTokensDetails),
		ThoughtsTokenCount:         a.ThoughtsTokenCount + b.ThoughtsTokenCount,
		ToolUsePromptTokenCount:    a.ToolUsePromptTokenCount + b.ToolUsePromptTokenCount,
		ToolUsePromptTokensDetails: addModalityTokenCounts(a.ToolUsePromptTokensDetails, b.ToolUsePromptTokensDetails),
		TotalTokenCount:            a.TotalTokenCount + b.TotalTokenCount,
		TrafficType:                b.TrafficType,
	}
	if sum.TrafficType == "" {
		sum.TrafficType = a.TrafficType
	}
	return sum
}

func addModalityTokenCounts(a, b []*genai.ModalityTokenCount) []*genai.ModalityTokenCount {
	if len(a) == 0 && len(b) == 0 {
		return nil
	}
	var order []genai.MediaModality
	counts := make(map[genai.MediaModality]int32)
	for _, details := range [][]*genai.ModalityTokenCount{a, b} {
		for _, detail := range details {
			if detail == nil {
				continue
			}
			if _, seen := counts[detail.Modality]; !seen {
				order = append(order, detail.Modality)
			}
			counts[detail.Modality] += detail.TokenCount
		}
	}
	sum := make([]*genai.ModalityTokenCount, 0, len(order))
	for _, modality := range order {
		sum = append(sum, &genai.ModalityTokenCount{Modality: modality, TokenCount: counts[modality]})
	}
	return sum
}

// ContinuationUsage follows the usage of a generation streamed over one or more
// continued requests. Within a request each chunk's usage is cumulative and replaces
// the previous one; a CONTINUATION finish closes the request, and the next one's
// usage is added on top.
type ContinuationUsage struct {
	closed  *genai.GenerateContentResponseUsageMetadata // requests already finished
	current *genai.GenerateContentResponseUsageMetadata // latest usage of the running request
	// InitialPromptTokens is the first request's prompt, the size of the context the
	// client sent: later requests read the output so far again as prompt.
	InitialPromptTokens int32
}

// Observe records one chunk's usage and returns the generation's usage so far.
func (u *ContinuationUsage) Observe(meta *genai.GenerateContentResponseUsageMetadata) *genai.GenerateContentResponseUsageMetadata {
	if meta != nil {
		u.current = meta
		if u.closed == nil && meta.PromptTokenCount > 0 {
			u.InitialPromptTokens = meta.PromptTokenCount
		}
	}
	return u.Total()
}

// CloseRequest ends the running request after its CONTINUATION finish.
func (u *ContinuationUsage) CloseRequest() {
	u.closed = AddUsageMetadata(u.closed, u.current)
	u.current = nil
}

// Total returns the usage of the finished requests plus the running one's latest.
func (u *ContinuationUsage) Total() *genai.GenerateContentResponseUsageMetadata {
	return AddUsageMetadata(u.closed, u.current)
}

// Continued reports whether the generation has gone past its first request.
func (u *ContinuationUsage) Continued() bool {
	return u.closed != nil
}
