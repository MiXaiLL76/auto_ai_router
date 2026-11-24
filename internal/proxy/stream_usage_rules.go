package proxy

import "github.com/mixaill76/auto_ai_router/internal/converter"

// streamUsageResolution is the provider-report state a streaming spend writer
// has to apply to the counts it merged while the stream passed. The finalizer
// (finalizeStreamingLogParsed) and the proxy-like writer
// (writeProxyStreamingResponseWithTokens) both keep this state, in different
// shapes; resolveStreamUsage is the one implementation of the rules so they
// cannot drift apart.
type streamUsageResolution struct {
	// usage holds the counts merged (MergeNonZero) as the stream passed. It is
	// mutated: a stale or corrected completion count is zeroed before the
	// local estimate is applied.
	usage *converter.TokenUsage
	// tracked is the last usage frame the provider's extractor saw, and
	// trackedProgress the accumulator's AnswerProgress at that moment. It is the
	// provider's last word only when no answer text followed it.
	tracked         *StreamUsageInfo
	trackedProgress int
	// trackedIsFinal marks a tracked frame that is also the stream's actual
	// final data chunk (the finalizer's already-parsed lastChunk). Such a frame
	// is never stale, even if the local estimate happens to be larger.
	trackedIsFinal bool
	// finalInfo is the stream's final data payload's usage. When it carries a
	// field, that field is the provider's last word: authoritative, zero
	// included.
	finalInfo *StreamUsageInfo
	// answerProgress is the accumulator's current AnswerProgress, answerTokens
	// the local completion estimate.
	answerProgress int
	answerTokens   int
	// promptEstimate is the lazy local prompt-token fallback. A function so a
	// provider-reported prompt count never pays for the tokenizer.
	promptEstimate func() int
}

// streamUsageOutcome is what the callers still need for sourcing the spend row.
type streamUsageOutcome struct {
	// providerUsage is true when the provider reported a usage object at all,
	// even an all-zero one.
	providerUsage bool
	// explicitZero is true when the provider's coherent all-zero report (both
	// fields present as zero in the stream's final data payload) is
	// authoritative: the request still bills its prompt as zero, not the local
	// estimate.
	explicitZero bool
}

// resolveStreamUsage applies the provider-report rules shared by both streaming
// spend writers and returns the usage to bill (creating it when the stream
// never produced one).
//
// A field the provider reported is never replaced by an estimate; a field it did
// not report falls back to the local estimate, except in an all-zero report. A
// tracked frame that predates streamed answer text is stale — a zero there said
// "nothing so far", and a positive count that the streamed answer outgrew was a
// running count — and its completion count is dropped so the estimate applies. A
// positive count larger than the local estimate is a final total sent early and
// is kept.
func resolveStreamUsage(in streamUsageResolution) (*converter.TokenUsage, streamUsageOutcome) {
	usage := in.usage
	if usage == nil {
		usage = &converter.TokenUsage{}
	}

	providerUsage := in.tracked != nil || in.finalInfo != nil ||
		usage.PromptTokens > 0 || usage.CompletionTokens > 0
	promptReported := usage.PromptTokens > 0
	completionReported := usage.CompletionTokens > 0

	if in.tracked != nil {
		if in.tracked.PromptTokens > 0 {
			promptReported = true
		}
		if in.tracked.CompletionTokensPresent {
			// A frame is the provider's last word unless answer text arrived
			// after it. A zero then said "nothing so far". A positive count is
			// ambiguous: a provider may send its final total before the answer
			// (the count then exceeds the local estimate), or a running count
			// that later text outgrew (the estimate exceeds the count). Trust
			// the count unless the streamed answer proves it too low.
			stale := !in.trackedIsFinal &&
				in.trackedProgress < in.answerProgress &&
				(in.tracked.CompletionTokens == 0 || in.tracked.CompletionTokens < in.answerTokens)
			switch {
			case stale:
				// The provider's count is not the final count. Drop it and let
				// the local estimate speak for the text that followed.
				usage.CompletionTokens = 0
				completionReported = false
			case in.tracked.CompletionTokens == 0:
				// An explicit zero in the provider's last word wins over whatever
				// an intermediate frame reported: MergeNonZero skipped that zero
				// while the stream passed, so the earlier non-zero survived.
				usage.CompletionTokens = 0
				completionReported = true
			default:
				completionReported = true
			}
		}
	}

	if in.finalInfo != nil {
		if in.finalInfo.PromptTokens > 0 {
			usage.PromptTokens = in.finalInfo.PromptTokens
			promptReported = true
		}
		if in.finalInfo.CompletionTokensPresent {
			usage.CompletionTokens = in.finalInfo.CompletionTokens
			completionReported = true
		}
	}

	// The all-zero report is authoritative only when it is the stream's final
	// data payload: a zero that a later usage-less frame (a Chat finish frame)
	// follows is a placeholder, not the provider's last word about the request.
	// A usage-bearing final payload that says both zero stays authoritative.
	explicitZero := in.finalInfo != nil &&
		in.finalInfo.PromptTokensPresent && in.finalInfo.PromptTokens == 0 &&
		in.finalInfo.CompletionTokensPresent && in.finalInfo.CompletionTokens == 0 &&
		usage.PromptTokens == 0 && usage.CompletionTokens == 0
	if !promptReported && !explicitZero && usage.PromptTokens == 0 && in.promptEstimate != nil {
		usage.PromptTokens = in.promptEstimate()
	}
	if !completionReported && !explicitZero && usage.CompletionTokens == 0 {
		usage.CompletionTokens = in.answerTokens
	}

	return usage, streamUsageOutcome{providerUsage: providerUsage, explicitZero: explicitZero}
}
