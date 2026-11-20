package vertex

import (
	"fmt"

	converterutil "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"google.golang.org/genai"
)

// This file refuses, before anything goes upstream, the requests a model profile
// says Google answers with 400. Such a 400 is replayed on every credential of the
// rotation (see capabilities.go), so it is cheaper and clearer to answer it here.

// validateChatRequest checks an OpenAI chat request against model's profile.
func validateChatRequest(req *openai.OpenAIRequest, model string) error {
	profile := lookupGeminiModelProfile(model)
	if profile == nil {
		return nil
	}
	if profile.thinkingLevelsOnly {
		if err := validateNativeThinking(req); err != nil {
			return err
		}
	}
	if profile.singleCandidate {
		if req.N != nil && *req.N > 1 {
			return singleCandidateError("n")
		}
		if gcMap, ok := req.ExtraBody["generation_config"].(map[string]interface{}); ok {
			for _, key := range []string{"candidate_count", "candidateCount"} {
				if count, ok := thinkingBudgetValue(gcMap[key]); ok && count > 1 {
					return singleCandidateError(key)
				}
			}
		}
	}
	if req.MaxCompletionTokens != nil {
		return ValidateMaxOutputTokens(model, "max_completion_tokens", req.MaxCompletionTokens)
	}
	return ValidateMaxOutputTokens(model, "max_tokens", req.MaxTokens)
}

// validateNativeThinking refuses the Gemini-native thinking parameters a
// thinking-levels-only model rejects: any thinking_budget and the MINIMAL level.
// The OpenAI- and Anthropic-style ways to ask for little or no thinking
// (reasoning_effort minimal/none, thinking disabled) stay accepted and get the
// lowest level, see geminiModelProfile.thinkingLevel.
func validateNativeThinking(req *openai.OpenAIRequest) error {
	for _, tcMap := range nativeThinkingConfigs(req) {
		if thinkingConfigField(tcMap, "thinking_budget", "thinkingBudget") != nil {
			return thinkingBudgetError()
		}
		level, _ := thinkingConfigField(tcMap, "thinking_level", "thinkingLevel").(string)
		if thinkingName(level) == "minimal" {
			return minimalThinkingError()
		}
	}
	for _, budget := range []interface{}{req.ThinkingBudget, req.ExtraBody["thinking_budget"], req.ExtraBody["thinkingBudget"]} {
		if budget != nil {
			return thinkingBudgetError()
		}
	}
	for _, level := range []string{req.ThinkingLevel, req.ThinkingLevelCamel,
		extraBodyString(req.ExtraBody, "thinking_level"), extraBodyString(req.ExtraBody, "thinkingLevel")} {
		if thinkingName(level) == "minimal" {
			return minimalThinkingError()
		}
	}
	return nil
}

func thinkingBudgetError() error {
	return &converterutil.RequestValidationError{
		Param:   "thinking_budget",
		Code:    "unsupported_parameter",
		Message: "thinking_budget is not supported by this model; set reasoning_effort or thinking_level to low, medium or high",
	}
}

func minimalThinkingError() error {
	return &converterutil.RequestValidationError{
		Param:   "thinking_level",
		Code:    "unsupported_value",
		Message: "Thinking level minimal is not supported by this model; use low, medium or high (thinking cannot be turned off, low is the lowest level)",
	}
}

func singleCandidateError(param string) error {
	return &converterutil.RequestValidationError{
		Param:   param,
		Code:    "unsupported_value",
		Message: "This model returns a single candidate; " + param + " must be 1",
	}
}

// ValidateMaxOutputTokens refuses an output cap above the token limit a model
// shares between input and output: Google answers it with 400. maxTokens may be
// nil. Exported for use by sub-packages (e.g. vertex/responses).
func ValidateMaxOutputTokens(model, param string, maxTokens *int) error {
	profile := lookupGeminiModelProfile(model)
	if profile == nil || profile.tokenLimit <= 0 || maxTokens == nil || *maxTokens <= profile.tokenLimit {
		return nil
	}
	return &converterutil.RequestValidationError{
		Param: param,
		Code:  "invalid_value",
		Message: fmt.Sprintf("%s must be at most %d for this model: input and output (thinking included) share one %d-token limit",
			param, profile.tokenLimit, profile.tokenLimit),
	}
}

// ValidateLastTurn refuses a conversation that ends with a model turn for a model
// that does not take assistant prefill. param names the request field the turns
// came from. Exported for use by sub-packages (e.g. vertex/responses).
func ValidateLastTurn(model string, contents []*genai.Content, param string) error {
	profile := lookupGeminiModelProfile(model)
	if profile == nil || !profile.noModelPrefill || len(contents) == 0 {
		return nil
	}
	if last := contents[len(contents)-1]; last == nil || last.Role != "model" {
		return nil
	}
	return &converterutil.RequestValidationError{
		Param:   param,
		Code:    "invalid_value",
		Message: "This model does not support assistant prefill: the conversation must end with a user or tool message",
	}
}
