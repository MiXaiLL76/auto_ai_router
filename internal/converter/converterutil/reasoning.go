package converterutil

// PickReasoningField returns the reasoning value of a Chat Completions message
// or stream delta. OpenAI-compatible providers spell the field either
// "reasoning_content" (DeepSeek, SiliconFlow, LiteLLM) or "reasoning"
// (OpenRouter, vLLM, Ollama, Groq). reasoning_content wins unless it is absent,
// null or an empty string, so an empty reasoning_content never hides a
// populated reasoning.
//
// Values are taken as interface{}: callers decode these fields untyped so a
// provider sending a non-string there cannot fail the whole unmarshal.
func PickReasoningField(reasoningContent, reasoning interface{}) interface{} {
	if reasoningContent == nil {
		return reasoning
	}
	if s, ok := reasoningContent.(string); ok && s == "" && reasoning != nil {
		return reasoning
	}
	return reasoningContent
}

// ReasoningText is PickReasoningField narrowed to a string; a non-string
// value yields "".
func ReasoningText(reasoningContent, reasoning interface{}) string {
	text, _ := PickReasoningField(reasoningContent, reasoning).(string)
	return text
}
