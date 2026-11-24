package vertex

import (
	"encoding/base64"
	"unicode/utf8"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"google.golang.org/genai"
)

// SignedText also recognizes the signature-only text part at the end of a stream.
func SignedText(part *genai.Part) bool {
	return len(part.ThoughtSignature) > 0 && !part.Thought &&
		part.FunctionCall == nil && part.FunctionResponse == nil && part.ToolCall == nil && part.ToolResponse == nil &&
		part.InlineData == nil && part.FileData == nil && part.ExecutableCode == nil && part.CodeExecutionResult == nil
}

func recordTextSignature(fields **openai.MessageProviderFields, part *genai.Part, start int) {
	if *fields == nil {
		*fields = &openai.MessageProviderFields{}
	}
	(*fields).ThoughtSignatures = append((*fields).ThoughtSignatures, openai.TextSignature{
		Start: start, End: start + len(part.Text), Signature: base64.StdEncoding.EncodeToString(part.ThoughtSignature),
	})
}

// Chat exposes a single content string. Split it at the original signed-part
// boundaries when the client echoes the message. An edited or malformed range
// falls back to the client text; it must never truncate it or split a rune.
func signedChatParts(text string, signatures []openai.TextSignature) []*genai.Part {
	var parts []*genai.Part
	pos := 0
	for _, sig := range signatures {
		if sig.Start < pos || sig.End < sig.Start || sig.End > len(text) ||
			(sig.Start < len(text) && !utf8.RuneStart(text[sig.Start])) ||
			(sig.End < len(text) && !utf8.RuneStart(text[sig.End])) {
			return []*genai.Part{{Text: text}}
		}
		raw, err := base64.StdEncoding.DecodeString(sig.Signature)
		if err != nil || len(raw) == 0 {
			return []*genai.Part{{Text: text}}
		}
		if sig.Start > pos {
			parts = append(parts, &genai.Part{Text: text[pos:sig.Start]})
		}
		parts = append(parts, &genai.Part{Text: text[sig.Start:sig.End], ThoughtSignature: raw})
		pos = sig.End
	}
	if pos < len(text) {
		parts = append(parts, &genai.Part{Text: text[pos:]})
	}
	return parts
}
