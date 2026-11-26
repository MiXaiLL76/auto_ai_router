package proxy

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"

	goccyjson "github.com/goccy/go-json"
)

// promptSessionRequest is the part of a request body the prompt-hash session ID is built from.
type promptSessionRequest struct {
	CacheControl map[string]any       `json:"cache_control"`
	Tools        any                  `json:"tools"`
	System       any                  `json:"system"`
	Instructions any                  `json:"instructions"`
	Messages     goccyjson.RawMessage `json:"messages"`
	Input        goccyjson.RawMessage `json:"input"`
}

// bindPromptSession sets a prompt-hash session ID on a cache_control request without one.
func (p *Proxy) bindPromptSession(logCtx *RequestLogContext, body []byte) {
	if !p.stickyPromptHash || p.sessionStore == nil || logCtx.SessionID != "" ||
		!bytes.Contains(body, []byte(`"cache_control"`)) {
		return
	}
	var req promptSessionRequest
	if goccyjson.Unmarshal(body, &req) != nil ||
		(logCtx.CacheMarkersIn.Total() == 0 && req.CacheControl["type"] != "ephemeral") {
		return
	}
	turn := append(firstUserTurn(req.Messages, false), firstUserTurn(req.Input, true)...)
	if len(turn) == 0 {
		return
	}
	prefix, _ := json.Marshal(normalizePrompt([]any{
		logCtx.Token, req.Tools,
		map[string]any{"role": "system", "content": req.System},
		map[string]any{"role": "system", "content": req.Instructions},
		turn,
	}))
	sum := sha256.Sum256(prefix)
	logCtx.SessionID = "p:" + hex.EncodeToString(sum[:16])
	logCtx.CacheEligibilitySource = "prompt_hash"
}

// firstUserTurn decodes conversation items up to and including the first user message.
func firstUserTurn(raw []byte, responsesInput bool) []any {
	if len(raw) > 0 && raw[0] == '"' {
		raw = slices.Concat([]byte(`[{"role":"user","content":`), raw, []byte(`}]`))
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('[') {
		return nil
	}
	var items []any
	for dec.More() {
		var item map[string]any
		if dec.Decode(&item) != nil {
			return nil
		}
		// Responses accepts message items with or without an explicit type.
		if responsesInput && item["type"] == "message" {
			delete(item, "type")
		}
		items = append(items, item)
		if item["role"] == "user" {
			return items
		}
	}
	return nil
}

// normalizePrompt drops cache_control markers and spells text content one way, in place.
func normalizePrompt(v any) any {
	switch t := v.(type) {
	case map[string]any:
		if cc, ok := t["cache_control"].(map[string]any); ok && cc["type"] == "ephemeral" {
			delete(t, "cache_control")
		}
		if t["type"] == "input_text" {
			t["type"] = "text"
		}
		if text, ok := t["content"].(string); ok {
			t["content"] = []any{map[string]any{"type": "text", "text": text}}
		}
		for _, val := range t {
			normalizePrompt(val)
		}
	case []any:
		for _, item := range t {
			normalizePrompt(item)
		}
	}
	return v
}
