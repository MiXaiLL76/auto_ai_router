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

// bindPromptSession gives a request that asks for prompt caching but has no session ID
// one hashed from its client key, tools, system prompt and conversation up to the first
// user message (session_sticky_prompt_hash).
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
	prefix, err := json.Marshal(stripCacheControl([]any{
		logCtx.Token, req.Tools, req.System, req.Instructions,
		firstUserTurn(req.Messages), firstUserTurn(req.Input),
	}))
	if err != nil {
		return
	}
	sum := sha256.Sum256(prefix)
	logCtx.SessionID = "p:" + hex.EncodeToString(sum[:16])
	logCtx.CacheEligibilitySource = "prompt_hash"
}

// firstUserTurn decodes conversation items up to and including the first user message,
// with string input and string content in block form; the rest is not read.
func firstUserTurn(raw []byte) []any {
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
		if text, ok := item["content"].(string); ok {
			item["content"] = []any{map[string]any{"type": "text", "text": text}}
		}
		items = append(items, item)
		if item["role"] == "user" {
			break
		}
	}
	return items
}

// stripCacheControl drops cache_control markers in place, so moving a breakpoint keeps the key.
func stripCacheControl(v any) any {
	switch t := v.(type) {
	case map[string]any:
		if cc, ok := t["cache_control"].(map[string]any); ok && cc["type"] == "ephemeral" {
			delete(t, "cache_control")
		}
		for _, val := range t {
			stripCacheControl(val)
		}
	case []any:
		for _, item := range t {
			stripCacheControl(item)
		}
	}
	return v
}
