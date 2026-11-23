package proxy

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash"

	goccyjson "github.com/goccy/go-json"
)

const (
	promptStickyKeyPrefix  = "p:"
	promptStickyKeyHexSize = 32

	cacheEligibilityPromptHash = "prompt_hash"
)

// resolvePromptStickyKey returns the session ID for a request that carries no session marker.
//
// With session_sticky_prompt_hash on, a request that asks for prompt caching
// (cache_control on a block or at the top level) gets a session ID hashed from its
// client key, model, tools, system prompt and the conversation up to the first
// user turn, and is then routed, forwarded and logged like any session.
func (p *Proxy) resolvePromptStickyKey(logCtx *RequestLogContext, body []byte, modelID string) string {
	if p.sessionStore == nil || logCtx == nil || !p.stickyPromptHash {
		return ""
	}
	return promptStickyKey(body, logCtx.Token, modelID, logCtx.CacheMarkersIn.Total() > 0)
}

// promptStickyKey hashes the stable prefix of a request; "" when the request has no cache_control.
func promptStickyKey(body []byte, token, modelID string, hasBlockMarkers bool) string {
	if !hasBlockMarkers && !bytes.Contains(body, []byte(`"cache_control"`)) {
		return ""
	}
	var req struct {
		CacheControl goccyjson.RawMessage `json:"cache_control"`
		Tools        goccyjson.RawMessage `json:"tools"`
		System       goccyjson.RawMessage `json:"system"`
		Instructions goccyjson.RawMessage `json:"instructions"`
		Messages     goccyjson.RawMessage `json:"messages"`
		Input        goccyjson.RawMessage `json:"input"`
	}
	if err := goccyjson.Unmarshal(body, &req); err != nil {
		return ""
	}
	if !hasBlockMarkers && !isEphemeralCacheControl(decodeJSON(req.CacheControl)) {
		return ""
	}

	system := decodeJSON(req.System)
	if system == nil {
		system = decodeJSON(req.Instructions)
	}
	conversation := bytes.TrimSpace(req.Messages)
	if len(conversation) == 0 || bytes.Equal(conversation, []byte("null")) {
		conversation = bytes.TrimSpace(req.Input)
	}

	h := sha256.New()
	writePromptStickyPart(h, token)
	writePromptStickyPart(h, modelID)
	for _, part := range []any{decodeJSON(req.Tools), canonicalContent(system), firstUserTurn(conversation)} {
		encoded, err := json.Marshal(stripCacheControl(part))
		if err != nil {
			return ""
		}
		writePromptStickyPart(h, string(encoded))
	}
	return promptStickyKeyPrefix + hex.EncodeToString(h.Sum(nil))[:promptStickyKeyHexSize]
}

func writePromptStickyPart(h hash.Hash, part string) {
	_, _ = fmt.Fprintf(h, "%d:%s", len(part), part)
}

func decodeJSON(raw []byte) any {
	if len(raw) == 0 {
		return nil
	}
	var v any
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(&v); err != nil {
		return nil
	}
	return v
}

// firstUserTurn decodes the conversation items up to and including the first user message,
// with string input and string content in their block form; the rest is not decoded.
func firstUserTurn(raw []byte) any {
	var items []any
	var text string
	if len(raw) > 0 && raw[0] == '"' {
		if err := json.Unmarshal(raw, &text); err != nil {
			return nil
		}
		items = []any{map[string]any{"role": "user", "content": text}}
	} else {
		dec := json.NewDecoder(bytes.NewReader(raw))
		dec.UseNumber()
		if tok, err := dec.Token(); err != nil || tok != json.Delim('[') {
			return nil
		}
		for dec.More() {
			var item any
			if err := dec.Decode(&item); err != nil {
				return nil
			}
			items = append(items, item)
			if m, ok := item.(map[string]any); ok && m["role"] == "user" {
				break
			}
		}
	}
	for _, item := range items {
		if m, ok := item.(map[string]any); ok && m["role"] != nil {
			if m["type"] == "message" {
				delete(m, "type")
			}
			m["content"] = canonicalContent(m["content"])
		}
	}
	return items
}

// canonicalContent turns string content into one text block and input_text blocks into text.
func canonicalContent(content any) any {
	switch t := content.(type) {
	case string:
		return []any{map[string]any{"type": "text", "text": t}}
	case []any:
		for _, block := range t {
			if m, ok := block.(map[string]any); ok && m["type"] == "input_text" {
				m["type"] = "text"
			}
		}
	}
	return content
}

// stripCacheControl drops cache_control markers in place, so moving a breakpoint keeps the key.
func stripCacheControl(v any) any {
	switch t := v.(type) {
	case map[string]any:
		if isEphemeralCacheControl(t["cache_control"]) {
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

func isEphemeralCacheControl(v any) bool {
	cc, ok := v.(map[string]any)
	return ok && cc["type"] == "ephemeral"
}
