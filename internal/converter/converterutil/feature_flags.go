package converterutil

import "sync/atomic"

// mirrorCacheWriteTokens gates whether a cache write reported only under a
// provider's own naming — Anthropic's cache_creation_input_tokens (surfaced
// as cache_creation_tokens), Requesty's caching_tokens — is also mirrored
// into cache_write_tokens. Some clients' own accounting reads only that
// name, and otherwise never sees the write at all (see PR #355 and the
// Requesty follow-up). Off by default: set once at startup from
// ServerConfig.MirrorCacheWriteTokens via SetMirrorCacheWriteTokens, before
// the server starts handling requests.
var mirrorCacheWriteTokens atomic.Bool

// SetMirrorCacheWriteTokens sets the flag from config. Intended to be called
// once at startup; the converters below only read it, concurrently, on the
// request path.
func SetMirrorCacheWriteTokens(enabled bool) {
	mirrorCacheWriteTokens.Store(enabled)
}

// MirrorCacheWriteTokensEnabled reports whether a cache write should be
// mirrored under cache_write_tokens in addition to its provider-native name.
func MirrorCacheWriteTokensEnabled() bool {
	return mirrorCacheWriteTokens.Load()
}
