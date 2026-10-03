package httputil

import (
	"net/http"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

// ApplyCredentialRequestHeaders applies cred.RequestHeaders to an outbound
// request. Call it after the client's headers were copied and the router set its
// own, so the configured values win. An empty value removes the header.
func ApplyCredentialRequestHeaders(header http.Header, cred *config.CredentialConfig) {
	if cred == nil {
		return
	}
	for name, value := range cred.RequestHeaders {
		if value != "" {
			header.Set(name, value)
			continue
		}
		header.Del(name)
		if http.CanonicalHeaderKey(name) == "User-Agent" {
			// Without the key net/http sends its own Go-http-client User-Agent;
			// an explicit empty value makes it send none.
			header["User-Agent"] = []string{""}
		}
	}
}
