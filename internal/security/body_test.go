package security

import (
	"net/http"
	"strings"
	"testing"
)

func TestRedactBodyForLogOmitsArbitrarySecrets(t *testing.T) {
	for _, body := range []string{
		`{"messages":[{"content":"synthetic-oauth-secret"}]}`,
		`{"error":{"message":"provider echoed synthetic-oauth-secret"}}`,
		`{"tools":[{"result":"synthetic-oauth-secret"}]}`,
		"not JSON: synthetic-oauth-secret",
		strings.Repeat("x", 20000) + "synthetic-oauth-secret",
	} {
		got := RedactBodyForLog(body)
		if strings.Contains(got, "synthetic-oauth-secret") || !strings.Contains(got, "REDACTED") {
			t.Fatal("body content escaped redaction")
		}
	}
	if RedactBodyForLog("") != "" {
		t.Fatal("empty bodies must remain empty")
	}
}

func TestMaskSensitiveHeadersCanonicalAndMixedCase(t *testing.T) {
	for _, name := range []string{"X-API-Key", "x-api-key", "X-Litellm-Api-Key", "Set-Cookie", "Cookie", "AUTHORIZATION"} {
		t.Run(name, func(t *testing.T) {
			for _, canonical := range []bool{false, true} {
				headers := http.Header{}
				if canonical {
					headers.Set(name, "synthetic-header-secret")
				} else {
					headers[name] = []string{"synthetic-header-secret"}
				}
				masked := MaskSensitiveHeaders(headers)
				for _, values := range masked {
					if strings.Contains(strings.Join(values, ","), "header-secret") {
						t.Fatal("header secret escaped redaction")
					}
				}
				if !strings.Contains(strings.Join(headers[name], ",")+headers.Get(name), "header-secret") {
					t.Fatal("masking must not mutate request/response headers")
				}
			}
		})
	}
}
