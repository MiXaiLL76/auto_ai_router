package security

import (
	"net/url"
	"strings"
	"testing"
)

func TestRedactURLForLog(t *testing.T) {
	raw := "https://example.com/v1/chat?model=gpt&api_key=secret-api&access_token=secret-access&token=secret-token&key=secret-key&client_secret=secret-client"
	got := RedactURLForLog(raw)
	for _, secret := range []string{"secret-api", "secret-access", "secret-token", "secret-key", "secret-client"} {
		if strings.Contains(got, secret) {
			t.Fatalf("logged URL contains %q: %s", secret, got)
		}
	}
	u, err := url.Parse(got)
	if err != nil || u.Query().Get("model") != "gpt" {
		t.Fatalf("non-sensitive query was lost: %s, %v", got, err)
	}
	if !strings.Contains(got, "REDACTED") {
		t.Fatalf("sensitive query values were not marked: %s", got)
	}
	if !strings.Contains(raw, "secret-api") {
		t.Fatal("original upstream URL was changed")
	}
}
