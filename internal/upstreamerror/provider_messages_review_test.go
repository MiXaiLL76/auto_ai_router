package upstreamerror

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// #1: правило со статусом 404 — статус и сообщение должны оба доехать
func TestProviderMessageReview_Non400Non400StatusLosesMessage(t *testing.T) {
	t.Cleanup(func() { SetProviderMessageRules(nil) })
	SetProviderMessageRules([]ProviderMessageRule{{
		ProviderCode: "InvalidParameter",
		Match:        "insufficient permission",
		Code:         "invalid_request",
		Message:      "Caller lacks permission",
		Status:       http.StatusNotFound,
	}})

	body := []byte(`{"error":{"code":"InvalidParameter","message":"insufficient permission to call this resource"}}`)
	classified, ok := ClassifyStatus(http.StatusNotFound, body)
	t.Logf("ClassifyStatus(404) -> %+v ok=%v", classified, ok)
	assert.True(t, ok)
	assert.Equal(t, "Caller lacks permission", classified.Message)
}

// #2: вложенный под detail код
func TestProviderMessageReview_NestedNestedDetailShape(t *testing.T) {
	body := []byte(`{"error":{"detail":{"code":"InvalidParameter","message":"Agent capabilities are not enabled for the current model."}}}`)
	info, ok := ExtractProviderErrorInfo(body)
	t.Logf("info.Signals=%q info.Codes=%q ok=%v", info.Signals, info.Codes, ok)
	assert.True(t, ok)
	assert.Equal(t, http.StatusBadRequest, ClassifyProviderErrorInfo(info))
}

// #3: зарезервированный код ломает идемпотентность
func TestProviderMessageReview_ReservedReservedCodeBreaksIdempotency(t *testing.T) {
	t.Cleanup(func() { SetProviderMessageRules(nil) })
	SetProviderMessageRules([]ProviderMessageRule{{
		Match:   "insufficient permission to call this resource",
		Code:    "account_blocked",
		Message: "Caller lacks permission for this resource",
		Status:  http.StatusBadRequest,
	}})

	body := []byte(`{"error":{"code":"InvalidParameter","message":"insufficient permission to call this resource"}}`)
	first := ClassifyBadRequest(body)
	t.Logf("first -> %+v", first)

	routerBody := []byte(`{"error":{"message":"` + first.Message + `","type":"invalid_request_error","param":null,"code":"` + first.Code + `"}}`)
	second := ClassifyBadRequest(routerBody)
	t.Logf("second -> %+v", second)
	assert.Equal(t, first.Message, second.Message)
}

// #4: content-policy guard — metadata.error_type должен побеждать
func TestProviderMessageReview_ContentContentGuardSignalMismatch(t *testing.T) {
	t.Cleanup(func() { SetProviderMessageRules(nil) })
	SetProviderMessageRules([]ProviderMessageRule{{
		Match:   "upstream error",
		Code:    "invalid_request",
		Message: "Upstream refused the request",
		Status:  http.StatusBadRequest,
	}})

	body := []byte(`{"error":{"message":"upstream error","code":502,"metadata":{"error_type":"content_filter"}}}`)
	info, _ := ExtractProviderErrorInfo(body)
	t.Logf("signals=%q", info.Signals)
	_, ok := ClassifyProviderMessage(info)
	t.Logf("ClassifyProviderMessage ok=%v", ok)
	assert.False(t, ok, "content policy в metadata должен побеждать")
}

// #5: правило, чей текст пересекается с ContentPolicyMarkers
func TestProviderMessageReview_MarkerRuleOverlapsContentMarker(t *testing.T) {
	t.Cleanup(func() { SetProviderMessageRules(nil) })
	SetProviderMessageRules([]ProviderMessageRule{{
		Match:   "model requires content moderation metadata to be supplied",
		Code:    "invalid_request",
		Message: "This model requires content moderation metadata; add the moderation field.",
		Status:  http.StatusBadRequest,
	}})

	body := []byte(`{"error":{"code":"InvalidParameter","message":"model requires content moderation metadata to be supplied"}}`)
	classified := ClassifyBadRequest(body)
	t.Logf("classified -> %+v", classified)
	assert.Equal(t, "This model requires content moderation metadata; add the moderation field.", classified.Message)
}

// #7: свежая ошибка не должна приниматься за тело роутера.
//
// Тело, написанное роутером для правила, несёт: message=сообщение
// правила, code=код правила, type=тип по статусу правила, param=null.
// Свежая ошибка провайдера, отличающаяся хоть одним из этих полей, не
// должна проходить как уже классифицированная.
func TestProviderMessageReview_FreshFreshErrorMistakenForClassified(t *testing.T) {
	t.Cleanup(func() { SetProviderMessageRules(nil) })
	SetProviderMessageRules([]ProviderMessageRule{{
		Match:   "some provider wording",
		Code:    "invalid_request",
		Message: "Model does not support this feature",
		Status:  http.StatusBadRequest,
	}})

	for _, tt := range []struct {
		name string
		body string
	}{
		{"другой код провайдера", `{"error":{"code":"ProviderSpecificCode","message":"Model does not support this feature"}}`},
		{"другой type", `{"error":{"message":"Model does not support this feature","type":"api_error","param":null,"code":"invalid_request"}}`},
		{"с параметром", `{"error":{"message":"Model does not support this feature","type":"invalid_request_error","param":"temperature","code":"invalid_request"}}`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, ok := extractRouterClassification([]byte(tt.body), "invalid_request_error")
			assert.False(t, ok, "свежая ошибка не должна проходить как тело роутера")
		})
	}
}

// #8: ProviderCode сверяется только с полем code
func TestProviderMessageReview_ProviderCodeProviderCodeScansAllSignals(t *testing.T) {
	t.Cleanup(func() { SetProviderMessageRules(nil) })
	SetProviderMessageRules([]ProviderMessageRule{{
		ProviderCode: "RESOURCE_EXHAUSTED",
		Code:         "rate_limit_error",
		Message:      "Rate limit exceeded",
		Status:       http.StatusTooManyRequests,
	}})

	body := []byte(`{"error":{"code":"429","message":"RESOURCE_EXHAUSTED"}}`)
	info, _ := ExtractProviderErrorInfo(body)
	t.Logf("codes=%q signals=%q", info.Codes, info.Signals)
	_, ok := ClassifyProviderMessage(info)
	assert.False(t, ok, "правило по коду не должно срабатывать по тексту сообщения")
}

// TestProviderMessageReview_ReplacesABuiltinRuleWithoutItsPunctuation checks
// that an operator's rule replaces the built-in one even when the texts
// differ only in the trailing period.
//
// The built-in texts end in a period and an operator's usually do not. An
// exact comparison left the built-in rule matched first, so the operator saw
// their rule in the config and a DIFFERENT message on the wire.
func TestProviderMessageReview_ReplacesABuiltinRuleWithoutItsPunctuation(t *testing.T) {
	t.Cleanup(func() { SetProviderMessageRules(nil) })

	for _, match := range []string{
		"agent capabilities are not enabled for the current model",
		"agent capabilities are not enabled for the current model.",
		"Agent capabilities are not enabled for the current model",
	} {
		t.Run(match, func(t *testing.T) {
			SetProviderMessageRules([]ProviderMessageRule{{
				Match:   match,
				Code:    "invalid_request",
				Message: "Reworded by the operator",
				Status:  http.StatusBadRequest,
			}})

			body := []byte(`{"error":{"code":"InvalidParameter",` +
				`"message":"<400> InternalError.Algo.InvalidParameter: Agent capabilities are not enabled for the current model."}}`)
			classified := ClassifyBadRequest([]byte(body))

			assert.Equal(t, "Reworded by the operator", classified.Message,
				"the operator's rule must replace the built-in one")
		})
	}
}

// TestProviderMessageReview_RetryableIsCarried checks that a rule's own
// retryable answer reaches the classification.
//
// The rule knows what the failure IS, which the status alone does not: a 400
// is retryable by default -- a provider account may lack a model another
// account serves -- but a rule about the REQUEST's own fault is not, because
// every credential refuses it identically.
func TestProviderMessageReview_RetryableIsCarried(t *testing.T) {
	t.Cleanup(func() { SetProviderMessageRules(nil) })

	notRetryable := false
	retryable := true

	for _, tt := range []struct {
		name      string
		retryable *bool
		want      bool
		said      bool
	}{
		{"not retryable", &notRetryable, false, true},
		{"retryable", &retryable, true, true},
		{"unset", nil, false, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			SetProviderMessageRules([]ProviderMessageRule{{
				ProviderCode: "InvalidParameter",
				Match:        "unsupported model",
				Code:         "invalid_model",
				Message:      "Invalid model",
				Status:       http.StatusBadRequest,
				Retryable:    tt.retryable,
			}})

			body := []byte(`{"error":{"code":"InvalidParameter","message":"Unsupported model: 'kimi-k2.7-code'"}}`)
			info, ok := ExtractProviderErrorInfo([]byte(body))
			require.True(t, ok)

			got, said := ProviderMessageRetryable(info)
			assert.Equal(t, tt.want, got)
			assert.Equal(t, tt.said, said, "whether a rule said must be reported")

			classified, ok := ClassifyProviderMessage(info)
			require.True(t, ok)
			assert.Equal(t, tt.retryable, classified.Retryable)
		})
	}
}

// TestProviderMessageReview_BuiltinRulesAreNotRetryable checks that the
// rules the router ships with are not retryable.
//
// They describe the request's own fault -- the model cannot do what was
// asked of it -- so every credential refuses them identically. Production
// measured it: 441 requests, 1 236 attempts across the pool, not one
// attempt succeeded.
func TestProviderMessageReview_BuiltinRulesAreNotRetryable(t *testing.T) {
	for _, builtin := range builtinProviderMessages() {
		t.Run(builtin.Match, func(t *testing.T) {
			require.NotNil(t, builtin.Retryable, "a built-in rule must say")
			assert.False(t, *builtin.Retryable)
		})
	}
}

// TestProviderMessageReview_DeploymentRuleWinsOverTheBuiltinOne checks that
// a deployment's rule wins over the built-in one it names -- including a
// rule whose wording is SHORTER, which is what an operator writes.
//
// The order the rules are tried in is the order they are set in, and the
// deployment's rules come first: a rule behind the built-in one could never
// say anything different about a failure the router already knows -- not
// its wording, and not whether the failure may be replayed.
func TestProviderMessageReview_DeploymentRuleWinsOverTheBuiltinOne(t *testing.T) {
	t.Cleanup(func() { SetProviderMessageRules(nil) })

	notRetryable := false
	SetProviderMessageRules([]ProviderMessageRule{{
		ProviderCode: "InvalidParameter",
		Match:        "agent capabilities are not enabled",
		Code:         "invalid_request",
		Message:      "Reworded by the operator",
		Status:       http.StatusBadRequest,
		Retryable:    &notRetryable,
	}})

	body := []byte(`{"error":{"code":"InvalidParameter",` +
		`"message":"<400> InternalError.Algo.InvalidParameter: Agent capabilities are not enabled for the current model."}}`)
	info, ok := ExtractProviderErrorInfo([]byte(body))
	require.True(t, ok)

	classified, ok := ClassifyProviderMessage(info)
	require.True(t, ok)
	assert.Equal(t, "Reworded by the operator", classified.Message)

	retryable, said := ProviderMessageRetryable(info)
	assert.True(t, said)
	assert.False(t, retryable)
}
