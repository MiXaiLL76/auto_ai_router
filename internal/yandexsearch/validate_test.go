package yandexsearch

import (
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateSearchRequestAcceptsMinimalRequest(t *testing.T) {
	err := validateSearchRequest([]byte(`{"query":{"queryText":"hello world"}}`))
	require.NoError(t, err)
}

func TestValidateSearchRequestAcceptsFullAllowlistedShape(t *testing.T) {
	body := `{
		"query": {"searchType":"SEARCH_TYPE_RU","queryText":"hello","familyMode":"FAMILY_MODE_NONE","page":0,"fixTypoMode":"FIX_TYPO_MODE_ON"},
		"sortSpec": {"sortMode":"SORT_MODE_BY_RELEVANCE","sortOrder":"SORT_ORDER_DESC"},
		"groupSpec": {"groupMode":"GROUP_MODE_FLAT","groupsOnPage":10,"docsInGroup":3},
		"maxPassages": 4,
		"region": "225",
		"l10n": "LOCALIZATION_RU",
		"period": "PERIOD_ALL_TIME",
		"responseFormat": "FORMAT_HTML",
		"folderId": "client-can-send-it-but-it-gets-overridden",
		"metadata": {"fields": {"some-other-flag": "on"}}
	}`
	require.NoError(t, validateSearchRequest([]byte(body)))
}

func TestValidateSearchRequestRejectsUnknownTopLevelField(t *testing.T) {
	err := validateSearchRequest([]byte(`{"query":{"queryText":"hi"},"notAField":true}`))
	require.ErrorIs(t, err, ErrInvalidBody)
}

func TestValidateSearchRequestRejectsUnknownNestedField(t *testing.T) {
	err := validateSearchRequest([]byte(`{"query":{"queryText":"hi","notAField":true}}`))
	require.ErrorIs(t, err, ErrInvalidBody)
}

func TestValidateSearchRequestRejectsFamilyModeAtTopLevel(t *testing.T) {
	// familyMode/page/fixTypoMode must be nested inside "query" per the ТЗ.
	err := validateSearchRequest([]byte(`{"query":{"queryText":"hi"},"familyMode":"FAMILY_MODE_NONE"}`))
	require.ErrorIs(t, err, ErrInvalidBody)
}

func TestValidateSearchRequestRequiresQuery(t *testing.T) {
	require.ErrorIs(t, validateSearchRequest([]byte(`{}`)), ErrInvalidBody)
	require.ErrorIs(t, validateSearchRequest([]byte(`{"query":{}}`)), ErrInvalidBody)
	require.ErrorIs(t, validateSearchRequest([]byte(`{"query":{"queryText":"   "}}`)), ErrInvalidBody)
}

func TestValidateSearchRequestRejectsMalformedJSON(t *testing.T) {
	require.ErrorIs(t, validateSearchRequest([]byte(`not json`)), ErrInvalidBody)
	require.ErrorIs(t, validateSearchRequest([]byte(`[1,2,3]`)), ErrInvalidBody)
	require.ErrorIs(t, validateSearchRequest([]byte(`{"query":{"queryText":"hi"}} trailing garbage`)), ErrInvalidBody)
}

func TestValidateSearchRequestRejectsOversizedQuery(t *testing.T) {
	longText := strings.Repeat("a", maxQueryTextRunes+1)
	err := validateSearchRequest([]byte(`{"query":{"queryText":"` + longText + `"}}`))
	require.ErrorIs(t, err, ErrQueryTooLong)

	manyWords := strings.Repeat("a ", maxQueryTextWords+1)
	err = validateSearchRequest([]byte(`{"query":{"queryText":"` + manyWords + `"}}`))
	require.ErrorIs(t, err, ErrQueryTooLong)
}

func TestValidateSearchRequestAcceptsExactLimits(t *testing.T) {
	exactRunes := strings.Repeat("a", maxQueryTextRunes)
	require.NoError(t, validateSearchRequest([]byte(`{"query":{"queryText":"`+exactRunes+`"}}`)))

	exactWords := strings.TrimSpace(strings.Repeat("a ", maxQueryTextWords))
	require.NoError(t, validateSearchRequest([]byte(`{"query":{"queryText":"`+exactWords+`"}}`)))
}

func TestValidateSearchRequestRejectsSmartSnippets(t *testing.T) {
	body := `{"query":{"queryText":"hi"},"metadata":{"fields":{"x-genesis-info-context":"on"}}}`
	err := validateSearchRequest([]byte(body))
	require.ErrorIs(t, err, ErrSmartSnippetsNotSupported)
}

func TestValidateSearchRequestAllowsSmartSnippetsFlagWhenOff(t *testing.T) {
	body := `{"query":{"queryText":"hi"},"metadata":{"fields":{"x-genesis-info-context":"off"}}}`
	require.NoError(t, validateSearchRequest([]byte(body)))
}

func TestValidateSearchRequestSmartSnippetsCaseInsensitive(t *testing.T) {
	body := `{"query":{"queryText":"hi"},"metadata":{"fields":{"x-genesis-info-context":"ON"}}}`
	err := validateSearchRequest([]byte(body))
	require.True(t, errors.Is(err, ErrSmartSnippetsNotSupported))
}

func TestValidateSearchRequestAllowsOtherMetadataFields(t *testing.T) {
	body := `{"query":{"queryText":"hi"},"metadata":{"fields":{"anything":"else"}}}`
	assert.NoError(t, validateSearchRequest([]byte(body)))
}
