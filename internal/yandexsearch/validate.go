package yandexsearch

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strings"
)

const (
	maxQueryTextRunes = 400
	maxQueryTextWords = 40
	smartSnippetsFlag = "x-genesis-info-context"
)

// ErrSmartSnippetsNotSupported is returned when a request asks for Yandex's
// smart-snippets mode (metadata.fields.x-genesis-info-context: "on") — a
// separate paid mode out of scope for this endpoint's first version.
var ErrSmartSnippetsNotSupported = errors.New("yandexsearch: smart snippets are not supported on this endpoint")

// ErrQueryTooLong is returned when query.queryText exceeds Yandex's own
// documented limits (400 characters / 40 words) — checked here rather than
// left to Yandex so an oversized query never costs a billable call.
var ErrQueryTooLong = errors.New("yandexsearch: query.queryText exceeds the 400-character/40-word limit")

// searchRequest is the allowlisted WebSearch.Search request shape (see
// https://aistudio.yandex.ru/ru/docs/search-api/api-ref/WebSearch/search).
// DisallowUnknownFields rejects anything outside this tree, recursively,
// in one Decode call — the same idiom internal/video/handler.go's decode()
// already uses in this codebase. Async/generative search and image search
// are out of scope by construction: this type only describes this one
// endpoint's request, and nothing routes other Yandex endpoints through it.
type searchRequest struct {
	Query          *searchQuery `json:"query"`
	SortSpec       *sortSpec    `json:"sortSpec"`
	GroupSpec      *groupSpec   `json:"groupSpec"`
	MaxPassages    *int64       `json:"maxPassages"`
	Region         *string      `json:"region"`
	L10n           *string      `json:"l10n"`
	Period         *string      `json:"period"`
	ResponseFormat *string      `json:"responseFormat"`
	// FolderID is accepted (so a client that copies Yandex's own example
	// payloads doesn't get a 400) but is always overridden downstream in
	// client.go with the server's own folder_id — never trusted from here.
	FolderID *string     `json:"folderId"`
	Metadata *searchMeta `json:"metadata"`
}

type searchQuery struct {
	SearchType  *string `json:"searchType"`
	QueryText   string  `json:"queryText"`
	FamilyMode  *string `json:"familyMode"`
	Page        *int64  `json:"page"`
	FixTypoMode *string `json:"fixTypoMode"`
}

type sortSpec struct {
	SortMode  *string `json:"sortMode"`
	SortOrder *string `json:"sortOrder"`
}

type groupSpec struct {
	GroupMode    *string `json:"groupMode"`
	GroupsOnPage *int64  `json:"groupsOnPage"`
	DocsInGroup  *int64  `json:"docsInGroup"`
}

type searchMeta struct {
	Fields map[string]string `json:"fields"`
}

// validateSearchRequest gates a client's raw request body against the
// WebSearch.Search schema and this endpoint's launch-scope restrictions.
// It does not modify or re-encode body — client.go's folderId swap keeps
// operating on the original bytes, so every field this function doesn't
// care about reaches Yandex byte-for-byte unchanged.
func validateSearchRequest(body []byte) error {
	var req searchRequest
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&req); err != nil {
		return ErrInvalidBody
	}
	var extra json.RawMessage
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return ErrInvalidBody // trailing garbage after the JSON value
	}

	if req.Query == nil || strings.TrimSpace(req.Query.QueryText) == "" {
		return ErrInvalidBody
	}
	if len([]rune(req.Query.QueryText)) > maxQueryTextRunes {
		return ErrQueryTooLong
	}
	if len(strings.Fields(req.Query.QueryText)) > maxQueryTextWords {
		return ErrQueryTooLong
	}

	if req.Metadata != nil && strings.EqualFold(req.Metadata.Fields[smartSnippetsFlag], "on") {
		return ErrSmartSnippetsNotSupported
	}

	return nil
}
