package spendlog

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
)

// dailySpendExecer is implemented by pgx.Tx. Daily aggregators deliberately
// accept this narrow interface so they cannot escape to a pool connection and
// accidentally commit independently from the raw row and entity counters.
type dailySpendExecer interface {
	Exec(context.Context, string, ...any) (pgconn.CommandTag, error)
}

type spendLogRecord struct {
	UserID                   string
	Date                     string
	APIKey                   string
	Model                    string
	ModelGroup               string
	CustomLLMProvider        string
	MCPNamespacedTool        string
	Endpoint                 string
	PromptTokens             int
	CompletionTokens         int
	CacheReadInputTokens     int64
	CacheCreationInputTokens int64
	Spend                    float64
	Status                   string
	RequestID                string
	TeamID                   string
	OrganizationID           string
	EndUser                  string
	SkipDaily                bool
}

// spendRoute ties together the three spellings AIR needs for one supported
// spend route: the LiteLLM call_type recorded on the raw spend row, the
// request-path fragment the proxy matches, and the LiteLLM
// ROUTE_ENDPOINT_MAPPING endpoint used as the daily dimension.
type spendRoute struct {
	callType     string
	pathFragment string
	endpoint     string
}

// spendRoutes lists every AIR-supported spend route, ordered most specific
// first: a path containing "/chat/completions" also contains "/completions",
// so the bare fragment must stay last.
var spendRoutes = []spendRoute{
	{"acompletion", "/chat/completions", "/chat/completions"},
	{"aembedding", "/embeddings", "/embeddings"},
	{"aresponses", "/responses", "/responses"},
	{"aimage_generation", "/images/generations", "/image/generations"},
	{"aimage_edit", "/images/edits", "/images/edits"},
	{"atranscription", "/audio/transcriptions", "/audio/transcriptions"},
	{"aspeech", "/audio/speech", "/audio/speech"},
	{"amoderation", "/moderations", "/moderations"},
	{"arerank", "/rerank", "/rerank"},
	{"atext_completion", "/completions", "/completions"},
}

var dailyEndpointByCallType = func() map[string]string {
	endpoints := make(map[string]string, len(spendRoutes))
	for _, route := range spendRoutes {
		endpoints[route.callType] = route.endpoint
	}
	return endpoints
}()

// LiteLLMCallTypeForPath translates an AIR request path into the call_type
// value the LiteLLM proxy records for the same operation (the async method
// name, e.g. "acompletion" — see litellm
// route_llm_request.ROUTE_ENDPOINT_MAPPING). The daily aggregation pipeline
// keys its endpoint dimension off these values, and the spend comparison
// relies on them matching the primary accounting. Unknown paths map to "" —
// the raw spend row is still written, only the daily aggregation is skipped
// for it.
func LiteLLMCallTypeForPath(path string) string {
	for _, route := range spendRoutes {
		if strings.Contains(path, route.pathFragment) {
			return route.callType
		}
	}
	return ""
}

// dailyEndpoint mirrors LiteLLM ROUTE_ENDPOINT_MAPPING for AIR-supported
// spend routes. Unknown call types intentionally yield an empty/NULL-like
// endpoint rather than leaking api_base into the daily dimension.
func dailyEndpoint(callType string) string {
	return dailyEndpointByCallType[callType]
}

// buildSpendLogRecords projects the just-inserted batch entries into daily
// aggregation records entirely in memory. Re-reading the inserted rows from
// LiteLLM_SpendLogs (the old QuerySelectUnprocessedSpendLogs) is a heavy query
// on the LiteLLM schema, and everything it returned is already known here: the
// writer owns every column it inserted.
//
// The date bucket is the UTC wall-clock date of StartTime, matching
// TO_CHAR("startTime", 'YYYY-MM-DD') on the UTC wall clock LiteLLM stores in
// the timestamp-without-time-zone column.
func buildSpendLogRecords(
	inserted []insertedSpendEntry,
	logger *slog.Logger,
	scope string,
) ([]spendLogRecord, error) {
	records := make([]spendLogRecord, 0, len(inserted))
	for _, item := range inserted {
		entry := item.entry
		if entry == nil {
			continue
		}

		originalCallType, cacheRead, cacheCreation := spendLogMetadataFields(entry.Metadata)
		record := spendLogRecord{
			UserID:                   entry.UserID,
			Date:                     entry.StartTime.UTC().Format("2006-01-02"),
			APIKey:                   entry.APIKey,
			Model:                    entry.Model,
			ModelGroup:               entry.ModelGroup,
			CustomLLMProvider:        entry.CustomLLMProvider,
			PromptTokens:             entry.PromptTokens,
			CompletionTokens:         entry.CompletionTokens,
			CacheReadInputTokens:     cacheRead,
			CacheCreationInputTokens: cacheCreation,
			Spend:                    entry.Spend,
			Status:                   entry.Status,
			RequestID:                item.requestID,
			TeamID:                   entry.TeamID,
			OrganizationID:           entry.OrganizationID,
			EndUser:                  entry.EndUser,
		}

		rawCallType := entry.CallType
		effectiveCallType := rawCallType
		if effectiveCallType == "" {
			effectiveCallType = originalCallType
		}
		if effectiveCallType == "" {
			record.SkipDaily = true
		} else if _, knownEndpoint := dailyEndpointByCallType[effectiveCallType]; !knownEndpoint {
			// A call_type outside the endpoint map is a permanent property of
			// the entry, not a transient fault: failing here would poison the
			// whole batch (retries → DLQ) and lose valid rows alongside it.
			// The raw spend row is already inserted in this transaction, so
			// only its daily aggregation is skipped.
			logger.Warn("[DB] "+scope+" aggregation: unknown call_type, skipping daily aggregation",
				"call_type", effectiveCallType, "request_id", record.RequestID)
			record.SkipDaily = true
		} else if rawCallType == "" {
			if record.Status != "failure" {
				return nil, fmt.Errorf(
					"empty raw LiteLLM call_type with status %q for request %q",
					record.Status, record.RequestID,
				)
			}
			// No nonzero-spend rejection here: LiteLLM legitimately writes
			// failure rows with partial cost and usage (interrupted streams,
			// recovered_response_cost in proxy_track_cost_callback).
			// LiteLLM failure rows use the public model as their daily model and
			// leave provider and endpoint empty. Keep the richer raw
			// AIR dimensions intact while projecting the compatible daily key.
			publicModel := record.ModelGroup
			if publicModel == "" {
				publicModel = record.Model
			}
			record.Model = publicModel
			record.CustomLLMProvider = ""
			record.Endpoint = ""
			if effectiveCallType == "aresponses" {
				record.ModelGroup = ""
			}
		} else {
			record.Endpoint = dailyEndpoint(rawCallType)
		}

		records = append(records, record)
	}

	return records, nil
}

// spendLogMetadataFields extracts the fields the daily aggregation used to read
// back out of the just-inserted row: the legacy LiteLLM original call type and
// the cache token counters. The cache extraction mirrors LiteLLM
// _extract_cache_read_tokens/_extract_cache_creation_tokens (and the COALESCE /
// NULLIF fallbacks of the retired SELECT): Anthropic-style top-level
// usage_object fields win when nonzero, then the OpenAI-compatible
// prompt_tokens_details fallbacks; zero is falsy and falls through.
func spendLogMetadataFields(metadata string) (originalCallType string, cacheReadInputTokens, cacheCreationInputTokens int64) {
	if metadata == "" {
		return "", 0, 0
	}
	var doc map[string]interface{}
	if err := json.Unmarshal([]byte(metadata), &doc); err != nil {
		return "", 0, 0
	}

	if spendLogsMetadata, ok := doc["spend_logs_metadata"].(map[string]interface{}); ok {
		originalCallType, _ = spendLogsMetadata["original_call_type"].(string)
	}

	usageObject, _ := doc["usage_object"].(map[string]interface{})
	promptTokensDetails, _ := usageObject["prompt_tokens_details"].(map[string]interface{})
	cacheReadInputTokens = firstNonZeroJSONInt(
		usageObject["cache_read_input_tokens"],
		promptTokensDetails["cached_tokens"],
	)
	cacheCreationInputTokens = firstNonZeroJSONInt(
		usageObject["cache_creation_input_tokens"],
		promptTokensDetails["cache_write_tokens"],
		promptTokensDetails["cache_creation_tokens"],
	)
	return originalCallType, cacheReadInputTokens, cacheCreationInputTokens
}

func firstNonZeroJSONInt(values ...interface{}) int64 {
	for _, value := range values {
		if n := jsonInt64(value); n != 0 {
			return n
		}
	}
	return 0
}

func jsonInt64(value interface{}) int64 {
	switch v := value.(type) {
	case float64:
		return int64(v)
	case string:
		n, err := strconv.ParseInt(v, 10, 64)
		if err == nil {
			return n
		}
	case json.Number:
		n, err := v.Int64()
		if err == nil {
			return n
		}
	}
	return 0
}

// dailyLockOrdered is implemented by every daily aggregation key. lockOrder
// serializes the key fields so keys sort identically across AIR replicas.
type dailyLockOrdered interface {
	comparable
	lockOrder() string
}

// sortedDailyKeys returns daily upsert keys in a stable order so concurrent
// transactions acquire row locks consistently and queue instead of
// deadlocking — Go map iteration order is randomized. Same rationale as
// sortedSpendKeys in spend_updater.go.
func sortedDailyKeys[K dailyLockOrdered, V any](aggregations map[K]V) []K {
	keys := make([]K, 0, len(aggregations))
	for key := range aggregations {
		keys = append(keys, key)
	}
	slices.SortFunc(keys, func(a, b K) int {
		return strings.Compare(a.lockOrder(), b.lockOrder())
	})
	return keys
}

// runAggregators runs all four daily aggregators sequentially on the transaction
// that inserted the source rows. The first error aborts the pipeline because a
// PostgreSQL transaction is unusable after a statement error and must roll back.
func (sl *Logger) runAggregators(aggCtx context.Context, tx dailySpendExecer, scope string, records []spendLogRecord) error {
	dailyRecords := make([]spendLogRecord, 0, len(records))
	for _, record := range records {
		if !record.SkipDaily {
			dailyRecords = append(dailyRecords, record)
		}
	}
	aggregators := []struct {
		name string
		fn   func() error
	}{
		{"User", func() error {
			c, cn := context.WithTimeout(aggCtx, 30*time.Second)
			defer cn()
			return aggregateDailyUserSpendLogs(c, tx, sl.logger, dailyRecords)
		}},
		{"Team", func() error {
			c, cn := context.WithTimeout(aggCtx, 30*time.Second)
			defer cn()
			return aggregateDailyTeamSpendLogs(c, tx, sl.logger, dailyRecords)
		}},
		{"Organization", func() error {
			c, cn := context.WithTimeout(aggCtx, 30*time.Second)
			defer cn()
			return aggregateDailyOrganizationSpendLogs(c, tx, sl.logger, dailyRecords)
		}},
		{"EndUser", func() error {
			c, cn := context.WithTimeout(aggCtx, 30*time.Second)
			defer cn()
			return aggregateDailyEndUserSpendLogs(c, tx, sl.logger, dailyRecords)
		}},
	}

	for _, agg := range aggregators {
		if err := agg.fn(); err != nil {
			sl.logger.Error("[DB] "+scope+": aggregator failed", "aggregator", agg.name, "error", err)
			return fmt.Errorf("%s daily aggregation: %w", agg.name, err)
		}
	}
	return nil
}
