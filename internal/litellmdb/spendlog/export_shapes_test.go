package spendlog

import (
	"encoding/csv"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Replays the daily-spend dumps of a real LiteLLM database against AIR's daily
// projection. Set LITELLM_EXPORT_DIR to a directory with _LiteLLM_DailyEndUserSpend__*.csv
// and/or _LiteLLM_DailyTeamSpend__*.csv; the tests are skipped otherwise.

func forEachDailyRow(t *testing.T, fn func(rec map[string]string)) (files int) {
	t.Helper()
	return forEachExportRow(t, []string{"_LiteLLM_DailyEndUserSpend__*.csv", "_LiteLLM_DailyTeamSpend__*.csv"}, fn)
}

// forEachExportRow streams every CSV matching the patterns (latest match per pattern).
func forEachExportRow(t *testing.T, patterns []string, fn func(rec map[string]string)) (files int) {
	t.Helper()
	dir := os.Getenv("LITELLM_EXPORT_DIR")
	if dir == "" {
		t.Skip("set LITELLM_EXPORT_DIR to run against a LiteLLM export")
	}
	for _, pattern := range patterns {
		matches, err := filepath.Glob(filepath.Join(dir, pattern))
		require.NoError(t, err)
		if len(matches) == 0 {
			continue
		}
		files++
		func() {
			file, err := os.Open(matches[len(matches)-1])
			require.NoError(t, err)
			defer func() { _ = file.Close() }()
			reader := csv.NewReader(file)
			reader.FieldsPerRecord = -1
			reader.LazyQuotes = true
			header, err := reader.Read()
			require.NoError(t, err)
			for {
				row, err := reader.Read()
				if err == io.EOF {
					return
				}
				require.NoError(t, err)
				rec := make(map[string]string, len(header))
				for i, name := range header {
					if i < len(row) {
						rec[name] = row[i]
					}
				}
				fn(rec)
			}
		}()
	}
	if files == 0 {
		t.Skip("no Daily*Spend dumps in LITELLM_EXPORT_DIR")
	}
	return files
}

// Every endpoint LiteLLM recorded must be one AIR can produce, or the daily tables
// would miss a dimension value the reports group by.
func TestExport_DailyEndpointsAreKnownToAIR(t *testing.T) {
	known := map[string]bool{}
	for _, endpoint := range dailyEndpointByCallType {
		known[endpoint] = true
	}

	seen := map[string]int{}
	forEachDailyRow(t, func(rec map[string]string) {
		if endpoint := rec["endpoint"]; endpoint != "" {
			seen[endpoint]++
		}
	})

	require.NotEmpty(t, seen)
	for endpoint, rows := range seen {
		assert.True(t, known[endpoint], "endpoint %q (%d rows) is recorded by LiteLLM but unknown to AIR", endpoint, rows)
	}
}

// LiteLLM writes a request that never reached a deployment as a row without endpoint whose
// model is (mostly) the public group name. AIR's projection has to produce the same key,
// otherwise failures land in rows that never merge with LiteLLM's.
//
// Observed on the work database: endpoint is empty in every failure-only row; model equals
// model_group in ~81% (the rest are requests that reached a deployment or a model-level
// fallback); provider is empty in ~95%, but LiteLLM began recording it for failures on
// 2026-09-04, so it is reported rather than asserted.
func TestExport_FailureRowsHaveTheShapeAIRProjects(t *testing.T) {
	var failureRows, endpointEmpty, providerEmpty, modelIsGroup int
	forEachDailyRow(t, func(rec map[string]string) {
		success, _ := strconv.ParseInt(rec["successful_requests"], 10, 64)
		failed, _ := strconv.ParseInt(rec["failed_requests"], 10, 64)
		if success != 0 || failed == 0 || rec["model_group"] == "" {
			return
		}
		failureRows++
		if rec["endpoint"] == "" {
			endpointEmpty++
		}
		if rec["custom_llm_provider"] == "" {
			providerEmpty++
		}
		if rec["model"] == rec["model_group"] {
			modelIsGroup++
		}
	})
	require.NotZero(t, failureRows, "the export has no failure-only rows")
	t.Logf("failure-only rows: %d; endpoint empty %.1f%%, provider empty %.1f%%, model == model_group %.1f%%",
		failureRows, pct(endpointEmpty, failureRows), pct(providerEmpty, failureRows), pct(modelIsGroup, failureRows))

	assert.Equal(t, failureRows, endpointEmpty, "AIR projects failures without an endpoint")
	assert.GreaterOrEqual(t, pct(modelIsGroup, failureRows), 75.0, "AIR projects the public name as the failure model")

	entry := &models.SpendLogEntry{
		StartTime:  time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC),
		Status:     "failure",
		Model:      "qwen397b-int4", // deployment the request was routed to
		ModelGroup: "qwen-ultra",    // what the client asked for
		APIKey:     "hash",
		UserID:     "u",
		// CallType empty: the request failed before a call type was recorded.
		CustomLLMProvider: "vllm",
	}
	metadata := `{"spend_logs_metadata":{"original_call_type":"acompletion"}}`
	entry.Metadata = metadata

	records, err := buildSpendLogRecords([]insertedSpendEntry{{entry: entry, requestID: "r1"}}, slog.New(slog.NewTextHandler(io.Discard, nil)), "test")
	require.NoError(t, err)
	require.Len(t, records, 1)
	assert.Equal(t, "qwen-ultra", records[0].Model, "failure rows use the public name as model")
	assert.Equal(t, "qwen-ultra", records[0].ModelGroup)
	assert.Empty(t, records[0].CustomLLMProvider)
	assert.Empty(t, records[0].Endpoint)
}

func pct(part, whole int) float64 { return 100 * float64(part) / float64(whole) }

// LiteLLM's daily rows always satisfy api_requests == successful + failed; AIR's
// aggregation must keep that invariant on the same kind of mix.
func TestExport_RequestCountInvariantHolds(t *testing.T) {
	var rows, broken int
	forEachDailyRow(t, func(rec map[string]string) {
		total, _ := strconv.ParseInt(rec["api_requests"], 10, 64)
		success, _ := strconv.ParseInt(rec["successful_requests"], 10, 64)
		failed, _ := strconv.ParseInt(rec["failed_requests"], 10, 64)
		rows++
		if total != success+failed {
			broken++
		}
	})
	require.NotZero(t, rows)
	assert.Zero(t, broken, "LiteLLM rows violating api_requests == successful + failed")

	agg := &aggregationValue{}
	for i := 0; i < 7; i++ {
		agg.addRecord(spendLogRecord{Status: "success"})
	}
	for i := 0; i < 3; i++ {
		agg.addRecord(spendLogRecord{Status: "failure"})
	}
	assert.Equal(t, agg.apiRequests, agg.successfulRequests+agg.failedRequests)
	assert.EqualValues(t, 10, agg.apiRequests)
}

// The dashboard's user report reads LiteLLM_DailyUserSpend for every key except the shared
// OpenWebUI key and the GenAI team. On what remains, the user of a row is the owner of the
// key (or nobody, for project service keys): only a small share of traffic carries a user
// taken from a request header instead. That is what makes "key owner unless an identity
// header says otherwise" the right default for AIR's user_id.
func TestExport_DailyUserSpendFollowsTheKeyOwner(t *testing.T) {
	const (
		openWebUIKey = "679688606f5364910254f35d00615896a9142f767e6d8b95def0b2a13169d9bd"
		genAITeam    = "f73764d6-b7a0-42e4-b219-99aee082cbcb"
	)
	owner := map[string]string{}
	team := map[string]string{}
	if forEachExportRow(t, []string{"_LiteLLM_VerificationToken__*.csv"}, func(rec map[string]string) {
		owner[rec["token"]] = rec["user_id"]
		team[rec["token"]] = rec["team_id"]
	}) == 0 {
		t.Skip("no VerificationToken dump in LITELLM_EXPORT_DIR")
	}

	var total, keyOwner, noUser, fromHeader int64
	if forEachExportRow(t, []string{"_LiteLLM_DailyUserSpend__*.csv"}, func(rec map[string]string) {
		requests, _ := strconv.ParseInt(rec["successful_requests"], 10, 64)
		key := rec["api_key"]
		keyTeam, known := team[key]
		// The same rows the report keeps: successful, real key with a team, not the
		// excluded key/team. (A NULL team is dropped by the report's SQL as well.)
		if requests <= 0 || key == "litellm-internal-health-check" || key == openWebUIKey ||
			!known || keyTeam == "" || keyTeam == genAITeam {
			return
		}
		total += requests
		switch rec["user_id"] {
		case owner[key]:
			keyOwner += requests
		case "":
			noUser += requests
		default:
			fromHeader += requests
		}
	}) == 0 {
		t.Skip("no DailyUserSpend dump in LITELLM_EXPORT_DIR")
	}
	require.NotZero(t, total)

	t.Logf("report requests=%d: key owner %.1f%%, no user %.1f%%, user from a header %.1f%%",
		total, 100*float64(keyOwner)/float64(total), 100*float64(noUser)/float64(total), 100*float64(fromHeader)/float64(total))
	assert.Less(t, float64(fromHeader)/float64(total), 0.05, "header-derived users are expected to be a small share")
}
