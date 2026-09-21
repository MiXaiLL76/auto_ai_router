package modeltable

import (
	"encoding/csv"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	cryptoutils "github.com/mixaill76/auto_ai_router/internal/litellmdb/crypto_utils"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// These tests replay a real LiteLLM database export (CSV dumps of the tables) through
// AIR's loader and compare the result with what LiteLLM actually served. They need
// data that must not live in the repository, so they run only when both are provided:
//
//	LITELLM_EXPORT_DIR   directory with _LiteLLM_ProxyModelTable__*.csv,
//	                     _LiteLLM_Config__*.csv and (optionally) the Daily*Spend dumps
//	LITELLM_MASTER_KEY   the master key the export's secrets are encrypted with
//
// The export is treated as read-only and its secrets are never logged.

func exportEnv(t *testing.T) (dir, key string) {
	t.Helper()
	dir = os.Getenv("LITELLM_EXPORT_DIR")
	key = os.Getenv("LITELLM_MASTER_KEY")
	if dir == "" || key == "" {
		t.Skip("set LITELLM_EXPORT_DIR and LITELLM_MASTER_KEY to run against a LiteLLM export")
	}
	return dir, key
}

// readExport streams a CSV dump, calling fn with each record keyed by column name.
func readExport(t *testing.T, dir, pattern string, fn func(rec map[string]string)) bool {
	t.Helper()
	matches, err := filepath.Glob(filepath.Join(dir, pattern))
	require.NoError(t, err)
	if len(matches) == 0 {
		return false
	}
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
			break
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
	return true
}

type loadedExport struct {
	rows     []queries.ModelTable
	router   queries.RouterSettings
	credRows []queries.CredentialTable
}

func loadExport(t *testing.T, dir, key string) loadedExport {
	t.Helper()
	var out loadedExport

	require.True(t, readExport(t, dir, "_LiteLLM_ProxyModelTable__*.csv", func(rec map[string]string) {
		raw, err := json.Marshal(map[string]json.RawMessage{
			"model_id":       mustJSON(t, rec["model_id"]),
			"model_name":     mustJSON(t, rec["model_name"]),
			"litellm_params": json.RawMessage(rec["litellm_params"]),
			"model_info":     json.RawMessage(rec["model_info"]),
		})
		require.NoError(t, err)
		var row queries.ModelTable
		require.NoError(t, json.Unmarshal(raw, &row))
		row.Blocked = strings.EqualFold(rec["blocked"], "true")
		out.rows = append(out.rows, row)
	}), "ProxyModelTable export not found in %s", dir)

	readExport(t, dir, "_LiteLLM_Config__*.csv", func(rec map[string]string) {
		if rec["param_name"] != "router_settings" {
			return
		}
		settings, err := queries.ParseRouterSettings([]byte(rec["param_value"]))
		require.NoError(t, err)
		out.router = settings
	})

	// The credentials table is not part of the export, so stand one in for every bound
	// name. Its connection details are dummies; only the wiring is under test. No
	// provider is set on purpose: it has to be inferred from the deployments.
	seen := map[string]bool{}
	for _, row := range out.rows {
		if row.LlmParams == nil || row.LlmParams.LiteLLMCredentialName == nil {
			continue
		}
		name, err := cryptoutils.DecryptValueHelper(*row.LlmParams.LiteLLMCredentialName, "litellm_credential_name", key)
		require.NoError(t, err, "the master key does not match the export")
		if seen[name] {
			continue
		}
		seen[name] = true
		apiBase, err := cryptoutils.EncryptValueHelper("http://"+name+".invalid:8000/v1", key)
		require.NoError(t, err)
		credName := name
		out.credRows = append(out.credRows, queries.CredentialTable{
			CredentialName:   &credName,
			CredentialParams: &queries.CredentialLiteLLMParams{APIBase: &apiBase},
		})
	}
	return out
}

func mustJSON(t *testing.T, v any) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(v)
	require.NoError(t, err)
	return raw
}

func TestExport_LoaderServesTheWholeFleet(t *testing.T) {
	dir, key := exportEnv(t)
	export := loadExport(t, dir, key)
	creds, models, prices, aliases := buildAIRModels(testhelpers.NewTestLogger(), export.credRows, export.rows, export.router, key)

	require.NotEmpty(t, export.rows)
	require.NotEmpty(t, models, "an export of a working proxy must yield models")

	// Every deployment is hosted_vllm in the export, so every credential must be vLLM
	// (nothing silently dropped as an unsupported provider).
	assert.Len(t, creds, len(export.credRows))
	for _, c := range creds {
		assert.Equal(t, config.ProviderTypeVLLM, c.Type, c.Name)
		assert.NotContains(t, c.BaseURL, "hosted_vllm")
	}

	// Independent expectation: serve every row that is neither blocked nor a rerank
	// model; aliases resolve to their target group instead of adding models.
	groups := map[string]int{}
	served := 0
	for _, row := range export.rows {
		if row.ModelName == nil || row.Blocked || row.Mode() == "rerank" {
			continue
		}
		groups[*row.ModelName]++
		served++
	}
	assert.Len(t, models, served)

	names := map[string]bool{}
	for _, m := range models {
		names[m.Name] = true
		assert.NotContains(t, m.Model, "hosted_vllm/", "the LiteLLM provider prefix must never reach vLLM (%s)", m.Name)
		assert.NotEmpty(t, m.Credential, m.Name)
	}
	for _, row := range export.rows {
		if row.ModelName != nil && (row.Blocked || row.Mode() == "rerank") && groups[*row.ModelName] == 0 {
			assert.False(t, names[*row.ModelName], "%s must not be served", *row.ModelName)
		}
	}
	// Every alias whose target is served must resolve to it; an alias that would shadow
	// a real group must not.
	for alias, target := range export.router.ModelGroupAlias {
		_, isGroup := groups[alias]
		switch {
		case isGroup || alias == target || groups[target] == 0:
			assert.NotContains(t, aliases, alias)
		default:
			assert.Equal(t, target, aliases[alias], "alias %s", alias)
			assert.False(t, names[alias], "alias %s must not be a model of its own", alias)
		}
	}
	t.Logf("export: %d rows -> %d credentials, %d models (%d aliases), %d prices",
		len(export.rows), len(creds), len(models), len(export.router.ModelGroupAlias), len(prices))
}

// TestExport_ModelPairsMatchObservedTraffic checks the model / model_group pair AIR
// would record against the pairs LiteLLM really wrote in its daily spend tables.
// LiteLLM stores the deployment's real name in model and the name the client asked for in
// model_group; AIR reproduces that by routing the group name to the real model.
func TestExport_ModelPairsMatchObservedTraffic(t *testing.T) {
	dir, key := exportEnv(t)
	export := loadExport(t, dir, key)
	_, models, _, aliases := buildAIRModels(testhelpers.NewTestLogger(), export.credRows, export.rows, export.router, key)

	realNames := map[string]map[string]bool{}
	for _, m := range models {
		real := m.Model
		if real == "" {
			real = m.Name
		}
		if realNames[m.Name] == nil {
			realNames[m.Name] = map[string]bool{}
		}
		realNames[m.Name][real] = true
	}
	// An alias is served by its target's deployments.
	for alias, target := range aliases {
		realNames[alias] = realNames[target]
	}

	var total, knownGroup, explained int64
	found := false
	for _, pattern := range []string{"_LiteLLM_DailyEndUserSpend__*.csv", "_LiteLLM_DailyTeamSpend__*.csv"} {
		if readExport(t, dir, pattern, func(rec map[string]string) {
			requests, _ := strconv.ParseInt(rec["successful_requests"], 10, 64)
			group := rec["model_group"]
			if requests <= 0 || group == "" {
				return
			}
			model := strings.TrimPrefix(rec["model"], "hosted_vllm/")
			total += requests
			if reals, ok := realNames[group]; ok {
				knownGroup += requests
				if reals[model] {
					explained += requests
				}
			}
		}) {
			found = true
		}
	}
	if !found {
		t.Skip("no Daily*Spend dumps in the export")
	}
	require.NotZero(t, total)

	groupShare := float64(knownGroup) / float64(total)
	modelShare := float64(explained) / float64(knownGroup)
	t.Logf("successful requests=%d, served by a group AIR now imports=%.1f%%, of those the recorded model equals a current deployment=%.1f%%",
		total, groupShare*100, modelShare*100)

	// The remainder is history: deployments that were replaced or removed since the
	// traffic was recorded (older Qwen/MiniMax builds).
	assert.GreaterOrEqual(t, groupShare, 0.99, "traffic goes to groups AIR would not serve")
	assert.GreaterOrEqual(t, modelShare, 0.90, "recorded model names diverge from the imported deployments")
}
