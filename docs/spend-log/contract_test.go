package spendsink

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"testing"
	"time"
)

const (
	contractVersion = "air-spend-log/v1"
	liteLLMCommit   = "10d5804b3ef4ead5abb77c3f5fafdd0c0159de0f"
	goldenDirectory = "../../testdata/golden/spend-log"
)

var routeContract = map[string]struct {
	file          string
	callType      string
	dailyEndpoint string
}{
	"/v1/chat/completions":   {"chat-completions.json", "acompletion", "/chat/completions"},
	"/v1/completions":        {"completions.json", "atext_completion", "/completions"},
	"/v1/embeddings":         {"embeddings.json", "aembedding", "/embeddings"},
	"/v1/responses":          {"responses.json", "aresponses", "/responses"},
	"/v1/images/generations": {"image-generation.json", "aimage_generation", "/image/generations"},
	"/v1/images/edits":       {"image-edit.json", "aimage_edit", "/images/edits"},
}

var expectedRawRowFields = []string{
	"request_id", "call_type", "api_key", "spend", "total_tokens",
	"prompt_tokens", "completion_tokens", "startTime", "endTime",
	"request_duration_ms", "completionStartTime", "model", "model_id",
	"model_group", "custom_llm_provider", "api_base", "user", "metadata",
	"cache_hit", "cache_key", "request_tags", "team_id", "organization_id",
	"end_user", "requester_ip_address", "messages", "response", "session_id",
	"status", "mcp_namespaced_tool_name", "agent_id", "proxy_server_request",
}

var expectedCounterTables = []string{
	"LiteLLM_AgentsTable",
	"LiteLLM_EndUserTable",
	"LiteLLM_OrganizationMembership",
	"LiteLLM_OrganizationTable",
	"LiteLLM_TagTable",
	"LiteLLM_TeamMembership",
	"LiteLLM_TeamTable",
	"LiteLLM_UserTable",
	"LiteLLM_VerificationToken",
}

var expectedDailyTables = []string{
	"LiteLLM_DailyAgentSpend",
	"LiteLLM_DailyEndUserSpend",
	"LiteLLM_DailyOrganizationSpend",
	"LiteLLM_DailyTagSpend",
	"LiteLLM_DailyTeamSpend",
	"LiteLLM_DailyUserSpend",
}

var secretLikeValue = regexp.MustCompile(`(?i)(bearer\s+|-----begin [a-z ]*private key-----|(?:^|[^a-z])sk-[a-z0-9_-]{8,})`)

type goldenFixture struct {
	ContractVersion string            `json:"contract_version"`
	Reference       contractReference `json:"reference"`
	Scenario        scenario          `json:"scenario"`
	RawRow          rawSpendRow       `json:"raw_row"`
	CounterDeltas   []counterDelta    `json:"counter_deltas"`
	Daily           dailyContract     `json:"daily"`
}

type contractReference struct {
	LiteLLMCommit string `json:"litellm_commit"`
	SchemaModel   string `json:"schema_model"`
}

type scenario struct {
	Name      string `json:"name"`
	Endpoint  string `json:"endpoint"`
	RouteType string `json:"route_type"`
	Transport string `json:"transport"`
}

type rawSpendRow struct {
	RequestID             string          `json:"request_id"`
	CallType              string          `json:"call_type"`
	APIKey                string          `json:"api_key"`
	Spend                 float64         `json:"spend"`
	TotalTokens           int             `json:"total_tokens"`
	PromptTokens          int             `json:"prompt_tokens"`
	CompletionTokens      int             `json:"completion_tokens"`
	StartTime             string          `json:"startTime"`
	EndTime               string          `json:"endTime"`
	RequestDurationMS     int             `json:"request_duration_ms"`
	CompletionStartTime   string          `json:"completionStartTime"`
	Model                 string          `json:"model"`
	ModelID               string          `json:"model_id"`
	ModelGroup            string          `json:"model_group"`
	CustomLLMProvider     string          `json:"custom_llm_provider"`
	APIBase               string          `json:"api_base"`
	User                  string          `json:"user"`
	Metadata              spendMetadata   `json:"metadata"`
	CacheHit              string          `json:"cache_hit"`
	CacheKey              string          `json:"cache_key"`
	RequestTags           []string        `json:"request_tags"`
	TeamID                string          `json:"team_id"`
	OrganizationID        string          `json:"organization_id"`
	EndUser               string          `json:"end_user"`
	RequesterIPAddress    string          `json:"requester_ip_address"`
	Messages              json.RawMessage `json:"messages"`
	Response              json.RawMessage `json:"response"`
	SessionID             string          `json:"session_id"`
	Status                string          `json:"status"`
	MCPNamespacedToolName json.RawMessage `json:"mcp_namespaced_tool_name"`
	AgentID               string          `json:"agent_id"`
	ProxyServerRequest    json.RawMessage `json:"proxy_server_request"`
}

type spendMetadata struct {
	UserAPIKey             string              `json:"user_api_key"`
	UserAPIKeyAlias        string              `json:"user_api_key_alias"`
	UserAPIKeyTeamID       string              `json:"user_api_key_team_id"`
	UserAPIKeyProjectID    string              `json:"user_api_key_project_id"`
	UserAPIKeyProjectAlias string              `json:"user_api_key_project_alias"`
	UserAPIKeyOrgID        string              `json:"user_api_key_org_id"`
	UserAPIKeyUserID       string              `json:"user_api_key_user_id"`
	UserAPIKeyTeamAlias    string              `json:"user_api_key_team_alias"`
	RequesterIPAddress     string              `json:"requester_ip_address"`
	LiteLLMCallID          string              `json:"litellm_call_id"`
	Status                 string              `json:"status"`
	AttemptedRetries       int                 `json:"attempted_retries"`
	MaxRetries             int                 `json:"max_retries"`
	UsageObject            usageObject         `json:"usage_object"`
	CostBreakdown          costBreakdown       `json:"cost_breakdown"`
	ModelMapInformation    modelMapInformation `json:"model_map_information"`
	SpendLogsMetadata      spendLogsExtensions `json:"spend_logs_metadata"`
}

type usageObject struct {
	PromptTokens            int                    `json:"prompt_tokens"`
	CompletionTokens        int                    `json:"completion_tokens"`
	TotalTokens             int                    `json:"total_tokens"`
	PromptTokensDetails     promptTokenDetails     `json:"prompt_tokens_details"`
	CompletionTokensDetails completionTokenDetails `json:"completion_tokens_details"`
}

type promptTokenDetails struct {
	CachedTokens        int `json:"cached_tokens"`
	CacheCreationTokens int `json:"cache_creation_tokens"`
}

type completionTokenDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

type costBreakdown struct {
	InputCost         float64 `json:"input_cost"`
	CacheReadCost     float64 `json:"cache_read_cost"`
	CacheCreationCost float64 `json:"cache_creation_cost"`
	OutputCost        float64 `json:"output_cost"`
	ReasoningCost     float64 `json:"reasoning_cost"`
	TotalCost         float64 `json:"total_cost"`
}

type modelMapInformation struct {
	ModelMapKey   string          `json:"model_map_key"`
	ModelMapValue json.RawMessage `json:"model_map_value"`
}

type spendLogsExtensions struct {
	ComparisonEligible bool          `json:"comparison_eligible"`
	ShadowContextState string        `json:"shadow_context_state"`
	ActualProvider     string        `json:"actual_provider"`
	ActualCredential   string        `json:"actual_credential"`
	ActualUpstreamHost string        `json:"actual_upstream_host"`
	PriceSnapshot      priceSnapshot `json:"price_snapshot"`
}

type priceSnapshot struct {
	Registry               string  `json:"registry"`
	Model                  string  `json:"model"`
	InputCostPerToken      float64 `json:"input_cost_per_token"`
	OutputCostPerToken     float64 `json:"output_cost_per_token"`
	CacheReadCostPerToken  float64 `json:"cache_read_cost_per_token"`
	CacheWriteCostPerToken float64 `json:"cache_write_cost_per_token"`
}

type counterDelta struct {
	Table  string             `json:"table"`
	Key    map[string]string  `json:"key"`
	Deltas map[string]float64 `json:"deltas"`
}

type dailyContract struct {
	Dimensions dailyDimensions `json:"dimensions"`
	Metrics    dailyMetrics    `json:"metrics"`
	Entities   []dailyEntity   `json:"entities"`
}

type dailyDimensions struct {
	Date                  string          `json:"date"`
	APIKey                string          `json:"api_key"`
	Model                 string          `json:"model"`
	ModelGroup            string          `json:"model_group"`
	CustomLLMProvider     string          `json:"custom_llm_provider"`
	MCPNamespacedToolName json.RawMessage `json:"mcp_namespaced_tool_name"`
	Endpoint              string          `json:"endpoint"`
}

type dailyMetrics struct {
	PromptTokens             int     `json:"prompt_tokens"`
	CompletionTokens         int     `json:"completion_tokens"`
	CacheReadInputTokens     int     `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int     `json:"cache_creation_input_tokens"`
	Spend                    float64 `json:"spend"`
	APIRequests              int     `json:"api_requests"`
	SuccessfulRequests       int     `json:"successful_requests"`
	FailedRequests           int     `json:"failed_requests"`
}

type dailyEntity struct {
	Table     string            `json:"table"`
	Key       map[string]string `json:"key"`
	RequestID string            `json:"request_id,omitempty"`
}

func TestGoldenFixturesMatchLiteLLMSpendContract(t *testing.T) {
	for endpoint, route := range routeContract {
		endpoint, route := endpoint, route
		t.Run(route.file, func(t *testing.T) {
			t.Parallel()

			path := filepath.Join(goldenDirectory, route.file)
			fixture, raw := loadFixture(t, path)

			if fixture.ContractVersion != contractVersion {
				t.Fatalf("contract_version = %q, want %q", fixture.ContractVersion, contractVersion)
			}
			if fixture.Reference.LiteLLMCommit != liteLLMCommit {
				t.Fatalf("litellm_commit = %q, want %q", fixture.Reference.LiteLLMCommit, liteLLMCommit)
			}
			if fixture.Reference.SchemaModel != "LiteLLM_SpendLogs" {
				t.Fatalf("schema_model = %q, want LiteLLM_SpendLogs", fixture.Reference.SchemaModel)
			}

			if fixture.Scenario.Endpoint != endpoint {
				t.Errorf("scenario.endpoint = %q, want %q", fixture.Scenario.Endpoint, endpoint)
			}
			if fixture.Scenario.Name == "" {
				t.Error("scenario.name is required")
			}
			if fixture.Scenario.RouteType != route.callType || fixture.RawRow.CallType != route.callType {
				t.Errorf("route type mismatch: scenario=%q raw_row=%q want=%q", fixture.Scenario.RouteType, fixture.RawRow.CallType, route.callType)
			}
			if fixture.Scenario.Transport != "http-json" && fixture.Scenario.Transport != "http-multipart" {
				t.Errorf("unsupported fixture transport %q", fixture.Scenario.Transport)
			}

			assertRawShape(t, raw)
			assertRawSemantics(t, fixture)
			assertCounterContract(t, fixture)
			assertDailyContract(t, fixture, route.dailyEndpoint)
			assertPrivacy(t, raw)
		})
	}
}

func TestGoldenDirectoryContainsOnlyContractFixtures(t *testing.T) {
	entries, err := os.ReadDir(goldenDirectory)
	if err != nil {
		t.Fatal(err)
	}

	want := make([]string, 0, len(routeContract))
	for _, route := range routeContract {
		want = append(want, route.file)
	}
	got := make([]string, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			got = append(got, entry.Name())
		}
	}
	assertSameStrings(t, "golden files", got, want)

	requestIDs := make(map[string]string, len(got))
	callIDs := make(map[string]string, len(got))
	for _, name := range got {
		fixture, _ := loadFixture(t, filepath.Join(goldenDirectory, name))
		assertUnique(t, "request_id", fixture.RawRow.RequestID, name, requestIDs)
		assertUnique(t, "litellm_call_id", fixture.RawRow.Metadata.LiteLLMCallID, name, callIDs)
	}
}

func loadFixture(t *testing.T, path string) (goldenFixture, map[string]any) {
	t.Helper()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	var fixture goldenFixture
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&fixture); err != nil {
		t.Fatalf("decode typed fixture: %v", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		t.Fatalf("fixture must contain exactly one JSON value: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("decode generic fixture: %v", err)
	}
	return fixture, raw
}

func assertRawShape(t *testing.T, fixture map[string]any) {
	t.Helper()

	raw, ok := fixture["raw_row"].(map[string]any)
	if !ok {
		t.Fatal("raw_row must be an object")
	}
	got := make([]string, 0, len(raw))
	for field := range raw {
		got = append(got, field)
	}
	assertSameStrings(t, "raw_row fields", got, expectedRawRowFields)

	for _, privateField := range []string{"messages", "response", "proxy_server_request"} {
		value, ok := raw[privateField].(map[string]any)
		if !ok || len(value) != 0 {
			t.Errorf("raw_row.%s = %#v, want empty JSON object", privateField, raw[privateField])
		}
	}
}

func assertRawSemantics(t *testing.T, fixture goldenFixture) {
	t.Helper()
	raw := fixture.RawRow
	metadata := raw.Metadata
	usage := metadata.UsageObject
	cost := metadata.CostBreakdown

	if raw.RequestID == "" || metadata.LiteLLMCallID == "" {
		t.Error("request_id and metadata.litellm_call_id are required")
	}
	if !isSHA256(raw.APIKey) {
		t.Errorf("api_key must be a lowercase SHA-256 hex digest, got %q", raw.APIKey)
	}
	if raw.APIBase != "http://air-ru01/v1" || raw.CustomLLMProvider != "openai" {
		t.Errorf("AIR route mismatch: api_base=%q provider=%q", raw.APIBase, raw.CustomLLMProvider)
	}
	if raw.Model == "" || raw.ModelID == "" || raw.ModelGroup == "" || raw.Model == raw.ModelGroup {
		t.Errorf("model fields must distinguish backend/public/deployment: model=%q model_group=%q model_id=%q", raw.Model, raw.ModelGroup, raw.ModelID)
	}
	if raw.Status != "success" || metadata.Status != raw.Status {
		t.Errorf("status mismatch: raw=%q metadata=%q", raw.Status, metadata.Status)
	}

	if raw.APIKey != metadata.UserAPIKey || raw.User != metadata.UserAPIKeyUserID || raw.TeamID != metadata.UserAPIKeyTeamID || raw.OrganizationID != metadata.UserAPIKeyOrgID {
		t.Error("raw identity columns must match metadata.user_api_key* fields")
	}
	if raw.RequesterIPAddress != metadata.RequesterIPAddress {
		t.Error("requester_ip_address must match metadata")
	}
	if metadata.UserAPIKeyProjectID == "" || raw.AgentID == "" || raw.EndUser == "" || len(raw.RequestTags) != 1 {
		t.Error("project, agent, end-user and tags must be represented")
	}

	if raw.PromptTokens != usage.PromptTokens || raw.CompletionTokens != usage.CompletionTokens || raw.TotalTokens != usage.TotalTokens {
		t.Error("raw token columns must match metadata.usage_object")
	}
	if raw.TotalTokens != raw.PromptTokens+raw.CompletionTokens {
		t.Errorf("total_tokens=%d, want prompt+completion=%d", raw.TotalTokens, raw.PromptTokens+raw.CompletionTokens)
	}
	if usage.PromptTokensDetails.CachedTokens+usage.PromptTokensDetails.CacheCreationTokens > usage.PromptTokens {
		t.Error("cache token details cannot exceed prompt_tokens")
	}
	if usage.CompletionTokensDetails.ReasoningTokens > usage.CompletionTokens {
		t.Error("reasoning_tokens cannot exceed completion_tokens")
	}

	computedCost := cost.InputCost + cost.CacheReadCost + cost.CacheCreationCost + cost.OutputCost
	if !closeEnough(cost.TotalCost, computedCost) || !closeEnough(raw.Spend, cost.TotalCost) {
		t.Errorf("cost mismatch: spend=%g total=%g components=%g", raw.Spend, cost.TotalCost, computedCost)
	}
	if metadata.ModelMapInformation.ModelMapKey != raw.Model || !isJSONNull(metadata.ModelMapInformation.ModelMapValue) {
		t.Error("model_map_information must identify backend model and have a null map value in anonymized fixtures")
	}
	if !metadata.SpendLogsMetadata.ComparisonEligible || metadata.SpendLogsMetadata.ShadowContextState != "valid" {
		t.Error("signed-context golden rows must be comparison eligible and valid")
	}
	if !strings.HasSuffix(metadata.SpendLogsMetadata.ActualUpstreamHost, ".invalid") {
		t.Errorf("actual_upstream_host must use RFC 2606 .invalid, got %q", metadata.SpendLogsMetadata.ActualUpstreamHost)
	}
	if metadata.SpendLogsMetadata.ActualProvider == "" || metadata.SpendLogsMetadata.ActualCredential == "" {
		t.Error("actual provider and credential identifiers are required")
	}
	if metadata.SpendLogsMetadata.PriceSnapshot.Model != raw.Model || metadata.SpendLogsMetadata.PriceSnapshot.Registry == "" {
		t.Error("price_snapshot must identify its registry and backend model")
	}

	start := parseTime(t, raw.StartTime)
	end := parseTime(t, raw.EndTime)
	completionStart := parseTime(t, raw.CompletionStartTime)
	if completionStart.Before(start) || completionStart.After(end) {
		t.Error("completionStartTime must be inside the request interval")
	}
	if got := int(end.Sub(start).Milliseconds()); got != raw.RequestDurationMS {
		t.Errorf("request_duration_ms=%d, want %d", raw.RequestDurationMS, got)
	}
	if !isEmptyJSONObject(raw.Messages) || !isEmptyJSONObject(raw.Response) || !isEmptyJSONObject(raw.ProxyServerRequest) {
		t.Error("messages, response and proxy_server_request must be empty JSON objects")
	}
}

func assertCounterContract(t *testing.T, fixture goldenFixture) {
	t.Helper()

	gotTables := make([]string, 0, len(fixture.CounterDeltas))
	for _, counter := range fixture.CounterDeltas {
		gotTables = append(gotTables, counter.Table)
		if !closeEnough(counter.Deltas["spend"], fixture.RawRow.Spend) {
			t.Errorf("%s spend delta=%g, want %g", counter.Table, counter.Deltas["spend"], fixture.RawRow.Spend)
		}
	}
	assertSameStrings(t, "counter tables", gotTables, expectedCounterTables)

	byTable := make(map[string]counterDelta, len(fixture.CounterDeltas))
	for _, counter := range fixture.CounterDeltas {
		byTable[counter.Table] = counter
	}
	assertKey(t, byTable["LiteLLM_VerificationToken"].Key, "token", fixture.RawRow.APIKey)
	assertKey(t, byTable["LiteLLM_UserTable"].Key, "user_id", fixture.RawRow.User)
	assertKey(t, byTable["LiteLLM_TeamTable"].Key, "team_id", fixture.RawRow.TeamID)
	assertKey(t, byTable["LiteLLM_OrganizationTable"].Key, "organization_id", fixture.RawRow.OrganizationID)
	assertKey(t, byTable["LiteLLM_OrganizationMembership"].Key, "organization_id", fixture.RawRow.OrganizationID)
	assertKey(t, byTable["LiteLLM_OrganizationMembership"].Key, "user_id", fixture.RawRow.User)
	assertKey(t, byTable["LiteLLM_EndUserTable"].Key, "user_id", fixture.RawRow.EndUser)
	assertKey(t, byTable["LiteLLM_AgentsTable"].Key, "agent_id", fixture.RawRow.AgentID)
	assertKey(t, byTable["LiteLLM_TagTable"].Key, "tag_name", fixture.RawRow.RequestTags[0])
	assertKey(t, byTable["LiteLLM_TeamMembership"].Key, "team_id", fixture.RawRow.TeamID)
	assertKey(t, byTable["LiteLLM_TeamMembership"].Key, "user_id", fixture.RawRow.User)
	if !closeEnough(byTable["LiteLLM_TeamMembership"].Deltas["total_spend"], fixture.RawRow.Spend) {
		t.Error("team membership total_spend delta must match spend")
	}
}

func assertDailyContract(t *testing.T, fixture goldenFixture, expectedEndpoint string) {
	t.Helper()
	daily := fixture.Daily
	raw := fixture.RawRow
	usage := raw.Metadata.UsageObject

	if daily.Dimensions.Date != strings.Split(raw.StartTime, "T")[0] {
		t.Errorf("daily date=%q does not match startTime=%q", daily.Dimensions.Date, raw.StartTime)
	}
	if daily.Dimensions.APIKey != raw.APIKey || daily.Dimensions.Model != raw.Model || daily.Dimensions.ModelGroup != raw.ModelGroup || daily.Dimensions.CustomLLMProvider != raw.CustomLLMProvider {
		t.Error("daily dimensions must match raw spend row")
	}
	if daily.Dimensions.Endpoint != expectedEndpoint {
		t.Errorf("daily endpoint=%q, want LiteLLM mapping %q", daily.Dimensions.Endpoint, expectedEndpoint)
	}
	if !isJSONNull(daily.Dimensions.MCPNamespacedToolName) {
		t.Error("mcp_namespaced_tool_name must be null for these fixtures")
	}

	metrics := daily.Metrics
	if metrics.PromptTokens != raw.PromptTokens || metrics.CompletionTokens != raw.CompletionTokens || metrics.CacheReadInputTokens != usage.PromptTokensDetails.CachedTokens || metrics.CacheCreationInputTokens != usage.PromptTokensDetails.CacheCreationTokens || !closeEnough(metrics.Spend, raw.Spend) {
		t.Error("daily token/cost metrics must match raw row and usage details")
	}
	if metrics.APIRequests != 1 || metrics.SuccessfulRequests != 1 || metrics.FailedRequests != 0 {
		t.Errorf("daily request counters=%+v, want one successful request", metrics)
	}

	gotTables := make([]string, 0, len(daily.Entities))
	for _, entity := range daily.Entities {
		gotTables = append(gotTables, entity.Table)
		if entity.Table == "LiteLLM_DailyTagSpend" && entity.RequestID != raw.RequestID {
			t.Error("daily tag request_id must match raw request_id")
		}
	}
	assertSameStrings(t, "daily tables", gotTables, expectedDailyTables)

	byTable := make(map[string]dailyEntity, len(daily.Entities))
	for _, entity := range daily.Entities {
		byTable[entity.Table] = entity
	}
	assertKey(t, byTable["LiteLLM_DailyUserSpend"].Key, "user_id", raw.User)
	assertKey(t, byTable["LiteLLM_DailyTeamSpend"].Key, "team_id", raw.TeamID)
	assertKey(t, byTable["LiteLLM_DailyOrganizationSpend"].Key, "organization_id", raw.OrganizationID)
	assertKey(t, byTable["LiteLLM_DailyEndUserSpend"].Key, "end_user_id", raw.EndUser)
	assertKey(t, byTable["LiteLLM_DailyAgentSpend"].Key, "agent_id", raw.AgentID)
	assertKey(t, byTable["LiteLLM_DailyTagSpend"].Key, "tag", raw.RequestTags[0])
}

func assertPrivacy(t *testing.T, fixture map[string]any) {
	t.Helper()

	forbiddenKeys := map[string]bool{
		"content": true, "prompt": true, "input": true, "output": true,
		"request_body": true, "response_body": true, "authorization": true,
		"api_key_plaintext": true, "private_key": true,
	}
	walkJSON(t, "$", fixture, forbiddenKeys)
}

func walkJSON(t *testing.T, path string, value any, forbiddenKeys map[string]bool) {
	t.Helper()
	switch typed := value.(type) {
	case map[string]any:
		for key, child := range typed {
			if forbiddenKeys[strings.ToLower(key)] {
				t.Errorf("forbidden content-bearing key at %s.%s", path, key)
			}
			walkJSON(t, path+"."+key, child, forbiddenKeys)
		}
	case []any:
		for i, child := range typed {
			walkJSON(t, fmt.Sprintf("%s[%d]", path, i), child, forbiddenKeys)
		}
	case string:
		if secretLikeValue.MatchString(typed) {
			t.Errorf("secret-like value at %s", path)
		}
	}
}

func assertKey(t *testing.T, key map[string]string, field, want string) {
	t.Helper()
	if got := key[field]; got != want {
		t.Errorf("key[%q]=%q, want %q", field, got, want)
	}
}

func assertSameStrings(t *testing.T, label string, got, want []string) {
	t.Helper()
	got = append([]string(nil), got...)
	want = append([]string(nil), want...)
	sort.Strings(got)
	sort.Strings(want)
	if !slices.Equal(got, want) {
		t.Fatalf("%s = %v, want %v", label, got, want)
	}
}

func assertUnique(t *testing.T, label, value, file string, seen map[string]string) {
	t.Helper()
	if previous, exists := seen[value]; exists {
		t.Errorf("duplicate %s %q in %s and %s", label, value, previous, file)
	}
	seen[value] = file
}

func parseTime(t *testing.T, value string) time.Time {
	t.Helper()
	parsed, err := time.Parse(time.RFC3339Nano, value)
	if err != nil {
		t.Fatalf("parse timestamp %q: %v", value, err)
	}
	return parsed
}

func isSHA256(value string) bool {
	if len(value) != 64 || strings.ToLower(value) != value {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func isJSONNull(value json.RawMessage) bool {
	return string(bytes.TrimSpace(value)) == "null"
}

func isEmptyJSONObject(value json.RawMessage) bool {
	return string(bytes.TrimSpace(value)) == "{}"
}

func closeEnough(a, b float64) bool {
	return math.Abs(a-b) <= 1e-12
}
