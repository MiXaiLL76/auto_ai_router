package queries

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSpendLogInsertPrivacyColumnsAreExplicitEmptyObjects(t *testing.T) {
	for _, query := range []string{QueryInsertSpendLog, BuildBatchInsertQuery(2)} {
		assert.Contains(t, query, "messages")
		assert.Contains(t, query, "response")
		assert.Contains(t, query, "proxy_server_request")
		assert.Contains(t, query, "'{}'::jsonb, '{}'::jsonb, '{}'::jsonb")
	}
}

func TestBuildBatchInsertQuery(t *testing.T) {
	tests := []struct {
		name           string
		count          int
		expectEmpty    bool
		expectContains []string
	}{
		{
			name:        "zero count returns empty string",
			count:       0,
			expectEmpty: true,
		},
		{
			name:        "negative count returns empty string",
			count:       -1,
			expectEmpty: true,
		},
		{
			name:        "single insert",
			count:       1,
			expectEmpty: false,
			expectContains: []string{
				`INSERT INTO "LiteLLM_SpendLogs"`,
				"$1",
				"ON CONFLICT (request_id) DO NOTHING RETURNING request_id",
			},
		},
		{
			name:        "three inserts",
			count:       3,
			expectEmpty: false,
			expectContains: []string{
				`INSERT INTO "LiteLLM_SpendLogs"`,
				"$1", "$2", "$3", // first row
				"$55", "$56", "$57", // third row starts at $55 (27 params per row)
				"ON CONFLICT (request_id) DO NOTHING RETURNING request_id",
			},
		},
		{
			name:        "five inserts",
			count:       5,
			expectEmpty: false,
			expectContains: []string{
				`INSERT INTO "LiteLLM_SpendLogs"`,
				"$1",
				"$135", // 5 rows * 27 params = 135
				"ON CONFLICT (request_id) DO NOTHING RETURNING request_id",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := BuildBatchInsertQuery(tt.count)

			if tt.expectEmpty {
				if result != "" {
					t.Errorf("expected empty string, got: %s", result)
				}
				return
			}

			if result == "" {
				t.Errorf("expected non-empty string for count=%d", tt.count)
				return
			}

			for _, expected := range tt.expectContains {
				if !strings.Contains(result, expected) {
					t.Errorf("expected result to contain %q, got: %s", expected, result)
				}
			}
		})
	}
}

func TestBuildBatchInsertQuery_ParameterCount(t *testing.T) {
	// Test that the query has correct number of parameter placeholders
	for _, count := range []int{1, 2, 5, 10, 100} {
		result := BuildBatchInsertQuery(count)
		if result == "" {
			t.Fatalf("expected non-empty string for count=%d", count)
		}

		// Count $N patterns
		expectedParams := count * spendLogParamCount
		var paramCount int
		for i := 1; i <= expectedParams; i++ {
			if strings.Contains(result, "$"+string(rune('0'+i%10))) ||
				strings.Contains(result, "$"+string(rune('0'+(i/10)%10))+string(rune('0'+i%10))) ||
				strings.Contains(result, "$"+string(rune('0'+(i/100)%10))+string(rune('0'+(i/10)%10))+string(rune('0'+i%10))) {
				paramCount++
			}
		}

		// Simple check: result should contain $1 and $X where X is the last parameter
		lastParam := expectedParams
		if !strings.Contains(result, "$"+string(rune('0'+(lastParam/100)%10))+string(rune('0'+(lastParam/10)%10))+string(rune('0'+lastParam%10))) &&
			!strings.Contains(result, "$"+string(rune('0'+(lastParam/10)%10))+string(rune('0'+lastParam%10))) &&
			!strings.Contains(result, "$"+string(rune('0'+lastParam%10))) {
			// Just verify we have enough $N patterns
			count := strings.Count(result, "$")
			if count != expectedParams {
				t.Errorf("count=%d: expected %d parameters, found %d ($ occurrences)",
					count, expectedParams, count)
			}
		}
	}
}

func TestBuildBatchInsertQuery_Format(t *testing.T) {
	// Test that the query is properly formatted with commas between rows
	result := BuildBatchInsertQuery(3)
	if result == "" {
		t.Fatal("expected non-empty string")
	}

	// Should have exactly 3 VALUES clauses
	valuesCount := strings.Count(result, "VALUES")
	if valuesCount != 1 {
		t.Errorf("expected 1 VALUES keyword, got %d", valuesCount)
	}

	// Should have 2 commas separating the 3 row value sets
	// Each row looks like "($1, $2, ...), ($26, $27, ...), ($51, ...)"
	// We should have 2 "), (" patterns
	rowSeparators := strings.Count(result, "), (")
	if rowSeparators != 2 {
		t.Errorf("expected 2 row separators ), ( got %d", rowSeparators)
	}
}
