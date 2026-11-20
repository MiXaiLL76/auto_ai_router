package proxy

import (
	"fmt"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/deepseektools"
)

// deepSeekToolsEnabledFor reports whether the router should run the search
// tools itself for this request.
//
// It is an explicit opt-in on the credential (deepseek_tools), not a guess
// from the base URL: the router must not start calling a provider's
// endpoints because its address happens to look like DeepSeek's.
func deepSeekToolsEnabledFor(modelID string, cred *config.CredentialConfig) bool {
	if cred == nil || !cred.DeepSeekTools {
		return false
	}
	return isDeepSeekModel(modelID)
}

// deepSeekAccountKey names the DeepSeek account the calls belong to, for the
// per-account QPS pools.
//
// The account is the credential's name: DeepSeek counts its limits per
// account, and one credential here is one account. The key is namespaced by
// the credential type so that two providers' credentials of the same name
// cannot share a pool.
func deepSeekAccountKey(cred *config.CredentialConfig) string {
	if cred == nil {
		return ""
	}
	return string(cred.Type) + ":" + cred.Name
}

// describeToolPlan renders the plan for a log line.
func describeToolPlan(plan *deepseektools.ToolPlan) string {
	if plan == nil {
		return "none"
	}
	return fmt.Sprintf("search=%v fetch=%v", plan.NeedsSearch, plan.NeedsFetch)
}
