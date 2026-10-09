package balancer

// noopHealthChecker reports every account alive; used as the default so the
// selection path never deals with a nil checker when the feature is off.
type noopHealthChecker struct{}

func NewNoopHealthChecker() *noopHealthChecker {
	return &noopHealthChecker{}
}

// IsDead (HealthChecker): feature disabled — nothing is dead.
func (_ *noopHealthChecker) IsDead(_ string) bool {
	return false
}
