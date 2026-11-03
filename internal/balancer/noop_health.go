package balancer

// NoopHealthChecker reports every account alive; used as the default so the
// selection path never deals with a nil checker when the feature is off.
type NoopHealthChecker struct{}

func NewNoopHealthChecker() *NoopHealthChecker {
	return &NoopHealthChecker{}
}

// IsDead (HealthChecker): feature disabled — nothing is dead.
func (n *NoopHealthChecker) IsDead(_ string) bool {
	return false
}
