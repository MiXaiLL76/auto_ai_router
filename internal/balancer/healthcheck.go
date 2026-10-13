package balancer

// HealthChecker is the account-level liveness source consumed while picking
// credentials: a cached answer to "is this account dead?". The ban reader
// (kafka path) and the no-op checker implement it structurally. Model-scoped
// bans are handled by fail2ban, not by this interface.
type HealthChecker interface {
	IsDead(name string) bool
}
