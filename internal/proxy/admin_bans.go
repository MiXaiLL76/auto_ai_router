package proxy

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
)

// maxAdminBodyBytes caps admin API request bodies; they are tiny JSON objects.
const maxAdminBodyBytes = 64 << 10

// adminBanRequest is the body of POST /api/ban.
type adminBanRequest struct {
	Credential string `json:"credential"`
	// Model is optional: empty or "*" bans every model of the credential.
	Model string `json:"model"`
	// TTL is a Go duration ("30m", "2h"). Omitted or empty bans until /api/unban
	// (or a router restart — bans are kept in memory only).
	TTL    string `json:"ttl"`
	Reason string `json:"reason"`
}

// adminUnbanRequest is the body of POST /api/unban.
type adminUnbanRequest struct {
	Credential string `json:"credential"`
	// Model is optional: empty lifts every ban of the credential, otherwise only
	// the ban on that exact model (use "*" for the credential-wide ban).
	Model string `json:"model"`
}

// AdminBanView is one ban as reported by the admin API.
type AdminBanView struct {
	Credential string `json:"credential"`
	// Model is "*" for a credential-wide ban.
	Model     string    `json:"model"`
	Origin    string    `json:"origin"` // "admin" or "fail2ban"
	Reason    string    `json:"reason,omitempty"`
	ErrorCode int       `json:"error_code"` // 0 for admin bans
	Since     time.Time `json:"since"`
	Permanent bool      `json:"permanent"`
	// Until is omitted for permanent bans.
	Until *time.Time `json:"until,omitempty"`
}

func newAdminBanView(bp fail2ban.BanPair) AdminBanView {
	view := AdminBanView{
		Credential: bp.Credential,
		Model:      bp.Model,
		Origin:     bp.Origin,
		Reason:     bp.Reason,
		ErrorCode:  bp.ErrorCode,
		Since:      bp.BanTime.UTC(),
		Permanent:  bp.BanUntil.IsZero(),
	}
	if !bp.BanUntil.IsZero() {
		until := bp.BanUntil.UTC()
		view.Until = &until
	}
	return view
}

// authorizeAdmin lets only the master key through. LiteLLM DB keys are never
// accepted: they identify end users, not operators. It writes the error
// response itself and reports whether the caller may proceed.
func (p *Proxy) authorizeAdmin(w http.ResponseWriter, r *http.Request) bool {
	token, state := extractClientToken(r)
	switch state {
	case clientCredentialMissing:
		WriteErrorUnauthorized(w, "master key required")
		return false
	case clientCredentialMalformed:
		WriteErrorUnauthorized(w, "malformed credentials")
		return false
	}
	if !p.isMasterKey(token) {
		WriteErrorForbidden(w, "master key required")
		return false
	}
	return true
}

func requireMethod(w http.ResponseWriter, r *http.Request, method string) bool {
	if r.Method == method {
		return true
	}
	w.Header().Set("Allow", method)
	WriteJSONError(w, http.StatusMethodNotAllowed, "Method Not Allowed", errorTypeForStatus(http.StatusMethodNotAllowed), nil, nil)
	return false
}

func decodeAdminBody(w http.ResponseWriter, r *http.Request, dst any) bool {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxAdminBodyBytes))
	// Unknown fields are rejected on purpose: a typo such as "ttl_seconds"
	// would otherwise silently turn a temporary ban into a permanent one.
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			WriteErrorTooLarge(w, "request body too large")
			return false
		}
		if errors.Is(err, io.EOF) {
			WriteErrorBadRequest(w, "request body required")
			return false
		}
		WriteErrorBadRequest(w, "invalid JSON: "+err.Error())
		return false
	}
	return true
}

func writeAdminJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// HandleAdminBan serves POST /api/ban: ban a credential (all models or one)
// for a while, or until it is explicitly unbanned.
func (p *Proxy) HandleAdminBan(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) || !p.authorizeAdmin(w, r) {
		return
	}

	var req adminBanRequest
	if !decodeAdminBody(w, r, &req) {
		return
	}
	if strings.TrimSpace(req.Credential) == "" {
		WriteErrorBadRequest(w, "credential is required")
		return
	}

	var ttl time.Duration
	if strings.TrimSpace(req.TTL) != "" {
		var err error
		ttl, err = time.ParseDuration(strings.TrimSpace(req.TTL))
		if err != nil {
			WriteErrorBadRequest(w, "invalid ttl: use a duration like \"30m\" or omit it to ban until unban")
			return
		}
		if ttl <= 0 {
			WriteErrorBadRequest(w, "ttl must be positive; omit it to ban until unban")
			return
		}
	}

	if !p.balancer.HasCredential(req.Credential) {
		WriteErrorNotFound(w, "unknown credential")
		return
	}

	ban := p.balancer.AdminBan(req.Credential, req.Model, ttl, req.Reason)
	writeAdminJSON(w, http.StatusOK, newAdminBanView(ban))
}

// HandleAdminUnban serves POST /api/unban. It is idempotent: lifting a ban
// that does not exist still returns 200 (with removed=0).
func (p *Proxy) HandleAdminUnban(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) || !p.authorizeAdmin(w, r) {
		return
	}

	var req adminUnbanRequest
	if !decodeAdminBody(w, r, &req) {
		return
	}
	if strings.TrimSpace(req.Credential) == "" {
		WriteErrorBadRequest(w, "credential is required")
		return
	}
	if !p.balancer.HasCredential(req.Credential) {
		WriteErrorNotFound(w, "unknown credential")
		return
	}

	removed := p.balancer.AdminUnban(req.Credential, req.Model)
	writeAdminJSON(w, http.StatusOK, map[string]any{
		"credential": req.Credential,
		"model":      req.Model,
		"removed":    removed,
	})
}

// HandleAdminBans serves GET /api/bans: every active ban with its origin,
// reason and expiry.
func (p *Proxy) HandleAdminBans(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) || !p.authorizeAdmin(w, r) {
		return
	}

	active := p.balancer.GetActiveBans()
	bans := make([]AdminBanView, 0, len(active))
	for _, bp := range active {
		bans = append(bans, newAdminBanView(bp))
	}
	writeAdminJSON(w, http.StatusOK, map[string]any{"bans": bans})
}
