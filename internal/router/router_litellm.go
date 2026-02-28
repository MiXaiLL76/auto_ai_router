package router

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/users"
)

func (r *Router) handleLitellm(w http.ResponseWriter, req *http.Request) bool {
	switch req.URL.Path {
	case "/litellm/.well-known/litellm-ui-config":
		// auth not need
		// http://localhost:34001/litellm/.well-known/litellm-ui-config
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		if _, err := w.Write([]byte(`{"server_root_path":"/","proxy_base_url":null,"auto_redirect_to_sso":false,"admin_ui_disabled":false}`)); err != nil {
			r.logger.Error("Failed request", "error", err)
			http.Error(w, "Bad Request: invalid JSON", http.StatusBadRequest)
		}
		return true
	case "/v2/login":
		r.handleLogin(w, req)
		return true
	case "/get_image":
		return true
	default:
		return false
	}
}

func (r *Router) handleLogin(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}

	var loginReq users.LoginRequest
	if err := json.NewDecoder(req.Body).Decode(&loginReq); err != nil {
		r.logger.Error("Failed to decode login request", "error", err)
		http.Error(w, "Bad Request: invalid JSON", http.StatusBadRequest)
		return
	}

	masterKey := r.proxy.GetMasterKey()

	// Get DB pool (may be nil if LiteLLM DB is disabled)
	var pool = r.proxy.LiteLLMDB.GetPool()

	result, err := users.AuthenticateUser(req.Context(), loginReq, masterKey, pool)
	if err != nil {
		if err == users.ErrInvalidCredentials {
			r.logger.Warn("Login failed: invalid credentials", "username", loginReq.Username)
			http.Error(w, "Unauthorized: invalid credentials", http.StatusUnauthorized)
			return
		}
		r.logger.Error("Login error", "error", err)
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		return
	}

	// Generate session JWT for cookie (contains the key)
	sessionClaims := &users.SessionClaims{
		UserID:    result.UserID,
		UserRole:  result.UserRole,
		UserEmail: result.UserEmail,
		Key:       result.Key,
		Exp:       time.Now().Add(users.SessionJWTDuration).Unix(),
		Iat:       time.Now().Unix(),
	}

	sessionJWT, err := users.GenerateSessionJWT(sessionClaims, masterKey)
	if err != nil {
		r.logger.Error("Failed to generate session JWT", "error", err)
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		return
	}

	// Set cookies
	maxAge := int(users.SessionJWTDuration.Seconds())

	http.SetCookie(w, &http.Cookie{
		Name:     "session",
		Value:    sessionJWT,
		Path:     "/",
		MaxAge:   maxAge,
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
	})
	http.SetCookie(w, &http.Cookie{
		Name:   "username",
		Value:  loginReq.Username,
		Path:   "/",
		MaxAge: maxAge,
	})
	http.SetCookie(w, &http.Cookie{
		Name:   "authenticated",
		Value:  "true",
		Path:   "/",
		MaxAge: maxAge,
	})

	r.logger.Info("Login successful", "username", loginReq.Username, "role", result.UserRole)

	// Response
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	if _, err := w.Write([]byte(`{"redirect_url":"/ui/?login=success"}`)); err != nil {
		r.logger.Error("Failed request", "error", err)
		http.Error(w, "Bad Request: invalid JSON", http.StatusBadRequest)
	}
}
