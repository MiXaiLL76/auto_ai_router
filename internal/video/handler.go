// Package video implements AIR video generation and artifact delivery.
package video

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

type HandlerConfig struct {
	Service          *Service
	Resolver         PrincipalResolver
	UploadSigningKey []byte
}
type Handler struct {
	service  *Service
	resolver PrincipalResolver
	key      []byte
	mux      *http.ServeMux
}

func NewHandler(cfg HandlerConfig) (*Handler, error) {
	if cfg.Service == nil || cfg.Resolver == nil || len(cfg.UploadSigningKey) < 32 {
		return nil, ErrInvalid
	}
	h := &Handler{service: cfg.Service, resolver: cfg.Resolver, key: append([]byte(nil), cfg.UploadSigningKey...), mux: http.NewServeMux()}
	h.mux.HandleFunc("POST /v1/videos", h.create)
	h.mux.HandleFunc("GET /v1/videos/{id}", h.get)
	h.mux.HandleFunc("DELETE /v1/videos/{id}", h.cancel)
	h.mux.HandleFunc("GET /v1/videos/{id}/content", h.content)
	h.mux.HandleFunc("HEAD /v1/videos/{id}/content", h.content)
	h.mux.HandleFunc("POST /v1/media/uploads", h.createUpload)
	h.mux.HandleFunc("PUT /v1/media/uploads/{id}/object", h.putUpload)
	h.mux.HandleFunc("POST /v1/media/uploads/{id}/complete", h.completeUpload)
	return h, nil
}
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) { h.mux.ServeHTTP(w, r) }
func (h *Handler) create(w http.ResponseWriter, r *http.Request) {
	var req CreateRequest
	if err := decode(w, r, &req); err != nil {
		writeDomainError(w, r, err)
		return
	}
	p, err := h.resolver.ResolvePrincipal(w, r, strings.TrimSpace(req.Model))
	if err != nil {
		return
	}
	j, _, err := h.service.Create(r.Context(), p, strings.TrimSpace(r.Header.Get("Idempotency-Key")), req)
	if err != nil {
		writeDomainError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, jobResponse(j, false))
}
func (h *Handler) get(w http.ResponseWriter, r *http.Request) {
	p, err := h.resolve(w, r)
	if err != nil {
		return
	}
	j, err := h.service.Get(r.Context(), p.OrganizationID, r.PathValue("id"))
	if err != nil {
		writeDomainError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, jobResponse(j, true))
}
func (h *Handler) cancel(w http.ResponseWriter, r *http.Request) {
	p, err := h.resolve(w, r)
	if err != nil {
		return
	}
	j, err := h.service.Cancel(r.Context(), p.OrganizationID, r.PathValue("id"))
	if err != nil {
		writeDomainError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, jobResponse(j, true))
}
func (h *Handler) content(w http.ResponseWriter, r *http.Request) {
	p, err := h.resolve(w, r)
	if err != nil {
		return
	}
	o, err := h.service.Content(r.Context(), p.OrganizationID, r.PathValue("id"))
	if err != nil {
		writeDomainError(w, r, err)
		return
	}
	defer func() { _ = o.Body.Close() }()
	w.Header().Set("Content-Type", o.ContentType)
	w.Header().Set("Content-Length", strconv.FormatInt(o.Size, 10))
	w.Header().Set("ETag", `"`+strings.Trim(o.ETag, `"`)+`"`)
	if r.Method == http.MethodHead {
		w.WriteHeader(http.StatusOK)
		return
	}
	if seeker, ok := o.Body.(io.ReadSeeker); ok {
		http.ServeContent(w, r, r.PathValue("id"), time.Time{}, seeker)
		return
	}
	if r.Header.Get("Range") != "" {
		tmp, createErr := os.CreateTemp("", "air-video-content-*")
		if createErr != nil {
			writeDomainError(w, r, createErr)
			return
		}
		name := tmp.Name()
		defer func() { _ = os.Remove(name) }()
		defer func() { _ = tmp.Close() }()
		n, copyErr := io.Copy(tmp, io.LimitReader(o.Body, MaxArtifactBytes+1))
		if copyErr != nil || n > MaxArtifactBytes {
			writeDomainError(w, r, ErrInvalid)
			return
		}
		if _, copyErr = tmp.Seek(0, io.SeekStart); copyErr != nil {
			writeDomainError(w, r, copyErr)
			return
		}
		http.ServeContent(w, r, r.PathValue("id"), time.Time{}, tmp)
		return
	}
	w.WriteHeader(http.StatusOK)
	_, _ = io.Copy(w, o.Body)
}
func (h *Handler) createUpload(w http.ResponseWriter, r *http.Request) {
	p, err := h.resolve(w, r)
	if err != nil {
		return
	}
	var req UploadRequest
	if err = decode(w, r, &req); err != nil {
		writeDomainError(w, r, err)
		return
	}
	u, err := h.service.CreateUpload(r.Context(), p.OrganizationID, req)
	if err != nil {
		writeDomainError(w, r, err)
		return
	}
	writeJSON(w, http.StatusCreated, h.uploadResponse(u))
}
func (h *Handler) putUpload(w http.ResponseWriter, r *http.Request) {
	p, err := h.resolve(w, r)
	if err != nil {
		return
	}
	u, err := h.service.store.GetUpload(r.Context(), p.OrganizationID, r.PathValue("id"))
	if err != nil {
		writeDomainError(w, r, err)
		return
	}
	if !h.validSignature(u, r.URL.Query().Get("signature")) {
		writeDomainError(w, r, ErrAuthentication)
		return
	}
	if err = h.service.PutUpload(r.Context(), p.OrganizationID, u.ID, r.Header.Get("Content-Type"), r.Body); err != nil {
		writeDomainError(w, r, err)
		return
	}
	w.WriteHeader(http.StatusOK)
}
func (h *Handler) completeUpload(w http.ResponseWriter, r *http.Request) {
	p, err := h.resolve(w, r)
	if err != nil {
		return
	}
	u, err := h.service.CompleteUpload(r.Context(), p.OrganizationID, r.PathValue("id"))
	if err != nil {
		writeDomainError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, h.uploadResponse(u))
}
func (h *Handler) resolve(w http.ResponseWriter, r *http.Request) (Principal, error) {
	p, err := h.resolver.ResolvePrincipal(w, r, "")
	if err != nil {
		return Principal{}, err
	}
	if p.OrganizationID == "" {
		writeDomainError(w, r, ErrAuthentication)
		return Principal{}, ErrAuthentication
	}
	return p, nil
}
func (h *Handler) signature(u *Upload) string {
	m := hmac.New(sha256.New, h.key)
	_, _ = io.WriteString(m, u.OrganizationID+"\n"+u.ID+"\n"+u.ObjectKey+"\n"+u.SHA256+"\n"+u.ExpiresAt.UTC().Format("2006-01-02T15:04:05.999999999Z07:00"))
	return hex.EncodeToString(m.Sum(nil))
}
func (h *Handler) validSignature(u *Upload, got string) bool {
	a, err := hex.DecodeString(got)
	if err != nil {
		return false
	}
	b, _ := hex.DecodeString(h.signature(u))
	return hmac.Equal(a, b)
}
func (h *Handler) uploadResponse(u *Upload) map[string]any {
	return map[string]any{"id": u.ID, "object": "media_upload", "purpose": u.Purpose, "status": u.State, "object_key": u.ObjectKey, "upload_url": "/v1/media/uploads/" + u.ID + "/object?signature=" + h.signature(u), "expires_at": u.ExpiresAt.UTC().Format("2006-01-02T15:04:05Z07:00"), "size_bytes": u.SizeBytes, "mime": u.MIME, "sha256": u.SHA256}
}
func jobResponse(j *Job, details bool) map[string]any {
	quality := j.Request.Quality
	if quality == "" {
		quality = "standard"
	}
	size := j.Request.Size
	if size == "" {
		size = "720p"
	}
	out := map[string]any{"id": j.ID, "object": "video", "model": j.Request.Model, "status": j.Status, "progress": progress(j.Status), "created_at": j.CreatedAt.Unix(), "size": size, "seconds": strconv.Itoa(j.Request.DurationSeconds), "quality": quality}
	if details {
		out["updated_at"] = j.UpdatedAt.Unix()
		if j.ErrorCode != "" {
			out["error"] = map[string]string{"code": j.ErrorCode, "message": j.ErrorMessage}
		}
		if j.Status == StatusCompleted {
			out["content"] = map[string]string{"url": "/v1/videos/" + j.ID + "/content"}
		}
	}
	return out
}
func progress(s Status) int {
	if s == StatusQueued {
		return 0
	}
	if s == StatusInProgress {
		return 50
	}
	return 100
}
func decode(w http.ResponseWriter, r *http.Request, out any) error {
	r.Body = http.MaxBytesReader(w, r.Body, MaxRequestBytes)
	d := json.NewDecoder(r.Body)
	d.DisallowUnknownFields()
	if err := d.Decode(out); err != nil {
		return ErrInvalid
	}
	var extra any
	if err := d.Decode(&extra); !errors.Is(err, io.EOF) {
		return ErrInvalid
	}
	return nil
}
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
func writeDomainError(w http.ResponseWriter, r *http.Request, err error) {
	status := http.StatusInternalServerError
	code := "internal_error"
	message := "request failed"
	switch {
	case errors.Is(err, ErrAuthentication):
		status = http.StatusUnauthorized
		code = "authentication_failed"
		message = "authentication required"
	case errors.Is(err, ErrInvalid):
		status = http.StatusBadRequest
		code = "invalid_request"
		message = "invalid request"
	case errors.Is(err, ErrUnsupportedModel):
		status = http.StatusUnprocessableEntity
		code = "unsupported_model"
		message = "model is not supported"
	case errors.Is(err, ErrNotFound):
		status = http.StatusNotFound
		code = "not_found"
		message = "resource not found"
	case errors.Is(err, ErrConflict), errors.Is(err, ErrStateChanged):
		status = http.StatusConflict
		code = "conflict"
		message = "request conflicts with current state"
	case errors.Is(err, ErrNotReady):
		status = http.StatusConflict
		code = "job_not_ready"
		message = "resource is not ready"
	case errors.Is(err, ErrQuota):
		status = http.StatusPaymentRequired
		code = "insufficient_quota"
		message = "usage reservation failed"
	}
	writeJSON(w, status, map[string]any{"error": map[string]string{"code": code, "message": message, "request_id": r.Header.Get("x-request-id")}})
}
