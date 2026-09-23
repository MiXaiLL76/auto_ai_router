package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/video"
)

type VideoPrincipalResolver struct {
	proxy *Proxy
}

func NewVideoPrincipalResolver(p *Proxy) *VideoPrincipalResolver {
	return &VideoPrincipalResolver{proxy: p}
}

func (r *VideoPrincipalResolver) ResolvePrincipal(w http.ResponseWriter, req *http.Request, model string) (video.Principal, error) {
	if r == nil || r.proxy == nil {
		WriteErrorServiceUnavailable(w, "Service unavailable")
		return video.Principal{}, errors.New("video admission unavailable")
	}
	tokenInfo, visibility, ok := r.proxy.AuthenticateClientRequestScoped(w, req)
	if !ok {
		return video.Principal{}, errors.New("video authentication failed")
	}
	policy, dangling := r.proxy.OrganizationPolicyForTokenInfo(tokenInfo)
	if dangling {
		WriteErrorForbidden(w, "Forbidden")
		return video.Principal{}, errors.New("video organization unavailable")
	}
	if policy == nil {
		WriteErrorNotFound(w, "Video is not enabled for this organization")
		return video.Principal{}, errors.New("video organization policy missing")
	}

	principal := video.Principal{
		OrganizationID:     policy.OrganizationID,
		APIKeyHash:         tokenInfo.Token,
		UserID:             tokenInfo.UserID,
		TeamID:             tokenInfo.TeamID,
		BillingTeamID:      tokenInfo.TeamID,
		PriceProfileID:     policy.PriceProfileID,
		PriceProfileSHA256: policy.ProfileSHA256,
		Currency:           "USD",
	}
	if model == "" {
		return principal, nil
	}
	if videoAdmissionHasUnsupportedLimits(tokenInfo) {
		WriteErrorServiceUnavailable(w, "Video generation is unavailable for budgeted or rate-limited keys")
		return video.Principal{}, errors.New("video admission limits require durable reservation support")
	}

	resolution, err := r.proxy.modelManager.ResolveOrganizationModelScoped(policy, model, visibility)
	if err != nil {
		WriteErrorNotFound(w, "Model "+model+" not found")
		return video.Principal{}, err
	}
	if !r.proxy.IsOrganizationModelAllowedForToken(tokenInfo, policy, model) {
		WriteErrorForbidden(w, "Model not allowed")
		return video.Principal{}, errors.New("video model not allowed")
	}
	if resolution.ModelPrice == nil || resolution.ModelPrice.OutputCostPerVideoPerSecond <= 0 {
		WriteErrorServiceUnavailable(w, "Model pricing unavailable")
		return video.Principal{}, errors.New("video model price unavailable")
	}
	principal.RatePerSecond = strconv.FormatFloat(resolution.ModelPrice.OutputCostPerVideoPerSecond, 'f', -1, 64)
	return principal, nil
}

func videoAdmissionHasUnsupportedLimits(info *dbmodels.TokenInfo) bool {
	for _, level := range budgetLevels(info) {
		if level.maxBudget != nil || level.rpm != nil || level.tpm != nil {
			return true
		}
	}
	return false
}

type VideoBilling struct {
	committer litellmdb.SpendCommitter
}

func NewVideoBilling(committer litellmdb.SpendCommitter) *VideoBilling {
	return &VideoBilling{committer: committer}
}

func (b *VideoBilling) Reserve(_ context.Context, job *video.Job, _ string) (video.Reservation, error) {
	if job == nil {
		return video.Reservation{}, video.ErrInvalid
	}
	return video.Reservation{
		ID:       job.ID,
		Handle:   job.ID,
		Amount:   job.QuotedAmount,
		Currency: job.Currency,
	}, nil
}

func (b *VideoBilling) Settle(ctx context.Context, job *video.Job, _ string) error {
	if job == nil || b == nil || b.committer == nil {
		return video.ErrInvalid
	}
	spend, err := strconv.ParseFloat(job.QuotedAmount, 64)
	if err != nil || spend < 0 {
		return video.ErrInvalid
	}
	metadata, err := json.Marshal(map[string]any{
		"air_event_id":                      job.ID,
		"requested_model":                   job.Request.Model,
		"video_duration_seconds":            job.Request.DurationSeconds,
		"output_cost_per_video_second":      job.Principal.RatePerSecond,
		"organization_price_profile_id":     job.Principal.PriceProfileID,
		"organization_price_profile_sha256": job.Principal.PriceProfileSHA256,
	})
	if err != nil {
		return err
	}
	end := time.Now().UTC()
	_, err = b.committer.CommitSpend(ctx, &dbmodels.SpendLogEntry{
		RequestID:         job.ID,
		AirEventID:        job.ID,
		StartTime:         job.CreatedAt,
		EndTime:           end,
		RequestDurationMS: int(end.Sub(job.CreatedAt).Milliseconds()),
		CallType:          "avideo_generation",
		APIBase:           "/v1/videos",
		Model:             job.Request.Model,
		ModelID:           "runway:" + job.Request.Model,
		ModelGroup:        job.Request.Model,
		CustomLLMProvider: "runway",
		Metadata:          string(metadata),
		Spend:             spend,
		APIKey:            job.Principal.APIKeyHash,
		UserID:            job.Principal.UserID,
		TeamID:            job.Principal.TeamID,
		BillingTeamID:     job.Principal.BillingTeamID,
		OrganizationID:    job.Principal.OrganizationID,
		Status:            "success",
	})
	return err
}

func (b *VideoBilling) Release(context.Context, *video.Job, string) error {
	return nil
}
