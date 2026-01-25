package monitoring

import (
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	RequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "auto_ai_router_requests_total",
			Help: "Total number of requests",
		},
		[]string{"credential", "endpoint", "status"},
	)

	RequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "auto_ai_router_requests_duration_seconds",
			Help:    "Request duration in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"credential", "endpoint"},
	)

	CredentialRPMCurrent = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_credential_rpm_current",
			Help: "Current RPM for each credential",
		},
		[]string{"credential"},
	)

	CredentialTPMCurrent = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_credential_tpm_current",
			Help: "Current TPM (tokens per minute) for each credential",
		},
		[]string{"credential"},
	)

	CredentialBanned = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_credential_banned",
			Help: "Ban status for each credential (1 = banned, 0 = active)",
		},
		[]string{"credential"},
	)

	CredentialErrorsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "auto_ai_router_credential_errors_total",
			Help: "Total number of errors for each credential",
		},
		[]string{"credential"},
	)
)

type Metrics struct {
	enabled bool
}

func New(enabled bool) *Metrics {
	return &Metrics{
		enabled: enabled,
	}
}

func (m *Metrics) RecordRequest(credential, endpoint string, statusCode int, duration time.Duration) {
	if !m.enabled {
		return
	}

	status := strconv.Itoa(statusCode)
	RequestsTotal.WithLabelValues(credential, endpoint, status).Inc()
	RequestDuration.WithLabelValues(credential, endpoint).Observe(duration.Seconds())

	if statusCode != 200 {
		CredentialErrorsTotal.WithLabelValues(credential).Inc()
	}
}

func (m *Metrics) UpdateCredentialRPM(credential string, rpm int) {
	if !m.enabled {
		return
	}
	CredentialRPMCurrent.WithLabelValues(credential).Set(float64(rpm))
}

func (m *Metrics) UpdateCredentialTPM(credential string, tpm int) {
	if !m.enabled {
		return
	}
	CredentialTPMCurrent.WithLabelValues(credential).Set(float64(tpm))
}

func (m *Metrics) UpdateCredentialBanStatus(credential string, banned bool) {
	if !m.enabled {
		return
	}
	value := 0.0
	if banned {
		value = 1.0
	}
	CredentialBanned.WithLabelValues(credential).Set(value)
}
