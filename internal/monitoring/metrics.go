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
			Buckets: []float64{1, 10, 30, 60, 120, 240, 600},
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

	ModelRPMCurrent = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_model_rpm_current",
			Help: "Current RPM for each model within a credential",
		},
		[]string{"credential", "model"},
	)

	ModelTPMCurrent = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "auto_ai_router_model_tpm_current",
			Help: "Current TPM (tokens per minute) for each model within a credential",
		},
		[]string{"credential", "model"},
	)

	CredentialSelectionRejected = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "auto_ai_router_credential_selection_rejected_total",
			Help: "Total number of times a credential was rejected during selection",
		},
		[]string{"reason"},
	)

	CredentialBanEvents = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "auto_ai_router_credential_ban_events_total",
			Help: "Total number of ban events for credentials",
		},
		[]string{"credential", "error_code"},
	)

	CredentialUnbanEvents = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "auto_ai_router_credential_unban_events_total",
			Help: "Total number of unban events for credentials",
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

func (m *Metrics) isEnabled() bool {
	return m.enabled
}

// updateCredentialMetric updates a credential-level gauge metric
func (m *Metrics) updateCredentialMetric(gauge *prometheus.GaugeVec, credential string, value int) {
	if !m.isEnabled() {
		return
	}
	gauge.WithLabelValues(credential).Set(float64(value))
}

// updateModelMetric updates a model-level gauge metric
func (m *Metrics) updateModelMetric(gauge *prometheus.GaugeVec, credential, model string, value int) {
	if !m.isEnabled() {
		return
	}
	gauge.WithLabelValues(credential, model).Set(float64(value))
}

func (m *Metrics) RecordRequest(credential, endpoint string, statusCode int, duration time.Duration) {
	if !m.isEnabled() {
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
	m.updateCredentialMetric(CredentialRPMCurrent, credential, rpm)
}

func (m *Metrics) UpdateCredentialTPM(credential string, tpm int) {
	m.updateCredentialMetric(CredentialTPMCurrent, credential, tpm)
}

func (m *Metrics) UpdateCredentialBanStatus(credential string, banned bool) {
	if !m.isEnabled() {
		return
	}
	value := 0.0
	if banned {
		value = 1.0
	}
	CredentialBanned.WithLabelValues(credential).Set(value)
}

func (m *Metrics) UpdateModelRPM(credential, model string, rpm int) {
	m.updateModelMetric(ModelRPMCurrent, credential, model, rpm)
}

func (m *Metrics) UpdateModelTPM(credential, model string, tpm int) {
	m.updateModelMetric(ModelTPMCurrent, credential, model, tpm)
}
