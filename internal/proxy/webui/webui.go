// Package webui serves the embedded web UI.
package webui

import (
	_ "embed"
	"fmt"
	"html/template"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/security"
)

//go:embed style.css
var styleCSS []byte

//go:embed health.html
var healthHTML []byte

//go:embed trace.html
var traceHTML []byte

//go:embed config.html
var configHTML string

// ConfigTemplate is the parsed server config dashboard template.
var ConfigTemplate *template.Template

func init() {
	ConfigTemplate = template.Must(
		template.New("config").Funcs(template.FuncMap{
			"fmtBool": func(b bool) string {
				if b {
					return "yes"
				}
				return "no"
			},
			"fmtLimit": func(n int) string {
				if n <= 0 {
					return "unlimited"
				}
				return strconv.Itoa(n)
			},
			"fmtDur": func(d time.Duration) string {
				if d == 0 {
					return "—"
				}
				return d.String()
			},
			"join": func(ss []string) string {
				return strings.Join(ss, ", ")
			},
			"joinInts": func(ns []int) string {
				parts := make([]string, len(ns))
				for i, n := range ns {
					parts[i] = strconv.Itoa(n)
				}
				return strings.Join(parts, ", ")
			},
			"credClass": func(t any) string {
				switch fmt.Sprint(t) {
				case "openai":
					return "ok"
				case "vertex-ai", "gemini", "proxy":
					return "info"
				case "anthropic", "bedrock":
					return "warn"
				default:
					return ""
				}
			},
			"slice7": func(s string) string {
				if len(s) <= 7 {
					return s
				}
				return s[:7]
			},
			"maskAPIKey": func(s string) string {
				if s == "" {
					return "not set"
				}
				return security.MaskAPIKey(s)
			},
			"maskURL": func(raw string) string {
				if raw == "" {
					return ""
				}
				return "***"
			},
			"maskDBURL": func(raw string) string {
				if raw == "" {
					return ""
				}
				return "***"
			},
		}).Parse(configHTML),
	)
}

func ServeCSS(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/css; charset=utf-8")
	w.Header().Set("Cache-Control", "public, max-age=3600")
	_, _ = w.Write(styleCSS)
}

func ServeHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write(healthHTML)
}

func ServeTrace(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write(traceHTML)
}
