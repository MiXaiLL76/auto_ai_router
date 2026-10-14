// Package main renders golangci-lint's gocyclo findings as a report.
//
// It reads golangci-lint JSON output (--output.json.path) on stdin and writes
// either a self-contained HTML page or a Markdown summary on stdout:
//
//	golangci-lint run --enable-only gocyclo --output.json.path gocyclo.json ./...
//	go run ./cmd/cycloreport < gocyclo.json > gocyclo.html
//	go run ./cmd/cycloreport -format markdown -top 20 < gocyclo.json
package main

import (
	_ "embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"html/template"
	"io"
	"os"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

//go:embed report.html.tmpl
var reportTemplate string

// gocycloText matches the message golangci-lint's gocyclo linter emits.
var gocycloText = regexp.MustCompile("^cyclomatic complexity (\\d+) of func `([^`]+)` is high \\(> (\\d+)\\)")

type lintOutput struct {
	Issues []struct {
		FromLinter string
		Text       string
		Pos        struct {
			Filename string
			Line     int
		}
	}
}

// Func is one function over the complexity threshold.
type Func struct {
	Name       string
	File       string
	Line       int
	Package    string
	Complexity int
	URL        string
	BarPct     string
}

// Bucket counts findings in one complexity range.
type Bucket struct {
	Label  string
	Min    int
	Max    int
	Count  int
	BarPct string
}

// Package aggregates findings of one directory.
type Package struct {
	Name   string
	Count  int
	Max    int
	BarPct string
}

// Report is everything the templates render.
type Report struct {
	Generated string
	Commit    string
	Threshold int
	Total     int
	Max       int
	Over50    int
	Funcs     []Func
	Buckets   []Bucket
	Packages  []Package
}

func main() {
	format := flag.String("format", "html", "output format: html or markdown")
	repoURL := flag.String("repo-url", "", "base URL for source links, e.g. https://github.com/org/repo/blob/<sha>")
	commit := flag.String("commit", "", "commit shown in the report header")
	top := flag.Int("top", 20, "markdown only: number of functions listed")
	flag.Parse()

	if err := run(os.Stdin, os.Stdout, *format, *repoURL, *commit, *top); err != nil {
		fmt.Fprintln(os.Stderr, "cycloreport:", err)
		os.Exit(1)
	}
}

func run(in io.Reader, out io.Writer, format, repoURL, commit string, top int) error {
	var lint lintOutput
	if err := json.NewDecoder(in).Decode(&lint); err != nil {
		return fmt.Errorf("decode golangci-lint JSON: %w", err)
	}
	report := Report{
		Generated: time.Now().UTC().Format("2006-01-02 15:04 UTC"),
		Commit:    commit,
	}
	for _, issue := range lint.Issues {
		if issue.FromLinter != "gocyclo" {
			continue
		}
		m := gocycloText.FindStringSubmatch(issue.Text)
		if m == nil {
			continue
		}
		complexity, _ := strconv.Atoi(m[1])
		report.Threshold, _ = strconv.Atoi(m[3])
		f := Func{
			Name:       m[2],
			File:       issue.Pos.Filename,
			Line:       issue.Pos.Line,
			Package:    path.Dir(issue.Pos.Filename),
			Complexity: complexity,
		}
		if repoURL != "" {
			f.URL = fmt.Sprintf("%s/%s#L%d", strings.TrimRight(repoURL, "/"), f.File, f.Line)
		}
		report.Funcs = append(report.Funcs, f)
	}
	summarize(&report)

	switch format {
	case "html":
		return renderHTML(out, &report)
	case "markdown":
		return renderMarkdown(out, &report, top)
	default:
		return errors.New("unknown -format " + strconv.Quote(format))
	}
}

func summarize(r *Report) {
	sort.Slice(r.Funcs, func(i, j int) bool {
		if r.Funcs[i].Complexity != r.Funcs[j].Complexity {
			return r.Funcs[i].Complexity > r.Funcs[j].Complexity
		}
		return r.Funcs[i].File+r.Funcs[i].Name < r.Funcs[j].File+r.Funcs[j].Name
	})
	r.Total = len(r.Funcs)
	if r.Total > 0 {
		r.Max = r.Funcs[0].Complexity
	}

	r.Buckets = []Bucket{
		{Label: "16–20", Min: 16, Max: 20},
		{Label: "21–30", Min: 21, Max: 30},
		{Label: "31–50", Min: 31, Max: 50},
		{Label: "51–100", Min: 51, Max: 100},
		{Label: "> 100", Min: 101, Max: int(^uint(0) >> 1)},
	}
	if r.Threshold > 0 {
		r.Buckets[0].Label = fmt.Sprintf("%d–20", r.Threshold+1)
		r.Buckets[0].Min = r.Threshold + 1
	}
	byPkg := map[string]*Package{}
	for i := range r.Funcs {
		f := &r.Funcs[i]
		f.BarPct = pct(f.Complexity, r.Max)
		if f.Complexity > 50 {
			r.Over50++
		}
		for b := range r.Buckets {
			if f.Complexity >= r.Buckets[b].Min && f.Complexity <= r.Buckets[b].Max {
				r.Buckets[b].Count++
			}
		}
		p := byPkg[f.Package]
		if p == nil {
			p = &Package{Name: f.Package}
			byPkg[f.Package] = p
		}
		p.Count++
		p.Max = max(p.Max, f.Complexity)
	}

	maxBucket := 0
	for _, b := range r.Buckets {
		maxBucket = max(maxBucket, b.Count)
	}
	for i := range r.Buckets {
		r.Buckets[i].BarPct = pct(r.Buckets[i].Count, maxBucket)
	}

	for _, p := range byPkg {
		r.Packages = append(r.Packages, *p)
	}
	sort.Slice(r.Packages, func(i, j int) bool {
		if r.Packages[i].Count != r.Packages[j].Count {
			return r.Packages[i].Count > r.Packages[j].Count
		}
		return r.Packages[i].Name < r.Packages[j].Name
	})
	for i := range r.Packages {
		r.Packages[i].BarPct = pct(r.Packages[i].Count, r.Packages[0].Count)
	}
}

func pct(v, of int) string {
	if of == 0 {
		return "0"
	}
	return strconv.FormatFloat(float64(v)*100/float64(of), 'f', 1, 64)
}

func renderHTML(out io.Writer, r *Report) error {
	tmpl, err := template.New("report").Parse(reportTemplate)
	if err != nil {
		return err
	}
	return tmpl.Execute(out, r)
}

func renderMarkdown(out io.Writer, r *Report, top int) error {
	if r.Total == 0 {
		_, err := io.WriteString(out, "### ✅ gocyclo: no functions above the threshold\n")
		return err
	}
	var b strings.Builder
	fmt.Fprintf(&b, "### ❌ gocyclo: %d function(s) above %d\n\n", r.Total, r.Threshold)
	b.WriteString("| Range | Functions |\n|---|---:|\n")
	for _, bk := range r.Buckets {
		fmt.Fprintf(&b, "| %s | %d |\n", bk.Label, bk.Count)
	}
	fmt.Fprintf(&b, "\nTop %d of %d (full list in the `gocyclo-report` artifact):\n\n", min(top, r.Total), r.Total)
	b.WriteString("| Complexity | Function | Location |\n|---:|---|---|\n")
	for _, f := range r.Funcs[:min(top, r.Total)] {
		loc := fmt.Sprintf("`%s:%d`", f.File, f.Line)
		if f.URL != "" {
			loc = fmt.Sprintf("[%s:%d](%s)", f.File, f.Line, f.URL)
		}
		fmt.Fprintf(&b, "| %d | `%s` | %s |\n", f.Complexity, f.Name, loc)
	}
	_, err := io.WriteString(out, b.String())
	return err
}
