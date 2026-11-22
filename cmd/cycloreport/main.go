// Command cycloreport prints gocyclo findings from golangci-lint JSON.
package main

import (
	"cmp"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"regexp"
	"slices"
	"strconv"
	"text/tabwriter"
)

var diagnostic = regexp.MustCompile("^cyclomatic complexity ([0-9]+) of func `(.+)` is high ")

type issue struct {
	FromLinter string
	Text       string
	Pos        struct {
		Filename string
		Line     int
	}
}

type finding struct {
	complexity int
	function   string
	file       string
	line       int
}

func main() {
	top := flag.Int("top", 25, "maximum number of findings to show")
	flag.Parse()
	log.SetFlags(0)
	if *top < 1 {
		log.Fatal("cycloreport: -top must be positive")
	}
	if err := printReport(os.Stdin, os.Stdout, *top); err != nil {
		log.Fatalf("cycloreport: %v", err)
	}
}

func printReport(input io.Reader, output io.Writer, top int) error {
	var report struct{ Issues []issue }
	if err := json.NewDecoder(input).Decode(&report); err != nil {
		return fmt.Errorf("read golangci-lint JSON: %w", err)
	}

	var findings []finding
	for _, item := range report.Issues {
		if item.FromLinter != "gocyclo" {
			continue
		}
		match := diagnostic.FindStringSubmatch(item.Text)
		if match == nil {
			return fmt.Errorf("unrecognized gocyclo finding: %q", item.Text)
		}
		complexity, err := strconv.Atoi(match[1])
		if err != nil {
			return err
		}
		findings = append(findings, finding{complexity, match[2], item.Pos.Filename, item.Pos.Line})
	}

	slices.SortFunc(findings, func(a, b finding) int {
		return cmp.Or(cmp.Compare(b.complexity, a.complexity), cmp.Compare(a.file, b.file),
			cmp.Compare(a.line, b.line), cmp.Compare(a.function, b.function))
	})
	if len(findings) == 0 {
		_, err := fmt.Fprintln(output, "No gocyclo findings.")
		return err
	}
	writer := tabwriter.NewWriter(output, 0, 0, 2, ' ', 0)
	if _, err := fmt.Fprintln(writer, "COMPLEXITY\tFUNCTION\tLOCATION"); err != nil {
		return err
	}
	for _, item := range findings[:min(top, len(findings))] {
		if _, err := fmt.Fprintf(writer, "%d\t%s\t%s:%d\n", item.complexity, item.function, item.file, item.line); err != nil {
			return err
		}
	}
	return writer.Flush()
}
