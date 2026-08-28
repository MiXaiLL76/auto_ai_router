// Command healthcheck is a Docker HEALTHCHECK probe for the distroless
// image: distroless/static has no shell and no wget/curl, so the Dockerfile
// can't shell out to check /health like the old alpine image did. This is
// a tiny stdlib-only HTTP GET instead, exit 0 on 2xx, exit 1 otherwise.
package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"time"
)

func main() {
	os.Exit(run())
}

// defaultURL builds the probe target from the environment.
//
// The Dockerfile's HEALTHCHECK runs this binary with no arguments (there is no
// shell in distroless to expand anything into argv), so a hardcoded :8080
// would leave any deployment that overrides server.port permanently unhealthy.
// Docker/Kubernetes do pass the container's env to HEALTHCHECK, so the port —
// or the whole URL, for anything more exotic — comes in that way instead:
//
//	HEALTHCHECK_URL   full URL, wins over everything (e.g. a non-loopback host)
//	HEALTHCHECK_PORT  port only; the path is /health, fixed in config
//
// An explicit -url flag still overrides both.
func defaultURL() string {
	if u := os.Getenv("HEALTHCHECK_URL"); u != "" {
		return u
	}
	port := os.Getenv("HEALTHCHECK_PORT")
	if port == "" {
		port = "8080"
	}
	return "http://localhost:" + port + "/health"
}

// run keeps os.Exit out of the function that holds the deferred Close: Exit
// never runs deferred calls, so calling it directly here would skip closing
// resp.Body on the unhealthy-status path.
func run() int {
	url := flag.String("url", defaultURL(), "health endpoint to probe")
	timeout := flag.Duration("timeout", 3*time.Second, "request timeout")
	flag.Parse()

	client := &http.Client{Timeout: *timeout}
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, *url, nil)
	if err != nil {
		fmt.Fprintln(os.Stderr, "healthcheck: ", err)
		return 1
	}
	resp, err := client.Do(req)
	if err != nil {
		fmt.Fprintln(os.Stderr, "healthcheck: ", err)
		return 1
	}
	defer func() {
		if cerr := resp.Body.Close(); cerr != nil {
			fmt.Fprintln(os.Stderr, "healthcheck: closing response body: ", cerr)
		}
	}()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		fmt.Fprintln(os.Stderr, "healthcheck: unhealthy status", resp.StatusCode)
		return 1
	}
	return 0
}
