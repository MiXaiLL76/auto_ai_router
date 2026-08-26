// Command healthcheck is a Docker HEALTHCHECK probe for the distroless
// image: distroless/static has no shell and no wget/curl, so the Dockerfile
// can't shell out to check /health like the old alpine image did. This is
// a tiny stdlib-only HTTP GET instead, exit 0 on 2xx, exit 1 otherwise.
package main

import (
	"flag"
	"fmt"
	"net/http"
	"os"
	"time"
)

func main() {
	url := flag.String("url", "http://localhost:8080/health", "health endpoint to probe")
	timeout := flag.Duration("timeout", 3*time.Second, "request timeout")
	flag.Parse()

	client := &http.Client{Timeout: *timeout}
	resp, err := client.Get(*url)
	if err != nil {
		fmt.Fprintln(os.Stderr, "healthcheck: ", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		fmt.Fprintln(os.Stderr, "healthcheck: unhealthy status", resp.StatusCode)
		os.Exit(1)
	}
}
