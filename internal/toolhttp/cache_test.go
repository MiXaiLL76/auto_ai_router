package toolhttp

import (
	"net/http"
	"testing"
	"time"
)

// TestTheClientIsReusedForTheSameDestination checks that the same
// destination gets the same client: the connections to the provider survive
// between the requests, so a tool-using request does not pay a fresh
// handshake every time.
func TestTheClientIsReusedForTheSameDestination(t *testing.T) {
	cache := NewCache()
	first := cache.Client("https://api.moonshot.ai", "", 30*time.Second)
	second := cache.Client("https://api.moonshot.ai", "", 30*time.Second)
	if first != second {
		t.Error("the same destination got two clients: the connections are not reused")
	}
}

// TestDifferentDestinationsGetDifferentClients checks that the cache keys
// cover everything that changes the client's behaviour.
func TestDifferentDestinationsGetDifferentClients(t *testing.T) {
	cache := NewCache()
	base := cache.Client("https://api.moonshot.ai", "", 30*time.Second)
	cases := []struct {
		name  string
		base  string
		proxy string
		until time.Duration
	}{
		{name: "another base URL", base: "https://api.deepseek.com", until: 30 * time.Second},
		{name: "another proxy", base: "https://api.moonshot.ai", proxy: "http://proxy:3128", until: 30 * time.Second},
		{name: "another timeout", base: "https://api.moonshot.ai", until: time.Minute},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			other := cache.Client(tc.base, tc.proxy, tc.until)
			if other == base {
				t.Error("a different destination got the same client")
			}
		})
	}
}

// TestTheCacheIsSafeForConcurrentUse checks that the cache is safe when
// several requests build their client at once: they must all get the same
// one, and none must be built twice.
func TestTheCacheIsSafeForConcurrentUse(t *testing.T) {
	cache := NewCache()
	const goroutines = 16
	clients := make(chan *http.Client, goroutines)
	start := make(chan struct{})
	for i := 0; i < goroutines; i++ {
		go func() {
			<-start
			clients <- cache.Client("https://api.moonshot.ai", "", 30*time.Second)
		}()
	}
	close(start)
	first := <-clients
	for i := 1; i < goroutines; i++ {
		if other := <-clients; other != first {
			t.Fatal("the goroutines got different clients")
		}
	}
}
