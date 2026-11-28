package proxy

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"
	"testing/iotest"
)

// readRequestBody must return what io.ReadAll(io.LimitReader(body, limit))
// returns for every body and every Content-Length a client can declare.
func TestReadRequestBodyMatchesReadAll(t *testing.T) {
	payload := strings.Repeat("0123456789abcdef", 4096) // 64 KiB
	for _, tc := range []struct {
		name          string
		body          string
		contentLength int64
		limit         int64
	}{
		{"exact length", payload, int64(len(payload)), 1 << 20},
		{"unknown length", payload, -1, 1 << 20},
		{"zero length header", "", 0, 1 << 20},
		{"empty body with unknown length", "", -1, 1 << 20},
		{"declared longer than sent", payload, int64(len(payload)) * 4, 1 << 20},
		{"declared shorter than sent", payload, 10, 1 << 20},
		{"over the limit", payload, int64(len(payload)), 1000},
		{"exactly the limit", payload[:1000], 1000, 1000},
		{"one byte", "x", 1, 1 << 20},
		{"declared above the presize cap", payload, maxPresizedRequestBody * 4, maxPresizedRequestBody * 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want, wantErr := io.ReadAll(io.LimitReader(strings.NewReader(tc.body), tc.limit))
			got, gotErr := readRequestBody(strings.NewReader(tc.body), tc.contentLength, tc.limit)
			if !bytes.Equal(got, want) || !errors.Is(gotErr, wantErr) {
				t.Fatalf("got %d bytes, err %v; io.ReadAll gives %d bytes, err %v", len(got), gotErr, len(want), wantErr)
			}
			if got == nil {
				t.Fatal("io.ReadAll never returns a nil slice; neither may readRequestBody")
			}
			// Same bytes when the transport delivers them one at a time.
			slow, slowErr := readRequestBody(iotest.OneByteReader(strings.NewReader(tc.body)), tc.contentLength, tc.limit)
			if !bytes.Equal(slow, want) || slowErr != nil {
				t.Fatalf("one-byte reads: got %d bytes, err %v", len(slow), slowErr)
			}
		})
	}
}

func TestReadRequestBodyReturnsReadErrorWithPartialData(t *testing.T) {
	boom := errors.New("connection reset")
	src := io.MultiReader(strings.NewReader("partial"), iotest.ErrReader(boom))
	want, wantErr := io.ReadAll(io.LimitReader(io.MultiReader(strings.NewReader("partial"), iotest.ErrReader(boom)), 100))
	got, err := readRequestBody(src, 50, 100)
	if !errors.Is(err, boom) || !errors.Is(wantErr, boom) || !bytes.Equal(got, want) {
		t.Fatalf("got %q, err %v; io.ReadAll gives %q, err %v", got, err, want, wantErr)
	}
}

func TestReadRequestBodyDoesNotTrustHugeContentLength(t *testing.T) {
	got, err := readRequestBody(strings.NewReader("tiny"), 1<<40, 1<<41)
	if err != nil || string(got) != "tiny" {
		t.Fatalf("got %q, err %v", got, err)
	}
	if cap(got) > maxPresizedRequestBody+bytes.MinRead {
		t.Fatalf("reserved %d bytes for a declared length, cap is %d", cap(got), maxPresizedRequestBody)
	}
}

func TestReadRequestBodyAllocatesOnce(t *testing.T) {
	payload := strings.Repeat("x", 1<<20)
	allocs := testing.AllocsPerRun(20, func() {
		body, err := readRequestBody(strings.NewReader(payload), int64(len(payload)), 35<<20)
		if err != nil || len(body) != len(payload) {
			t.Fatal("short read")
		}
	})
	// The body buffer, the bytes.Buffer header, the strings.Reader and the LimitReader.
	if allocs > 4 {
		t.Fatalf("%v allocations for a body of known length", allocs)
	}
}
