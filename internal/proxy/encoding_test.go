package proxy

import (
	"bytes"
	"compress/flate"
	"compress/gzip"
	"io"
	"testing"
)

func TestParseAcceptEncoding(t *testing.T) {
	tests := []struct {
		name     string
		header   string
		expected []AcceptedEncoding
	}{
		{
			name:   "single encoding",
			header: "gzip",
			expected: []AcceptedEncoding{
				{Encoding: "gzip", Quality: 1.0},
			},
		},
		{
			name:   "multiple encodings",
			header: "gzip, deflate",
			expected: []AcceptedEncoding{
				{Encoding: "gzip", Quality: 1.0},
				{Encoding: "deflate", Quality: 1.0},
			},
		},
		{
			name:   "with quality values",
			header: "deflate, gzip;q=1.0, *;q=0.5",
			expected: []AcceptedEncoding{
				{Encoding: "deflate", Quality: 1.0},
				{Encoding: "gzip", Quality: 1.0},
				{Encoding: "*", Quality: 0.5},
			},
		},
		{
			name:   "quality values with spaces",
			header: "gzip; q=0.8, deflate; q=0.5",
			expected: []AcceptedEncoding{
				{Encoding: "gzip", Quality: 0.8},
				{Encoding: "deflate", Quality: 0.5},
			},
		},
		{
			name:   "wildcard",
			header: "*",
			expected: []AcceptedEncoding{
				{Encoding: "*", Quality: 1.0},
			},
		},
		{
			name:   "quality zero (not preferred)",
			header: "gzip, deflate;q=0",
			expected: []AcceptedEncoding{
				{Encoding: "gzip", Quality: 1.0},
				{Encoding: "deflate", Quality: 0.0},
			},
		},
		{
			name:     "empty header",
			header:   "",
			expected: []AcceptedEncoding{},
		},
		{
			name:   "case insensitive",
			header: "GZIP, Deflate",
			expected: []AcceptedEncoding{
				{Encoding: "gzip", Quality: 1.0},
				{Encoding: "deflate", Quality: 1.0},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ParseAcceptEncoding(tt.header)
			if len(result) != len(tt.expected) {
				t.Errorf("got %d encodings, want %d", len(result), len(tt.expected))
				return
			}
			for i, enc := range result {
				if enc.Encoding != tt.expected[i].Encoding {
					t.Errorf("encoding %d: got %s, want %s", i, enc.Encoding, tt.expected[i].Encoding)
				}
				if enc.Quality != tt.expected[i].Quality {
					t.Errorf("quality %d: got %f, want %f", i, enc.Quality, tt.expected[i].Quality)
				}
			}
		})
	}
}

func TestSelectBestEncoding(t *testing.T) {
	tests := []struct {
		name      string
		encodings []AcceptedEncoding
		expected  string
	}{
		{
			name:      "prefer gzip",
			encodings: []AcceptedEncoding{{Encoding: "gzip", Quality: 1.0}},
			expected:  "gzip",
		},
		{
			name:      "prefer deflate when gzip not available",
			encodings: []AcceptedEncoding{{Encoding: "deflate", Quality: 1.0}},
			expected:  "deflate",
		},
		{
			name: "choose highest quality",
			encodings: []AcceptedEncoding{
				{Encoding: "deflate", Quality: 0.8},
				{Encoding: "gzip", Quality: 0.9},
			},
			expected: "gzip",
		},
		{
			name: "wildcard returns gzip",
			encodings: []AcceptedEncoding{
				{Encoding: "*", Quality: 1.0},
			},
			expected: "gzip",
		},
		{
			name: "skip zero quality",
			encodings: []AcceptedEncoding{
				{Encoding: "gzip", Quality: 0.0},
				{Encoding: "deflate", Quality: 1.0},
			},
			expected: "deflate",
		},
		{
			name: "unsupported encoding fallback to identity",
			encodings: []AcceptedEncoding{
				{Encoding: "br", Quality: 1.0},
				{Encoding: "zstd", Quality: 0.9},
			},
			expected: "identity",
		},
		{
			name:      "empty list returns identity",
			encodings: []AcceptedEncoding{},
			expected:  "identity",
		},
		{
			name: "identity preferred",
			encodings: []AcceptedEncoding{
				{Encoding: "identity", Quality: 1.0},
			},
			expected: "identity",
		},
		{
			name: "wildcard with gzip excluded returns deflate",
			encodings: []AcceptedEncoding{
				{Encoding: "*", Quality: 1.0},
				{Encoding: "gzip", Quality: 0.0},
			},
			expected: "deflate",
		},
		{
			name: "wildcard with gzip and deflate excluded returns identity",
			encodings: []AcceptedEncoding{
				{Encoding: "*", Quality: 1.0},
				{Encoding: "gzip", Quality: 0.0},
				{Encoding: "deflate", Quality: 0.0},
			},
			expected: "identity",
		},
		{
			name: "wildcard with zero quality returns identity",
			encodings: []AcceptedEncoding{
				{Encoding: "*", Quality: 0.0},
			},
			expected: "identity",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := SelectBestEncoding(tt.encodings)
			if result != tt.expected {
				t.Errorf("got %s, want %s", result, tt.expected)
			}
		})
	}
}

func TestCompressBody(t *testing.T) {
	testData := []byte("Hello, World! This is test data for compression.")

	tests := []struct {
		name     string
		encoding string
		verify   func(t *testing.T, compressed []byte, encoding string)
	}{
		{
			name:     "gzip compression",
			encoding: "gzip",
			verify: func(t *testing.T, compressed []byte, encoding string) {
				if encoding != "gzip" {
					t.Errorf("got encoding %s, want gzip", encoding)
				}
				// Verify it's valid gzip
				reader, err := gzip.NewReader(bytes.NewReader(compressed))
				if err != nil {
					t.Fatalf("failed to create gzip reader: %v", err)
				}
				decompressed, err := io.ReadAll(reader)
				if err != nil {
					t.Fatalf("failed to decompress: %v", err)
				}
				if !bytes.Equal(decompressed, testData) {
					t.Errorf("decompressed data doesn't match original")
				}
			},
		},
		{
			name:     "deflate compression",
			encoding: "deflate",
			verify: func(t *testing.T, compressed []byte, encoding string) {
				if encoding != "deflate" {
					t.Errorf("got encoding %s, want deflate", encoding)
				}
				// Verify it's valid deflate
				reader := flate.NewReader(bytes.NewReader(compressed))
				defer func() { _ = reader.Close() }()
				decompressed, err := io.ReadAll(reader)
				if err != nil {
					t.Fatalf("failed to decompress: %v", err)
				}
				if !bytes.Equal(decompressed, testData) {
					t.Errorf("decompressed data doesn't match original")
				}
			},
		},
		{
			name:     "identity (no compression)",
			encoding: "identity",
			verify: func(t *testing.T, compressed []byte, encoding string) {
				if encoding != "identity" {
					t.Errorf("got encoding %s, want identity", encoding)
				}
				if !bytes.Equal(compressed, testData) {
					t.Errorf("data should not be compressed with identity")
				}
			},
		},
		{
			name:     "empty encoding defaults to identity",
			encoding: "",
			verify: func(t *testing.T, compressed []byte, encoding string) {
				if encoding != "identity" {
					t.Errorf("got encoding %s, want identity", encoding)
				}
				if !bytes.Equal(compressed, testData) {
					t.Errorf("data should not be compressed with empty encoding")
				}
			},
		},
		{
			name:     "unsupported encoding defaults to identity",
			encoding: "br",
			verify: func(t *testing.T, compressed []byte, encoding string) {
				if encoding != "identity" {
					t.Errorf("got encoding %s, want identity", encoding)
				}
				if !bytes.Equal(compressed, testData) {
					t.Errorf("data should not be compressed with unsupported encoding")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			compressed, encoding, err := CompressBody(testData, tt.encoding)
			if err != nil {
				t.Fatalf("compression failed: %v", err)
			}
			tt.verify(t, compressed, encoding)
		})
	}
}

func TestCompressBodyEmptyInput(t *testing.T) {
	emptyData := []byte{}

	compressed, encoding, err := CompressBody(emptyData, "gzip")
	if err != nil {
		t.Fatalf("compression failed: %v", err)
	}

	// Empty input should still produce valid gzip output
	reader, err := gzip.NewReader(bytes.NewReader(compressed))
	if err != nil {
		t.Fatalf("failed to create gzip reader: %v", err)
	}
	decompressed, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("failed to decompress: %v", err)
	}

	if len(decompressed) != 0 {
		t.Errorf("expected empty decompressed data, got %d bytes", len(decompressed))
	}
	if encoding != "gzip" {
		t.Errorf("got encoding %s, want gzip", encoding)
	}
}
