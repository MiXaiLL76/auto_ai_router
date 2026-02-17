package proxy

import (
	"bytes"
	"compress/flate"
	"compress/gzip"
	"sort"
	"strconv"
	"strings"
)

// AcceptedEncoding represents a single encoding from Accept-Encoding header
type AcceptedEncoding struct {
	Encoding string  // gzip, deflate, br, zstd, identity, *
	Quality  float64 // 0.0 to 1.0, default 1.0
}

// ParseAcceptEncoding parses the Accept-Encoding header and returns a sorted list
// of accepted encodings by quality (highest first)
// Examples:
//
//	"gzip"
//	"gzip, deflate"
//	"deflate, gzip;q=1.0, *;q=0.5"
func ParseAcceptEncoding(header string) []AcceptedEncoding {
	if header == "" {
		return []AcceptedEncoding{}
	}

	var encodings []AcceptedEncoding
	parts := strings.Split(header, ",")

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		// Split encoding and quality value
		tokens := strings.Split(part, ";")
		encoding := strings.TrimSpace(tokens[0])
		quality := 1.0 // Default quality

		// Parse quality value if present
		if len(tokens) > 1 {
			for _, token := range tokens[1:] {
				token = strings.TrimSpace(token)
				if strings.HasPrefix(token, "q=") {
					qStr := strings.TrimPrefix(token, "q=")
					if q, err := strconv.ParseFloat(qStr, 64); err == nil {
						if q < 0 {
							q = 0
						} else if q > 1 {
							q = 1
						}
						quality = q
					}
				}
			}
		}

		encodings = append(encodings, AcceptedEncoding{
			Encoding: strings.ToLower(encoding),
			Quality:  quality,
		})
	}

	// Sort by quality (descending), then by specificity (exact encodings before *)
	sort.Slice(encodings, func(i, j int) bool {
		if encodings[i].Quality != encodings[j].Quality {
			return encodings[i].Quality > encodings[j].Quality
		}
		// If quality is equal, prefer specific encodings over wildcard
		if encodings[i].Encoding == "*" && encodings[j].Encoding != "*" {
			return false
		}
		if encodings[i].Encoding != "*" && encodings[j].Encoding == "*" {
			return true
		}
		return false
	})

	return encodings
}

// SelectBestEncoding selects the best encoding that we support
// Supported encodings: gzip, deflate, identity (no compression)
// Returns empty string if no compatible encoding found (shouldn't happen with proper Accept-Encoding)
func SelectBestEncoding(acceptedEncodings []AcceptedEncoding) string {
	supported := map[string]bool{
		"gzip":     true,
		"deflate":  true,
		"identity": true,
	}

	// Sort by quality (highest first) to find the best match
	sortedEncodings := make([]AcceptedEncoding, len(acceptedEncodings))
	copy(sortedEncodings, acceptedEncodings)
	sort.Slice(sortedEncodings, func(i, j int) bool {
		if sortedEncodings[i].Quality != sortedEncodings[j].Quality {
			return sortedEncodings[i].Quality > sortedEncodings[j].Quality
		}
		// If quality is equal, prefer specific encodings over wildcard
		if sortedEncodings[i].Encoding == "*" && sortedEncodings[j].Encoding != "*" {
			return false
		}
		if sortedEncodings[i].Encoding != "*" && sortedEncodings[j].Encoding == "*" {
			return true
		}
		return false
	})

	// Find first supported encoding
	for _, enc := range sortedEncodings {
		if enc.Quality == 0 {
			continue // Skip encodings with quality 0
		}

		if enc.Encoding == "*" {
			// Wildcard - return best supported (prefer gzip)
			return "gzip"
		}

		if supported[enc.Encoding] {
			return enc.Encoding
		}
	}

	// No compatible encoding found, return identity (no compression)
	return "identity"
}

// CompressBody compresses the body according to the specified encoding
// Returns the compressed body and the encoding used (or "identity" if no compression)
func CompressBody(body []byte, encoding string) ([]byte, string, error) {
	encoding = strings.ToLower(encoding)

	switch encoding {
	case "gzip":
		var buf bytes.Buffer
		gzWriter := gzip.NewWriter(&buf)
		if _, err := gzWriter.Write(body); err != nil {
			return nil, "", err
		}
		if err := gzWriter.Close(); err != nil {
			return nil, "", err
		}
		return buf.Bytes(), "gzip", nil

	case "deflate":
		var buf bytes.Buffer
		flateWriter, err := flate.NewWriter(&buf, flate.DefaultCompression)
		if err != nil {
			return nil, "", err
		}
		if _, err := flateWriter.Write(body); err != nil {
			return nil, "", err
		}
		if err := flateWriter.Close(); err != nil {
			return nil, "", err
		}
		return buf.Bytes(), "deflate", nil

	case "identity", "":
		// No compression
		return body, "identity", nil

	default:
		// Unsupported encoding - return uncompressed
		return body, "identity", nil
	}
}
