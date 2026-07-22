package openai

import (
	"bytes"
	"io"
	"mime"
	"mime/multipart"
	"net/textproto"
	"strings"
)

// detectImageMIME inspects the first bytes of data and returns the image MIME type.
// Returns ("", false) if the data does not match a known image magic sequence.
func detectImageMIME(data []byte) (string, bool) {
	if len(data) >= 4 && data[0] == 0x89 && data[1] == 'P' && data[2] == 'N' && data[3] == 'G' {
		return "image/png", true
	}
	if len(data) >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF {
		return "image/jpeg", true
	}
	if len(data) >= 12 &&
		data[0] == 'R' && data[1] == 'I' && data[2] == 'F' && data[3] == 'F' &&
		data[8] == 'W' && data[9] == 'E' && data[10] == 'B' && data[11] == 'P' {
		return "image/webp", true
	}
	return "", false
}

// mimeTypeToExt maps image MIME types to file extensions.
func mimeTypeToExt(mimeType string) string {
	switch mimeType {
	case "image/png":
		return ".png"
	case "image/jpeg":
		return ".jpg"
	case "image/webp":
		return ".webp"
	default:
		return ""
	}
}

// isImageFieldName reports whether the multipart field name is an image field
// (e.g. "image", "image[0]", "mask").
func isImageFieldName(name string) bool {
	if name == "image" || name == "mask" {
		return true
	}
	// image[N] pattern
	if strings.HasPrefix(name, "image[") && strings.HasSuffix(name, "]") {
		return true
	}
	return false
}

// collectedPart holds a fully-read multipart part so we can do two passes.
type collectedPart struct {
	header    textproto.MIMEHeader
	fieldName string
	filename  string
	data      []byte
}

// RewriteImageEditMultipart rewrites a multipart/form-data body for /v1/images/edits:
//  1. Replaces the model field with the provider-facing model name.
//  2. Fixes image parts whose Content-Type is "application/octet-stream" by
//     detecting the actual image format from magic bytes and setting the correct
//     Content-Type (image/png, image/jpeg, or image/webp) and filename.
//  3. Strips the "response_format" field when stripResponseFormat is true
//     (used for gpt-image-1 which rejects that parameter).
//  4. Renames multiple "image" fields to "image[]" — when the client sends
//     more than one image using the same field name "image", OpenAI requires
//     the array syntax "image[]" instead.
//
// Returns the rewritten body bytes and the new Content-Type header value
// (multipart/form-data with updated boundary).  On any parse error the original
// body and contentType are returned unchanged so the caller can still forward them.
func RewriteImageEditMultipart(body []byte, contentType, modelID string, stripResponseFormat bool) (newBody []byte, newContentType string) {
	// Parse the boundary from Content-Type.
	mediaType, params, err := mime.ParseMediaType(contentType)
	if err != nil || !strings.HasPrefix(mediaType, "multipart/") {
		return body, contentType
	}
	boundary, ok := params["boundary"]
	if !ok || boundary == "" {
		return body, contentType
	}

	// --- First pass: collect all parts into memory. ---
	reader := multipart.NewReader(bytes.NewReader(body), boundary)
	var parts []collectedPart
	imageCount := 0

	for {
		part, err := reader.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			return body, contentType
		}

		fieldName := part.FormName()
		data, readErr := io.ReadAll(part)
		_ = part.Close()
		if readErr != nil {
			return body, contentType
		}

		if fieldName == "image" {
			imageCount++
		}

		parts = append(parts, collectedPart{
			header:    part.Header,
			fieldName: fieldName,
			filename:  part.FileName(),
			data:      data,
		})
	}

	// --- Second pass: write rewritten parts. ---
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	for _, p := range parts {
		fieldName := p.fieldName
		filename := p.filename
		partData := p.data
		if fieldName == "model" && modelID != "" {
			partData = []byte(modelID)
		}

		// Skip response_format field when requested.
		if stripResponseFormat && fieldName == "response_format" {
			continue
		}

		// Rename "image" → "image[]" when multiple images are present.
		if fieldName == "image" && imageCount > 1 {
			fieldName = "image[]"
		}

		// Fix octet-stream MIME type for image fields.
		partContentType := p.header.Get("Content-Type")
		if isImageFieldName(fieldName) && strings.EqualFold(strings.TrimSpace(partContentType), "application/octet-stream") {
			if detected, ok := detectImageMIME(partData); ok {
				partContentType = detected
				if filename == "" {
					filename = "image" + mimeTypeToExt(detected)
				}
			}
		}

		// Build the new part header.
		h := make(textproto.MIMEHeader)

		// Reconstruct Content-Disposition.
		cd := `form-data; name="` + escapeQuotes(fieldName) + `"`
		if filename != "" {
			cd += `; filename="` + escapeQuotes(filename) + `"`
		}
		h.Set("Content-Disposition", cd)

		if partContentType != "" {
			h.Set("Content-Type", partContentType)
		}

		// Copy any other headers from the original part (except the ones we
		// just handled).
		for key, vals := range p.header {
			canonKey := textproto.CanonicalMIMEHeaderKey(key)
			if canonKey == "Content-Disposition" || canonKey == "Content-Type" {
				continue
			}
			for _, v := range vals {
				h.Add(key, v)
			}
		}

		pw, createErr := writer.CreatePart(h)
		if createErr != nil {
			return body, contentType
		}
		if _, writeErr := pw.Write(partData); writeErr != nil {
			return body, contentType
		}
	}

	if closeErr := writer.Close(); closeErr != nil {
		return body, contentType
	}

	return buf.Bytes(), writer.FormDataContentType()
}

// escapeQuotes escapes double-quote characters in header parameter values.
func escapeQuotes(s string) string {
	return strings.ReplaceAll(s, `"`, `\"`)
}
